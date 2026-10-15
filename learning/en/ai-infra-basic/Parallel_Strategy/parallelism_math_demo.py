#!/usr/bin/env python3
"""CPU math lab / CPU 数学实验: no PyTorch, devices, or real collectives.

Run from the repository root:
    python3 learning/zh/ai-infra-basic/Parallel_Strategy/parallelism_math_demo.py

Assertions compare independent decompositions with unsharded references.
This teaches semantics; it is not SGLang's runtime or a performance benchmark.
"""
import math


def close(a, b):
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), (a, b)
        for x, y in zip(a, b):
            close(x, y)
    else:
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def attention(scores, values, width=2):
    """Return normalized output and natural-log LSE, including empty shards."""
    assert len(scores) == len(values)
    if not scores:
        return [0.0] * width, -math.inf
    maximum = max(scores)
    weights = [math.exp(s - maximum) for s in scores]
    denominator = sum(weights)
    output = [
        sum(w * v[d] for w, v in zip(weights, values)) / denominator
        for d in range(width)
    ]
    return output, maximum + math.log(denominator)


def merge_attention(parts):
    """Exactly merge normalized outputs, using LSE instead of sum/average."""
    valid = [(out, lse) for out, lse in parts if lse != -math.inf]
    if not valid:
        raise ValueError("No visible KV: an all-masked query has no softmax.")
    maximum = max(lse for _, lse in valid)
    weights = [math.exp(lse - maximum) for _, lse in valid]
    return [
        sum(w * out[d] for w, (out, _) in zip(weights, valid)) / sum(weights)
        for d in range(len(valid[0][0]))
    ]


def rms_norm(row, epsilon=1e-6):
    scale = math.sqrt(sum(x * x for x in row) / len(row) + epsilon)
    return [x / scale for x in row]


def demo_sp():
    # Two partial row-parallel GEMM outputs. Add contributions, then normalize.
    partials = [
        [[float((rank + 1) * (t + 1)), 2.0 - rank, t - rank, 0.5]
         for t in range(5)]
        for rank in range(2)
    ]
    full = [[sum(p[t][d] for p in partials) for d in range(4)] for t in range(5)]
    reference = [rms_norm(row) for row in full]
    # Simulate reduce-scatter over row blocks with padding to six rows.
    padded = [p + [[0.0] * 4] for p in partials]
    shards = [
        [[sum(p[t][d] for p in padded) for d in range(4)]
         for t in range(rank * 3, (rank + 1) * 3)]
        for rank in range(2)
    ]
    gathered = [rms_norm(row) for shard in shards for row in shard][:5]
    close(gathered, reference)
    wrong = [sum(rms_norm(p[0])[d] for p in partials) for d in range(4)]
    assert any(abs(a - b) > 0.01 for a, b in zip(wrong, reference[0]))
    print("SP: reduce-scatter -> row RMSNorm -> all-gather matches full RMSNorm.")


def zigzag_indices(length, degree):
    # Split into 2C chunks, placing the remainder into early chunks.
    quotient, remainder = divmod(length, 2 * degree)
    chunks, start = [], 0
    for block in range(2 * degree):
        size = quotient + (block < remainder)
        chunks.append(list(range(start, start + size)))
        start += size
    return [chunks[r] + chunks[2 * degree - 1 - r] for r in range(degree)]


def interleave_indices(lengths, degree):
    # Flatten NEW tokens across requests; modulo does not restart per request.
    flat = [(request, pos) for request, n in enumerate(lengths) for pos in range(n)]
    return [flat[r::degree] for r in range(degree)]


def request_attention(request, new_pos, prefix):
    # Distinct request values expose accidental cross-request attention.
    query_position = prefix + new_pos
    scores = [((query_position + 1) * (j + 2) % 11 - 5) / 3
              for j in range(query_position + 1)]
    values = [[100.0 * request + j, float(j * j)]
              for j in range(query_position + 1)]
    return attention(scores, values)[0]


def demo_pcp():
    lengths, prefixes, degree = [5, 3], [2, 0], 2
    reference = {(req, p): request_attention(req, p, prefixes[req])
                 for req, n in enumerate(lengths) for p in range(n)}
    zigzag = [[] for _ in range(degree)]
    for req, n in enumerate(lengths):
        for rank, positions in enumerate(zigzag_indices(n, degree)):
            zigzag[rank].extend((req, p) for p in positions)
    interleave = interleave_indices(lengths, degree)
    assert interleave[0] == [(0, 0), (0, 2), (0, 4), (1, 1)]
    # One packed KV array holds multiple requests plus their cached prefixes.
    # Local queries must filter BOTH request id and original causal position.
    packed_kv = [(req, j, [100.0 * req + j, float(j * j)])
                 for req, n in enumerate(lengths)
                 for j in range(prefixes[req] + n)]
    for name, assignment in [("zigzag", zigzag), ("interleave", interleave)]:
        restored = {}
        for shard in assignment:
            for req, pos in shard:
                assert (req, pos) not in restored
                original_pos = prefixes[req] + pos
                visible = [(j, value) for kv_req, j, value in packed_kv
                           if kv_req == req and j <= original_pos]
                scores = [((original_pos + 1) * (j + 2) % 11 - 5) / 3
                          for j, _ in visible]
                restored[req, pos] = attention(scores, [v for _, v in visible])[0]
        assert restored.keys() == reference.keys()
        for key in reference:
            close(restored[key], reference[key])
        print("PCP", name, "request/position assignments:", assignment)
    for n in range(1, 18):
        for c in range(1, 5):
            assigned = zigzag_indices(n, c)
            assert sorted(p for shard in assigned for p in shard) == list(range(n))
    contiguous = [sum(p + 1 for p in shard) for shard in [range(4), range(4, 8)]]
    balanced = [sum(p + 1 for p in shard) for shard in zigzag_indices(8, 2)]
    assert contiguous == [10, 26] and balanced == [18, 18]
    print("PCP causal score counts: contiguous", contiguous, "zigzag", balanced)


def demo_dcp():
    count = 0
    for length in [1, 2, 7, 17]:
        for degree in [1, 2, 4]:
            # Includes empty shards and scores that overflow a naive exp().
            for offset in [-1000.0, 0.0, 1000.0]:
                scores = [offset + ((j * 7) % 13 - 6) for j in range(length)]
                values = [[float(j), float(1 - 2 * j)] for j in range(length)]
                reference = attention(scores, values)[0]
                parts = [attention(scores[r::degree], values[r::degree])
                         for r in range(degree)]
                close(merge_attention(parts), reference)
                count += 1
    example = [([2.0, 0.0], 0.0), ([10.0, 0.0], math.log(3.0))]
    close(merge_attention(example), [8.0, 0.0])
    assert sum(out[0] for out, _ in example) == 12.0
    try:
        merge_attention([([0.0, 0.0], -math.inf)])
    except ValueError:
        pass
    else:
        raise AssertionError("All-empty partitions must not silently normalize.")
    for degree in [1, 2, 4]:
        for virtual_slot in range(23):
            owner, local_slot = virtual_slot % degree, virtual_slot // degree
            assert local_slot * degree + owner == virtual_slot
    print("DCP:", count, "stable merge cases passed; LSE merge=8, sum=12, mean=6.")


def demo_ep_dwdp():
    # Experts f_e(x)=w_e*x. Two source ranks own separate requests.
    weights = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
    tokens = {"A": (0, 2.0, [(0, 0.25), (2, 0.75)]),
              "B": (1, 3.0, [(2, 0.5), (3, 0.5)])}
    expert_owner = {e: e // 2 for e in weights}
    resident = [{e: w for e, w in weights.items() if expert_owner[e] == r}
                for r in range(2)]
    # EP: dispatch token contributions to resident expert owners.
    dispatched = [[], []]
    for token_id, (_, x, routes) in tokens.items():
        for expert, probability in routes:
            dispatched[expert_owner[expert]].append((token_id, x, expert, probability))
    ep = {token_id: 0.0 for token_id in tokens}
    for rank, items in enumerate(dispatched):
        for token_id, x, expert, probability in items:
            ep[token_id] += probability * resident[rank][expert] * x
    # DWDP teaching analogue: copy planned remote layer weights first;
    # tokens stay at source owners. This is not a VMM/stream implementation.
    layer_cache = [dict(shard) for shard in resident]
    for rank in range(2):
        for remote in range(2):
            if remote != rank:
                layer_cache[rank].update(resident[remote])
    dwdp = {}
    for token_id, (rank, x, routes) in tokens.items():
        dwdp[token_id] = sum(p * layer_cache[rank][e] * x for e, p in routes)
        close(dwdp[token_id], ep[token_id])
    close([ep["A"], ep["B"]], [5.0, 10.5])
    # One logical expert chooses one of its identical physical copies.
    copies = {"slot_a": weights[2], "slot_b": weights[2]}
    close(0.75 * copies["slot_b"] * 2.0, 0.75 * weights[2] * 2.0)
    remote_bytes, hidden, element_bytes, top_k, remote_fraction = (
        96 * 1024 * 1024, 4096, 2, 8, 0.75)
    crossover = remote_bytes / (2 * top_k * remote_fraction * hidden * element_bytes)
    close(crossover, 1024.0)
    print("EP token dispatch == DWDP weight prefetch:", ep,
          "; simplified traffic crossover tokens =", int(crossover))


def demo_topology_pd_encoder():
    dpa, cp, atp = 2, 2, 2
    grid = {(d, c, a): (d * cp + c) * atp + a
            for d in range(dpa) for c in range(cp) for a in range(atp)}
    assert sorted(grid.values()) == list(range(8))
    # A separate example: DCP is a subgroup of existing Attention TP ranks.
    dpa_groups = [list(range(4)), list(range(4, 8))]
    dcp_groups = [group[start:start + 2] for group in dpa_groups for start in [0, 2]]
    assert sorted(r for group in dcp_groups for r in group) == list(range(8))
    assert all(any(set(g).issubset(set(d)) for d in dpa_groups) for g in dcp_groups)
    # Pure GQA head remap: P-TP=2 -> D-TP=4, eight KV heads.
    p_cache = [{h: ("head", h) for h in range(r * 4, (r + 1) * 4)}
               for r in range(2)]
    d_cache = [{h: p_cache[h // 4][h] for h in range(r * 2, (r + 1) * 2)}
               for r in range(4)]
    restored = {h: value for shard in d_cache for h, value in shard.items()}
    assert restored == {h: ("head", h) for h in range(8)}
    # Greedy encoder assignment by estimated work, then gather by original id.
    costs, assignments, loads = [9, 8, 1, 1], [[], []], [0, 0]
    for i in sorted(range(len(costs)), key=lambda i: -costs[i]):
        rank = min(range(2), key=lambda r: loads[r])
        assignments[rank].append(i)
        loads[rank] += costs[i]
    assert loads == [10, 9]
    gathered = [(i, "embedding_" + str(i)) for shard in assignments for i in shard]
    assert [v for _, v in sorted(gathered)] == ["embedding_" + str(i) for i in range(4)]
    print("Topology: 8 ranks, nested DCP", dcp_groups,
          "; P/D heads restored; encoder loads", loads)


def demo_ulysses_cfg():
    # [token][head][one scalar feature]; no causal mask in this DiT example.
    sequence, heads, degree = 4, 2, 2
    q = [[0.2 * (t + h + 1) for h in range(heads)] for t in range(sequence)]
    k = [[0.3 * (t - h) for h in range(heads)] for t in range(sequence)]
    v = [[float(10 * h + t) for h in range(heads)] for t in range(sequence)]

    def attend_head(h):
        return [attention([q[t][h] * k[j][h] for j in range(sequence)],
                          [[v[j][h]] for j in range(sequence)], width=1)[0][0]
                for t in range(sequence)]

    reference_heads = [attend_head(h) for h in range(heads)]
    token_shards = [list(range(r * 2, (r + 1) * 2)) for r in range(degree)]
    # Input AllToAll: each destination receives its heads from every source.
    exchanged = []
    for destination in range(degree):
        h = destination
        received = [(t, q[t][h], k[t][h], v[t][h])
                    for source in range(degree) for t in token_shards[source]]
        assert [item[0] for item in received] == list(range(sequence))
        exchanged.append([
            attention([query * key for _, _, key, _ in received],
                      [[value] for _, _, _, value in received], width=1)[0][0]
            for _, query, _, _ in received
        ])
    close(exchanged, reference_heads)
    # Inverse AllToAll: head owners send output rows back to token owners.
    restored = [[[exchanged[h][t] for h in range(heads)] for t in shard]
                for shard in token_shards]
    close([row for shard in restored for row in shard],
          [[reference_heads[h][t] for h in range(heads)] for t in range(sequence)])
    close(2.0 + 3.0 * (5.0 - 2.0), 11.0)
    print("Diffusion: Ulysses exchange/attention/inverse matches full MHA; CFG=11.")


if __name__ == "__main__":
    demo_sp()
    demo_pcp()
    demo_dcp()
    demo_ep_dwdp()
    demo_topology_pd_encoder()
    demo_ulysses_cfg()
    print("All math assertions passed. No hardware performance claims.")
