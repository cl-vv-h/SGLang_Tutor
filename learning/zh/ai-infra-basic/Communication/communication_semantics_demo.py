"""Single-process communication semantics; no network or hardware benchmark.

单进程通信语义演示；不调用真实通信库，不测量网络或加速卡性能。
Run from the repository root with Python 3; only the standard library is used.
"""

from math import isclose


def all_gather(shards):
    """Return the concatenated vector seen by every participating rank."""
    return [value for shard in shards for value in shard]


def all_reduce_sum(inputs):
    """Return the elementwise sum seen by every participating rank."""
    if not inputs or any(len(row) != len(inputs[0]) for row in inputs):
        raise ValueError("AllReduce requires nonempty, equal-width rank inputs")
    return [sum(column) for column in zip(*inputs)]


def reduce_scatter_sum(inputs):
    """Equal contiguous output shards, one per rank."""
    reduced = all_reduce_sum(inputs)
    ranks = len(inputs)
    if len(reduced) % ranks:
        raise ValueError("This example requires equal-sized output shards")
    width = len(reduced) // ranks
    return [reduced[r * width : (r + 1) * width] for r in range(ranks)]


def all_to_all_v(inputs, counts):
    """inputs[src] is packed in destination order; outputs use source order."""
    ranks = len(inputs)
    if len(counts) != ranks or any(len(row) != ranks for row in counts):
        raise ValueError("counts must be a square rank-by-rank matrix")
    if any(c < 0 for row in counts for c in row):
        raise ValueError("Counts cannot be negative")
    outputs = [[] for _ in range(ranks)]
    for src, row in enumerate(counts):
        if sum(row) != len(inputs[src]):
            raise ValueError("Send splits must cover the input exactly")
        offset = 0
        for dst, count in enumerate(row):
            outputs[dst].extend(inputs[src][offset : offset + count])
            offset += count
    return outputs


def moe_example():
    """Two source ranks, four experts, top-2; retain inverse routing metadata."""
    owners = [0, 0, 1, 1]
    # token: (source rank, scalar hidden value, [(expert, weight), ...])
    tokens = {
        "A": (0, 2.0, [(0, 0.25), (2, 0.75)]),
        "B": (1, 3.0, [(2, 0.50), (3, 0.50)]),
    }
    inboxes = [[], []]
    remote_assignments = 0
    for token, (src, x, routes) in tokens.items():
        for expert, weight in routes:
            dst = owners[expert]
            inboxes[dst].append((src, token, expert, weight, x))
            remote_assignments += int(src != dst)

    # A deliberately simple expert function: expert_e(x) = (e+1)*x.
    replies = [[], []]
    for inbox in inboxes:
        for src, token, expert, weight, x in inbox:
            replies[src].append((token, weight * (expert + 1) * x))
    restored = {token: 0.0 for token in tokens}
    for reply in replies:
        for token, contribution in reply:
            restored[token] += contribution
    reference = {
        token: sum(weight * (expert + 1) * x for expert, weight in routes)
        for token, (_, x, routes) in tokens.items()
    }
    assert all(isclose(restored[t], reference[t]) for t in tokens)
    assert remote_assignments == 1
    assert restored == {"A": 5.0, "B": 10.5}
    print("MoE destination assignment counts:", [len(x) for x in inboxes])
    print("MoE remote dispatch assignments:", remote_assignments)
    print("MoE restored weighted outputs:", restored)


def ring_all_reduce_us(ranks, payload_bytes, alpha_us, bandwidth_gb_per_s):
    """Teaching estimate: bandwidth_gb_per_s means decimal GB/s, not Gbit/s."""
    if ranks < 1 or payload_bytes < 0 or alpha_us < 0 or bandwidth_gb_per_s <= 0:
        raise ValueError("Invalid cost-model parameters")
    sent_bytes = 2 * (ranks - 1) / ranks * payload_bytes
    return 2 * (ranks - 1) * alpha_us + sent_bytes / (bandwidth_gb_per_s * 1e9) * 1e6


def main():
    inputs = [[1, 2, 3, 4], [10, 20, 30, 40]]
    reduced = all_reduce_sum(inputs)
    shards = reduce_scatter_sum(inputs)
    assert reduced == [11, 22, 33, 44]
    assert shards == [[11, 22], [33, 44]]
    assert all_gather(shards) == reduced
    print("AllGather of original inputs:", all_gather(inputs))
    print("AllReduce sum:", reduced)
    print("ReduceScatter sum:", shards)
    print("AllGather after ReduceScatter:", all_gather(shards))

    counts = [[1, 2], [3, 0]]
    exchanged = all_to_all_v([[0, 1, 2], [10, 11, 12]], counts)
    assert exchanged == [[0, 10, 11, 12], [1, 2]]
    assert [len(row) for row in exchanged] == [sum(col) for col in zip(*counts)]
    print("AllToAllV outputs by rank:", exchanged)
    moe_example()

    for size in (16 * 1024**2, 4 * 1024):
        estimate = ring_all_reduce_us(8, size, 3, 50)
        print(f"Estimated ring AllReduce ({size} bytes/rank): {estimate:.2f} us")
    assert ring_all_reduce_us(1, 1024, 3, 50) == 0
    kv_bytes = 2 * 32 * 8192 * 8 * 128 * 2
    assert kv_bytes == 1024**3
    print(f"GQA KV payload: {kv_bytes} bytes; ideal at 50 GB/s: {kv_bytes / 50e9 * 1e3:.2f} ms")
    print("Semantic checks passed. Timings are estimates, not measurements.")


if __name__ == "__main__":
    main()
