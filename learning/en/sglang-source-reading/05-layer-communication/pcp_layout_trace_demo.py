#!/usr/bin/env python3
"""Trace PCP metadata and row layouts / 跟踪 PCP metadata 与行布局.

Standard-library simulation of the pinned chapter's layout rules. This does
NOT import SGLang strategies or run device kernels/collectives.
"""
import math


def zigzag_trace(lengths, prefixes, degree, verbose=False):
    assert len(lengths) == len(prefixes)
    blocks_per_request = 2 * degree
    blocks, sizes = [], []
    for request, length in enumerate(lengths):
        base, extra = divmod(length, blocks_per_request)
        block_sizes = [base + (b < extra) for b in range(blocks_per_request)]
        sizes.append(block_sizes)
        start = 0
        for size in block_sizes:
            blocks.append([(request, i) for i in range(start, start + size)])
            start += size

    rank_block_ids = [
        [b * blocks_per_request + rank for b in range(len(lengths))]
        + [b * blocks_per_request + blocks_per_request - 1 - rank
           for b in range(len(lengths))]
        for rank in range(degree)
    ]
    rank_rows = [[row for block_id in ids for row in blocks[block_id]]
                 for ids in rank_block_ids]
    gathered_block_ids = [block for ids in rank_block_ids for block in ids]
    inverse = [gathered_block_ids.index(b) for b in range(len(blocks))]
    reverse_lengths = [len(blocks[b]) for b in gathered_block_ids]
    logical = [len(rows) for rows in rank_rows]
    physical = math.ceil(max(logical) / blocks_per_request) * blocks_per_request

    # Simulate _all_gather_reorganized: crop physical rows to logical lengths,
    # pad each participant to the maximum logical length for this collective,
    # gather, then remove collective padding.
    local_physical = [rows + [None] * (physical - len(rows)) for rows in rank_rows]
    collective_len = max(logical)
    collective_inputs = [
        padded[:n] + [None] * (collective_len - n)
        for padded, n in zip(local_physical, logical)
    ]
    rank_major = [row for buf, n in zip(collective_inputs, logical) for row in buf[:n]]
    chunks, cursor = [], 0
    for size in reverse_lengths:
        chunks.append(rank_major[cursor:cursor + size])
        cursor += size
    restored = [row for b in inverse for row in chunks[b]]
    expected = [(req, i) for req, n in enumerate(lengths) for i in range(n)]
    assert restored == expected
    assert all(row is not None for row in restored)

    fields = []
    outputs = []
    for rank in range(degree):
        early = [s[rank] for s in sizes]
        late = [s[blocks_per_request - 1 - rank] for s in sizes]
        kv_early = [prefix + sum(s[:rank + 1])
                    for prefix, s in zip(prefixes, sizes)]
        kv_late = [prefix + sum(s[:blocks_per_request - rank])
                   for prefix, s in zip(prefixes, sizes)]
        metadata = dict(
            zigzag_index=rank_block_ids[rank],
            actual_seq_q_prev_list=early,
            actual_seq_q_next_list=late,
            kv_len_prev_list=kv_early,
            kv_len_next_list=kv_late,
            total_q_prev_tokens=sum(early),
        )
        fields.append(metadata)
        rank_outputs = []
        for group, counts, kv_lens in [
            (0, early, kv_early), (1, late, kv_late)
        ]:
            for req, (q_count, kv_len) in enumerate(zip(counts, kv_lens)):
                block_id = rank_block_ids[rank][group * len(lengths) + req]
                for local_row, (request, new_index) in enumerate(blocks[block_id]):
                    last_key = kv_len - q_count + local_row
                    assert request == req
                    assert last_key == prefixes[req] + new_index
                    # Equal scores: independent full-history reference.
                    value = sum(2 * (j + 1) for j in range(last_key + 1)) / (
                        last_key + 1
                    )
                    reference = prefixes[req] + new_index + 2.0
                    assert math.isclose(value, reference)
                    rank_outputs.append(value)
        assert len(rank_outputs) == logical[rank]
        outputs.append(rank_outputs)

    if verbose:
        print("lengths =", lengths, "prefixes =", prefixes, "CP =", degree)
        print("split_list =", [n for request_sizes in sizes for n in request_sizes])
        print("reverse_split_len =", reverse_lengths)
        print("cp_reverse_index =", inverse)
        print("per_rank_logical_token =", logical)
        print("per_rank_actual_token AFTER padding =", [physical] * degree)
        print("gather collective rows per rank =", collective_len)
        for rank in range(degree):
            print("rank", rank, "metadata:", fields[rank])
            print("rank", rank, "query rows:", rank_rows[rank])
            print("rank", rank, "outputs:", outputs[rank])
        print("Rank-major gathered rows:", rank_major)
        print("Restored original rows:", restored)
    return fields, inverse, logical, physical


def interleave_trace(lengths, prefixes, degree, verbose=False):
    flat = [(req, i) for req, n in enumerate(lengths) for i in range(n)]
    legal_lengths = [prefixes[req] + i + 1 for req, i in flat]
    rank_rows = [flat[rank::degree] for rank in range(degree)]
    rank_legal = [legal_lengths[rank::degree] for rank in range(degree)]
    counts = [[sum(req == b for req, _ in rows) for b in range(len(lengths))]
              for rows in rank_rows]
    logical = [len(rows) for rows in rank_rows]
    physical = math.ceil(max(logical) / degree) * degree
    padded = [rows + [None] * (physical - len(rows)) for rows in rank_rows]
    gathered = [row for rows in padded for row in rows]
    inverse = [(f % degree) * physical + f // degree for f in range(len(flat))]
    restored = [gathered[index] for index in inverse]
    assert restored == flat
    for rows, valid in zip(rank_rows, rank_legal):
        assert valid == [prefixes[req] + i + 1 for req, i in rows]
    if verbose:
        print("lengths =", lengths, "prefixes =", prefixes, "CP =", degree)
        print("full seqlens_expanded =", legal_lengths)
        for rank in range(degree):
            print("rank", rank, "queries:", rank_rows[rank],
                  "query counts:", counts[rank], "KV lengths:", rank_legal[rank])
        print("Rank-major with padding:", gathered)
        print("gather indices =", inverse)
        print("Restored original rows:", restored)
    return counts, rank_legal, inverse


if __name__ == "__main__":
    print("1. Zigzag example: prefix 2, eight new tokens, CP2")
    fields, inverse, logical, physical = zigzag_trace([8], [2], 2, verbose=True)
    assert inverse == [0, 2, 3, 1]
    assert fields[0]["kv_len_prev_list"] == [4]
    assert fields[0]["kv_len_next_list"] == [10]
    assert fields[1]["kv_len_prev_list"] == [6]
    assert fields[1]["kv_len_next_list"] == [8]

    print("\n2. Zigzag padding: nine new tokens")
    _, _, logical, physical = zigzag_trace([9], [0], 2, verbose=True)
    assert logical == [5, 4] and physical == 8

    print("\n3. Multiple requests: early group is not always half the local tensor")
    fields, _, _, _ = zigzag_trace([9, 7], [2, 1], 2, verbose=True)
    assert fields[0]["total_q_prev_tokens"] == 5
    assert fields[0]["actual_seq_q_next_list"] == [2, 1]

    print("\n4. Interleave: request lengths 5/3, prefix lengths 2/1")
    counts, legal, _ = interleave_trace([5, 3], [2, 1], 2, verbose=True)
    assert counts == [[3, 1], [2, 2]]
    assert legal == [[3, 5, 7, 3], [4, 6, 2, 4]]

    print("\n5. Interleave gather with one padding row")
    _, _, inverse = interleave_trace([7], [0], 2, verbose=True)
    assert inverse == [0, 4, 1, 5, 2, 6, 3]

    cases = 0
    for degree in [2, 3]:
        for length in range(2 * degree, 2 * degree + 4):
            for prefix in [0, 2]:
                zigzag_trace([length], [prefix], degree)
                interleave_trace([length], [prefix], degree)
                cases += 1
    print(f"\nPASS: all chapter values plus {cases} additional layouts.")
    print("Simulation only: no SGLang class, distributed collective, or device kernel ran.")
