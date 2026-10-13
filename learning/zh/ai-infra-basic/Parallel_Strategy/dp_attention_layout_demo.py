"""CPU-only DP Attention layout simulator; no PyTorch or distributed runtime.

中文：模拟 token 所有权与求和，不能用于衡量 NCCL/HCCL 性能。
Run directly from either bilingual tutorial directory.
"""

from itertools import accumulate
from math import isclose, sqrt


def zeros(rows, width):
    return [[0.0] * width for _ in range(rows)]


def add_matrices(matrices, rows, width):
    result = zeros(rows, width)
    for matrix in matrices:
        assert len(matrix) == rows
        for i, row in enumerate(matrix):
            assert len(row) == width
            for j, value in enumerate(row):
                result[i][j] += value
    return result


def rank_coordinates(rank, world_size, dp_size, cp_size=1):
    assert world_size % (dp_size * cp_size) == 0
    tp_size = world_size // (dp_size * cp_size)
    return (
        rank // (tp_size * cp_size),
        (rank // tp_size) % cp_size,
        rank % tp_size,
    )


def align_counts(counts, tp_size, cp_size=1):
    # Match the two sequential alignments in this source snapshot.
    def align(value, unit):
        return (value + unit - 1) // unit * unit

    return [align(align(n, tp_size), cp_size * 2) for n in counts]


def padding_mode(counts, is_extend=False):
    if is_extend and len(counts) > 1:
        return "SUM_LEN"
    return (
        "MAX_LEN"
        if 2 * sum(counts) >= len(counts) * max(counts)
        else "SUM_LEN"
    )


def gather_sum_len(rank_inputs, counts, tp_size, width, partial):
    # Each rank contributes to its DP interval of a zero global buffer.
    total = sum(counts)
    starts = [0] + list(accumulate(counts))[:-1]
    contributions = []
    for rank, local in enumerate(rank_inputs):
        dp_rank, tp_rank = divmod(rank, tp_size)
        assert len(local) == counts[dp_rank]
        contribution = zeros(total, width)
        if partial or tp_rank == 0:
            for i, row in enumerate(local):
                contribution[starts[dp_rank] + i] = list(row)
        contributions.append(contribution)
    return add_matrices(contributions, total, width)


def gather_max_len(rank_inputs, counts, tp_size, width, partial):
    # Model attn-TP reduce-scatter, then world all-gather, CP=1.
    maximum = max(counts)
    assert maximum % tp_size == 0
    chunk = maximum // tp_size
    world_pieces = []
    for dp_rank, count in enumerate(counts):
        padded_group = []
        for tp_rank in range(tp_size):
            local = rank_inputs[dp_rank * tp_size + tp_rank]
            assert len(local) == count
            if partial or tp_rank == 0:
                padded = [list(row) for row in local] + zeros(maximum - count, width)
            else:
                padded = zeros(maximum, width)
            padded_group.append(padded)
        reduced = add_matrices(padded_group, maximum, width)
        # Each subgroup rank gets one reduced row block.
        for tp_rank in range(tp_size):
            world_pieces.extend(reduced[tp_rank * chunk : (tp_rank + 1) * chunk])
    return world_pieces


def strip_max_padding(global_rows, counts):
    maximum = max(counts)
    return [
        row
        for dp_rank, count in enumerate(counts)
        for row in global_rows[dp_rank * maximum : dp_rank * maximum + count]
    ]


def local_scatter(global_rows, lengths, dp_rank):
    # dp_scatter is local slicing/copying, not a network collective.
    start = sum(lengths[:dp_rank])
    return global_rows[start : start + lengths[dp_rank]]


def global_reduce_scatter_then_attn_gather(rank_partials, dp_size, tp_size, width):
    # Model dp_reduce_scatter_tensor: world RS -> subgroup AG when T>1.
    world_size = dp_size * tp_size
    total = len(rank_partials[0])
    assert total % world_size == 0
    reduced = add_matrices(rank_partials, total, width)
    chunk = total // world_size
    rank_chunks = [
        reduced[r * chunk : (r + 1) * chunk] for r in range(world_size)
    ]
    outputs = []
    for rank in range(world_size):
        dp_rank = rank // tp_size
        local = [
            row
            for piece in rank_chunks[dp_rank * tp_size : (dp_rank + 1) * tp_size]
            for row in piece
        ]
        outputs.append(local)
    return reduced, outputs


def check_close(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert len(a) == len(b)
        assert all(isclose(x, y, rel_tol=1e-10, abs_tol=1e-10) for x, y in zip(a, b))


def rmsnorm(row):
    denominator = sqrt(sum(x * x for x in row) / len(row) + 1e-6)
    return [x / denominator for x in row]


def run():
    print("Rank topology: P=4, D=2, C=1, T=2")
    for rank in range(4):
        print(f"  rank{rank}: (DP,CP,TP)={rank_coordinates(rank, 4, 2)}")
    assert rank_coordinates(6, 8, 2, 2) == (1, 1, 0)

    width, tp_size = 2, 2
    # Each local row has two hidden channels, with distinct rank contributions.
    for counts in ([2, 2], [4, 2], [4, 0], [0, 2]):
        for partial in (True, False):
            inputs = []
            expected = []
            for dp_rank, count in enumerate(counts):
                full = [[10.0 * (dp_rank + 1) + i, 100.0 + i] for i in range(count)]
                expected.extend(full)
                for tp_rank in range(tp_size):
                    # Partial values add to full; replicas each equal full.
                    factor = (0.25 if tp_rank == 0 else 0.75) if partial else 1.0
                    inputs.append([[factor * value for value in row] for row in full])
            summed = gather_sum_len(inputs, counts, tp_size, width, partial)
            max_padded = gather_max_len(inputs, counts, tp_size, width, partial)
            check_close(summed, expected)
            check_close(strip_max_padding(max_padded, counts), expected)
            for dp_rank, count in enumerate(counts):
                check_close(
                    local_scatter(max_padded, [max(counts)] * len(counts), dp_rank)[:count],
                    local_scatter(summed, counts, dp_rank),
                )
            print(f"  counts={counts}, partial={partial}: SUM/MAX agree")

    # Repeated complete values must contribute once per attention subgroup.
    replicas = [[[10.0, 20.0]], [[10.0, 20.0]]]
    wrong = gather_sum_len(replicas, [1], 2, 2, partial=True)
    correct = gather_sum_len(replicas, [1], 2, 2, partial=False)
    check_close(wrong, [[20.0, 40.0]])
    check_close(correct, [[10.0, 20.0]])
    print(f"Replica deduplication: wrong={wrong}, correct={correct}")

    # Residual must be added once; nonlinear normalization must follow reduction.
    partial0, partial1, residual = [1.0, 2.0], [9.0, 18.0], [100.0, 200.0]
    combined = [a + b + x for a, b, x in zip(partial0, partial1, residual)]
    duplicated = [a + b + 2 * x for a, b, x in zip(partial0, partial1, residual)]
    assert combined == [110.0, 220.0] and duplicated == [210.0, 420.0]
    norm_after = rmsnorm([2.0, 1.0])
    sum_of_norms = [a + b for a, b in zip(rmsnorm([1.0, 0.0]), rmsnorm([1.0, 1.0]))]
    assert any(abs(a - b) > 0.1 for a, b in zip(norm_after, sum_of_norms))
    print(f"Residual once={combined}; duplicated residual={duplicated}")

    real = [3, 1]
    aligned = align_counts(real, tp_size=2)
    assert aligned == [4, 2]
    assert padding_mode([4, 2, 6, 0]) == "MAX_LEN"
    assert padding_mode([12, 2, 0, 0]) == "SUM_LEN"
    assert padding_mode([4, 4], is_extend=True) == "SUM_LEN"
    print(f"Real counts {real} -> aligned {aligned} -> eager {padding_mode(aligned)}")

    # Global FFN TP partials: sum across world, return local DP interval.
    for tp_size in (1, 2):
        dp_size = 2
        world_size = dp_size * tp_size
        full = [[float(i + 1), float(10 * (i + 1))] for i in range(8)]
        partials = [[[v / world_size for v in row] for row in full] for _ in range(world_size)]
        reduced, outputs = global_reduce_scatter_then_attn_gather(
            partials, dp_size, tp_size, 2
        )
        check_close(reduced, full)
        for rank, output in enumerate(outputs):
            check_close(output, local_scatter(full, [4, 4], rank // tp_size))
        print(f"World reduce-scatter + subgroup gather: T={tp_size}, correct local outputs")

    print("PASS: ownership, padding, partial reduction, and output return invariants")


if __name__ == "__main__":
    run()
