#!/usr/bin/env python3
"""SP step by step / 逐步理解 SP: a Python-standard-library CPU simulator.

Read chapter 02 alongside this output. Every collective here is simulated
inside one process: there is no distributed runtime or device benchmark.
Weights use the X @ W teaching convention (PyTorch Linear stores W transposed).
"""
import math


def close_matrix(actual, expected):
    assert len(actual) == len(expected), (len(actual), len(expected))
    for row_a, row_b in zip(actual, expected):
        assert len(row_a) == len(row_b)
        for a, b in zip(row_a, row_b):
            assert math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-10), (a, b)


def show(label, matrix):
    print(label, "shape =", (len(matrix), len(matrix[0]) if matrix else 0))
    for row in matrix:
        print("   ", "[" + ", ".join(f"{x:8.4f}" for x in row) + "]")


def add(a, b):
    assert len(a) == len(b)
    assert all(len(x) == len(y) for x, y in zip(a, b))
    return [[x + y for x, y in zip(row_a, row_b)]
            for row_a, row_b in zip(a, b)]


def matmul(a, b):
    assert a and b and len(a[0]) == len(b)
    return [[sum(row[k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for row in a]


def norm_rows(matrix, epsilon=1e-6):
    result = []
    for row in matrix:
        rms = math.sqrt(sum(x * x for x in row) / len(row) + epsilon)
        result.append([x / rms for x in row])
    return result


def relu(matrix):
    return [[max(0.0, x) for x in row] for row in matrix]


def pad_rows(matrix, degree):
    padded_count = math.ceil(len(matrix) / degree) * degree
    return [row[:] for row in matrix] + [
        [0.0] * len(matrix[0]) for _ in range(padded_count - len(matrix))
    ]


def split_rows(matrix, degree):
    assert len(matrix) % degree == 0
    rows_per_rank = len(matrix) // degree
    return [[row[:] for row in matrix[r * rows_per_rank:(r + 1) * rows_per_rank]]
            for r in range(degree)]


def gather_rows(shards, real_rows=None):
    # AllGather concatenates different COMPLETE rows; it does not add them.
    result = [row[:] for shard in shards for row in shard]
    return result if real_rows is None else result[:real_rows]


def sum_partials(partials):
    # Same row/feature positions are numerical contributions to one result.
    result = [[0.0] * len(partials[0][0]) for _ in partials[0]]
    for partial in partials:
        result = add(result, partial)
    return result


def reduce_scatter_sum(partials, degree):
    assert len(partials) == degree
    return split_rows(sum_partials([pad_rows(p, degree) for p in partials]), degree)


def demo_norm_and_collectives():
    print("\n1. Four tokens, two TP ranks: partial != complete row")
    x = [[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0],
         [1.0, 1.0, 1.0, 1.0], [2.0, 0.0, 0.0, 0.0]]
    partial_0 = [[1.0] * 4 for _ in x]
    partial_1 = [[value - 1.0 for value in row] for row in x]
    show("Expected complete X", x)
    show("rank 0 partial Y0", partial_0)
    show("rank 1 partial Y1", partial_1)

    summed = sum_partials([partial_0, partial_1])
    close_matrix(summed, x)
    print("AllReduce: rank 0 AND rank 1 each receive the whole X above.")
    shards = reduce_scatter_sum([partial_0, partial_1], degree=2)
    for rank, shard in enumerate(shards):
        show(f"ReduceScatter output rank {rank}: COMPLETE assigned rows", shard)

    # Epsilon zero is used only on these nonzero rows for hand calculation.
    normalized_shards = [norm_rows(shard, epsilon=0.0) for shard in shards]
    for rank, shard in enumerate(normalized_shards):
        show(f"Local RMSNorm rank {rank}", shard)
    gathered = gather_rows(normalized_shards)
    expected = [[1.2, 1.6, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0],
                [1.0, 1.0, 1.0, 1.0], [2.0, 0.0, 0.0, 0.0]]
    close_matrix(gathered, expected)
    close_matrix(gathered, norm_rows(x, epsilon=0.0))
    show("AllGather AFTER local Norm: identical to full RMSNorm", gathered)

    a, b = [[1.0, 0.0]], [[0.0, 1.0]]
    correct = norm_rows(add(a, b), epsilon=0.0)
    wrong = add(norm_rows(a, epsilon=0.0), norm_rows(b, epsilon=0.0))
    close_matrix(correct, [[1.0, 1.0]])
    close_matrix(wrong, [[math.sqrt(2), math.sqrt(2)]])
    show("Correct: Norm(a+b)", correct)
    show("Wrong: Norm(a)+Norm(b)", wrong)
    print("Nonlinearity means the SUM must happen before Norm.")
    return x


def make_weights(hidden=4, expanded=8):
    up = [[((i + 2) * (j + 1) % 7 - 3) / 4.0
           for j in range(expanded)] for i in range(hidden)]
    down = [[((i + 1) * (j + 3) % 11 - 5) / 5.0
             for j in range(hidden)] for i in range(expanded)]
    return up, down


def weight_shards(up, down, degree):
    assert len(up[0]) == len(down) and len(down) % degree == 0
    width = len(down) // degree
    return [(
        [row[r * width:(r + 1) * width] for row in up],
        down[r * width:(r + 1) * width],
    ) for r in range(degree)]


def full_ffn_residual(x, up, down):
    # Independent unsharded reference: pre-Norm -> FFN -> residual add.
    return add(x, matmul(relu(matmul(norm_rows(x), up)), down))


def ordinary_tp(x, up, down, degree):
    # Every TP rank repeats Norm over all tokens.
    partials = [
        matmul(relu(matmul(norm_rows(x), up_shard)), down_shard)
        for up_shard, down_shard in weight_shards(up, down, degree)
    ]
    # AllReduce makes the summed result available on all ranks.
    complete_ffn = sum_partials(partials)
    return add(x, complete_ffn)


def sp_tp(x, up, down, degree, verbose=False):
    real_rows = len(x)
    residual_shards = split_rows(pad_rows(x, degree), degree)
    normalized_shards = [norm_rows(rows) for rows in residual_shards]
    # Gather rows so EACH weight shard sees EVERY real token.
    gathered = gather_rows(normalized_shards, real_rows=real_rows)
    partials = []
    for rank, (up_shard, down_shard) in enumerate(weight_shards(up, down, degree)):
        local_features = relu(matmul(gathered, up_shard))
        partial = matmul(local_features, down_shard)
        partials.append(partial)
        if verbose:
            show(f"rank {rank} SP residual rows", residual_shards[rank])
            show(f"rank {rank} local Norm", normalized_shards[rank])
            show(f"rank {rank} after AG: all normalized token rows", gathered)
            show(f"rank {rank} up+ReLU: local FEATURES for all tokens", local_features)
            show(f"rank {rank} down: PARTIALS for all tokens", partial)
    ffn_shards = reduce_scatter_sum(partials, degree)
    output_shards = [add(residual, ffn)
                     for residual, ffn in zip(residual_shards, ffn_shards)]
    if verbose:
        for rank in range(degree):
            show(f"rank {rank} RS: complete local FFN rows", ffn_shards[rank])
            show(f"rank {rank} add local residual ONCE", output_shards[rank])
    # Exit gather restores the external interface and drops padding.
    return gather_rows(output_shards, real_rows=real_rows)


def demo_ffn(x):
    print("\n2. Full matrices vs ordinary TP vs TP+SP: a real small FFN")
    print("FFN uses ReLU for clarity, no bias, and ordinary nonzero epsilon.")
    up, down = make_weights()
    for rows, degree in [(4, 2), (5, 2), (9, 4)]:
        inputs = [x[i % len(x)][:] for i in range(rows)]
        reference = full_ffn_residual(inputs, up, down)
        tp = ordinary_tp(inputs, up, down, degree)
        sp = sp_tp(inputs, up, down, degree, verbose=(rows == 4 and degree == 2))
        close_matrix(tp, reference)
        close_matrix(sp, reference)
        print(f"PASS: tokens={rows}, TP={degree}, "
              f"padded rows={len(pad_rows(inputs, degree))}; "
              "full == TP == TP+SP, including residual.")
    print("Padding is removed before real-token compute and at the final interface.")


def attention(scores, values):
    assert len(scores) == len(values) and scores
    maximum = max(scores)
    masses = [math.exp(s - maximum) for s in scores]
    total = sum(masses)
    output = sum(w * v for w, v in zip(masses, values)) / total
    return output, maximum + math.log(total)


def demo_cp():
    print("\n3. Same four tokens, attention needs remote context")
    values = [2.0, 4.0, 8.0, 10.0]
    full, _ = attention([0.0] * 4, values)
    wrong, _ = attention([0.0] * 2, values[2:])
    close_matrix([[full, wrong]], [[6.0, 9.0]])
    print("t3 full history:", values, "->", full)
    print("t3 rank-1-only KV:", values[2:], "->", wrong, "(WRONG)")

    # Unequal softmax mass makes even averaging local outputs incorrect.
    scores = [0.0, 0.0, math.log(3.0), math.log(3.0)]
    left, left_lse = attention(scores[:2], values[:2])
    right, right_lse = attention(scores[2:], values[2:])
    reference, _ = attention(scores, values)
    maximum = max(left_lse, right_lse)
    a, b = math.exp(left_lse - maximum), math.exp(right_lse - maximum)
    merged = (a * left + b * right) / (a + b)
    close_matrix([[left, right, merged, reference]], [[3.0, 9.0, 7.5, 7.5]])
    print("Local outputs:", left, right, "; unnormalized masses:", 2.0, 6.0)
    print("LSE-weighted merge =", merged, "; naive mean =", (left + right) / 2)
    print("CP must communicate KV or exact attention contributions.")
    print("SP's local Norm cannot substitute for that attention protocol.")


if __name__ == "__main__":
    data = demo_norm_and_collectives()
    demo_ffn(data)
    demo_cp()
    print("\nAll assertions passed. This validates math, not device performance.")
