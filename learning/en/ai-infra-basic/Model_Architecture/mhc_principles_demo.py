#!/usr/bin/env python3
"""mHC principles: standard-library mathematics, not a production kernel.

No downloads, PyTorch, or accelerator are required. Coefficients and examples
are synthetic, not checkpoint weights. C[i][j] maps source stream i to
destination j, so the old-stream update is transpose(C) @ R.
"""

import math
import random
import unittest


SGLANG_REVISION = "92580581da6fc8542a997a00c1a8b71b03497680"


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1 + ex)


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def matmul(a, b):
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def rms_norm(vector, eps=0.0):
    divisor = math.sqrt(sum(x * x for x in vector) / len(vector) + eps)
    if divisor == 0:
        raise ValueError("Zero RMS needs a positive epsilon")
    return [x / divisor for x in vector]


def row_normalize(matrix, eps=0.0):
    return [[x / (sum(row) + eps) for x in row] for row in matrix]


def column_normalize(matrix, eps=0.0):
    return transpose(row_normalize(transpose(matrix), eps))


def sinkhorn(logits, iterations=20, eps=1e-6):
    """Pinned-reference-style row-softmax start, then alternating rescaling.

    Setting eps=0 gives an idealized teaching version, not BF16/FP32 behavior.
    """
    if iterations < 1:
        raise ValueError("At least one iteration is required")
    size = len(logits)
    assert size and all(len(row) == size for row in logits)
    matrix = []
    for row in logits:
        exps = [math.exp(x - max(row)) for x in row]
        matrix.append([x / sum(exps) + eps for x in exps])
    matrix = column_normalize(matrix, eps)
    for _ in range(iterations - 1):
        matrix = row_normalize(matrix, eps)
        matrix = column_normalize(matrix, eps)
    return matrix


def coefficients(streams, weight, bias, scales, norm_eps=1e-5,
                 hc_eps=1e-6, iterations=20):
    """One token's dynamic pre, post, and source-to-destination matrix."""
    count, width = len(streams), len(streams[0])
    flat = [x for row in streams for x in row]
    coefficient_count = count * count + 2 * count
    assert len(weight) == len(bias) == coefficient_count
    assert all(len(row) == count * width for row in weight)
    assert len(scales) == 3
    denominator = math.sqrt(sum(x * x for x in flat) / len(flat) + norm_eps)
    z = [sum(x * w for x, w in zip(flat, row)) / denominator for row in weight]
    pre = [sigmoid(scales[0] * z[i] + bias[i]) + hc_eps for i in range(count)]
    post = [2 * sigmoid(scales[1] * z[count + i] + bias[count + i])
            for i in range(count)]
    raw = [scales[2] * z[2 * count + i] + bias[2 * count + i]
           for i in range(count * count)]
    mixing = sinkhorn([raw[i * count:(i + 1) * count] for i in range(count)],
                     iterations, hc_eps)
    return pre, post, mixing


def readout(streams, pre):
    return [sum(p * streams[i][d] for i, p in enumerate(pre))
            for d in range(len(streams[0]))]


def old_mix(streams, mixing):
    return matmul(transpose(mixing), streams)


def writeback(streams, output, post, mixing):
    old = old_mix(streams, mixing)
    return [[old[j][d] + post[j] * output[d] for d in range(len(output))]
            for j in range(len(streams))]


def full_step(streams, pre, post, mixing, sublayer, norm_eps=0.0):
    value = readout(streams, pre)
    normalized = rms_norm(value, norm_eps)
    output = sublayer(normalized)
    return writeback(streams, output, post, mixing)


def channel_sums(streams):
    return [sum(column) for column in zip(*streams)]


def squared_norm(matrix):
    return sum(x * x for row in matrix for x in row)


def matrix_error(matrix):
    rows = max(abs(sum(row) - 1) for row in matrix)
    columns = max(abs(sum(column) - 1) for column in zip(*matrix))
    return rows, columns


def worked_example():
    streams = [[1.0, 0.0], [0.0, 1.0]]
    pre, post = [0.5, 0.5], [1.0, 0.5]
    mixing = [[0.75, 0.25], [0.25, 0.75]]
    value = readout(streams, pre)
    normalized = rms_norm(value)
    output = [2 * normalized[0], -normalized[1]]
    final = writeback(streams, output, post, mixing)
    print("Convention: C[source][destination]; old path = transpose(C) @ R")
    print("Readout:", value, "Norm:", normalized, "Sublayer output:", output)
    print("Old-stream mix:", old_mix(streams, mixing))
    print("Final residual:", final)
    print("Old total:", channel_sums(streams), "Final total:", channel_sums(final))
    first = sinkhorn([[math.log(1), math.log(2)],
                      [math.log(3), math.log(4)]], iterations=1, eps=0)
    final_c = sinkhorn([[math.log(1), math.log(2)],
                        [math.log(3), math.log(4)]], iterations=20, eps=0)
    print("One Sinkhorn iteration:", first, "row/column errors:", matrix_error(first))
    print("Twenty idealized iterations:", final_c)


class MHCChecks(unittest.TestCase):
    def assertVectorClose(self, actual, expected, places=10):
        self.assertEqual(len(actual), len(expected))
        for x, y in zip(actual, expected):
            self.assertAlmostEqual(x, y, places=places)

    def assertMatrixClose(self, actual, expected, places=10):
        self.assertEqual(len(actual), len(expected))
        for x, y in zip(actual, expected):
            self.assertVectorClose(x, y, places)

    def test_complete_numerical_example(self):
        streams = [[1.0, 0.0], [0.0, 1.0]]
        pre, post = [0.5, 0.5], [1.0, 0.5]
        mixing = [[0.75, 0.25], [0.25, 0.75]]
        self.assertVectorClose(readout(streams, pre), [0.5, 0.5])
        result = full_step(streams, pre, post, mixing,
                           lambda v: [2 * v[0], -v[1]])
        self.assertMatrixClose(result, [[2.75, -0.75], [1.25, 0.25]])

    def test_non_symmetric_orientation(self):
        mixing = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
        self.assertEqual(old_mix([[1], [2], [3]], mixing), [[3], [1], [2]])

    def test_row_stochastic_only_can_amplify_a_destination(self):
        streams = [[1.0], [1.0]]
        mixing = [[0.9, 0.1], [0.9, 0.1]]
        result = old_mix(streams, mixing)
        self.assertEqual(result, [[1.8], [0.2]])
        self.assertGreater(squared_norm(result), squared_norm(streams))

    def test_old_path_preserves_sums_and_range(self):
        streams = [[-2.0, 5.0], [4.0, -1.0]]
        result = old_mix(streams, [[0.75, 0.25], [0.25, 0.75]])
        self.assertVectorClose(channel_sums(result), channel_sums(streams))
        for d in range(2):
            lo, hi = min(row[d] for row in streams), max(row[d] for row in streams)
            self.assertTrue(all(lo <= row[d] <= hi for row in result))

    def test_old_path_nonexpansion_for_many_positive_matrices(self):
        rng = random.Random(20261028)
        for size in (2, 3, 4):
            for _ in range(10):
                logits = [[rng.uniform(-1, 1) for _ in range(size)]
                          for _ in range(size)]
                mixing = sinkhorn(logits, iterations=100, eps=0)
                streams = [[rng.uniform(-5, 5) for _ in range(3)]
                           for _ in range(size)]
                self.assertLess(max(matrix_error(mixing)), 1e-10)
                self.assertLessEqual(squared_norm(old_mix(streams, mixing)),
                                     squared_norm(streams) + 1e-9)

    def test_averaging_can_erase_stream_differences(self):
        self.assertMatrixClose(old_mix([[1.0], [-1.0]], [[0.5] * 2] * 2),
                               [[0.0], [0.0]])

    def test_composition_is_doubly_stochastic(self):
        first = [[0.75, 0.25], [0.25, 0.75]]
        second = [[0.2, 0.8], [0.8, 0.2]]
        product = matmul(first, second)
        self.assertLess(max(matrix_error(product)), 1e-12)
        self.assertTrue(all(x >= 0 for row in product for x in row))

    def test_single_stream_recovers_residual_with_matching_gates(self):
        streams = [[1.0, 1.0]]
        result = full_step(streams, [1.0], [1.0], [[1.0]], lambda v: v)
        self.assertMatrixClose(result, [[2.0, 2.0]])

    def test_equal_copies_stay_equal_under_old_mixing(self):
        streams = [[1.0, -2.0]] * 2
        result = old_mix(streams, [[0.3, 0.7], [0.7, 0.3]])
        self.assertMatrixClose(result, streams)

    def test_new_writes_can_increase_total_and_norm(self):
        streams = [[1.0, 0.0], [0.0, 1.0]]
        result = writeback(streams, [2.0, -1.0], [1.0, 0.5],
                           [[0.75, 0.25], [0.25, 0.75]])
        self.assertVectorClose(channel_sums(result), [4.0, -0.5])
        self.assertGreater(squared_norm(result), squared_norm(streams))

    def test_one_sinkhorn_iteration(self):
        logits = [[math.log(1), math.log(2)], [math.log(3), math.log(4)]]
        first = sinkhorn(logits, iterations=1, eps=0)
        self.assertMatrixClose(first, [[7 / 16, 7 / 13], [9 / 16, 6 / 13]])
        row_error, column_error = matrix_error(first)
        self.assertGreater(row_error, 0.02)
        self.assertLess(column_error, 1e-12)

    def test_sinkhorn_limit_for_two_by_two_example(self):
        logits = [[math.log(1), math.log(2)], [math.log(3), math.log(4)]]
        matrix = sinkhorn(logits, iterations=100, eps=0)
        a = 2 / (2 + math.sqrt(6))
        self.assertMatrixClose(matrix, [[a, 1 - a], [1 - a, a]])

    def test_large_logits_are_stabilized(self):
        matrix = sinkhorn([[1000.0, 999.0], [999.0, 1000.0]])
        self.assertTrue(all(math.isfinite(x) and x > 0
                            for row in matrix for x in row))
        self.assertLess(max(matrix_error(matrix)), 1e-4)

    def test_finite_epsilon_iterations_are_not_exact_constraints(self):
        matrix = sinkhorn([[0.0, 1.0], [2.0, 0.0]], iterations=20, eps=1e-6)
        error = max(matrix_error(matrix))
        self.assertGreater(error, 0.0)
        self.assertLess(error, 1e-4)

    def test_pre_and_post_are_not_stream_softmax(self):
        pre, post = [sigmoid(0)] * 4, [2 * sigmoid(0)] * 4
        self.assertEqual(sum(pre), 2)
        self.assertEqual(sum(post), 4)

    def test_norm_reordering_does_not_move_bias(self):
        vector, weight, bias = [1.0, 3.0], [2.0, -1.0], 0.7
        divisor = math.sqrt(sum(x * x for x in vector) / len(vector) + 1e-5)
        first = sum(x * w for x, w in zip(rms_norm(vector, 1e-5), weight)) + bias
        second = sum(x * w for x, w in zip(vector, weight)) / divisor + bias
        wrong = (sum(x * w for x, w in zip(vector, weight)) + bias) / divisor
        self.assertAlmostEqual(first, second)
        self.assertNotAlmostEqual(first, wrong)

    def test_dynamic_coefficient_shapes(self):
        streams = [[1.0, 0.0], [0.0, 1.0]]
        weight = [[((i + 1) * (j + 2) % 11 - 5) / 10 for j in range(4)]
                  for i in range(8)]
        pre, post, mixing = coefficients(streams, weight, [0.0] * 8, [0.2] * 3)
        self.assertEqual((len(pre), len(post)), (2, 2))
        self.assertEqual([len(row) for row in mixing], [2, 2])
        self.assertTrue(all(0 < x < 1.000001 for x in pre))
        self.assertTrue(all(0 < x < 2 for x in post))
        other = coefficients([[3.0, 1.0], [2.0, -1.0]], weight,
                             [0.0] * 8, [0.2] * 3)
        self.assertNotEqual(pre, other[0])

    def test_token_isolation_and_one_sublayer_call_per_token(self):
        batch = [[[1.0, 0.0], [0.0, 1.0]],
                 [[2.0, 1.0], [1.0, 2.0]]]
        calls = []
        def sublayer(value):
            calls.append(value[:])
            return [2 * value[0], -value[1]]
        outputs = [full_step(r, [0.5, 0.5], [1.0, 0.5],
                            [[0.75, 0.25], [0.25, 0.75]], sublayer)
                   for r in batch]
        self.assertEqual(len(calls), len(batch))  # Not token count times streams.
        before = [row[:] for row in outputs[0]]
        batch[1][0][0] = 99.0
        after = full_step(batch[0], [0.5, 0.5], [1.0, 0.5],
                          [[0.75, 0.25], [0.25, 0.75]], sublayer)
        self.assertMatrixClose(before, after)

    def test_shape_parameter_and_memory_arithmetic(self):
        tokens, streams, hidden = 5, 4, 4096
        count = streams * streams + 2 * streams
        self.assertEqual((count, streams * hidden), (24, 16384))
        self.assertEqual(count * streams * hidden + count + 3, 393243)
        self.assertEqual(tokens * hidden * 2, 40 * 1024)
        self.assertEqual(tokens * streams * hidden * 2, 160 * 1024)

    def test_dynamic_old_path_derivative_need_not_be_nonexpansive(self):
        # R(t)=[t,-t]; a=sigmoid(2t), C=[[a,1-a],[1-a,a]] is always DS.
        # Old-path output is [f(t),-f(t)], f(t)=t*tanh(t).
        def f(t):
            a = sigmoid(2 * t)
            return (2 * a - 1) * t
        step = 1e-6
        derivative = (f(1 + step) - f(1 - step)) / (2 * step)
        self.assertLess(abs(f(1)), 1)  # Pointwise old-path norm decreases.
        self.assertGreater(derivative, 1.18)  # Full dynamic Jacobian may grow.


if __name__ == "__main__":
    worked_example()
    print("Pinned runtime reference:", SGLANG_REVISION, flush=True)
    print("Synthetic CPU math only; not GPU/NPU accuracy/performance validation.",
          flush=True)
    unittest.main(verbosity=2)
