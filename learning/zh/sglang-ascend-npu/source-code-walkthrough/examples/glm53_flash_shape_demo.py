#!/usr/bin/env python3
"""CPU math/shape checks accompanying the GLM-5.3-Flash walkthrough.

Standard library only; no weights, network, PyTorch, or accelerator required.
The constants are a documented subset of the pinned model configuration, not
a live download. Passing these tests is NOT validation of SGLang kernels or
full-model accuracy/performance. Small vectors illustrate the equations;
they are not checkpoint activations or a replacement model implementation.
"""

from __future__ import annotations

import math
import unittest


MODEL_REVISION = "46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
SGLANG_REVISION = "92580581da6fc8542a997a00c1a8b71b03497680"
HIDDEN = 4096
LAYERS = 45
KDA_HEADS = 64
KDA_DIM = 128
MLA_HEADS = 64
MLA_QK_DIM = 256
Q_RANK = 1536
KV_RANK = 512
TP = 8
DSA_LAYERS = tuple(range(3, LAYERS, 4))
KDA_LAYERS = tuple(i for i in range(LAYERS) if i not in DSA_LAYERS)


def sigmoid(x):
    """Stable scalar sigmoid, including large finite inputs."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def safe_gate(forget_raw, dt_bias, a_log, lower_bound=-5.0):
    """Return (log_decay, retention) for the checkpoint's safe-gate branch."""
    log_decay = lower_bound * sigmoid(math.exp(a_log) * (forget_raw + dt_bias))
    return log_decay, math.exp(log_decay)


def dot(a, b):
    assert len(a) == len(b)
    return sum(x * y for x, y in zip(a, b))


def normalized(x):
    norm = math.sqrt(dot(x, x))
    assert norm > 0
    return [v / norm for v in x]


def kda_step(state, q, k, v, alpha, beta):
    """Value-row/key-column convention; Q/K already preprocessed.

    Only the recurrent equation is represented: not convolution, output RMSNorm,
    output gating, quantization, or the optimized chunk algorithm.
    """
    key_dim = len(k)
    assert len(q) == len(alpha) == key_dim
    assert len(state) == len(v) and all(len(row) == key_dim for row in state)
    decayed = [[x * a for x, a in zip(row, alpha)] for row in state]
    predicted = [dot(row, k) for row in decayed]
    error = [target - estimate for target, estimate in zip(v, predicted)]
    updated = [
        [old + beta * err * key for old, key in zip(row, k)]
        for row, err in zip(decayed, error)
    ]
    output = [dot(row, q) / math.sqrt(key_dim) for row in updated]
    return updated, predicted, output


def conv_step(history, current, weights):
    """Tiny causal depthwise conv + SiLU; window order is oldest to newest."""
    assert len(weights) == len(current)
    window = history + [current]
    assert all(len(w) == len(window) for w in weights)
    linear = [
        sum(row[channel] * weights[channel][t] for t, row in enumerate(window))
        for channel in range(len(current))
    ]
    return [x * sigmoid(x) for x in linear], window[1:]


def sinkhorn(logits, iterations=20, eps=1e-6):
    """Teaching equivalent of softmax + stabilized row/column normalization."""
    size = len(logits)
    assert size > 0 and all(len(row) == size for row in logits)
    matrix = []
    for row in logits:
        largest = max(row)
        exps = [math.exp(x - largest) for x in row]
        total = sum(exps)
        matrix.append([x / total + eps for x in exps])

    def normalize_columns(values):
        totals = [sum(row[j] for row in values) + eps for j in range(size)]
        return [[x / totals[j] for j, x in enumerate(row)] for row in values]

    matrix = normalize_columns(matrix)
    for _ in range(iterations - 1):
        matrix = [[x / (sum(row) + eps) for x in row] for row in matrix]
        matrix = normalize_columns(matrix)
    return matrix


def mhc_read(streams, pre):
    return [
        sum(weight * row[d] for weight, row in zip(pre, streams))
        for d in range(len(streams[0]))
    ]


def mhc_write(streams, output, post, mixing):
    """mixing[i][j] maps old stream i to new stream j: C transpose times R."""
    count = len(streams)
    return [
        [
            sum(mixing[i][j] * streams[i][d] for i in range(count))
            + post[j] * output[d]
            for d in range(len(output))
        ]
        for j in range(count)
    ]


def pool_key(keys, gate_logits):
    """Feature-wise softmax pooling, before Hadamard/FP8 quantization."""
    assert len(keys) == len(gate_logits) == 4
    result = []
    for c in range(len(keys[0])):
        column = [row[c] for row in gate_logits]
        largest = max(column)
        exps = [math.exp(x - largest) for x in column]
        result.append(
            sum(w * key[c] for w, key in zip(exps, keys)) / sum(exps)
        )
    return result


def kpool_indices(visible, group_scores, token_budget=2048):
    """Logical positions only, not physical page IDs or padded kernel buffers."""
    assert visible >= 0 and token_budget > 0 and token_budget % 4 == 0
    complete = visible // 4
    assert len(group_scores) >= complete
    selected = sorted(
        range(complete), key=lambda group: (-group_scores[group], group)
    )[: token_budget // 4]
    expanded = [4 * group + offset for group in selected for offset in range(4)]
    return expanded + list(range(4 * complete, visible))


def route(logits, correction, topk=8):
    """Bias affects selection; original sigmoid scores determine combine weights."""
    assert len(logits) == len(correction) and 0 < topk <= len(logits)
    scores = [sigmoid(x) for x in logits]
    chosen = sorted(
        range(len(scores)), key=lambda e: (-(scores[e] + correction[e]), e)
    )[:topk]
    denominator = sum(scores[e] for e in chosen)
    return chosen, [scores[e] / denominator for e in chosen]


class WalkthroughChecks(unittest.TestCase):
    def assertVectorClose(self, actual, expected, places=10):
        self.assertEqual(len(actual), len(expected))
        for x, y in zip(actual, expected):
            self.assertAlmostEqual(x, y, places=places)

    def assertMatrixClose(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertVectorClose(a, b)

    def test_layer_partition(self):
        self.assertEqual(len(DSA_LAYERS), 11)
        self.assertEqual(len(KDA_LAYERS), 34)
        self.assertEqual(DSA_LAYERS[-1], 43)
        self.assertIn(44, KDA_LAYERS)
        self.assertTrue(set(range(3)).issubset(KDA_LAYERS))
        self.assertEqual(3 + 42, LAYERS)  # Independent dense/MoE partition.

    def test_projection_shapes(self):
        self.assertEqual((KDA_HEADS // TP, KDA_DIM), (8, 128))
        self.assertEqual(3 * KDA_HEADS * KDA_DIM // TP, 3072)
        self.assertEqual(KDA_HEADS * KDA_DIM // TP, 1024)
        self.assertEqual(MLA_HEADS * MLA_QK_DIM // TP, 2048)
        self.assertEqual(MLA_HEADS * (256 + 256) // TP, 4096)
        self.assertEqual(Q_RANK + KV_RANK, 2048)
        self.assertEqual(4 * HIDDEN, 16384)
        self.assertEqual(4 + 4 + 4 * 4, 24)
        self.assertEqual((288 // 8, 2048 // 8), (36, 256))

    def test_safe_gate_bounds_and_bias(self):
        for raw in (-20.0, -1.0, 0.0, 1.0, 20.0):
            g, retention = safe_gate(raw, 0.1, 0.2)
            self.assertGreater(g, -5.0)
            self.assertLess(g, 0.0)
            self.assertGreater(retention, math.exp(-5.0))
            self.assertLess(retention, 1.0)
        self.assertAlmostEqual(safe_gate(0, 0, 0)[0], -2.5)
        self.assertAlmostEqual(safe_gate(-0.3, 0.3, 0.7)[0], -2.5)

    def test_kda_prediction_is_from_decayed_memory(self):
        state, prediction, output = kda_step(
            [[2.0, 0.0], [0.0, 4.0]],
            normalized([1.0, 1.0]), [1.0, 0.0], [3.0, 2.0],
            [0.5, 0.25], 0.5,
        )
        self.assertVectorClose(prediction, [1.0, 0.0])
        self.assertMatrixClose(state, [[2.0, 0.0], [1.0, 1.0]])
        self.assertVectorClose(output, [1.0, 1.0])

    def test_kda_already_explained_value_has_no_extra_write(self):
        old = [[1.0, 0.0], [0.0, 1.0]]
        updated, prediction, _ = kda_step(
            old, [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0], 0.8,
        )
        self.assertVectorClose(prediction, [1.0, 0.0])
        self.assertMatrixClose(updated, old)

    def test_kda_prefix_boundary_restore_and_chunk_carry(self):
        state = [[0.0, 0.0], [0.0, 0.0]]
        snapshots = [[row[:] for row in state]]
        tokens = ([1.0, 2.0], [3.0, 1.0], [0.0, 4.0])
        for value in tokens:
            state, _, _ = kda_step(
                state, [1.0, 0.0], [1.0, 0.0], value, [0.9, 0.8], 0.6
            )
            snapshots.append([row[:] for row in state])
        # Accept two tokens: restore after token b, not after rejected token c.
        restored = snapshots[2]
        self.assertNotEqual(restored, snapshots[3])
        carried = snapshots[1]
        for value in tokens[1:]:
            carried, _, _ = kda_step(
                carried, [1.0, 0.0], [1.0, 0.0], value, [0.9, 0.8], 0.6
            )
        # This checks recurrence carry, not a parallel chunk-kernel algorithm.
        self.assertMatrixClose(carried, state)
        self.assertEqual(snapshots[0], [[0.0, 0.0], [0.0, 0.0]])

    def test_short_convolution_does_not_cross_request_boundaries(self):
        zeros = [[0.0], [0.0], [0.0]]
        weights = [[1.0, 1.0, 1.0, 1.0]]
        _, request_a_history = conv_step(zeros, [10.0], weights)
        output_b, _ = conv_step(zeros, [1.0], weights)
        contaminated_b, _ = conv_step(request_a_history, [1.0], weights)
        self.assertAlmostEqual(output_b[0], sigmoid(1.0))
        self.assertGreater(contaminated_b[0], output_b[0] + 9.0)

    def test_mhc_worked_example(self):
        streams = [[1, 0], [0, 1], [1, 1], [2, 0]]
        self.assertVectorClose(mhc_read(streams, [0.1, 0.2, 0.3, 0.4]), [1.2, 0.5])
        identity = [[float(i == j) for j in range(4)] for i in range(4)]
        written = mhc_write(streams, [2, 1], [1, 0.5, 1.5, 0.25], identity)
        self.assertMatrixClose(written, [[3, 1], [1, 1.5], [4, 2.5], [2.5, 0.25]])

    def test_mhc_mixing_orientation(self):
        # Non-symmetric cyclic permutation catches C@R versus transpose(C)@R.
        mixing = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
        result = mhc_write([[1], [2], [3]], [0], [0, 0, 0], mixing)
        self.assertEqual(result, [[3], [1], [2]])

    def test_sinkhorn_approximately_doubly_stochastic(self):
        mixing = sinkhorn([[0.2 * i - 0.1 * j + (i == j) for j in range(4)]
                           for i in range(4)])
        for row in mixing:
            self.assertLess(abs(sum(row) - 1), 1e-4)
            self.assertTrue(all(x > 0 for x in row))
        for column in zip(*mixing):
            self.assertLess(abs(sum(column) - 1), 1e-4)

    def test_kpool_featurewise_weighting(self):
        result = pool_key(
            [[1, 10], [2, 20], [3, 30], [4, 40]],
            [[math.log(7), 0], [0, 0], [0, 0], [0, 0]],
        )
        self.assertVectorClose(result, [1.6, 25.0])

    def test_kpool_eleven_token_example(self):
        self.assertEqual(kpool_indices(11, [0.2, 0.9], 4), list(range(4, 11)))
        # At position 5, the later group 4/5/6/7 is NOT a complete group.
        self.assertEqual(kpool_indices(6, [0.2, 999.0], 4), list(range(6)))

    def test_kpool_causality_tail_and_budget(self):
        for visible in list(range(33)) + [2048, 2049, 2051, 4099]:
            indices = kpool_indices(visible, list(range(visible // 4)))
            self.assertEqual(len(indices), min(512, visible // 4) * 4 + visible % 4)
            self.assertEqual(len(indices), len(set(indices)))
            self.assertTrue(all(0 <= i < visible for i in indices))
            tail = set(range(visible // 4 * 4, visible))
            self.assertTrue(tail.issubset(indices))
            self.assertLessEqual(len(indices), 2051)

    def test_moe_bias_selects_but_does_not_weight(self):
        chosen, weights = route([2.0, 0.0, -2.0], [0.0, 0.0, 2.0], topk=2)
        self.assertEqual(chosen, [2, 0])
        expected = sigmoid(-2.0) / (sigmoid(-2.0) + sigmoid(2.0))
        self.assertAlmostEqual(weights[0], expected)
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertLess(weights[0], weights[1])

    def test_cache_and_vision_arithmetic(self):
        recurrent = len(KDA_LAYERS) * KDA_HEADS * KDA_DIM * KDA_DIM * 4
        self.assertEqual(recurrent, 142606336)
        self.assertEqual(recurrent // 2**20, 136)
        self.assertEqual(recurrent // TP // 2**20, 17)
        self.assertEqual(len(DSA_LAYERS) * KV_RANK * 2, 11264)
        self.assertEqual((448 // 14) ** 2, 1024)
        self.assertEqual(3 * 2 * 14 * 14, 1176)
        self.assertEqual(1024 // 2**2, 256)


if __name__ == "__main__":
    print("GLM-5.3-Flash CPU teaching checks (not accelerator validation)", flush=True)
    print(f"Model revision:  {MODEL_REVISION}", flush=True)
    print(f"SGLang revision: {SGLANG_REVISION}", flush=True)
    unittest.main(verbosity=2)
