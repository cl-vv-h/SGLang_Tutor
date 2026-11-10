"""CPU reference for GDN/KDA recurrence and the tutorial's chunk derivation.

Python standard library only. One head per function call; q/k are already
normalized, alpha contains retention factors, and beta is already activated.
State layout is [Dv, Dk]. This is an arithmetic oracle, not an optimized kernel.
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import unittest


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def matvec(matrix, vector):
    return [dot(row, vector) for row in matrix]


def scale_columns(matrix, alpha):
    return [[x * a for x, a in zip(row, alpha)] for row in matrix]


def interval_decay(alpha, start, stop, dk):
    """Product over [start, stop); an empty interval is identity."""
    result = [1.0] * dk
    for row in alpha[start:stop]:
        result = [a * b for a, b in zip(result, row)]
    return result


def step(state, q, k, v, alpha, beta):
    decayed = scale_columns(state, alpha)
    prediction = matvec(decayed, k)
    residual = [value - pred for value, pred in zip(v, prediction)]
    updated = [
        [decayed[r][c] + beta * residual[r] * k[c] for c in range(len(k))]
        for r in range(len(v))
    ]
    output = [x / math.sqrt(len(k)) for x in matvec(updated, q)]
    return output, updated


def recurrent(state, q, k, v, alpha, beta):
    state = copy.deepcopy(state)
    outputs = []
    for args in zip(q, k, v, alpha, beta):
        output, state = step(state, *args)
        outputs.append(output)
    return outputs, state


def chunk(state, q, k, v, alpha, beta):
    """Solve (I+A)U=B, then independently form outputs and boundary state.

    This deliberately does not call step/recurrent. It uses products directly
    instead of prefix-product division, so even zero retention is well-defined.
    """
    count, dv, dk = len(q), len(state), len(state[0])
    writes = []
    for i in range(count):
        initial_prediction = matvec(
            scale_columns(state, interval_decay(alpha, 0, i + 1, dk)), k[i]
        )
        rhs = [beta[i] * (v[i][r] - initial_prediction[r]) for r in range(dv)]
        # Forward substitution: only earlier writes occur in this row of A.
        for j in range(i):
            decay = interval_decay(alpha, j + 1, i + 1, dk)
            coefficient = beta[i] * dot(k[j], [a * x for a, x in zip(decay, k[i])])
            rhs = [x - coefficient * u for x, u in zip(rhs, writes[j])]
        writes.append(rhs)

    outputs = []
    for i in range(count):
        result = matvec(
            scale_columns(state, interval_decay(alpha, 0, i + 1, dk)), q[i]
        )
        for j in range(i + 1):
            decay = interval_decay(alpha, j + 1, i + 1, dk)
            coefficient = dot(k[j], [a * x for a, x in zip(decay, q[i])])
            result = [x + coefficient * u for x, u in zip(result, writes[j])]
        outputs.append([x / math.sqrt(dk) for x in result])

    final = scale_columns(state, interval_decay(alpha, 0, count, dk))
    for j in range(count):
        decay = interval_decay(alpha, j + 1, count, dk)
        for r in range(dv):
            for c in range(dk):
                final[r][c] += writes[j][r] * k[j][c] * decay[c]
    return outputs, final


def chunked(state, q, k, v, alpha, beta, chunk_size):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    outputs, state = [], copy.deepcopy(state)
    for start in range(0, len(q), chunk_size):
        stop = start + chunk_size
        out, state = chunk(state, q[start:stop], k[start:stop], v[start:stop],
                           alpha[start:stop], beta[start:stop])
        outputs.extend(out)
    return outputs, state


def fixture(seed, length=9, dk=3, dv=2):
    rng = random.Random(seed)
    def vector(width):
        return [rng.uniform(-1, 1) for _ in range(width)]
    def unit(width):
        row = vector(width)
        norm = math.sqrt(dot(row, row))
        return [x / norm for x in row]
    state = [vector(dk) for _ in range(dv)]
    q, k = [unit(dk) for _ in range(length)], [unit(dk) for _ in range(length)]
    v = [vector(dv) for _ in range(length)]
    alpha = [[rng.uniform(0.15, 0.99) for _ in range(dk)] for _ in range(length)]
    beta = [rng.uniform(0.05, 0.95) for _ in range(length)]
    return state, q, k, v, alpha, beta


def flatten(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten(item)
    else:
        yield value


class DeltaAttentionChecks(unittest.TestCase):
    def close(self, a, b):
        left, right = list(flatten(a)), list(flatten(b))
        self.assertEqual(len(left), len(right))
        for x, y in zip(left, right):
            self.assertAlmostEqual(x, y, places=11)

    def test_worked_gdn_and_kda_values(self):
        args = ([[1, .5], [-.5, 1]], [.8, .6], [.6, .8], [1, -.5])
        gdn = step(*args, [.8, .8], .5)
        kda = step(*args, [.8, .25], .5)
        self.close(gdn[1], [[.86, .48], [-.67, .44]])
        self.close(kda[1], [[.926, .293], [-.538, .066]])
        self.close(gdn[0], [.976 / math.sqrt(2), -.272 / math.sqrt(2)])
        self.close(kda[0], [.9166 / math.sqrt(2), -.3908 / math.sqrt(2)])

    def test_scalar_gdn_is_uniform_kda(self):
        state, q, k, v, alpha, beta = fixture(7)
        scalars = [row[0] for row in alpha]
        uniform = [[a] * len(k[0]) for a in scalars]
        # Independent scalar expression, not merely the vector step twice.
        expected, scalar_state = [], copy.deepcopy(state)
        for qi, ki, vi, a, b in zip(q, k, v, scalars, beta):
            decay = [[a * x for x in row] for row in scalar_state]
            error = [x - y for x, y in zip(vi, matvec(decay, ki))]
            scalar_state = [[row[c] + b * error[r] * ki[c] for c in range(len(ki))]
                            for r, row in enumerate(decay)]
            expected.append([x / math.sqrt(len(ki)) for x in matvec(scalar_state, qi)])
        self.close((expected, scalar_state), recurrent(state, q, k, v, uniform, beta))

    def test_random_heads_rectangular_states_and_chunk_sizes(self):
        for head in range(4):
            for dk, dv in [(1, 3), (2, 2), (3, 2), (4, 5)]:
                args = fixture(head * 20 + dk, dk=dk, dv=dv)
                expected = recurrent(*args)
                for size in [1, 2, 3, 4, 64]:
                    with self.subTest(head=head, dk=dk, dv=dv, chunk=size):
                        self.close(expected, chunked(*args, chunk_size=size))

    def test_zero_beta_only_decays(self):
        state, q, k, v, alpha, beta = fixture(8)
        outputs, final = recurrent(state, q, k, v, alpha, [0] * len(beta))
        self.close(final, scale_columns(state, interval_decay(alpha, 0, len(q), 3)))
        self.close((outputs, final), chunk(state, q, k, v, alpha, [0] * len(beta)))

    def test_no_forgetting_delta_rule(self):
        state, q, k, v, alpha, beta = fixture(11)
        alpha = [[1] * 3 for _ in q]
        self.close(recurrent(state, q, k, v, alpha, beta),
                   chunk(state, q, k, v, alpha, beta))

    def test_zero_and_tiny_retention_need_no_division(self):
        state, q, k, v, alpha, beta = fixture(12)
        alpha[2] = [0, 1e-200, 0.4]
        alpha[5] = [0.1, 0, 1e-100]
        self.close(recurrent(state, q, k, v, alpha, beta),
                   chunked(state, q, k, v, alpha, beta, 4))

    def test_unit_key_full_write_matches_value(self):
        _, state = step([[1, .5], [-.5, 1]], [.8, .6], [.6, .8], [1, -.5], [.8, .25], 1)
        self.close(matvec(state, [.6, .8]), [1, -.5])

    def test_output_reads_after_current_write(self):
        out, state = step([[0, 0]], [1, 0], [1, 0], [2], [1, 1], 1)
        self.close(out, [math.sqrt(2)])
        self.close(state, [[2, 0]])

    def test_prediction_must_use_decayed_state(self):
        state, q, k, v, alpha, beta = fixture(15, length=1)
        _, correct = step(state, q[0], k[0], v[0], alpha[0], beta[0])
        wrong_error = [x - y for x, y in zip(v[0], matvec(state, k[0]))]
        wrong = [[state[r][c] * alpha[0][c] + beta[0] * wrong_error[r] * k[0][c]
                  for c in range(3)] for r in range(2)]
        self.assertGreater(max(abs(x-y) for x,y in zip(flatten(correct), flatten(wrong))), .01)

    def test_prefill_boundary_reproduces_next_decode(self):
        state, q, k, v, alpha, beta = fixture(16, length=12)
        expected = recurrent(state, q, k, v, alpha, beta)
        prompt_out, prompt_state = chunked(state, q[:-1], k[:-1], v[:-1], alpha[:-1], beta[:-1], 4)
        next_out, next_state = step(prompt_state, q[-1], k[-1], v[-1], alpha[-1], beta[-1])
        self.close(expected, (prompt_out + [next_out], next_state))

    def test_empty_sequence_preserves_state(self):
        args = fixture(17, length=0)
        self.close(recurrent(*args), chunk(*args))
        self.close(args[0], chunk(*args)[1])

    def test_inputs_and_initial_state_are_not_mutated(self):
        args = fixture(18)
        before = copy.deepcopy(args)
        recurrent(*args)
        chunked(*args, chunk_size=4)
        self.assertEqual(args, before)


def show_comparison():
    state, q, k, v = [[1, .5], [-.5, 1]], [.8, .6], [.6, .8], [1, -.5]
    for name, alpha in [("GDN", [.8, .8]), ("KDA", [.8, .25])]:
        output, updated = step(state, q, k, v, alpha, .5)
        print(f"{name}: output={[round(x, 6) for x in output]}, state={updated}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    options = parser.parse_args()
    if options.self_test:
        unittest.main(argv=[__file__], verbosity=2)
    else:
        show_comparison()
