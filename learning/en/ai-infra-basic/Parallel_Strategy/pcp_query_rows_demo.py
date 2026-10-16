#!/usr/bin/env python3
"""PCP query rows / PCP query 行: hand-worked examples using only Python.

Single process, one request, one attention head. Collectives are simulated;
short example lengths do not imply production PCP strategy admission.
"""
import math


def close(actual, expected):
    if isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            close(a, b)
    else:
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (
            actual, expected
        )


def dot(q, k):
    assert len(q) == len(k)
    return sum(a * b for a, b in zip(q, k))


def softmax(scores):
    maximum = max(scores)
    assert maximum != -math.inf, "Query has no visible key."
    masses = [math.exp(s - maximum) for s in scores]
    total = sum(masses)
    return [m / total for m in masses]


def full_matrix_reference(queries, keys, values, prefix):
    """Dense score matrix + explicit mask: an unsharded reference."""
    assert len(keys) == len(values) == prefix + len(queries)
    scores = [[dot(q, k) / math.sqrt(len(q)) for k in keys] for q in queries]
    weights = [
        softmax([s if j <= prefix + i else -math.inf
                 for j, s in enumerate(row)])
        for i, row in enumerate(scores)
    ]
    outputs = [
        [sum(w * v[d] for w, v in zip(row, values))
         for d in range(len(values[0]))]
        for row in weights
    ]
    return outputs, weights


def selected_query(q, keys, values, visible_positions):
    """Read one query's original legal columns; no dependence on other Q rows."""
    positions = list(visible_positions)
    scores = [dot(q, keys[j]) / math.sqrt(len(q)) for j in positions]
    weights = softmax(scores)
    output = [
        sum(w * values[j][d] for w, j in zip(weights, positions))
        for d in range(len(values[0]))
    ]
    return output, weights


def sharded_queries(queries, keys, values, prefix, assignment, verbose=False):
    restored = [None] * len(queries)
    concatenated = []
    for rank, new_indices in enumerate(assignment):
        rank_outputs = []
        for local_row, new_index in enumerate(new_indices):
            original_position = prefix + new_index
            output, weights = selected_query(
                queries[new_index], keys, values, range(original_position + 1)
            )
            assert restored[new_index] is None, "Query assigned twice."
            restored[new_index] = output
            rank_outputs.append(output)
            if verbose:
                print(f"rank {rank}, local row {local_row}: "
                      f"new query i={new_index}, original position={original_position}")
                print("  visible original positions:", list(range(original_position + 1)))
                print("  weights:", [round(w, 6) for w in weights], "output:", output)
        concatenated.extend(rank_outputs)
    assert all(row is not None for row in restored), "Query never assigned."
    return restored, concatenated


def main_example():
    names = ["P0", "P1", "A", "B", "C"]
    prefix = 2
    queries = [[0.0], [0.0], [0.0]]
    keys = [[float(j + 1)] for j in range(5)]
    values = [[float(2 * (j + 1))] for j in range(5)]
    print("1. Cached prefix P0/P1 + new tokens A/B/C")
    print("Q shape=(3, 1); K/V shape=(5, 1); all indices start at zero.")
    print("Values:", [v[0] for v in values])
    reference, weights = full_matrix_reference(queries, keys, values, prefix)
    close(reference, [[4.0], [5.0], [6.0]])
    for i in range(3):
        print(f"query {names[prefix + i]}, i={i}, original position={prefix + i}:")
        print("  visible:", names[:prefix + i + 1])
        print("  weights over ALL columns:", [round(w, 6) for w in weights[i]])
        print("  output:", reference[i])

    print("\n2. Give A/C to rank 0 and B to rank 1")
    assignment = [[0, 2], [1]]
    restored, concatenated = sharded_queries(
        queries, keys, values, prefix, assignment, verbose=True
    )
    close(restored, reference)
    close(concatenated, [[4.0], [6.0], [5.0]])
    assert concatenated != restored
    print("Concatenate rank outputs ->", [o[0] for o in concatenated], "(A/C/B order)")
    print("Restore original indices ->", [o[0] for o in restored], "(A/B/C order)")

    print("\n3. Two mistakes that change the answer")
    # Rank 0 owns new A/C; dropping rank 1's B removes original column 3.
    missing_b, _ = selected_query(queries[2], keys, values, [0, 1, 2, 4])
    close(missing_b, [5.5])
    print("Keep prefix + rank-0 new KV only -> C output", missing_b, "!= [6]")
    # C's rank-local row is 1; original new-query index is 2.
    wrong_position = prefix + 1
    wrong_mask, _ = selected_query(
        queries[2], keys, values, range(wrong_position + 1)
    )
    close(wrong_mask, [5.0])
    print("Mask C using local row 1 -> visible", names[:wrong_position + 1],
          "->", wrong_mask, "!= [6]")

    print("\n4. Unequal scores: attention is generally not an average")
    nonuniform_q = [[1.0], [1.0], [1.0]]
    nonuniform_k = [[0.0], [0.0], [math.log(2)], [0.0], [math.log(4)]]
    full, weights = full_matrix_reference(nonuniform_q, nonuniform_k, values, prefix)
    close(weights[0], [0.25, 0.25, 0.5, 0.0, 0.0])
    close(full, [[4.5], [5.2], [22.0 / 3.0]])
    restored, _ = sharded_queries(
        nonuniform_q, nonuniform_k, values, prefix, assignment
    )
    close(restored, full)
    print("A scores on P0/P1/A: [0, 0, log(2)]")
    print("A weights:", weights[0], "-> output", full[0])
    print("Unequal-score outputs also match after sharding and restoring.")


def additional_checks():
    # Independent dense reference versus row-wise evaluation, with nontrivial
    # head widths, value widths, prefixes, permutations, and empty query shards.
    for prefix in [0, 2]:
        for new_count in [1, 3, 5]:
            for degree in [1, 2, 4]:
                queries = [[0.3 * (i + 1), -0.2 * i] for i in range(new_count)]
                keys = [[0.1 * j, 0.4 - 0.05 * j]
                        for j in range(prefix + new_count)]
                values = [[float(j), float(j * j), 1.0]
                          for j in range(prefix + new_count)]
                assignment = [list(range(r, new_count, degree))[::-1]
                              for r in range(degree)]
                reference, _ = full_matrix_reference(queries, keys, values, prefix)
                restored, _ = sharded_queries(
                    queries, keys, values, prefix, assignment
                )
                close(restored, reference)
    print("\nPASS: 18 additional layouts, including nonzero scores and empty shards.")
    print("All assertions passed. Math correctness only; no hardware benchmark.")


if __name__ == "__main__":
    main_example()
    additional_checks()
