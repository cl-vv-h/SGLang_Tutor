"""Regression tests for distinctions found during the math migration."""

import unittest

from formalize_math import equation, literal_algebra, migrate


class FormalizeMathTests(unittest.TestCase):
    def test_parallel_assignments_are_not_chained_vectors(self):
        self.assertEqual(equation("B=1, L=8"), r"B=1,\qquad L=8")

    def test_shape_annotation_is_not_indexing(self):
        result = equation("Y = X @ W       [B,S,D]")
        self.assertIn(r"Y=X\,W\qquad\mathrm{shape}:", result)
        self.assertNotIn("{W}_{", result)

    def test_outer_product_has_transposed_key(self):
        self.assertIn(r"{k_{t}}^{\mathsf T}", equation("S_t = S_decay + u_t outer k_t"))

    def test_negated_sum_keeps_parentheses(self):
        self.assertEqual(equation("x = -(a+b)"), r"x=-\left(a+b\right)")

    def test_alias_slash_is_not_a_fraction(self):
        text = "Use `Q/K`, `prefill/decode`, and `W_UK/W_UV`.\n"
        self.assertEqual(migrate(text)[0], text)

    def test_source_and_mermaid_are_untouched(self):
        text = "```python\nx = a / b\n```\n```mermaid\nA --> B\n```\n"
        self.assertEqual(migrate(text)[0], text)

    def test_multiline_source_call_is_untouched(self):
        text = "```text\ncausal_conv1d_fn(\n    x,\n    ...,\n    cache_indices=indices,\n)\n```\n"
        self.assertEqual(migrate(text)[0], text)

    def test_bitwise_operations_are_not_inequalities(self):
        with self.assertRaises(ValueError):
            literal_algebra("word_idx = token_id >> 5")

    def test_sum_condition_is_a_subscript(self):
        self.assertIn(r"\sum_{", literal_algebra("y = sum_(e in TopK(t)) p_e * x_e"))

    def test_exponent_and_subscript_are_distinct(self):
        self.assertIn(r"q^{I}_{t,j}", literal_algebra("y = q^I_{t,j} dot k^I_s"))

    def test_shape_dimensions_are_not_thousands_separators(self):
        self.assertIn("448,512", literal_algebra("local weight: [448,512] = [448,512]"))

    def test_swiglu_product_is_elementwise(self):
        self.assertIn(r"\odot", equation("z = SiLU(gate) * up"))

    def test_idempotence(self):
        text = "```text\ny = x @ W\n```\nUse `N * D` channels.\n"
        converted = migrate(text)[0]
        self.assertEqual(migrate(converted)[0], converted)

    def test_existing_math_does_not_shift_code_span_boundaries(self):
        text = 'Use $`n=2`$ entries in `INT32`; if bit=0 write `-inf`.\n'
        self.assertEqual(migrate(text)[0], text)


if __name__ == "__main__":
    unittest.main()
