#!/usr/bin/env python3
"""Regression tests for public names, independent of generated-header freshness."""

import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "generator", Path(__file__).with_name("generate-experimental-tensor.py")
)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class NamespaceNameTests(unittest.TestCase):
    def test_public_names_from_actual_signatures(self):
        expected = {
            "isin_elements_Tensor_test_elements_Tensor": (None, "isin"),
            "instance_norm_input_Tensor_": (None, "instance_norm"),
            "tensordot_self_Tensor_": (None, "tensordot"),
            # The operation and its first parameter both contain 'out'.
            "abs_out_out_Tensor_": (None, "abs_out"),
            # A later parameter also starts with 'out'.
            "bmm_out_out_Tensor_self_Tensor_mat2_Tensor_out_dtype_": (None, "bmm_out"),
            # The dispatch suffix can order parameters differently.
            "result_type_other_Tensor_tensor_Tensor": (None, "result_type"),
            "_cast_Byte_self_Tensor": (None, "_cast_Byte"),
            "linalg_cholesky_self_Tensor": ("linalg", "cholesky"),
        }
        functions = list(generator.functions(generator.SOURCE.read_text(), "namespace"))
        for prefix, public_name in expected.items():
            with self.subTest(prefix=prefix):
                matching = [entry for entry in functions if entry[1].startswith(prefix)]
                self.assertTrue(matching, f"Missing source fixture: {prefix}")
                for _, encoded_name, signature, body in matching:
                    self.assertEqual(
                        generator.namespace_name(
                            encoded_name, generator.split_parameters(signature), body
                        ),
                        public_name,
                    )

    def test_rejects_mismatched_lantern_operation(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            generator.namespace_name(
                "isin_elements_Tensor", ["XPtrTorchTensor elements"],
                "auto r_out = lantern_relu_tensor(elements.get());",
            )


if __name__ == "__main__":
    unittest.main()
