# SPDX-License-Identifier: Apache-2.0
"""Regression tests for source-boundary errors seen during reconciliation."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("inventory", Path(__file__).with_name("generate-inventory.py"))
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


class ReferenceParsing(unittest.TestCase):
    def test_module_context_nested_classes_and_conditional_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "doc/source/reference"
            reference.mkdir(parents=True)
            (reference / "index.rst").write_text(""".. automodule:: numpy.dtypes
.. autosummary::
   :toctree: generated/

   register_dlpack_dtype

.. currentmodule:: numpy
.. autoclass:: ndarray

   .. automethod:: __getitem__

.. attribute:: float96
               float128

   Platform aliases.
""")
            names, modules = inventory.documentation(root, "numpy")
            self.assertEqual(set(names), {"numpy.dtypes", "numpy.dtypes.register_dlpack_dtype", "numpy.ndarray", "numpy.ndarray.__getitem__", "numpy.float96", "numpy.float128"})
            self.assertEqual(modules, {"numpy.dtypes"})

    def test_none_is_a_real_ufunc_attribute_not_an_unresolved_symbol(self):
        self.assertIsNone(inventory.resolve("numpy.add.signature"))
        self.assertIs(inventory.resolve("numpy.this_name_does_not_exist"), inventory.MISSING)

    def test_accessor_expansion_does_not_expand_container_properties(self):
        self.assertIs(inventory.accessor(inventory.LIBRARIES["pandas"].PeriodIndex.start_time), inventory.LIBRARIES["pandas"].PeriodIndex.start_time)
        self.assertTrue(isinstance(inventory.accessor(inventory.LIBRARIES["polars"].Expr.str), type))


if __name__ == "__main__":
    unittest.main()
