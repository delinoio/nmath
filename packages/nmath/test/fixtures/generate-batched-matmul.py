# SPDX-License-Identifier: Apache-2.0
"""Generate exact small-value matrix products with the fixed NumPy oracle.

No tolerance is needed: all finite products and partial sums in these cases
are representable exactly in the result dtype. This is not a criterion for
arbitrary long floating-point products or platform acceptance.
"""
import json
import platform
import sys
from pathlib import Path
import numpy as np

assert np.__version__ == "2.5.4", np.__version__
cases = []

def pack(a):
    dtype = str(a.dtype)
    values = a.ravel().tolist()
    if a.dtype.kind == "c":
        values = [component for z in values for component in [z.real, z.imag]]
    elif dtype in ["int64", "uint64"]:
        values = [str(x) for x in values]
    return {"shape": list(a.shape), "dtype": dtype, "values": values}

def operand(shape, dtype, shift):
    data = np.arange(np.prod(shape), dtype=np.int64) + shift
    if dtype == "bool":
        data = data % 2 == 0
    elif dtype.startswith("uint"):
        data = data % 7
    elif dtype.startswith("complex"):
        data = (data % 7 - 3) + (data % 3 - 1) * 1j
    else:
        data = data % 7 - 3
    return np.array(data, dtype=dtype).reshape(shape)

shapes = [
    ([2, 1, 2, 3], [4, 3, 2]),
    ([3], [2, 3, 4]),
    ([2, 4, 3], [3]),
    ([3], [3]),
    ([2, 3], [2, 3, 2]),
    ([2, 1, 2, 3], [3, 4]),
    ([0, 2, 3], [1, 3, 4]),
    ([2, 3, 0], [1, 0, 4]),
    ([2, 0, 3], [3, 4]),
    ([2, 4, 3], [3, 0]),
]
dtypes = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64", "complex64", "complex128"]
for dtype in dtypes:
    for i, (a, b) in enumerate(shapes):
        left, right = operand(a, dtype, 0), operand(b, dtype, 1)
        cases.append({"id": f"{dtype}/{i}", "a": pack(left), "b": pack(right), "expected": pack(left @ right)})
for a, b in [("int8", "uint8"), ("float16", "int16"), ("float32", "int32"), ("complex64", "float64")]:
    left, right = operand([2, 1, 2, 3], a, 0), operand([4, 3, 2], b, 1)
    cases.append({"id": f"promotion/{a}/{b}", "a": pack(left), "b": pack(right), "expected": pack(left @ right)})
for dtype, data in [
    ("int64", [2**60 + 3, 2**63 - 1, -2**63, -1]),
    ("uint64", [2**60 + 3, 2**64 - 1, 2**63, 1]),
    ("int8", [127, 127, -128, -128]),
]:
    left = np.array(data, dtype=dtype).reshape(2, 1, 2)
    right = np.array([1, 2], dtype=dtype)
    cases.append({"id": f"integer-boundary/{dtype}", "a": pack(left), "b": pack(right), "expected": pack(left @ right)})
Path(__file__).with_name("batched-matmul.json").write_text(json.dumps({"numpy": np.__version__, "environment": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(), "byteorder": sys.byteorder}, "criterion": "exact-small-values", "cases": cases}, indent=2) + "\n")
