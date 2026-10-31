# SPDX-License-Identifier: Apache-2.0
"""Development-only fixed NumPy oracle for variance and standard deviation.

Run with numpy 2.5.4 on PATH, then check in the little-endian JSON output.
"""
import json
import platform
import warnings
from pathlib import Path
import numpy as np

assert np.__version__ == "2.5.4"
DTYPES = "bool int8 uint8 int16 uint16 int32 uint32 int64 uint64 float16 float32 float64 complex64 complex128".split()


def packed(a):
    a = np.asarray(a)
    return {
        "dtype": a.dtype.name,
        "shape": list(a.shape),
        "hex": a.astype(a.dtype.newbyteorder("<")).tobytes().hex(),
    }


cases = []


def case(op, x, axis, ddof):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = getattr(x, op)(axis=axis, ddof=ddof)
    cases.append(
        {
            "op": op,
            "input": packed(x),
            "axis": axis,
            "ddof": ddof,
            "expected": packed(expected),
        }
    )


def data(dtype):
    if dtype == "bool":
        return np.array(
            [[True, True, False, False], [True, True, False, False]], dtype=dtype
        )
    values = np.array([[1, 3, 1, 3], [1, 3, 1, 3]], dtype=dtype)
    if dtype.startswith("complex"):
        values = values * np.array(1 + 1j, dtype=dtype)
    return values


for dtype in DTYPES:
    x = data(dtype)
    for op in ("var", "std"):
        case(op, x, None, 0)
        case(op, x, 0, 0)
        case(op, x, 1, 0)
        case(op, x, -1, 0)

# Exact ddof boundaries: divisor zero becomes infinity, a negative divisor stays
# negative, and an empty reduction divides zero by zero.
exact = np.array([1.0, -1.0])
for ddof in (0, 1, 2, 3):
    case("var", exact, None, ddof)
    case("std", exact, None, ddof)

for dtype in ("int32", "float16", "float64", "complex128"):
    case("var", np.empty((0,), dtype=dtype), None, 0)
    case("var", np.empty((2, 0), dtype=dtype), 1, 0)
    case("var", np.empty((0, 3), dtype=dtype), 0, 0)

for dtype in ("int32", "float32", "complex64"):
    case("var", np.array(5, dtype=dtype), None, 0)

output = {
    "numpy": np.__version__,
    "referenceRevision": "a3d0ef1df045b858e68115001fdabc930bb3b68c",
    "environment": {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    },
    "criterion": "Exact dtype, shape and recorded-value agreement on integer-valued inputs whose two-pass result is representable. Broad floating-point variance criteria remain pending.",
    "cases": cases,
}
Path(__file__).with_name("oracle.json").write_text(json.dumps(output, indent=2) + "\n")
