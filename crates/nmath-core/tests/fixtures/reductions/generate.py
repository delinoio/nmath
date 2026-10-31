# SPDX-License-Identifier: Apache-2.0
"""Development-only fixed NumPy oracle for product and index reductions.

Run with numpy 2.5.4 on PATH, then check in the little-endian JSON output.
"""
import json
import platform
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


def case(op, x, axis):
    expected = getattr(x, op)(axis=axis)
    cases.append(
        {
            "op": op,
            "input": packed(x),
            "axis": axis,
            "expected": packed(expected),
        }
    )


def data(dtype):
    if dtype == "bool":
        return np.array([[True, True, False], [True, True, True]], dtype=dtype)
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    if dtype.startswith("complex"):
        values = values + np.array([[0, 1j, 0], [1j, 0, 1j]], dtype=dtype)
    return values


for dtype in DTYPES:
    x = data(dtype)
    for op in ("prod", "argmin", "argmax"):
        for axis in (None, 0, 1):
            case(op, x, axis)

# Empty product reductions use the multiplicative identity; selected axes keep
# their original length. NumPy rejects index selection over a zero-size axis.
for dtype in ("bool", "int8", "uint64", "float16", "float32", "complex128"):
    case("prod", np.empty((0,), dtype=dtype), None)
    case("prod", np.empty((3, 0), dtype=dtype), 1)
    case("prod", np.empty((0, 3), dtype=dtype), 0)
    case("prod", np.empty((2, 0), dtype=dtype), 1)

# Rank-0 inputs reduce to themselves.
for dtype in ("int32", "float64", "complex64"):
    case("prod", np.array(5, dtype=dtype), None)
    case("argmin", np.array(5, dtype=dtype), None)

# First-occurrence ties, sticky NaNs and lexicographic complex ordering.
for dtype in ("float32", "float64"):
    case("argmin", np.array([3, 1, 1, 2], dtype=dtype), None)
    case("argmax", np.array([3, 1, 1, 2], dtype=dtype), None)
    case("argmin", np.array([1, np.nan, 0, np.nan], dtype=dtype), None)
    case("argmax", np.array([1, np.nan, 0, np.nan], dtype=dtype), None)
    case("argmin", np.array([[1, np.nan], [0, 2]], dtype=dtype), 0)
    case("argmax", np.array([[1, np.nan], [0, 2]], dtype=dtype), 1)
for dtype in ("complex64", "complex128"):
    case("argmin", np.array([1 + 2j, 1 + 1j, 0 + 5j], dtype=dtype), None)
    case("argmax", np.array([1 + 2j, 1 + 1j, 0 + 5j], dtype=dtype), None)

# Exact 64-bit products stay in the checked-in fixture, not JS numbers.
case("prod", np.array([2**40, 3], dtype="int64"), None)
case("prod", np.array([2**43, 5], dtype="uint64"), None)

output = {
    "numpy": np.__version__,
    "referenceRevision": "a3d0ef1df045b858e68115001fdabc930bb3b68c",
    "environment": {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    },
    "criterion": "Exact integer/bool/shape/dtype agreement; floating values are exact for the recorded integer-valued inputs. Broad floating-point product criteria remain pending.",
    "cases": cases,
}
Path(__file__).with_name("oracle.json").write_text(json.dumps(output, indent=2) + "\n")
