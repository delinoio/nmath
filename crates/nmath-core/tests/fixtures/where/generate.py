# SPDX-License-Identifier: Apache-2.0
"""Development-only fixed NumPy oracle for `where(condition, x, y)`.

Run with numpy 2.5.4 on PATH, then check in the little-endian JSON output.
"""
import json
import platform
from pathlib import Path
import numpy as np

assert np.__version__ == "2.5.4"


def packed(a):
    a = np.asarray(a)
    return {
        "dtype": a.dtype.name,
        "shape": list(a.shape),
        "hex": a.astype(a.dtype.newbyteorder("<")).tobytes().hex(),
    }


def operand(value):
    # Python scalars are NEP 50 weak operands; arrays are strong.
    if isinstance(value, bool):
        return {"scalar": "bool", "value": value}
    if isinstance(value, int):
        return {"scalar": "int", "value": value}
    if isinstance(value, float):
        return {"scalar": "float", "value": value}
    if isinstance(value, complex):
        return {"scalar": "complex", "value": [value.real, value.imag]}
    return {"array": packed(value)}


cases = []


def case(condition, x, y):
    expected = np.where(condition, x, y)
    cases.append(
        {
            "condition": packed(condition),
            "x": operand(x),
            "y": operand(y),
            "expected": packed(expected),
        }
    )


mask = np.array([[True, False, True], [False, True, False]])
DTYPES = [
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]


def values(dtype, offset):
    a = np.array([[1, 2, 3], [4, 5, 6]]) + offset
    if dtype == "bool":
        return (a % 2).astype(dtype)
    out = a.astype(dtype)
    if dtype.startswith("complex"):
        out = out + np.array([[0, 1j, 0], [1j, 0, 1j]], dtype=dtype)
    return out


# Same-dtype branches for every numeric dtype.
for dtype in DTYPES:
    case(mask, values(dtype, 0), values(dtype, 10))

# Mixed branch promotion.
PAIRS = [
    ("int8", "uint8"),
    ("int32", "float32"),
    ("uint64", "int64"),
    ("float16", "int16"),
    ("bool", "int8"),
    ("complex64", "float64"),
]
for left, right in PAIRS:
    case(mask, values(left, 0), values(right, 10))

# Non-bool conditions use nonzero truthiness, including NaN and complex.
case(np.array([0, 2, -1, 0], dtype="int32"), np.arange(4, dtype="int32"), np.int32(-1) * np.ones(4, "int32"))
case(np.array([0.0, np.nan, -0.0, 0.5]), np.arange(4.0), np.full(4, -1.0))
case(np.array([0, 1j, 0, 1 + 0j]), np.arange(4, dtype="int16"), np.full(4, -1, dtype="int16"))

# Three-way broadcasting and empty shapes.
case(np.array([[True], [False]]), np.array([1.0, 2.0, 3.0]), np.array([[10.0], [20.0]]))
case(np.array(True), np.arange(3, dtype="int32"), np.zeros((2, 1), dtype="int32"))
case(np.empty((0, 3), dtype=bool), np.arange(3, dtype="float32"), np.zeros(3, dtype="float32"))

# Weak scalar branches follow NEP 50.
small = np.array([1, 2, 3], dtype="int8")
f32 = np.array([1, 2, 3], dtype="float32")
cond = np.array([True, False, True])
case(cond, small, 7)
case(cond, 7, small)
case(cond, small, 1.5)
case(cond, f32, 2.5)
case(cond, f32, 1j)
case(cond, small, True)
case(cond, 1, 2)
case(cond, 1, 2.5)
case(cond, True, False)

output = {
    "numpy": np.__version__,
    "environment": {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    },
    "criterion": "Exact value, dtype and shape agreement under NumPy where promotion and broadcasting rules.",
    "cases": cases,
}
Path(__file__).with_name("oracle.json").write_text(json.dumps(output, indent=2) + "\n")
