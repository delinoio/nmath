# SPDX-License-Identifier: Apache-2.0
"""Development-only fixed NumPy oracle for elementwise comparisons.

Run with numpy 2.5.4 on PATH, then check in the little-endian JSON output.
"""
import json
import platform
import warnings
from pathlib import Path
import numpy as np

assert np.__version__ == "2.5.4"
OPS = ["eq", "ne", "lt", "le", "gt", "ge"]
UFUNC = {
    "eq": np.equal,
    "ne": np.not_equal,
    "lt": np.less,
    "le": np.less_equal,
    "gt": np.greater,
    "ge": np.greater_equal,
}


def packed(a):
    a = np.asarray(a)
    return {
        "dtype": a.dtype.name,
        "shape": list(a.shape),
        "hex": a.astype(a.dtype.newbyteorder("<")).tobytes().hex(),
    }


cases = []


def case(op, a, b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = UFUNC[op](a, b)
    cases.append(
        {
            "op": op,
            "a": packed(a),
            "b": packed(b),
            "expected": packed(expected),
        }
    )


def pair(dtype):
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    if dtype.startswith("complex"):
        values = values + np.array([[0, 1j, 0], [1j, 0, 1j]], dtype=dtype)
    return values


PAIRS = [
    ("int8", "int32"),
    ("uint8", "int64"),
    ("int32", "float32"),
    ("float32", "float64"),
    ("float16", "float32"),
    ("bool", "int16"),
    ("complex64", "complex128"),
    ("complex128", "float64"),
]
for left, right in PAIRS:
    a = pair(left)
    b = pair(right)
    for op in OPS:
        complex_input = left.startswith("complex") or right.startswith("complex")
        if complex_input and op not in {"eq", "ne"}:
            continue
        case(op, a, b)

# Broadcasting and NaN ordering.
a = np.array([[2.0], [5.0]])
b = np.array([1.0, 3.0, 5.0])
for op in OPS:
    case(op, a, b)
nan = np.array([[np.nan, 1.0], [2.0, np.nan]])
other = np.array([[1.0], [np.nan]])
for op in OPS:
    case(op, nan, other)

# Empty operands keep the broadcast shape.
empty = np.empty((0,), dtype="float64")
for op in OPS:
    case(op, empty, empty)

output = {
    "numpy": np.__version__,
    "referenceRevision": "a3d0ef1df045b858e68115001fdabc930bb3b68c",
    "environment": {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    },
    "criterion": "Exact boolean agreement under NumPy promotion and broadcasting rules.",
    "cases": cases,
}
Path(__file__).with_name("oracle.json").write_text(json.dumps(output, indent=2) + "\n")
