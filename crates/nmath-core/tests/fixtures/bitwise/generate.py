# SPDX-License-Identifier: Apache-2.0
"""Development-only fixed NumPy oracle for bitwise, shift and logical ufuncs.

Run with numpy 2.5.4 on PATH, then check in the little-endian JSON output.
"""
import json
import platform
import warnings
from pathlib import Path

import numpy as np

assert np.__version__ == "2.5.4"
OPS = {
    "bitwise_and": np.bitwise_and,
    "bitwise_or": np.bitwise_or,
    "bitwise_xor": np.bitwise_xor,
    "left_shift": np.left_shift,
    "right_shift": np.right_shift,
    "logical_and": np.logical_and,
    "logical_or": np.logical_or,
    "logical_xor": np.logical_xor,
}
INTS = ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"]


def packed(a):
    a = np.asarray(a)
    return {
        "dtype": a.dtype.name,
        "shape": list(a.shape),
        "hex": a.astype(a.dtype.newbyteorder("<")).tobytes().hex(),
    }


def values(dtype):
    if dtype == "bool":
        return np.array([True, False, True, False])
    info = np.iinfo(dtype)
    return np.array([0, 1, 5, int(info.max), int(info.min) if info.min else 200 % (info.max + 1), 3], dtype=dtype)


def counts(dtype, bits):
    if dtype == "bool":
        return np.array([True, False, False, True])
    raw = [0, 1, 3, bits - 1, bits, bits + 3]
    if np.iinfo(dtype).min < 0:
        raw[2] = -1
    return np.array(raw, dtype=dtype)


cases = []


def case(op, a, b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = OPS[op](a, b)
    cases.append({"op": op, "a": packed(a), "b": packed(b), "expected": packed(expected)})


for dtype in ["bool"] + INTS:
    a = values(dtype)
    bits = 8 if dtype == "bool" else np.iinfo(dtype).bits
    b = counts(dtype, bits)
    for op in OPS:
        case(op, a, b)

# Mixed integer promotion and broadcasting.
for left, right in [("int8", "uint8"), ("uint16", "int32"), ("bool", "int16"), ("uint32", "uint64")]:
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=left) if left != "bool" else np.array([[True, False, True], [False, True, False]])
    b = np.array([3, 1, 2], dtype=right)
    for op in OPS:
        case(op, a, b)

# Logical operators accept floats and complex values by truthiness.
f = np.array([0.0, -0.0, np.nan, 2.5])
z = np.array([0, 1j, 0, 0], dtype="complex64")
for op in ["logical_and", "logical_or", "logical_xor"]:
    case(op, f, z)
    case(op, np.empty((0, 2)), np.ones(2))

output = {
    "numpy": np.__version__,
    "environment": {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    },
    "criterion": "Exact value, dtype and shape agreement with NumPy.",
    "cases": cases,
}
Path(__file__).with_name("oracle.json").write_text(json.dumps(output, indent=1) + "\n")
