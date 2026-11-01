# SPDX-License-Identifier: Apache-2.0
"""Development-only fixed NumPy oracle for binary math ufuncs.

Run with numpy 2.5.4 on PATH, then check in the little-endian JSON output.
NumPy's accuracy suite defines no ULP bound for arctan2, hypot, logaddexp or
logaddexp2, so those cases record exact host results and are marked
`criterion: "host-exact-pending"`.
"""
import json
import platform
import warnings
from pathlib import Path

import numpy as np

assert np.__version__ == "2.5.4"
OPS = ["maximum", "minimum", "fmax", "fmin", "copysign", "nextafter", "fmod",
       "floor_divide", "arctan2", "hypot", "heaviside", "logaddexp", "logaddexp2"]
LIBM = {"arctan2", "hypot", "logaddexp", "logaddexp2"}
REAL = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64",
        "uint64", "float16", "float32", "float64"]


def packed(a):
    a = np.asarray(a)
    return {"dtype": a.dtype.name, "shape": list(a.shape),
            "hex": a.astype(a.dtype.newbyteorder("<")).tobytes().hex()}


def operands(dtype):
    if dtype == "bool":
        return np.array([True, False, True, False]), np.array([True, True, False, False])
    if dtype.startswith(("int", "uint")):
        info = np.iinfo(dtype)
        a = [7, 0, 5, int(info.max), int(info.min), 9, 100]
        b = [2, 0, 0, 1, 3, 4, 7]
        if info.min < 0:
            a += [-7, -7, 7, int(info.min), -1]
            b += [2, -2, -2, -1, 5]
        return np.array(a, dtype=dtype), np.array(b, dtype=dtype)
    special = [0.0, -0.0, 0.0, 1.0, -1.0, np.nan, 1.0, np.inf, -np.inf, 7.0, -7.0, 2.5, 1e-3, 5.0, np.inf]
    other = [-0.0, 0.0, 0.0, np.nan, 2.0, 1.0, 1.0, 2.0, np.inf, 2.0, 2.0, -0.5, 3.0, np.inf, np.inf]
    rng = np.random.default_rng(20261102)
    a = np.concatenate([np.array(special, dtype=dtype), rng.uniform(-30, 30, 40).astype(dtype)])
    b = np.concatenate([np.array(other, dtype=dtype), rng.uniform(-30, 30, 40).astype(dtype)])
    return a, b


cases = []


def case(op, a, b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = getattr(np, op)(a, b)
    item = {"op": op, "a": packed(a), "b": packed(b), "expected": packed(expected)}
    if op in LIBM and expected.dtype.kind == "f":
        item["criterion"] = "host-exact-pending"
    cases.append(item)


for op in OPS:
    for dtype in REAL:
        a, b = operands(dtype)
        case(op, a, b)
    for left, right in [("int8", "uint8"), ("int32", "float32"), ("float16", "float64"), ("bool", "int16")]:
        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=left) if left != "bool" else np.array([[True, False, True], [False, True, False]])
        b = np.array([3, 1, 2], dtype=right)
        case(op, a, b)
    if op in {"maximum", "minimum", "fmax", "fmin"}:
        for dtype in ["complex64", "complex128"]:
            a = np.array([1 + 2j, 1 + 3j, complex(np.nan, 0), 2 + 0j, complex(0, np.nan)], dtype=dtype)
            b = np.array([1 + 3j, 1 + 2j, 1 + 0j, complex(np.nan, 1), 5 + 5j], dtype=dtype)
            case(op, a, b)
    case(op, np.empty((0, 2)), np.ones(2))

output = {
    "numpy": np.__version__,
    "environment": {"python": platform.python_version(), "system": platform.system(),
                    "machine": platform.machine()},
    "criterion": ("Exact value, dtype and shape agreement with NumPy, ignoring NaN payloads. "
                  "Cases marked host-exact-pending use the platform libm and have no NumPy "
                  "accuracy bound; their cross-platform criterion needs an explicit decision."),
    "cases": cases,
}
Path(__file__).with_name("oracle.json").write_text(json.dumps(output, indent=1) + "\n")
