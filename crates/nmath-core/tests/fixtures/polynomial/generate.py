# SPDX-License-Identifier: Apache-2.0
"""Development-only fixed NumPy oracle; tests consume the little-endian JSON."""
import json
import platform
from pathlib import Path
import numpy as np

assert np.__version__ == "2.5.4"
DTYPES = "bool int8 uint8 int16 uint16 int32 uint32 int64 uint64 float16 float32 float64 complex64 complex128".split()

def packed(a):
    a = np.asarray(a)
    return {"dtype": a.dtype.name, "shape": list(a.shape), "hex": a.astype(a.dtype.newbyteorder("<")).tobytes().hex()}

cases = []
def case(c, x, ascending, tensor=True):
    with np.errstate(all="ignore"):
        y = np.polynomial.polynomial.polyval(x, c, tensor=tensor) if ascending else np.polyval(c, x)
    cases.append({"c": packed(c), "x": packed(x), "ascending": ascending,
                  "tensor": tensor if ascending else False, "expected": packed(y)})

for cd in DTYPES:
    for xd in DTYPES:
        c = np.array([1, 2, 3], dtype=cd)
        x = np.array([0, 1, 2, 3], dtype=xd)
        if cd.startswith("complex"):
            c += np.array([1j, -2j, 3j], dtype=cd)
        if xd.startswith("complex"):
            x += np.array([1j, 0j, -1j, 2j], dtype=xd)
        for ascending in (False, True):
            case(c, x, ascending)
for ascending in (False, True):
    for tensor in (False, True) if ascending else (False,):
        case(np.array([[0., 1.], [2., 3.]], dtype="float32"), np.array([1., 2.], dtype="float32"), ascending, tensor)
        case(np.arange(12, dtype="float64").reshape(3, 2, 2), np.array([[0.], [2.]]), ascending, tensor)
        case(np.empty((3, 0), dtype="float32"), np.empty((0,), dtype="float32"), ascending, tensor)
        case(np.array([1., 2., 3.]), np.empty((2, 0)), ascending, tensor)
        case(np.array([1., 2., 3.]), np.array(2.), ascending, tensor)
    for cd in ("float32", "complex64", "complex128"):
        case(np.array([1+2j] if cd.startswith("complex") else [1.], dtype=cd), np.array([np.inf, -np.inf, np.nan, -0.]), ascending)
    case(np.array([2**60 + 1, 3], dtype="int64"), np.array([1, 2], dtype="int64"), ascending)
    case(np.array([2**63 + 1, 3], dtype="uint64"), np.array([1, 2], dtype="uint64"), ascending)
case(np.array(3, dtype="int32"), np.array([1., 2.]), True)
case(np.array([], dtype="complex128"), np.array([np.inf, -0.], dtype="float32"), False)
output = {"numpy": np.__version__, "referenceRevision": "a3d0ef1df045b858e68115001fdabc930bb3b68c",
          "environment": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine()},
          "criterion": "Exact integer/bool/shape/dtype contract; floating bits with NaN equivalence are diagnostic, not a general numerical acceptance criterion.",
          "cases": cases}
Path(__file__).with_name("oracle.json").write_text(json.dumps(output, indent=2) + "\n")
