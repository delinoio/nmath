# SPDX-License-Identifier: Apache-2.0
"""Record derivative outputs from the fixed development-only NumPy oracle."""
import json
import platform
from pathlib import Path
import numpy as np

assert np.__version__ == "2.5.4"
cases = []


def packed(a):
    a = np.asarray(a)
    return {"dtype": a.dtype.name, "shape": list(a.shape),
            "hex": a.astype(a.dtype.newbyteorder("<")).tobytes().hex()}


def case(c, m=1, scale=1.0, axis=0, ascending=True):
    with np.errstate(all="ignore"):
        y = (np.polynomial.polynomial.polyder(c, m=m, scl=scale, axis=axis)
             if ascending else np.polyder(c, m=m))
    cases.append({"c": packed(c), "m": m, "axis": axis, "ascending": ascending,
                  "scale": [float(np.real(scale)), float(np.imag(scale))],
                  "complexScale": isinstance(scale, complex), "expected": packed(y)})


for dtype in "bool int8 uint8 int16 uint16 int32 uint32 int64 uint64 float16 float32 float64 complex64 complex128".split():
    for shape in [(4,), (0,), (1,)]:
        c = np.arange(np.prod(shape)).astype(dtype).reshape(shape)
        if dtype.startswith("complex"):
            c += np.asarray(0.25j, dtype=dtype)
        for m in [0, 1, 2, 4, 2**40]:
            # The legacy implementation recurses once per order even for empty
            # results; record its mathematically stable empty limit at order 4.
            case(c, m, ascending=True)
            if m < 100:
                case(c, m, ascending=False)
    for shape in [(), (2, 4, 3), (2, 0, 3)]:
        c = np.arange(np.prod(shape) if shape else 1).astype(dtype).reshape(shape)
        for axis in ([-1, 0] if not shape else [-1, 1]):
            for m in [0, 1, 9]:
                case(c, m, axis=axis)

for dtype in ["float16", "float32", "float64", "complex64", "complex128"]:
    c = np.array([1.25, 2.5, -3.75, 4.125, -0.0], dtype=dtype)
    for m in [1, 2, 3]:
        for scale in [0.1, -0.25, 0.0]:
            case(c, m, scale)
        if dtype.startswith("complex"):
            case(c + np.array([1j, -2j, 3j, 0j, -0j], dtype=dtype), m, 0.25-0.5j)
    for m in [0, 1, 4]:
        case(np.array([np.inf], dtype=dtype), m)
        case(np.array([np.inf, -np.inf, np.nan, -0.0], dtype=dtype), m)
        case(np.array([np.inf, -np.inf, np.nan, -0.0], dtype=dtype), m, ascending=False)

case(np.array([2**60+1, 2**62+1, -2**60+1], dtype="int64"), ascending=False)
case(np.array([2**63+1, 2**64-1, 3], dtype="uint64"), ascending=False)
for m in [0, 9]:
    case(np.array([1., 2., 3.]), m, 1+0j)

output = {"numpy": np.__version__,
          "referenceRevision": "a3d0ef1df045b858e68115001fdabc930bb3b68c",
          "environment": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine()},
          "criterion": "Exact dtype, shape and finite floating bits; NaNs compare by classification. These fixtures do not establish all-platform numerical acceptance.",
          "cases": cases}
Path(__file__).with_name("derivative-oracle.json").write_text(json.dumps(output, indent=2)+"\n")
