# SPDX-License-Identifier: Apache-2.0
"""Generate fixtures with NumPy 2.5.4; Python is an oracle only.

Exact comparisons cover dyadic arithmetic, reductions, matmul, promotion,
casts and exceptional component classifications. Fractional complex128 power
uses the same inputs as numpy/_core/tests/test_umath.py::TestPower::
test_power_complex and its assert_almost_equal default decimal=7 criterion
(component absolute error < 1.5 * 10**-7). No criterion is inferred for other
fractional-power inputs or complex64 fractional power.
"""
import base64
import io
import json
import math
from pathlib import Path
import warnings
import numpy as np

assert np.__version__ == "2.5.4"
warnings.simplefilter("ignore")
np.seterr(all="ignore")


def component(x):
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x == 0 and math.copysign(1, x) < 0:
        return "-0"
    return x


def record(x):
    a = np.asarray(x)
    return {"dtype": str(a.dtype), "shape": list(a.shape),
            "values": [[component(z.real), component(z.imag)] for z in a.flat]}


def npy(a):
    out = io.BytesIO()
    np.save(out, a)
    return base64.b64encode(out.getvalue()).decode()


dtypes = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64", "complex64", "complex128"]
result = {"numpy": np.__version__, "criterion": "exact components, NaN classification, shape and dtype; signed zero where marked",
          "promotion": {a: {b: str(np.result_type(a, b)) for b in dtypes} for a in ["complex64", "complex128"]},
          "weak": {d: str(np.result_type(np.ones(1, dtype=d), 1j)) for d in dtypes}, "cases": []}
for dtype in ["complex64", "complex128"]:
    a = np.array([[1+2j, 3-4j], [-2+1j, 0-2j]], dtype=dtype)
    b = np.array([2+0j, 0+2j], dtype=dtype)
    operations = {name: record(op(a, b)) for name, op in [("add", np.add), ("sub", np.subtract), ("mul", np.multiply), ("div", np.divide), ("pow", np.power)] if name != "pow"}
    operations.update({name: record(op(a, axis=0)) for name, op in [("sum", np.sum), ("mean", np.mean), ("min", np.min), ("max", np.max)]})
    operations["neg"] = record(-a)
    operations["pos"] = record(+a)
    operations["matmul"] = record(a @ b)
    operations["pow"] = record(a ** 2)
    values = [complex(0, 0), complex(-0.0, -0.0), complex(1, -2), complex(float("inf"), 0), complex(0, float("inf")), complex(float("nan"), 1), complex(1, float("nan"))]
    exceptional = []
    for x in values:
        for y in values:
            for name, op in [("div", np.divide), ("pow", np.power)]:
                if name == "pow" and x == 1-2j and y == 1-2j:
                    continue  # No matching upstream tolerance for this finite complex exponent.
                exceptional.append({"a": record(np.array([x], dtype=dtype)), "b": record(np.array([y], dtype=dtype)), "op": name, "result": record(op(np.array([x], dtype=dtype), np.array([y], dtype=dtype)))})
    casts_input = np.array([complex(x,7) for x in [float("nan"),float("inf"),-float("inf"),-1.5,256.5,1e30]],dtype=dtype)
    casts = {d: [str(v) if d in ["int64","uint64"] else v for v in casts_input.astype(d).tolist()] for d in dtypes if d not in ["float32","float64","complex64","complex128"]}
    cancellation = np.array([1e20+1e20j, 1+2j, -1e20-1e20j, 3+4j]*65,dtype=dtype)
    reduction = {op: record(getattr(np,op)(cancellation)) for op in ["sum","mean"]}
    result["cases"].append({"dtype": dtype, "castsInput": record(casts_input), "casts": casts, "pairwiseInput": record(cancellation), "pairwise": reduction, "a": record(a), "b": record(b), "operations": operations, "exceptional": exceptional, "npy": npy(a), "bigEndianNpy": npy(a.astype(a.dtype.newbyteorder('>'))), "fortranNpy": npy(np.asfortranarray(a))})
x = np.array([1+2j, 2+3j, 3+4j], dtype='complex128')
result["fractionalPower"] = {"source": "https://github.com/numpy/numpy/blob/v2.5.4/numpy/_core/tests/test_umath.py#L1139", "criterion": "assert_almost_equal decimal=7, absolute component error < 1.5e-7", "a": record(x), "result": record(x**0.5)}
Path(__file__).with_name('complex-dtypes.json').write_text(json.dumps(result, indent=2) + '\n')
