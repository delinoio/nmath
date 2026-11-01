# SPDX-License-Identifier: Apache-2.0
"""Development-only fixed NumPy oracle for elementwise math ufuncs.

Run with numpy 2.5.4 and mpmath 1.3.0 on PATH, then check in the
little-endian JSON output.

Transcendental cases record a correctly rounded reference computed with
mpmath. Their tolerance is the maximum ULP error that NumPy's own accuracy
suite (numpy/_core/tests/test_umath_accuracy.py and its validation CSVs)
allows for the same function and dtype. float16 uses the 1-ULP bound of
NumPy's fp16 transcendental test. Other functions must match NumPy exactly.
"""
import csv
import json
import platform
import warnings
from pathlib import Path

import mpmath
import numpy as np

assert np.__version__ == "2.5.4"
assert mpmath.__version__ == "1.3.0"
mpmath.mp.prec = 256

TRANSCENDENTAL = {
    "cbrt": lambda x: mpmath.sign(x) * mpmath.cbrt(abs(x)),
    "exp": mpmath.exp,
    "exp2": lambda x: mpmath.power(2, x),
    "expm1": mpmath.expm1,
    "log": mpmath.log,
    "log2": lambda x: mpmath.log(x, 2),
    "log10": mpmath.log10,
    "log1p": mpmath.log1p,
    "sin": mpmath.sin,
    "cos": mpmath.cos,
    "tan": mpmath.tan,
    "arcsin": mpmath.asin,
    "arccos": mpmath.acos,
    "arctan": mpmath.atan,
    "sinh": mpmath.sinh,
    "cosh": mpmath.cosh,
    "tanh": mpmath.tanh,
    "arcsinh": mpmath.asinh,
    "arccosh": mpmath.acosh,
    "arctanh": mpmath.atanh,
}
EXACT = [
    # IEEE 754 requires correctly rounded square roots.
    "sqrt",
    "deg2rad",
    "rad2deg",
    "fabs",
    "rint",
    "spacing",
    "absolute",
    "square",
    "reciprocal",
    "sign",
    "floor",
    "ceil",
    "trunc",
    "conjugate",
    "isnan",
    "isinf",
    "isfinite",
    "signbit",
    "logical_not",
]
REAL = [
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
]
COMPLEX_OPS = {"conjugate", "isnan", "isinf", "isfinite", "logical_not"}


def numpy_tolerances():
    """Maximum ULP error per function and dtype from NumPy's validation sets."""
    data = Path(np.__file__).parent / "_core/tests/data"
    out = {}
    for name in TRANSCENDENTAL:
        path = data / f"umath-validation-set-{name}.csv"
        if not path.exists():
            continue
        bounds = {}
        with path.open() as f:
            for row in csv.reader(line for line in f if line.startswith("np.")):
                dtype = row[0].removeprefix("np.")
                bounds[dtype] = max(bounds.get(dtype, 0), int(row[3]))
        out[name] = bounds
    return out


TOLERANCES = numpy_tolerances()


def packed(a):
    a = np.asarray(a)
    return {
        "dtype": a.dtype.name,
        "shape": list(a.shape),
        "hex": a.astype(a.dtype.newbyteorder("<")).tobytes().hex(),
    }


def inputs(dtype):
    if dtype == "bool":
        return np.array([[True, False], [False, True]])
    if dtype.startswith(("int", "uint")):
        info = np.iinfo(dtype)
        values = [0, 1, 2, 7, 100, int(info.max), int(info.min)]
        if info.min < 0:
            values += [-1, -2, -7, -100]
        return np.array(values, dtype=dtype)
    rng = np.random.default_rng(20261101)
    special = [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.0, 1.5, 2.5, -2.5, 0.999, 3.0, 10.0,
               -3.7, 1e-5, 100.0, np.inf, -np.inf, np.nan]
    big = {"float16": 60000.0, "float32": 3e38, "float64": 1e300}[dtype]
    values = np.array(special + [big, -big], dtype=dtype)
    spread = rng.uniform(-20, 20, 48).astype(dtype)
    small = rng.uniform(-1, 1, 24).astype(dtype)
    return np.concatenate([values, spread, small])


def complex_inputs(dtype):
    re = np.array([0.0, -0.0, 1.0, np.nan, np.inf, -2.5, 0.0])
    im = np.array([0.0, 1.0, -0.0, 0.0, 0.0, np.inf, np.nan])
    return (re + 1j * im).astype(dtype)


def reference(name, values, dtype):
    """Correctly rounded reference values in the result dtype."""
    fn = TRANSCENDENTAL[name]
    out = []
    for x in values.astype(np.float64).ravel().tolist():
        try:
            if np.isnan(x):
                y = float("nan")
            else:
                r = fn(mpmath.mpf(x))
                y = float("nan") if isinstance(r, mpmath.mpc) else float(r)
        except (ValueError, ZeroDivisionError, OverflowError):
            y = float("nan")
        out.append(y)
    return np.array(out, dtype=np.float64).reshape(values.shape).astype(dtype)


cases = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name in list(TRANSCENDENTAL) + EXACT:
        ufunc = getattr(np, name)
        dtypes = REAL + (["complex64", "complex128"] if name in COMPLEX_OPS else [])
        for dtype in dtypes:
            if name == "sign" and dtype == "bool":
                continue
            if name == "reciprocal" and not dtype.startswith("float"):
                continue
            x = complex_inputs(dtype) if dtype.startswith("complex") else inputs(dtype)
            expected = ufunc(x)
            case = {"op": name, "input": packed(x), "numpy": packed(expected)}
            if name in TRANSCENDENTAL:
                result = expected.dtype.name
                case["reference"] = packed(reference(name, x, result))
                case["maxUlp"] = 1 if result == "float16" else TOLERANCES[name][result]
            cases.append(case)

output = {
    "numpy": np.__version__,
    "mpmath": mpmath.__version__,
    "environment": {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    },
    "criterion": (
        "Transcendental functions: within maxUlp of the correctly rounded mpmath reference, "
        "where maxUlp is NumPy 2.5.4's accuracy-suite bound for the function and dtype "
        "(1 ULP for float16). Other functions: exact agreement with NumPy, with NaN "
        "payloads ignored. Result dtypes and shapes must match NumPy exactly."
    ),
    "tolerances": TOLERANCES,
    "cases": cases,
}
Path(__file__).with_name("oracle.json").write_text(json.dumps(output, indent=1) + "\n")
