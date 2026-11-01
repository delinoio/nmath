# SPDX-License-Identifier: Apache-2.0
"""Development-only fixed NumPy oracle for any, all, count_nonzero, isclose,
allclose, array_equal and array_equiv.

Run with numpy 2.5.4 from this directory:
uv run --no-project --python 3.13 --with numpy==2.5.4 python generate-logic.py
"""
import json
import platform
from pathlib import Path

import numpy as np

assert np.__version__ == "2.5.4"
rng = np.random.default_rng(20261105)
INF = float("inf")
NAN = float("nan")


def real(v):
    # JSON has no NaN or infinity literals; they travel as strings.
    v = float(v)
    if np.isnan(v):
        return "nan"
    if np.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def enc(a):
    a = np.asarray(a)
    flat = a.ravel()
    if a.dtype.kind == "c":
        values = [[real(z.real), real(z.imag)] for z in flat]
    elif a.dtype.kind in "iu":
        values = [str(int(v)) for v in flat]
    elif a.dtype.kind == "b":
        values = [bool(v) for v in flat]
    else:
        values = [real(v) for v in flat]
    return {"dtype": a.dtype.name, "shape": list(a.shape), "values": values}


def operand(x):
    if isinstance(x, np.ndarray):
        return {"array": enc(x)}
    if isinstance(x, bool):
        return {"scalar": "bool", "value": x}
    if isinstance(x, int):
        return {"scalar": "int", "value": str(x)}
    if isinstance(x, complex):
        return {"scalar": "complex", "value": [real(x.real), real(x.imag)]}
    return {"scalar": "float", "value": real(x)}


cases = []


def add(fn, inputs, args, expected):
    expected = np.asarray(expected)
    assert expected.dtype.kind in "biu", (fn, expected.dtype)
    cases.append({
        "fn": fn,
        "inputs": [operand(x) for x in inputs],
        "args": args,
        "expected": enc(expected),
    })


def reduced(x, axis, keepdims):
    # One input per case keeps the fixture small; each result names its arguments.
    key = tuple(axis) if isinstance(axis, list) else axis
    return {"args": {"axis": axis, "keepdims": keepdims},
            **{fn: enc(getattr(np, fn)(x, axis=key, keepdims=keepdims))
               for fn in ["any", "all", "count_nonzero"]}}


def reductions(x, axes=None):
    if axes is None:
        axes = [None, 0, -1, [0], [0, 2] if x.ndim > 2 else [-1, 0], []] if x.ndim > 1 else [None, 0, []]
    results = [reduced(x, axis, keepdims) for axis in axes for keepdims in [False, True]]
    for result in results:
        for fn in ["any", "all", "count_nonzero"]:
            assert np.dtype(result[fn]["dtype"]).kind in "biu"
    cases.append({"fn": "reductions", "inputs": [operand(x)], "results": results})


def sparse(shape, dtype):
    # Mostly zeros so that any, all and count_nonzero differ along each axis.
    a = rng.integers(-3, 4, shape) * (rng.uniform(size=shape) < 0.5)
    if dtype == "bool":
        return a.astype(bool)
    if dtype.startswith("uint"):
        return np.abs(a).astype(dtype)
    if dtype.startswith("complex"):
        im = rng.integers(-2, 3, shape) * (rng.uniform(size=shape) < 0.3)
        return (a + 1j * im).astype(dtype)
    return a.astype(dtype)


for dtype in ["bool", "int8", "uint8", "int32", "int64", "uint64", "float16", "float32",
              "float64", "complex64", "complex128"]:
    reductions(sparse((2, 3, 4), dtype))
    reductions(sparse((5,), dtype))

special = np.array([[0.0, -0.0, NAN], [INF, -INF, 0.0], [0.0, 0.0, 0.0]])
reductions(special)
reductions(np.array([[0j, complex(0, -0.0)], [complex(NAN, 0), complex(0, 1e-300)]]))
reductions(np.ones((3, 2), dtype=bool))
for shape, dtype in [((0,), "float64"), ((0, 3), "float32"), ((3, 0), "bool"), ((2, 0, 3), "int32")]:
    reductions(np.zeros(shape, dtype=dtype),
               [None] + list(range(len(shape))) + ([[0, 2]] if len(shape) > 2 else []))
for scalar in [np.array(0.0), np.array(NAN), np.array(3, dtype="int16"), np.array(False)]:
    reductions(scalar, [None])


def close(a, b, **kw):
    add("isclose", [a, b], kw, np.isclose(a, b, **kw))
    add("allclose", [a, b], kw, np.allclose(a, b, **kw))


base = rng.uniform(-10, 10, 12)
noise = base * (1 + rng.choice([0, 1e-7, 5e-6, 2e-5, 1e-3], 12))
for dtype in ["float16", "float32", "float64"]:
    a = noise.astype(dtype)
    b = base.astype(dtype)
    close(a, b)
    close(a, b, rtol=1e-3)
    close(a, b, rtol=0.0, atol=1e-4)
    close(b, a, rtol=1e-6, atol=0.0)
fx = np.array([1.0, NAN, INF, -INF, INF, NAN, 0.0, 1e-9, 1e-7, -0.0])
fy = np.array([1.0, NAN, INF, INF, 1e300, 1.0, 1e-9, 0.0, 0.0, 0.0])
for dtype in ["float32", "float64"]:
    with np.errstate(over="ignore"):  # 1e300 becomes inf in float32
        fyd = fy.astype(dtype)
    close(fx.astype(dtype), fyd)
    close(fx.astype(dtype), fyd, equal_nan=True)
    close(fx.astype(dtype), fyd, atol=0.0, equal_nan=True)
# Broadcasting.
close(rng.uniform(size=(3, 1)), rng.uniform(size=(4,)), atol=0.3)
close(np.array([[1.0], [NAN]]), np.array([NAN, 1.0, 1.0 + 1e-6]), equal_nan=True)
close(np.zeros((0, 3)), np.zeros((3,)))
close(np.array(1.0), np.array(1.0 + 1e-6))
close(np.array(1.0), np.array([1.0, 2.0, NAN]), equal_nan=True)
# Integer and bool inputs: the reference becomes float64 before subtraction.
ints = np.array([0, 1, 100, -100, 2**31 - 1, -2**31], dtype="int32")
close(ints, ints)
close(ints, (ints + 1).astype("int32"), rtol=0.0, atol=1.0)
close(ints, ints.astype("float32"))
close(np.array([0, 255, 3], dtype="uint8"), np.array([0, 1, 3], dtype="int8"), atol=0.0)
close(np.array([2**62, -2**62, 2**53 + 1], dtype="int64"), np.array([2**62, -2**62, 2**53], dtype="float64"))
close(np.array([2**63 + 5, 0], dtype="uint64"), np.array([2**63, 1], dtype="uint64"), rtol=0.0, atol=0.0)
close(np.array([True, False, True]), np.array([True, True, False]))
close(np.array([True, False]), np.array([1.0, 1e-9]))
close(np.array([1.0, 1e-9, 2.0]), np.array([True, False, True]))
# Complex inputs; abs is the modulus.
cx = np.array([1 + 1j, 1 + 1j, complex(INF, 0), complex(INF, 0), complex(NAN, 0), complex(0, NAN),
               1e-9j, 3 - 4j])
cy = np.array([1 + 1j, 1 + 1.00001j, complex(INF, 0), complex(INF, 1), complex(NAN, 0),
               complex(NAN, 0), 0j, 3 - 4.0001j])
for dtype in ["complex64", "complex128"]:
    close(cx.astype(dtype), cy.astype(dtype))
    close(cx.astype(dtype), cy.astype(dtype), equal_nan=True, rtol=1e-4)
close(cx, np.array([1.0, 1.0, INF, INF, NAN, NAN, 0.0, 5.0]))
close(np.array([1.0, 2.0, NAN]), cy[:3].astype("complex64"))
# Weak JS scalars on either side.
fz = np.array([1.0, 1.0 + 1e-6, 1.1, NAN, INF, 0.0], dtype="float32")
for scalar in [1.0, 1, True, False, 0, NAN, INF, complex(1, 0), complex(1, 1e-6)]:
    close(fz, scalar)
    close(fz, scalar, equal_nan=True, rtol=0.2)
for scalar in [1.0, 1, NAN, complex(1, 0)]:
    close(scalar, fz)
    close(scalar, fz, equal_nan=True)
close(ints, 100, rtol=0.0, atol=0.5)
close(ints, 100.25, rtol=0.0, atol=0.5)
close(np.array([3, 4], dtype="int16"), complex(3, 0))


def equal(a, b, **kw):
    add("array_equal", [a, b], kw, np.array_equal(a, b, **kw))


m = rng.integers(-5, 5, (2, 3))
equal(m, m.copy())
equal(m, m.astype("float32"))
equal(m, m + (np.arange(6).reshape(2, 3) == 4))
equal(m, m.T)
equal(m, m.ravel())
equal(np.zeros((0,)), np.zeros((0,)))
equal(np.zeros((0,)), np.zeros((0, 1)))
equal(np.zeros((2, 0)), np.zeros((0, 2)))
equal(np.array(5.0), np.array(5, dtype="int8"))
nanv = np.array([1.0, NAN, 3.0])
for kw in [{}, {"equal_nan": True}]:
    equal(nanv, nanv.copy(), **kw)
    equal(nanv, np.array([1.0, 2.0, NAN]), **kw)
    equal(np.array([NAN, NAN]), np.array([NAN, NAN]), **kw)
    equal(np.array([NAN, NAN]), np.array([NAN, 1.0]), **kw)
    equal(np.array([complex(NAN, 0), 1j]), np.array([complex(0, NAN), 1j]), **kw)
    equal(np.array([1, 2], dtype="int32"), np.array([1, 2], dtype="uint8"), **kw)
    equal(np.array([True, False]), np.array([1.0, 0.0]), **kw)


def equiv(a, b):
    add("array_equiv", [a, b], {}, np.array_equiv(a, b))


row = np.array([1, 2, 3])
equiv(row, np.array([[1, 2, 3], [1, 2, 3]]))
equiv(row, np.array([[1, 2, 3], [1, 2, 4]]))
equiv(row, np.array([[1, 2], [1, 2]]))
equiv(row, np.array([1.0, 2.0, 3.0]))
equiv(np.array([[1], [1]]), np.array([1, 1, 1]))
equiv(np.ones((2, 1, 3)), np.ones((4, 1)))
equiv(np.zeros((0, 3)), np.zeros((2, 1, 3)))
equiv(np.zeros((0,)), np.zeros((2,)))
equiv(np.array([NAN]), np.array([NAN]))
equiv(np.array([2, 2]), 2)
equiv(np.array([2.0, 2.5]), 2.0)
equiv(np.array([1 + 0j, 1 + 0j]), True)

Path(__file__).with_name("logic.json").write_text(json.dumps({
    "numpy": np.__version__,
    "environment": {"python": platform.python_version(), "machine": platform.machine()},
    "criterion": ("Exact: bool results of any, all, isclose, allclose, array_equal and "
                  "array_equiv and int64 counts of count_nonzero match NumPy element by "
                  "element, including dtype and shape."),
    "cases": cases,
}, indent=1) + "\n")
