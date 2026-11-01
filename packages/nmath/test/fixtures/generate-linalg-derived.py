# SPDX-License-Identifier: Apache-2.0
"""Development-only fixed NumPy oracle for norm, matrix_rank, pinv, cond and matrix_power.

Run with numpy 2.5.4 on PATH from this directory.
"""
import json
import platform
from pathlib import Path

import numpy as np

assert np.__version__ == "2.5.4"
rng = np.random.default_rng(20261104)


def enc(a):
    a = np.asarray(a)
    flat = a.ravel()
    if a.dtype.kind == "c":
        values = [[float(z.real), float(z.imag)] for z in flat]
    elif a.dtype.kind in "iu":
        values = [int(v) for v in flat]
    elif a.dtype.kind == "b":
        values = [bool(v) for v in flat]
    else:
        values = [None if np.isnan(v) else (str(v) if np.isinf(v) else float(v)) for v in flat]
    return {"dtype": a.dtype.name, "shape": list(a.shape), "values": values}


def dense(shape, dtype):
    a = rng.uniform(-1.5, 1.5, shape)
    if dtype.startswith("complex"):
        a = a + 1j * rng.uniform(-1.5, 1.5, shape)
    return a.astype(dtype)


cases = []
INF = float("inf")


def add(fn, x, args, expected):
    # JSON has no infinity literal; infinite orders travel as strings.
    args = {k: ("inf" if v == INF else "-inf" if v == -INF else v) if isinstance(v, float) else v
            for k, v in args.items()}
    cases.append({"fn": fn, "input": enc(x), "args": args, "expected": enc(expected)})


for dtype in ["float32", "float64", "complex64", "complex128"]:
    v = dense((5,), dtype)
    for ord_ in [None, 2, 1, INF, -INF, 0, 3, 0.5, -1]:
        add("norm", v, {"ord": ord_}, np.linalg.norm(v, ord_))
    m = dense((3, 4), dtype)
    for ord_ in [None, "fro", "nuc", 1, -1, 2, -2, INF, -INF]:
        add("norm", m, {"ord": ord_}, np.linalg.norm(m, ord_))
    t = dense((2, 3, 4), dtype)
    add("norm", t, {}, np.linalg.norm(t))
    add("norm", t, {"keepdims": True}, np.linalg.norm(t, keepdims=True))
    for axis in [0, -1, 1]:
        add("norm", t, {"ord": 1, "axis": axis}, np.linalg.norm(t, 1, axis=axis))
    for axis, ord_ in [((0, 2), "fro"), ((2, 0), 1), ((1, 2), 2), ((-1, -2), INF), ((0, 1), "nuc")]:
        add("norm", t, {"ord": ord_, "axis": list(axis)}, np.linalg.norm(t, ord_, axis=axis))
        add("norm", t, {"ord": ord_, "axis": list(axis), "keepdims": True},
            np.linalg.norm(t, ord_, axis=axis, keepdims=True))
    sq = dense((3, 3), dtype)
    tall = dense((2, 4, 3), dtype)
    add("pinv", tall, {}, np.linalg.pinv(tall))
    add("pinv", sq, {"rtol": None}, np.linalg.pinv(sq, rtol=None))
    add("cond", sq, {}, np.linalg.cond(sq))
    add("cond", tall, {"p": -2}, np.linalg.cond(tall, -2))
    low = np.outer(np.arange(1, 5), np.arange(1, 4)).astype(dtype)
    add("matrix_rank", low, {}, np.linalg.matrix_rank(low))
    add("matrix_rank", tall, {}, np.linalg.matrix_rank(tall))
    add("matrix_rank", tall, {"tol": 10.0}, np.linalg.matrix_rank(tall, tol=10.0))
    small = (dense((3, 3), dtype) * 0.5).astype(dtype)
    for n in [0, 1, 2, 3, 5, 6, -1, -3]:
        add("matrix_power", small, {"n": n}, np.linalg.matrix_power(small, n))

ints = np.array([[1, 2], [3, 4]], dtype="int32")
for n in [0, 1, 4, 7]:
    add("matrix_power", ints, {"n": n}, np.linalg.matrix_power(ints, n))
flags = np.array([[True, False], [True, True]])
add("matrix_power", flags, {"n": 3}, np.linalg.matrix_power(flags, 3))
add("norm", ints, {"ord": 1}, np.linalg.norm(ints, 1))
add("matrix_rank", np.zeros((3, 0)), {}, np.linalg.matrix_rank(np.zeros((3, 0))))
singular = np.array([[1.0, 2.0], [2.0, 4.0]])
add("cond", singular, {}, np.linalg.cond(singular))
# The singular condition number is rounding noise: NumPy's test_linalg only
# requires it to exceed 1e15, so the test applies that bound instead.
cases[-1]["atLeast"] = 1e15
add("pinv", singular, {}, np.linalg.pinv(singular))

Path(__file__).with_name("linalg-derived.json").write_text(json.dumps({
    "numpy": np.__version__,
    "environment": {"python": platform.python_version(), "machine": platform.machine()},
    "criterion": ("NumPy test_linalg decimals: norm, cond and matrix_power 6 (single) or 12; "
                  "pinv 5 or 11; integer, bool and rank results exact. Matrix-power criteria are "
                  "relative to max(1, |expected|)."),
    "cases": cases,
}, indent=1) + "\n")
