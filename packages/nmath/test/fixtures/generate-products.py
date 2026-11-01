# SPDX-License-Identifier: Apache-2.0
"""Development-only fixed NumPy oracle for dot, vdot, inner, outer, diagonal,
trace, clip and diff.

Run from this directory:
uv run --no-project --python 3.13 --with numpy==2.5.4 python generate-products.py

No tolerance is needed: the operands are small integers or dyadic fractions,
so every product, partial sum, difference and bound is exactly representable
in the result dtype. This is not a criterion for arbitrary long
floating-point products or platform acceptance.
"""
import json
import platform
import sys
from pathlib import Path

import numpy as np

assert np.__version__ == "2.5.4", np.__version__


def pack(a):
    a = np.asarray(a)
    flat = a.ravel().tolist()
    if a.dtype.kind == "c":
        values = [[float(z.real), float(z.imag)] for z in flat]
    elif a.dtype.kind in "iu":
        values = [str(v) for v in flat] if a.dtype.itemsize == 8 else [int(v) for v in flat]
    elif a.dtype.kind == "b":
        values = [bool(v) for v in flat]
    else:
        # JSON has no NaN or infinity literals.
        values = [None if v != v else ("inf" if v == float("inf") else "-inf" if v == float("-inf") else v)
                  for v in flat]
    return {"dtype": a.dtype.name, "shape": list(a.shape), "values": values}


def operand(shape, dtype, shift=0):
    data = np.arange(int(np.prod(shape)), dtype=np.int64) + shift
    if dtype == "bool":
        data = data % 3 != 1
    elif dtype.startswith("uint"):
        data = data % 7
    elif dtype.startswith("complex"):
        data = (data % 7 - 3) + (data % 3 - 1) * 1j
    elif dtype.startswith("float"):
        data = (data % 7 - 3) * 0.5
    else:
        data = data % 7 - 3
    return np.array(data, dtype=dtype).reshape(shape)


def bound(value):
    """Encode a clip bound; Python ints travel as JS bigint, floats as number."""
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": str(value)}
    if isinstance(value, float):
        return {"kind": "float", "value": pack(np.array(value))["values"][0]}
    if isinstance(value, complex):
        return {"kind": "complex", "value": [value.real, value.imag]}
    return {"kind": "array", **pack(value)}


cases = []


def add(fn, operands, expected, **kwargs):
    cases.append({"id": f"{fn}/{len(cases)}", "fn": fn, "operands": [pack(x) for x in operands],
                  "kwargs": kwargs, "expected": pack(expected)})


DTYPES = ["bool", "int32", "int64", "float32", "float64", "complex128"]

dot_shapes = [
    ([], [2, 3]), ([2, 3], []), ([3], [3]), ([2, 3], [3]), ([3], [3, 4]), ([2, 3], [3, 4]),
    ([2, 2, 3], [3]), ([2, 2, 3], [3, 4]), ([2, 3], [2, 3, 4]), ([2, 1, 3], [4, 3, 2]),
    ([3], [2, 3, 2]), ([0, 3], [3, 2]), ([2, 0], [0, 3]), ([2, 3, 0], [2, 0, 4]), ([3], [2, 3, 0]),
]
for dtype in DTYPES:
    for a_shape, b_shape in dot_shapes:
        a, b = operand(a_shape, dtype), operand(b_shape, dtype, 1)
        add("dot", [a, b], np.dot(a, b))
for a_dtype, b_dtype in [("int8", "uint8"), ("int32", "float32"), ("float32", "complex64"), ("bool", "int16")]:
    a, b = operand([2, 2, 3], a_dtype), operand([3, 2], b_dtype, 1)
    add("dot", [a, b], np.dot(a, b))

for dtype in DTYPES + ["complex64"]:
    for a_shape, b_shape in [([3], [3]), ([2, 3], [3, 2]), ([2, 2], [4]), ([0], [0]), ([], [])]:
        a, b = operand(a_shape, dtype), operand(b_shape, dtype, 2)
        add("vdot", [a, b], np.vdot(a, b))

for dtype in DTYPES:
    for a_shape, b_shape in [([], [2, 3]), ([3], [3]), ([2, 3], [4, 3]), ([2, 2, 3], [3]),
                             ([2, 3], [2, 2, 3]), ([0, 3], [2, 3]), ([2, 0], [3, 0])]:
        a, b = operand(a_shape, dtype), operand(b_shape, dtype, 1)
        add("inner", [a, b], np.inner(a, b))
    for a_shape, b_shape in [([3], [4]), ([2, 2], [3]), ([0], [3]), ([], [2])]:
        a, b = operand(a_shape, dtype), operand(b_shape, dtype, 1)
        add("outer", [a, b], np.outer(a, b))
a, b = operand([3], "int8"), operand([2], "float32", 1)
add("outer", [a, b], np.outer(a, b))

diagonal_cases = [
    ([3, 4], 0, 0, 1), ([3, 4], 1, 0, 1), ([3, 4], 3, 0, 1), ([3, 4], 4, 0, 1), ([3, 4], 9, 0, 1),
    ([3, 4], -1, 0, 1), ([3, 4], -2, 0, 1), ([3, 4], -3, 0, 1), ([3, 4], -5, 0, 1),
    ([4, 3], 1, 1, 0), ([4, 3], -1, 1, 0), ([2, 3, 4], 0, 0, 1), ([2, 3, 4], 1, 0, 2),
    ([2, 3, 4], -1, 2, 1), ([2, 3, 4], 0, -1, -3), ([2, 3, 4], 2, 1, 2), ([3, 0], 0, 0, 1),
    ([0, 0], 0, 0, 1), ([2, 0, 3], 1, 0, 2), ([2, 2, 2, 2], 0, 3, 1),
]
for dtype in DTYPES + ["uint8", "int8"]:
    for shape, offset, axis1, axis2 in diagonal_cases:
        a = operand(shape, dtype)
        kwargs = {"offset": offset, "axis1": axis1, "axis2": axis2}
        if dtype in DTYPES:
            add("diagonal", [a], np.diagonal(a, offset, axis1, axis2), **kwargs)
        add("trace", [a], np.trace(a, offset, axis1, axis2), **kwargs)

nan = float("nan")
inf = float("inf")
special = np.array([nan, -inf, -2.5, -0.5, 0.0, 0.5, 1.5, inf])
clip_cases = [
    (special, 0.0, 1.0), (special, 1.0, 0.0), (special, nan, 1.0), (special, 0.0, nan),
    (special, None, 0.5), (special, -0.5, None), (special, None, None),
    (special.astype(np.float32), -1.0, 1.0), (special.astype(np.float32), 2, None),
    (special.astype(np.float16), -1.0, 1.0),
    (operand([2, 4], "int32"), -1, 2), (operand([2, 4], "int32"), 2, -1),
    (operand([2, 4], "int32"), -1.5, 1.5), (operand([2, 4], "int32"), None, None),
    (np.array([-128, -5, 0, 100, 127], np.int8), -1000, 1000),
    (np.array([-128, -5, 0, 100, 127], np.int8), -1000, 50),
    (np.array([-128, -5, 0, 100, 127], np.int8), -3, 1000),
    (np.array([0, 3, 200, 255], np.uint8), -1, 100), (np.array([0, 3, 200, 255], np.uint8), 2, 256),
    (np.array([-(2**63), 0, 2**63 - 1], np.int64), -(2**63), 2**62),
    (np.array([0, 2**63, 2**64 - 1], np.uint64), 1, 2**64 - 1),
    (operand([4], "bool"), True, True), (operand([4], "bool"), False, None),
    (operand([2, 3], "int16"), np.array([-1, 0, 1], np.int16), np.array([[2], [0]], np.int16)),
    (operand([2, 3], "int32"), np.array([0, 1, -5], np.int8), 2),
    (operand([3], "float32"), np.array([0.0, nan, -1.0], np.float32), np.array([0.5], np.float64)),
    (operand([4], "complex128"), 0 + 0j, 1 + 0.5j), (operand([4], "complex64"), -1 - 1j, None),
]
for a, lo, hi in clip_cases:
    cases.append({"id": f"clip/{len(cases)}", "fn": "clip", "operands": [pack(a)],
                  "kwargs": {"a_min": bound(lo), "a_max": bound(hi)}, "expected": pack(np.clip(a, lo, hi))})

diff_cases = [
    (operand([6], "int32"), 1, -1), (operand([6], "int32"), 2, -1), (operand([6], "int32"), 5, -1),
    (operand([6], "int32"), 7, -1), (operand([6], "int32"), 0, -1), (operand([3, 4], "int64"), 1, 0),
    (operand([3, 4], "int64"), 2, 1), (operand([3, 4], "float32"), 2, 0), (operand([2, 3, 4], "float64"), 1, 1),
    (operand([2, 3, 4], "float64"), 3, -1), (operand([7], "bool"), 1, 0), (operand([7], "bool"), 2, 0),
    (operand([3, 3], "bool"), 1, 0), (np.array([5, 1, 0, 255], np.uint8), 1, 0),
    (np.array([5, 1, 0, 255], np.uint8), 2, 0), (special, 1, 0), (operand([4], "complex128"), 2, 0),
    (operand([3, 0], "int32"), 1, 1), (operand([0, 3], "float64"), 1, 0), (operand([2, 3], "int8"), 1, 0),
]
for a, n, axis in diff_cases:
    add("diff", [a], np.diff(a, n, axis), n=n, axis=axis)

Path(__file__).with_name("products.json").write_text(json.dumps({
    "numpy": np.__version__,
    "environment": {"python": platform.python_version(), "system": platform.system(),
                    "machine": platform.machine(), "byteorder": sys.byteorder},
    "criterion": "exact-small-values",
    "cases": cases,
}, indent=1) + "\n")
