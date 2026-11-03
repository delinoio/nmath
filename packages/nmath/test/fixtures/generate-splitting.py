# SPDX-License-Identifier: Apache-2.0
"""Development-only fixed NumPy oracle for array_split, split, hsplit, vsplit,
dsplit, repeat, tile, roll and diff with prepend and append.

Run from this directory:
uv run --no-project --python 3.13 --with numpy==2.5.4 python generate-splitting.py

No tolerance is needed: these functions only select, copy and cast elements,
and the diff operands are small integers or dyadic fractions whose
differences are exact. This is not a criterion for platform acceptance.
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
    elif a.dtype.kind == "U":
        values = [str(v) for v in flat]
    elif a.dtype.kind in "mM":
        values = [str(v) for v in a.ravel().view(np.int64).tolist()]
    else:
        # JSON has no NaN or infinity literals.
        values = [None if v != v else ("inf" if v == float("inf") else "-inf" if v == float("-inf") else v)
                  for v in flat]
    packed = {"dtype": a.dtype.name, "shape": list(a.shape), "values": values}
    if a.dtype.kind == "U":
        packed["dtype"] = "str"
        packed["width"] = a.dtype.itemsize // 4
    if a.dtype.kind in "mM":
        packed["dtype"] = "datetime64" if a.dtype.kind == "M" else "timedelta64"
        packed["unit"] = np.datetime_data(a.dtype)[0]
    return packed


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
    elif dtype == "str":
        return np.array(["ab"[: v % 3] + chr(0x61 + v % 26) for v in data.tolist()]).reshape(shape)
    elif dtype.startswith("datetime64") or dtype.startswith("timedelta64"):
        return (data * 86400 - 1000).astype(dtype).reshape(shape)
    else:
        data = data * 3 % 11 - 5
    return np.array(data, dtype=dtype).reshape(shape)


def edge(value):
    """Encode a diff edge; Python ints travel as JS bigint, floats as number."""
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": str(value)}
    if isinstance(value, float):
        return {"kind": "float", "value": value}
    if isinstance(value, complex):
        return {"kind": "complex", "value": [value.real, value.imag]}
    return {"kind": "array", **pack(value)}


cases = []


def add(fn, operand_, expected, **kwargs):
    if isinstance(expected, list):
        expected = [pack(x) for x in expected]
    else:
        expected = pack(expected)
    cases.append({"id": f"{fn}/{len(cases)}", "fn": fn, "operand": pack(operand_), "kwargs": kwargs,
                  "expected": expected})


# One dtype per storage family keeps the fixture below the Git LFS threshold.
DTYPES = ["bool", "int8", "uint16", "int64", "uint64", "float16", "float64", "complex128", "str",
          "datetime64[s]", "timedelta64[ms]"]

split_cases = [
    ([6], 3, 0), ([6], 2, 0), ([6], [2, 4], 0), ([6], [4, 2], 0), ([6], [-2], 0), ([6], [10], 0),
    ([6], [], 0), ([6], [0, 0, 6], 0), ([4, 6], 2, 1), ([4, 6], [1, 5], -1), ([4, 6], 4, 0),
    ([2, 3, 4], [1, 3], 2), ([2, 3, 4], 3, -2), ([0, 3], 2, 0), ([3, 0], 3, 1),
]
for dtype in DTYPES:
    for shape, sections, axis in split_cases:
        a = operand(shape, dtype)
        add("split", a, np.split(a, sections, axis), sections=sections, axis=axis)
for shape, sections, axis in [([7], 3, 0), ([5], 5, 0), ([2], 4, 0), ([3, 7], 4, 1), ([0], 3, 0),
                              ([4, 6], [1, 5], 0), ([10], 4, -1)]:
    for dtype in ["int32", "float64", "str"]:
        a = operand(shape, dtype)
        add("array_split", a, np.array_split(a, sections, axis), sections=sections, axis=axis)
for fn, shapes in [("hsplit", [[6], [2, 6], [2, 6, 2]]), ("vsplit", [[6, 2], [6, 2, 3]]), ("dsplit", [[2, 2, 6], [1, 2, 6, 2]])]:
    for shape in shapes:
        for sections in [2, [1, 4]]:
            for dtype in ["uint8", "complex64"]:
                a = operand(shape, dtype)
                add(fn, a, getattr(np, fn)(a, sections), sections=sections)

repeat_cases = [
    ([3], 2, None), ([3], 0, None), ([2, 3], 2, None), ([2, 3], [1, 0, 2], 1), ([2, 3], [3, 1], 0),
    ([2, 3], 2, -1), ([2, 3], [2], 0), ([], 3, None), ([0, 2], 2, 0), ([2, 0], 3, 1), ([2, 2, 2], [0, 1], 2),
]
for dtype in DTYPES:
    for shape, repeats, axis in repeat_cases:
        a = operand(shape, dtype)
        add("repeat", a, np.repeat(a, repeats, axis), repeats=repeats, axis=axis)
# A 0-d input with an axis repeats as its 1-D form.
for dtype in ["int64", "str"]:
    for repeats, axis in [(3, 0), ([2], -1), (0, 0)]:
        a = operand([], dtype)
        add("repeat", a, np.repeat(a, repeats, axis), repeats=repeats, axis=axis)

tile_cases = [
    ([3], 2), ([3], [2, 2]), ([2, 3], 2), ([2, 3], [2, 1]), ([2, 3], [1, 1]), ([2, 3], [3, 1, 2]),
    ([2, 1, 2], [2, 2]), ([], 3), ([], [2, 2]), ([3], 0), ([2, 3], [0, 2]), ([0, 3], [2, 2]), ([3], []),
]
for dtype in DTYPES:
    for shape, reps in tile_cases:
        a = operand(shape, dtype)
        add("tile", a, np.tile(a, reps), reps=reps)

roll_cases = [
    ([5], 2, None), ([5], -1, None), ([5], 7, None), ([5], 0, None), ([2, 3], 1, None), ([2, 3], -4, None),
    ([2, 3], 1, 0), ([2, 3], 1, 1), ([2, 3], -1, -1), ([2, 3], [1, 2], [0, 1]), ([2, 3], 1, [0, 1]),
    ([2, 3], [1, 1], 1), ([2, 3], [2, -1], [1, 1]), ([3, 4, 2], [1, -3, 5], [2, 1, 0]), ([0, 3], 2, 0),
    ([3, 0], 2, 1), ([], 3, None), ([4], [1, 2], None),
]
for dtype in DTYPES:
    for shape, shift, axis in roll_cases:
        a = operand(shape, dtype)
        add("roll", a, np.roll(a, shift, axis), shift=shift, axis=axis)
# Empty shift or axis lists broadcast to zero pairs and roll nothing.
for dtype in ["int64", "str"]:
    for shape, shift, axis in [([5], [], 0), ([5], 1, []), ([2, 3], [], []), ([5], [], None)]:
        a = operand(shape, dtype)
        add("roll", a, np.roll(a, shift, axis), shift=shift, axis=axis)

diff_cases = [
    ("int32", [4], 1, -1, 0, None), ("int32", [4], 1, -1, None, 5), ("int32", [4], 2, -1, 1, 1),
    ("int8", [4], 1, -1, 2, None), ("uint8", [4], 1, -1, None, 200), ("int16", [4], 1, -1, 0.5, None),
    ("float32", [2, 3], 1, 0, 1.5, None), ("float32", [2, 3], 1, 1, None, -2.0),
    ("float64", [2, 3], 2, 1, -1.0, 1.0), ("bool", [4], 1, -1, True, None), ("bool", [4], 1, -1, None, False),
    ("bool", [4], 1, -1, 1, None), ("complex128", [3], 1, -1, 1j, None), ("float32", [3], 1, -1, 1j, None),
    ("int64", [2, 3], 1, 0, 0, 0), ("uint64", [3], 1, -1, 0, None),
]
for dtype, shape, n, axis, prepend, append in diff_cases:
    a = operand(shape, dtype)
    kwargs = {}
    if prepend is not None:
        kwargs["prepend"] = prepend
    if append is not None:
        kwargs["append"] = append
    add("diff", a, np.diff(a, n, axis, **kwargs), n=n, axis=axis,
        **{key: edge(value) for key, value in kwargs.items()})
for dtype, shape, axis, prepend_shape, append_shape in [
    ("int32", [2, 3], 1, [2, 1], [2, 2]), ("float64", [2, 3], 0, [1, 3], None), ("int16", [3], -1, [], [2]),
    ("float32", [2, 3], 1, [], []), ("uint8", [2, 3, 2], 2, [2, 3, 1], []),
]:
    a = operand(shape, dtype)
    kwargs = {}
    if prepend_shape is not None:
        kwargs["prepend"] = operand(prepend_shape, "int8", 3)
    if append_shape is not None:
        kwargs["append"] = operand(append_shape, dtype, 5)
    add("diff", a, np.diff(a, 1, axis, **kwargs), n=1, axis=axis,
        **{key: edge(value) for key, value in kwargs.items()})
a = operand([3], "datetime64[s]")
add("diff", a, np.diff(a, 1, -1, prepend=operand([1], "datetime64[s]", 9)), n=1, axis=-1,
    prepend=edge(operand([1], "datetime64[s]", 9)))
a = operand([3], "int32")
add("diff", a, np.diff(a, 0, -1, prepend=7), n=0, axis=-1, prepend=edge(7))

Path(__file__).with_name("splitting.json").write_text(json.dumps({
    "numpy": np.__version__,
    "environment": {"python": platform.python_version(), "system": platform.system(),
                    "machine": platform.machine(), "byteorder": sys.byteorder},
    "criterion": "exact",
    "cases": cases,
}, indent=1) + "\n")
