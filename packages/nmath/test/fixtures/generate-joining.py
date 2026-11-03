# SPDX-License-Identifier: Apache-2.0
"""Development-only fixed NumPy oracle for concatenate, stack, vstack, hstack,
dstack, column_stack, append and atleast_1d/2d/3d.

Run from this directory:
uv run --no-project --python 3.13 --with numpy==2.5.4 python generate-joining.py

No tolerance is needed: joining only copies elements and casts them to the
promoted dtype, and every operand value is exactly representable in every
dtype it is cast to. This is not a criterion for platform acceptance.
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
        data = data % 7 - 3
    return np.array(data, dtype=dtype).reshape(shape)


cases = []


def add(fn, operands, expected, **kwargs):
    if isinstance(expected, tuple):
        expected = [pack(x) for x in expected]
    else:
        expected = pack(expected)
    cases.append({"id": f"{fn}/{len(cases)}", "fn": fn, "operands": [pack(x) for x in operands],
                  "kwargs": kwargs, "expected": expected})


DTYPES = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
          "float16", "float32", "float64", "complex64", "complex128"]

# One dtype, several shapes and axes.
concat_shapes = [
    ([[3], [2]], 0), ([[3], [0], [1]], 0), ([[2, 3], [1, 3]], 0), ([[2, 3], [2, 1]], 1),
    ([[2, 3], [2, 2]], -1), ([[2, 3, 4], [2, 1, 4], [2, 2, 4]], 1), ([[2, 3, 4], [2, 3, 2]], -1),
    ([[0, 3], [0, 3]], 0), ([[2, 0], [2, 3]], 1), ([[1, 2, 2, 2], [3, 2, 2, 2]], -4), ([[4]], 0),
]
for dtype in DTYPES:
    for shapes, axis in concat_shapes:
        arrays = [operand(s, dtype, i) for i, s in enumerate(shapes)]
        add("concatenate", arrays, np.concatenate(arrays, axis), axis=axis)
    arrays = [operand([2, 3], dtype), operand([], dtype, 1), operand([4], dtype, 2)]
    add("concatenate", arrays, np.concatenate(arrays, None), axis=None)

# Mixed dtypes promote like `np.result_type` over every operand.
MIXED = [
    ["bool", "int8"], ["int8", "uint8"], ["int16", "uint32"], ["int64", "uint64"], ["uint8", "float16"],
    ["int16", "float16"], ["int32", "float32"], ["float16", "float32"], ["float32", "complex64"],
    ["float64", "complex64"], ["int64", "complex64"], ["bool", "uint16", "int8"],
    ["uint8", "int8", "float32", "complex64"], ["bool", "bool"],
    # Pairwise promotion is not associative; NumPy resolves these in any order.
    ["int8", "uint8", "float16"], ["float16", "uint8", "int8"], ["int16", "uint16", "float16"],
    ["uint64", "int8", "float16"],
]
for dtypes in MIXED:
    arrays = [operand([2, 2], d, i) for i, d in enumerate(dtypes)]
    add("concatenate", arrays, np.concatenate(arrays, 0), axis=0)
    add("concatenate", arrays, np.concatenate(arrays, 1), axis=1)

# Strings widen to the longest width; datetimes and timedeltas of one unit join.
for dtype in ["str", "datetime64[s]", "timedelta64[ms]", "datetime64[D]"]:
    arrays = [operand([2, 1], dtype), operand([2, 3], dtype, 5)]
    add("concatenate", arrays, np.concatenate(arrays, 1), axis=1)
    add("stack", arrays[:1] * 3, np.stack(arrays[:1] * 3, 1), axis=1)
arrays = [np.array(["x"]), np.array(["longer", "mid"])]
add("concatenate", arrays, np.concatenate(arrays), axis=0)

for dtype in ["int32", "float64", "complex128", "bool", "uint8"]:
    for shape in [[], [3], [2, 3], [2, 0, 3]]:
        rank = len(shape) + 1
        for axis in sorted({0, rank - 1, -1, -rank, min(1, rank - 1)}):
            arrays = [operand(shape, dtype, i) for i in range(3)]
            add("stack", arrays, np.stack(arrays, axis), axis=axis)
arrays = [operand([2, 3], "int8"), operand([2, 3], "float32", 1)]
add("stack", arrays, np.stack(arrays, 1), axis=1)
add("stack", arrays[:1], np.stack(arrays[:1], -1), axis=-1)

# vstack/hstack/dstack/column_stack accept mixed ranks after their promotions.
STACK_SHAPES = [
    [[3], [3]], [[], []], [[2, 3], [3]], [[3], [2, 3], [1, 3]], [[2, 3, 2], [1, 3, 2]],
    [[4], [2]], [[2, 3], [2, 1]], [[0], [3]], [[2], []], [[2, 1], [2, 2, 1]],
]
for fn in ["vstack", "hstack", "dstack", "column_stack"]:
    for dtype in ["int16", "float64", "complex64", "bool"]:
        for shapes in STACK_SHAPES:
            arrays = [operand(s, dtype, i) for i, s in enumerate(shapes)]
            try:
                expected = getattr(np, fn)(arrays)
            except ValueError:
                continue
            add(fn, arrays, expected)
    arrays = [operand([2], "uint8"), operand([2], "int8", 1)]
    add(fn, arrays, getattr(np, fn)(arrays))

for shapes, axis in [([[2, 3], [4]], None), ([[2, 3], [2, 3]], None), ([[3], [2]], None), ([[], [2]], None),
                     ([[2, 3], [1, 3]], 0), ([[2, 3], [2, 2]], 1), ([[2, 3], [2, 2]], -1)]:
    for dtypes in [["int32", "int32"], ["uint8", "float32"], ["bool", "complex128"]]:
        a, v = (operand(s, d, i) for i, (s, d) in enumerate(zip(shapes, dtypes)))
        add("append", [a, v], np.append(a, v, axis), axis=axis)

for fn in ["atleast_1d", "atleast_2d", "atleast_3d"]:
    for shape in [[], [3], [2, 3], [2, 3, 4], [1, 2, 3, 4], [0], [2, 0]]:
        a = operand(shape, "float32")
        add(fn, [a], getattr(np, fn)(a))
    arrays = [operand([], "int8"), operand([3], "complex64", 1), operand([2, 3], "bool", 2)]
    add(fn, arrays, getattr(np, fn)(*arrays))

Path(__file__).with_name("joining.json").write_text(json.dumps({
    "numpy": np.__version__,
    "environment": {"python": platform.python_version(), "system": platform.system(),
                    "machine": platform.machine(), "byteorder": sys.byteorder},
    "criterion": "exact",
    "cases": cases,
}, indent=1) + "\n")
