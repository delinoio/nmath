# SPDX-License-Identifier: Apache-2.0
"""Generate exact shape-transform values and sharing decisions with NumPy."""
import json
import platform
import sys
from pathlib import Path
import numpy as np

assert np.__version__ == "2.5.4", np.__version__

def pack(a):
    values = a.ravel().tolist()
    if a.dtype.kind == "c":
        values = [component for z in values for component in [z.real, z.imag]]
    elif str(a.dtype) in ["int64", "uint64"]:
        values = list(map(str, values))
    return {"shape": list(a.shape), "dtype": str(a.dtype), "values": values}

def apply(a, operations):
    for op in operations:
        if op["kind"] == "transpose":
            a = a.transpose(op.get("axes"))
        elif op["kind"] == "slice":
            a = a[tuple(slice(*s) for s in op["selectors"])]
        else:
            a = a.reshape(op["shape"], order="C", copy=op.get("copy"))
    return a

cases = []
dtypes = "bool int8 uint8 int16 uint16 int32 uint32 int64 uint64 float16 float32 float64 complex64 complex128".split()
scenarios = [
    ("transpose-default", [2, 3, 4], [{"kind": "transpose"}]),
    ("transpose-negative", [2, 3, 4], [{"kind": "transpose", "axes": [-1, 0, 1]}]),
    ("reshape-inferred", [2, 3, 4], [{"kind": "reshape", "shape": [6, -1]}]),
    ("reshape-copy", [2, 3], [{"kind": "reshape", "shape": [3, 2], "copy": True}]),
    ("transpose-copy", [2, 3], [{"kind": "transpose"}, {"kind": "reshape", "shape": [3, 2], "copy": True}]),
    ("transpose-flat", [2, 3], [{"kind": "transpose"}, {"kind": "reshape", "shape": [-1]}]),
    ("regular-step", [12], [{"kind": "slice", "selectors": [[None, None, 2]]}, {"kind": "reshape", "shape": [2, 3], "copy": False}]),
    ("reverse", [12], [{"kind": "slice", "selectors": [[None, None, -1]]}, {"kind": "reshape", "shape": [3, -1], "copy": False}]),
    ("reverse-matrix", [2, 3], [{"kind": "slice", "selectors": [[None, None, -1], [None, None, -1]]}, {"kind": "reshape", "shape": [6], "copy": False}]),
    ("singleton", [2, 1, 3], [{"kind": "transpose", "axes": [1, 0, 2]}, {"kind": "reshape", "shape": [6], "copy": False}]),
    ("empty", [0, 3], [{"kind": "reshape", "shape": [-1, 2]}]),
    ("scalar", [], [{"kind": "transpose", "axes": []}, {"kind": "reshape", "shape": [1, 1]}]),
]
for dtype in dtypes:
    for name, shape, operations in scenarios:
        data = np.arange(int(np.prod(shape)), dtype=np.int64)
        if dtype == "bool":
            data = data % 3 == 1
        elif dtype.startswith("complex"):
            data = data % 7 - 3 + (data % 3 - 1) * 1j
        elif dtype.startswith("uint"):
            data %= 7
        else:
            data = data % 7 - 3
        a = np.array(data, dtype=dtype).reshape(shape)
        result = apply(a, operations)
        cases.append({"id": f"{dtype}/{name}", "input": pack(a), "operations": operations, "expected": pack(result), "sharesMemory": bool(np.shares_memory(a, result)) if a.size else None})
Path(__file__).with_name("layout.json").write_text(json.dumps({"numpy": np.__version__, "criterion": "exact-values-and-sharing", "environment": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(), "byteorder": sys.byteorder}, "cases": cases}, indent=2) + "\n")
