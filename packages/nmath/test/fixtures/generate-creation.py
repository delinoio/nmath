# SPDX-License-Identifier: Apache-2.0
"""Development-only fixed NumPy oracle for linspace, logspace, eye and identity.

Run with numpy 2.5.4 from this directory, for example:
uv run --no-project --python 3.13 --with numpy==2.5.4 python generate-creation.py
"""
import json
import math
import platform
import struct
from pathlib import Path

import numpy as np

assert np.__version__ == "2.5.4"


def bits64(x):
    """Lossless big-endian IEEE-754 binary64 encoding, including signed zeros and NaN."""
    return struct.pack(">d", float(x)).hex()


def enc(a):
    a = np.asarray(a)
    flat = a.ravel()
    if a.dtype.kind == "c":
        width = a.dtype.itemsize // 2
        values = [[np.asarray(z.real).view(f"u{width}").item(), np.asarray(z.imag).view(f"u{width}").item()]
                  for z in flat]
        values = [[f"{re:0{width * 2}x}", f"{im:0{width * 2}x}"] for re, im in values]
    elif a.dtype.kind == "f":
        width = a.dtype.itemsize
        values = [f"{v:0{width * 2}x}" for v in flat.view(f"u{width}").tolist()]
    elif a.dtype.kind in "iu":
        # Strings keep 64-bit integers exact through JSON.
        values = [str(int(v)) for v in flat]
    else:
        values = [bool(v) for v in flat]
    return {"dtype": a.dtype.name, "shape": list(a.shape), "values": values}


INF = math.inf
cases = []


def linspace(start, stop, num, endpoint=True, dtype=None):
    y, step = np.linspace(start, stop, num, endpoint=endpoint, retstep=True, dtype=dtype)
    cases.append({
        "fn": "linspace",
        "args": {"start": bits64(start), "stop": bits64(stop), "num": num, "endpoint": endpoint,
                 "dtype": None if dtype is None else np.dtype(dtype).name},
        "expected": enc(y),
        "step": bits64(step),
    })


def logspace(start, stop, num, endpoint=True, base=10.0, dtype=None):
    y = np.logspace(start, stop, num, endpoint=endpoint, base=base, dtype=dtype)
    cases.append({
        "fn": "logspace",
        "args": {"start": bits64(start), "stop": bits64(stop), "num": num, "endpoint": endpoint,
                 "base": bits64(base), "dtype": None if dtype is None else np.dtype(dtype).name},
        "expected": enc(y),
    })


def eye(n, m=None, k=0, dtype=float):
    cases.append({
        "fn": "eye",
        "args": {"N": n, "M": m, "k": k, "dtype": np.dtype(dtype).name},
        "expected": enc(np.eye(n, m, k=k, dtype=dtype)),
    })


def identity(n, dtype=float):
    cases.append({
        "fn": "identity",
        "args": {"n": n, "dtype": np.dtype(dtype).name},
        "expected": enc(np.identity(n, dtype=dtype)),
    })


# linspace: defaults, counts, endpoints, reversed and degenerate ranges.
linspace(0.0, 1.0, 50)
linspace(2.0, 3.0, 5)
linspace(2.0, 3.0, 5, endpoint=False)
linspace(0.1, 0.7, 7)
linspace(0.1, 0.7, 7, endpoint=False)
linspace(1.0, 2.0, 0)
linspace(1.0, 2.0, 0, endpoint=False)
linspace(1.0, 2.0, 1)
linspace(1.0, 2.0, 1, endpoint=False)
linspace(-3.0, 4.0, 2)
linspace(5.0, 5.0, 4)
linspace(5.0, 5.0, 4, endpoint=False)
linspace(-0.0, -0.0, 3)
linspace(-0.0, 1.0, 1)
linspace(10.0, -3.0, 7)
linspace(10.0, -3.0, 7, endpoint=False)
linspace(-2.5, 7.25, 13)
linspace(1e-10, 1e10, 11)
linspace(1e300, -1e300, 9)
linspace(-1.7e308, 1.7e308, 5)
linspace(0.0, INF, 3)
linspace(0.0, 5e-324, 3)
linspace(0.0, 1e-320, 10)
linspace(-1e-315, 1e-315, 6, endpoint=False)
linspace(123456789.123, 123456789.987, 9)
linspace(0.0, 1.0, 1001)
# Floating dtypes evaluate in float64 and round once.
linspace(0.0, 1.0, 7, dtype=np.float32)
linspace(1e-3, 1e3, 9, dtype=np.float32)
linspace(-1.0, 1.0, 10, endpoint=False, dtype=np.float32)
linspace(0.0, 1.0, 11, dtype=np.float16)
linspace(0.0, 1.0, 3, dtype=np.complex128)
linspace(-1.0, 3.0, 5, endpoint=False, dtype=np.complex64)
# Integer dtypes floor before the cast.
linspace(-1.5, 2.5, 5, dtype=np.int32)
linspace(0.0, 10.0, 4, dtype=np.int32)
linspace(10.0, 0.0, 4, dtype=np.int32)
linspace(-7.0, 7.0, 5, endpoint=False, dtype=np.int32)
linspace(-0.5, 0.5, 1, dtype=np.int32)
linspace(0.0, 4294967295.0, 4, dtype=np.uint32)
linspace(0.0, 255.0, 6, dtype=np.uint8)
linspace(-128.0, 127.0, 4, dtype=np.int8)
linspace(-3.0, 3.0, 5, dtype=np.int16)
linspace(0.0, 65535.0, 3, dtype=np.uint16)
linspace(0.0, 2.0 ** 62, 5, dtype=np.int64)
linspace(-1e18, 1e18, 7, dtype=np.int64)
linspace(-9.2e18, 9.2e18, 3, dtype=np.int64)
linspace(0.0, 1.8e19, 4, dtype=np.uint64)
linspace(0.0, 1.0, 3, dtype=np.bool_)
linspace(-1.0, 1.0, 3, dtype=np.bool_)

# logspace: base ** linspace(start, stop) in float64, then one cast.
logspace(2.0, 3.0, 4)
logspace(2.0, 3.0, 4, endpoint=False)
logspace(2.0, 3.0, 4, base=2.0)
logspace(0.0, 1.0, 50)
logspace(-3.0, 3.0, 13)
logspace(0.1, 1.0, 10)
logspace(1.0, -1.0, 5, base=0.5)
logspace(0.0, 10.0, 11, base=math.e)
logspace(0.0, 2.0, 5, base=-2.0)
logspace(-300.0, 300.0, 7)
logspace(1.0, 2.0, 0)
logspace(1.0, 2.0, 1)
logspace(0.0, 2.0, 9, dtype=np.float32)
logspace(0.0, 3.0, 4, dtype=np.int32)
logspace(0.0, 18.0, 7, dtype=np.int64)
logspace(0.0, 10.0, 11, base=2.0, dtype=np.uint16)
# C pow returns 1 for a base of 1 with any exponent and for -1 ** +-inf, where JS
# Math.pow returns NaN; non-finite endpoints and an overflowing span produce them.
logspace(0.0, INF, 3, base=1.0)
logspace(0.0, -INF, 3, base=1.0)
logspace(0.0, INF, 3, base=-1.0)
logspace(0.0, -INF, 3, base=-1.0)
logspace(-1e308, 1e308, 3, base=1.0)
logspace(math.nan, 1.0, 3, base=1.0)
logspace(0.0, INF, 3, base=1.0, dtype=np.int32)
logspace(0.0, INF, 3, base=math.nan)

# eye and identity: square, non-square, and every diagonal offset class.
for k in [0, 1, -1, 2, -2, 3, -3, 5, -5]:
    eye(3, k=k)
for n, m, k in [(2, 4, 0), (2, 4, 1), (2, 4, 3), (2, 4, -1), (4, 2, 0), (4, 2, 1), (4, 2, -1), (4, 2, -3),
                (4, 2, -4), (0, None, 0), (0, 3, 0), (3, 0, 0), (1, None, 0), (5, 3, -2)]:
    eye(n, m, k=k)
for dtype in [np.int32, np.int64, np.uint8, np.bool_, np.float32, np.float16, np.complex128, np.complex64]:
    eye(3, 4, k=1, dtype=dtype)
identity(3)
identity(1, np.int8)
identity(0)
identity(4, np.bool_)

Path(__file__).with_name("creation.json").write_text(json.dumps({
    "numpy": np.__version__,
    "environment": {"python": platform.python_version(), "machine": platform.machine(),
                    "system": platform.system()},
    "encoding": "Floats are big-endian IEEE-754 bit strings in hexadecimal; integers are decimal strings.",
    "criterion": "linspace, eye and identity: exact shape, dtype and element bits (any NaN matches NaN). "
                 "logspace: exact shape and dtype; float64 within 4 ULP, float32 within 1 ULP, "
                 "integer results exact for the recorded inputs.",
    "cases": cases,
}, indent=1) + "\n")
