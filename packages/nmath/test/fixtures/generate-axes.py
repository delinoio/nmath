# SPDX-License-Identifier: Apache-2.0
"""Development-only fixed NumPy oracle for axis-manipulation helpers.

Run with numpy 2.5.4 on PATH from this directory, for example:
uv run --no-project --python 3.13 --with numpy==2.5.4 python generate-axes.py
"""
import json
import math
import platform
from pathlib import Path

import numpy as np

assert np.__version__ == "2.5.4"

# (function, input shape, positional arguments after the array).
CASES = [
    ("squeeze", [1, 3, 1, 4], []),
    ("squeeze", [1, 3, 1, 4], [0]),
    ("squeeze", [1, 3, 1, 4], [-2]),
    ("squeeze", [1, 3, 1, 4], [[0, 2]]),
    ("squeeze", [1, 3, 1, 4], [[-4, -2]]),
    ("squeeze", [1, 3, 1, 4], [[]]),
    ("squeeze", [1, 3, 1, 4], [1]),
    ("squeeze", [1, 3, 1, 4], [4]),
    ("squeeze", [1, 3, 1, 4], [-5]),
    ("squeeze", [1, 3, 1, 4], [[0, 0]]),
    ("squeeze", [1, 3, 1, 4], [[0, -4]]),
    ("squeeze", [1, 3, 1, 4], [1.5]),
    ("squeeze", [1, 1], []),
    ("squeeze", [0, 1, 3], []),
    ("squeeze", [0, 1, 3], [1]),
    ("squeeze", [0, 1, 3], [0]),
    ("squeeze", [], []),
    ("squeeze", [], [0]),
    ("squeeze", [], [-1]),
    ("squeeze", [], [[0]]),
    ("squeeze", [], [1]),
    ("squeeze", [1, 3, 1, 4], [[1, 5]]),
    ("squeeze", [1, 3, 1, 4], [[1, 1]]),
    ("expand_dims", [2, 3], [0]),
    ("expand_dims", [2, 3], [2]),
    ("expand_dims", [2, 3], [-1]),
    ("expand_dims", [2, 3], [-3]),
    ("expand_dims", [2, 3], [[0, 1]]),
    ("expand_dims", [2, 3], [[0, -1]]),
    ("expand_dims", [2, 3], [[1, 3]]),
    ("expand_dims", [2, 3], [[0, 2, 4]]),
    ("expand_dims", [2, 3], [[-1, -5]]),
    ("expand_dims", [2, 3], [[]]),
    ("expand_dims", [2, 3], [3]),
    ("expand_dims", [2, 3], [-4]),
    ("expand_dims", [2, 3], [[0, 4]]),
    ("expand_dims", [2, 3], [[0, 0]]),
    ("expand_dims", [2, 3], [[0, -4]]),
    ("expand_dims", [2, 3], [1.5]),
    ("expand_dims", [], [0]),
    ("expand_dims", [], [-1]),
    ("expand_dims", [], [[0, 1]]),
    ("expand_dims", [0, 2], [1]),
    ("swapaxes", [2, 3, 4], [0, 2]),
    ("swapaxes", [2, 3, 4], [-1, 0]),
    ("swapaxes", [2, 3, 4], [1, 1]),
    ("swapaxes", [2, 3, 4], [1, -2]),
    ("swapaxes", [2, 3, 4], [0, 3]),
    ("swapaxes", [2, 3, 4], [-4, 0]),
    ("swapaxes", [2, 3, 4], [0.5, 1]),
    ("swapaxes", [0, 3], [0, 1]),
    ("swapaxes", [5], [0, -1]),
    ("moveaxis", [2, 3, 4], [0, -1]),
    ("moveaxis", [2, 3, 4], [-1, 0]),
    ("moveaxis", [2, 3, 4], [1, 1]),
    ("moveaxis", [2, 3, 4], [[0, 1], [-1, -2]]),
    ("moveaxis", [2, 3, 4], [[0, 1, 2], [2, 0, 1]]),
    ("moveaxis", [2, 3, 4], [[2, 0], [0, 1]]),
    ("moveaxis", [2, 3, 4], [[], []]),
    ("moveaxis", [2, 3, 4], [0, 3]),
    ("moveaxis", [2, 3, 4], [-4, 0]),
    ("moveaxis", [2, 3, 4], [[0, 0], [1, 2]]),
    ("moveaxis", [2, 3, 4], [[0, 1], [2, -1]]),
    ("moveaxis", [2, 3, 4], [[0, 1], [2]]),
    ("moveaxis", [2, 3, 4], [0, 1.5]),
    ("moveaxis", [3, 0, 2], [1, 0]),
    ("moveaxis", [], [[], []]),
    ("ravel", [2, 3, 4], []),
    ("ravel", [], []),
    ("ravel", [0, 3], []),
    ("ravel", [1], []),
    ("flip", [2, 3, 4], []),
    ("flip", [2, 3, 4], [0]),
    ("flip", [2, 3, 4], [-1]),
    ("flip", [2, 3, 4], [[0, 2]]),
    ("flip", [2, 3, 4], [[-3, 1]]),
    ("flip", [2, 3, 4], [[]]),
    ("flip", [2, 3, 4], [3]),
    ("flip", [2, 3, 4], [-4]),
    ("flip", [2, 3, 4], [[1, -2]]),
    ("flip", [2, 3, 4], [0.5]),
    ("flip", [], []),
    ("flip", [], [0]),
    ("flip", [0, 3], []),
    ("flip", [0, 3], [1]),
    ("fliplr", [2, 3, 4], []),
    ("fliplr", [3, 2], []),
    ("fliplr", [0, 2], []),
    ("fliplr", [3], []),
    ("fliplr", [], []),
    ("flipud", [2, 3, 4], []),
    ("flipud", [3], []),
    ("flipud", [0, 2], []),
    ("flipud", [], []),
]


def to_python(value):
    """Python tuples are NumPy's axis-sequence spelling; JS uses lists."""
    return tuple(value) if isinstance(value, list) else value


cases = []
for name, shape, args in CASES:
    a = np.arange(math.prod(shape), dtype=np.int32).reshape(shape)
    case = {"function": name, "shape": shape, "args": args}
    try:
        r = getattr(np, name)(a, *[to_python(x) for x in args])
        case["result"] = {"shape": list(r.shape), "values": r.ravel().tolist()}
    except Exception as error:  # noqa: BLE001 - the error class is the oracle.
        case["error"] = {"class": type(error).__name__, "message": str(error)}
    cases.append(case)

Path(__file__).with_name("axes.json").write_text(json.dumps({
    "numpy": np.__version__,
    "environment": {"python": platform.python_version(), "machine": platform.machine()},
    "source": "np.arange(prod(shape), dtype=np.int32).reshape(shape)",
    "criterion": "Exact shape and C-order values.",
    "cases": cases,
}, indent=1) + "\n")
