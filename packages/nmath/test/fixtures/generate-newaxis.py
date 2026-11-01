# SPDX-License-Identifier: Apache-2.0
"""Development-only fixed NumPy oracle for newaxis and ellipsis indexing.

Run with numpy 2.5.4 on PATH from this directory.
"""
import json
import platform
from pathlib import Path

import numpy as np

assert np.__version__ == "2.5.4"
E = Ellipsis
N = None


def sl(start=None, stop=None, step=None):
    return slice(start, stop, step)


INDEXES = [
    (N,), (N, N), (E,), (E, N), (N, E), (N, E, 0), (0, E), (E, 0), (0, E, 1), (1, E, 1, 2),
    (0, N, 1), (sl(None, None, 2), N, 1), (N, 0, N, sl(1, 3)),
    (0, N, [1, 2]), (N, [0, 1], 0), ([0, 1], N, 0), (0, [1, 2]), (0, sl(), [1, 2]),
    (N, 0, N, [1, 2]), ([0, 1], N), (sl(), N, [1, 2], N), (0, E, [1, 2]), (E, [0, 2]),
    (np.array([True, False]), N, 0), (sl(), np.array([True, False, True]), N),
    (N, sl(), 0, [3, 0]), (sl(), 0, N, [1, 3]), (-1, N, E, N),
]


def encode(s):
    if s is None:
        return {"kind": "newaxis"}
    if s is Ellipsis:
        return {"kind": "ellipsis"}
    if isinstance(s, int):
        return {"kind": "index", "value": s}
    if isinstance(s, slice):
        return {"kind": "slice", "start": s.start, "stop": s.stop, "step": s.step}
    if isinstance(s, np.ndarray):
        return {"kind": "mask", "values": s.tolist()}
    return {"kind": "indices", "values": list(s)}


a = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
cases = []
for index in INDEXES:
    r = a[index]
    cases.append({"index": [encode(s) for s in index], "shape": list(r.shape), "values": r.ravel().tolist()})

Path(__file__).with_name("newaxis.json").write_text(json.dumps({
    "numpy": np.__version__,
    "environment": {"python": platform.python_version(), "machine": platform.machine()},
    "source": "np.arange(24, dtype=np.int32).reshape(2, 3, 4)",
    "criterion": "Exact shape and C-order values.",
    "cases": cases,
}, indent=1) + "\n")
