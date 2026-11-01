# SPDX-License-Identifier: Apache-2.0
"""Development-only fixed NumPy oracle for general eigendecomposition.

Run with numpy 2.5.4 on PATH, then check in the little-endian JSON output.
Eigenvalue order and eigenvector phases are not unique; the Rust test
matches eigenvalues as a multiset and checks NumPy's A v = v w criterion.
"""
import json
import platform
from pathlib import Path

import numpy as np

assert np.__version__ == "2.5.4"
rng = np.random.default_rng(20261105)


def packed(a):
    a = np.asarray(a)
    return {"dtype": a.dtype.name, "shape": list(a.shape),
            "hex": a.astype(a.dtype.newbyteorder("<")).tobytes().hex()}


def dense(shape, dtype):
    def part():
        return rng.uniform(0.5, 1.5, shape) * rng.choice([-1, 1], shape)
    a = part()
    if dtype.startswith("complex"):
        a = a + 1j * part()
    return a.astype(dtype)


cases = []


def case(a):
    w = np.linalg.eigvals(a)
    cases.append({"input": packed(a), "eigenvalues": packed(w.astype(np.result_type(w, np.complex64)))})


for dtype in ["float32", "float64", "complex64", "complex128"]:
    for shape in [(1, 1), (2, 2), (4, 4), (6, 6), (2, 3, 3)]:
        case(dense(shape, dtype))
    case(np.array([[0, -1], [1, 0]], dtype=dtype))
    case(np.diag(np.array([3, -1, 2], dtype=dtype)))
    case(np.array([[1, 1], [0, 1]], dtype=dtype))
    case(np.array([[2, 1, 0], [0, 2, 1], [0, 0, 3]], dtype=dtype))
case(np.array([[4, 1], [2, 3]], dtype="int32"))

output = {
    "numpy": np.__version__,
    "environment": {"python": platform.python_version(), "system": platform.system(),
                    "machine": platform.machine()},
    "criterion": ("NumPy 2.5.4 test_linalg: eigenvalues assert_almost_equal (decimal 6 for "
                  "single, 12 otherwise) as a multiset; A v allclose to v w with rtol 1e-5 or "
                  "1e-11 and atol 0."),
    "cases": cases,
}
Path(__file__).with_name("oracle.json").write_text(json.dumps(output, indent=1) + "\n")
