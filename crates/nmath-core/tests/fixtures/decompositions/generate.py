# SPDX-License-Identifier: Apache-2.0
"""Development-only fixed NumPy oracle for linalg factorizations.

Run with numpy 2.5.4 on PATH, then check in the little-endian JSON output.
QR, eigenvector and singular-vector signs are not unique, so the Rust test
checks NumPy's own test_linalg invariants and compares only unique results
(eigenvalues, singular values, Cholesky factors and slogdet) with NumPy.
"""
import json
import platform
from pathlib import Path

import numpy as np

assert np.__version__ == "2.5.4"
rng = np.random.default_rng(20261103)


def packed(a):
    a = np.asarray(a)
    return {"dtype": a.dtype.name, "shape": list(a.shape),
            "hex": a.astype(a.dtype.newbyteorder("<")).tobytes().hex()}


def dense(shape, dtype):
    # Entries in ±[0.5, 1.5] keep NumPy's rtol-only reconstruction checks meaningful.
    def part():
        return rng.uniform(0.5, 1.5, shape) * rng.choice([-1, 1], shape)
    a = part()
    if dtype.startswith("complex"):
        a = a + 1j * part()
    return a.astype(dtype)


def hermitian(n, dtype, batch=()):
    a = dense(batch + (n, n), dtype)
    return ((a + np.conj(np.swapaxes(a, -1, -2))) / 2).astype(dtype)


def spd(n, dtype, batch=()):
    a = dense(batch + (n, n), dtype)
    return (a @ np.conj(np.swapaxes(a, -1, -2)) + n * np.eye(n)).astype(dtype)


cases = []
for dtype in ["float32", "float64", "complex64", "complex128"]:
    for shape in [(1, 1), (3, 3), (4, 4), (2, 3, 3), (5, 3), (3, 5), (2, 4, 2)]:
        a = dense(shape, dtype)
        s = np.linalg.svd(a, compute_uv=False)
        u, _, vh = np.linalg.svd(a, full_matrices=True)
        ur, _, vhr = np.linalg.svd(a, full_matrices=False)
        q, r = np.linalg.qr(a)
        qc, rc = np.linalg.qr(a, mode="complete")
        case = {"kind": "general", "input": packed(a), "svdvals": packed(s),
                "shapes": {"svdU": list(u.shape), "svdVh": list(vh.shape), "svdUReduced": list(ur.shape),
                           "svdVhReduced": list(vhr.shape), "qrQ": list(q.shape), "qrR": list(r.shape),
                           "qrQComplete": list(qc.shape), "qrRComplete": list(rc.shape)}}
        if shape[-1] == shape[-2]:
            sign, logabs = np.linalg.slogdet(a)
            case["slogdetSign"] = packed(sign)
            case["slogdetLogabs"] = packed(logabs)
        cases.append(case)
    for n, batch in [(1, ()), (3, ()), (4, ()), (3, (2,))]:
        h = hermitian(n, dtype, batch)
        cases.append({"kind": "hermitian", "input": packed(h),
                      "eigvalsh": packed(np.linalg.eigvalsh(h)),
                      "eigvalshUpper": packed(np.linalg.eigvalsh(h, UPLO="U"))})
        p = spd(n, dtype, batch)
        cases.append({"kind": "spd", "input": packed(p), "cholesky": packed(np.linalg.cholesky(p)),
                      "choleskyUpper": packed(np.linalg.cholesky(p, upper=True))})
# Integer inputs compute in float64; empty matrices keep NumPy shapes.
ints = np.array([[4, 1], [2, 3]], dtype="int32")
cases.append({"kind": "general", "input": packed(ints), "svdvals": packed(np.linalg.svd(ints, compute_uv=False)),
              "slogdetSign": packed(np.linalg.slogdet(ints)[0]), "slogdetLogabs": packed(np.linalg.slogdet(ints)[1]),
              "shapes": {"svdU": [2, 2], "svdVh": [2, 2], "svdUReduced": [2, 2], "svdVhReduced": [2, 2],
                         "qrQ": [2, 2], "qrR": [2, 2], "qrQComplete": [2, 2], "qrRComplete": [2, 2]}})
singular = np.array([[1.0, 2.0], [2.0, 4.0]])
cases.append({"kind": "general", "input": packed(singular),
              "svdvals": packed(np.linalg.svd(singular, compute_uv=False)),
              "slogdetSign": packed(np.linalg.slogdet(singular)[0]),
              "slogdetLogabs": packed(np.linalg.slogdet(singular)[1]),
              "shapes": {"svdU": [2, 2], "svdVh": [2, 2], "svdUReduced": [2, 2], "svdVhReduced": [2, 2],
                         "qrQ": [2, 2], "qrR": [2, 2], "qrQComplete": [2, 2], "qrRComplete": [2, 2]}})

output = {
    "numpy": np.__version__,
    "environment": {"python": platform.python_version(), "system": platform.system(),
                    "machine": platform.machine()},
    "criterion": ("NumPy 2.5.4 test_linalg criteria: assert_almost_equal with decimal 6 for "
                  "float32/complex64 and 12 otherwise; reconstruction allclose with rtol 1e-5 or "
                  "1e-11 and atol 0; Cholesky reconstruction atol 500*n*eps."),
    "cases": cases,
}
Path(__file__).with_name("oracle.json").write_text(json.dumps(output, indent=1) + "\n")
