# SPDX-License-Identifier: Apache-2.0
"""Generate exact-byte mixed-call fixtures with the pinned NumPy oracle only."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np

assert np.__version__ == "2.5.4", np.__version__
root = Path(__file__).parent
rng = np.random.default_rng(12345)
calls = [
    ("random", "float32", 0, 0, False, 3),
    ("random", "float64", 0, 0, False, 7),
    ("integers", "int64", -17, 27, False, 19),
    ("integers", "uint64", 0, 2**64, False, 7),
    ("integers", "int64", -2**63, 2**63-1, True, 5),
    ("integers", "uint64", 0, 2**63+1, False, 19),
    ("integers", "int32", -2**31, 2**31, False, 7),
    ("integers", "uint32", 0, 2**32-1, True, 3),
]
for dtype, low, high in [("int8", -128, 128), ("uint8", 0, 256), ("int16", -32768, 32768), ("uint16", 0, 65536)]:
    calls += [("integers", dtype, low, high, False, 19), ("integers", dtype, 0, high//2+1, False, 29)]
calls += [("integers", "bool", 0, 2, False, 37), ("integers", "int64", 5, 5, True, 3), ("random", "float32", 0, 0, False, 1)]
calls += [(op, dtype, a, b, False, 4096) for op, dtype, a, b in [
    ("uniform", "float64", -1.5, 4.5),
    ("standard_normal", "float64", 0, 0),
    ("standard_normal", "float32", 0, 0),
    ("normal", "float64", 3, 2),
    ("exponential", "float64", 2, 0),
]]
calls += [("random", "float32", 0, 0, False, 3), ("random", "float64", 0, 0, False, 0)]
lines = []
for index, (op, dtype, a, b, endpoint, size) in enumerate(calls):
    if op in ("random", "standard_normal"):
        values = getattr(rng, op)(size=size, dtype=dtype)
    elif op == "integers":
        values = rng.integers(a, b, size=size, dtype=dtype, endpoint=endpoint)
    elif op == "exponential":
        values = rng.exponential(a, size=size)
    else:
        values = getattr(rng, op)(a, b, size=size)
    state = rng.bit_generator.state
    raw = values.astype(values.dtype.newbyteorder("<")).tobytes()
    filename = f"{index:02d}.bin"
    (root / filename).write_bytes(raw)
    lines.append("|".join(map(str, [op, dtype, a, b, int(endpoint), size, filename, state["state"]["state"], state["state"]["inc"], state["has_uint32"], state["uinteger"], hashlib.sha256(raw).hexdigest()])))
(root / "mixed.txt").write_text("\n".join(lines) + "\n")

(root / "provenance.json").write_text(json.dumps({
    "numpy": np.__version__,
    "numpyRevision": np.version.git_revision,
    "platform": platform.system(),
    "machine": platform.machine(),
    "python": platform.python_version(),
    "seed": "12345",
    "comparison": "exact little-endian output bytes and complete PCG64 state after every call",
    "scope": "scalar-parameter Generator methods; not full API or cross-platform acceptance",
}, indent=2) + "\n")
