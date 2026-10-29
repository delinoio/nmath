"""Generate synthetic NPZ interoperability fixtures with NumPy 2.5.4 only."""
from pathlib import Path
import hashlib
import json
import numpy as np

assert np.__version__ == "2.5.4", np.__version__
root = Path(__file__).parent
arrays = {
    "small": np.array([[-128, 0, 127], [1, -2, 3]], dtype=np.int8),
    "wide": np.array([0, 2**64 - 1], dtype=np.uint64),
    "float": np.array([1.5, -0.0, float("inf")], dtype=np.float32),
    "empty": np.empty((2, 0, 3), dtype=np.int16),
    "scalar": np.array(True),
    "한글": np.array([65535, 0], dtype=np.uint16),
}
np.savez(root / "stored.npz", **arrays, allow_pickle=False)
np.savez_compressed(root / "deflated.npz", **arrays, allow_pickle=False)
np.savez(root / "positional.npz", np.array([3, 4], dtype=np.int32), np.array(1.25), allow_pickle=False)
np.savez(root / "object.npz", bad=np.array([{"forbidden": 1}], dtype=object))
np.savez_compressed(root / "bomb.npz", zeros=np.zeros(1 << 20, dtype=np.uint8), allow_pickle=False)
(root / "manifest.json").write_text(json.dumps({"numpy": np.__version__, "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.npz"))}}, indent=2) + "\n")
