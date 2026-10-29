# SPDX-License-Identifier: Apache-2.0
"""Generate reference data with NumPy 2.5.4; never used by the runtime."""
import base64
import io
import json
from pathlib import Path

import numpy as np

assert np.__version__ == "2.5.4"
dtypes = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64"]
result = {"numpy": np.__version__, "promotion": {a: {b: np.result_type(a, b).name for b in dtypes} for a in dtypes}, "cases": []}
for dtype in ["int8", "int16", "uint8", "uint16"]:
    info = np.iinfo(dtype)
    a = np.array([[info.min, info.max], [2, 3]], dtype=dtype)
    b = np.array([1, 2], dtype=dtype)
    operations = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b, "mod": a % b, "pow": a ** b, "neg": -a, "sum": a.sum(axis=0), "mean": a.mean(axis=0), "min": a.min(axis=0), "max": a.max(axis=0), "matmul": a @ b}
    stream = io.BytesIO()
    np.save(stream, a, allow_pickle=False)
    result["cases"].append({"dtype": dtype, "a": a.tolist(), "b": b.tolist(), "min": int(info.min), "max": int(info.max), "npy": base64.b64encode(stream.getvalue()).decode(), "operations": {op: {"dtype": value.dtype.name, "values": value.tolist()} for op, value in operations.items()}})
Path(__file__).with_name("integer-dtypes.json").write_text(json.dumps(result, indent=2) + "\n")
