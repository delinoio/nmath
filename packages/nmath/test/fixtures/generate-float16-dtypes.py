# SPDX-License-Identifier: Apache-2.0
"""NumPy 2.5.4 binary16 oracle. Exact bits for the recorded finite inputs.

NaNs compare by classification; no tolerance is inferred for arbitrary power,
long reductions, or other unrecorded inputs. Those numerical criteria remain
uncovered. Python is an oracle only, never a production dependency.
"""
import base64
import io
import json
from pathlib import Path
import warnings
import numpy as np

assert np.__version__ == "2.5.4"
warnings.simplefilter("ignore")
np.seterr(all="ignore")


def record(a):
    a = np.asarray(a, dtype="float16")
    return {"shape": list(a.shape), "bits": a.view("uint16").ravel().tolist()}


def npy(a):
    stream = io.BytesIO()
    np.save(stream, a)
    return base64.b64encode(stream.getvalue()).decode()


dtypes = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64", "complex64", "complex128"]
a = np.array([[1, -2, 0.5], [3, 4, -0.25]], dtype="float16")
b = np.array([2, -0.5, 4], dtype="float16")
cases = []
for name, op in [("add", np.add), ("sub", np.subtract), ("mul", np.multiply), ("div", np.divide), ("mod", np.remainder), ("pow", np.power)]:
    cases.append({"op": name, "a": record(a), "b": record(b), "result": record(op(a, b))})
for values in [a, np.array([65504, 1, -65504, 1]*65, dtype="float16"), np.full((257, 3), 0.1, dtype="float16"), np.empty((0, 2), dtype="float16"), np.array([-0., 0.], dtype="float16"), np.array([0., -0.], dtype="float16")]:
    for name in ["sum", "mean", "min", "max"]:
        for axis in [None, 0, -1, ()]:
            try:
                result = getattr(np, name)(values, axis=axis)
            except ValueError:
                continue
            cases.append({"op": name, "a": record(values), "axis": axis if axis != () else [], "result": record(result)})
for left, right in [(a, b), (a.T, a), (np.full(257, .1, dtype="float16"), np.full(257, .1, dtype="float16"))]:
    cases.append({"op": "matmul", "a": record(left), "b": record(right), "result": record(left @ right)})
for name, op in [("pos", np.positive), ("neg", np.negative)]:
    values = np.array([0, 0x8000, 1, 0x3c00, 0x7bff, 0x7c00, 0xfc00, 0x7e01], dtype="uint16").view("float16")
    cases.append({"op": name, "a": record(values), "result": record(op(values))})
values = [0., -0., 1., 1.00048828125, 1.00146484375, 2**-24, 2**-25, 65504., 65520., -65520.]
result = {"numpy": np.__version__, "criterion": "exact finite bits and NaN classification for recorded inputs only", "uncovered": ["arbitrary fractional power tolerances", "arbitrary long reduction and matrix accumulation tolerances", "other CPU architectures"], "promotion": {d: str(np.result_type("float16", d)) for d in dtypes}, "casts": {"values": values, "result": record(values)}, "cases": cases, "npy": [npy(a), npy(a.astype(">f2")), npy(np.asfortranarray(a))]}
cast_input = np.array([np.nan, np.inf, -np.inf, -1.5, 65504, -65504], dtype="float16")
result["integerCasts"] = {"input": record(cast_input), "outputs": {d: [str(x) if d in ["int64", "uint64"] else x for x in cast_input.astype(d).tolist()] for d in dtypes if "int" in d}}
Path(__file__).with_name("float16-dtypes.json").write_text(json.dumps(result, indent=2) + "\n")
