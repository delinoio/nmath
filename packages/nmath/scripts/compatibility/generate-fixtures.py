# SPDX-License-Identifier: Apache-2.0
"""Generate deterministic native differential inputs and exact oracle outputs."""
import argparse
import struct
import warnings
from pathlib import Path
from oracle import ROOT, SOURCES, VERSIONS, environment, libraries, source, write_json

np = libraries()["numpy"]
DTYPES = ["bool", "int32", "uint32", "int64", "uint64", "float32", "float64"]


def scalar(value, dtype):
    if dtype == "bool":
        return bool(value)
    if dtype.startswith(("int", "uint")):
        return str(int(value))
    # Store IEEE bits: JSON numbers cannot preserve NaN, infinities or signed zero.
    return struct.pack(">f" if dtype == "float32" else ">d", float(value)).hex()


def result(value):
    value = np.asarray(value)
    dtype = str(value.dtype)
    return {"dtype": dtype, "shape": list(value.shape),
            "values": [scalar(x, dtype) for x in value.reshape(-1)]}


def array(data, dtype):
    return {"data": data, "dtype": dtype}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "fixtures")
    args = parser.parse_args()
    cases = []

    def add(identifier, operation, inputs, run, **extra):
        case = {"id": identifier, "operation": operation, "inputs": inputs, **extra}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                case["expected"] = result(run())
            except (TypeError, ValueError, IndexError, OverflowError) as error:
                case["expectedError"] = type(error).__name__
        floating = case.get("expected", {}).get("dtype", "").startswith("float")
        case["criterion"] = "floating-diagnostic" if floating else "exact-contract"
        cases.append(case)

    for left in DTYPES:
        for right in DTYPES:
            a = np.array([1, 2, 3], dtype=left)
            b = np.array([2, 1, 2], dtype=right)
            for operation in ["add", "subtract", "multiply", "divide", "remainder", "power"]:
                add(f"binary/{operation}/{left}/{right}", operation,
                    [array([1, 2, 3], left), array([2, 1, 2], right)],
                    lambda: getattr(np, operation)(a, b))
        for kind, value in [("number", 0.5), ("bigint", 2), ("boolean", True)]:
            a = np.array([1, 2, 3], dtype=left)
            add(f"weak/add/{left}/{kind}", "add",
                [array([1, 2, 3], left), {"scalar": str(value) if kind == "bigint" else value, "kind": kind}],
                lambda: np.add(a, value))
        for operation in ["positive", "negative", "sum", "mean", "min", "max"]:
            a = np.array([1, 2, 3], dtype=left)
            add(f"unary/{operation}/{left}", operation, [array([1, 2, 3], left)],
                lambda: getattr(np, operation)(a))
        a = np.array([[1, 2], [3, 4]], dtype=left)
        add(f"matmul/{left}", "matmul", [array([[1, 2], [3, 4]], left)] * 2, lambda: a @ a)
    for operation in ["divide", "remainder"]:
        a, b = np.array([-5, 5, -5], dtype="int32"), np.array([2, -2, -2], dtype="int32")
        add(f"signed/{operation}", operation,
            [array(a.tolist(), "int32"), array(b.tolist(), "int32")], lambda: getattr(np, operation)(a, b))
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype="int32")
    for name, selectors, index in [
        ("negative", [1, -1], (1, -1)),
        ("reverse", [{"slice": [None, None, -1]}, 0], (slice(None, None, -1), 0)),
        ("advanced", [{"indices": [1, 0]}, 2], ([1, 0], 2)),
        ("mask", [{"mask": [False, True]}], (np.array([False, True]),)),
        ("empty", [{"slice": [0, 0, 1]}], (slice(0, 0),)),
        ("out-of-bounds", [2], (2,)),
    ]:
        add(f"index/{name}", "index", [array(a.tolist(), "int32")], lambda: a[index], selectors=selectors)
        if "expected" in cases[-1]:
            view = a[index]
            # Empty arrays do not share elements, although they can retain a base.
            cases[-1]["sharesStorage"] = bool(view.base is not None) if isinstance(view, np.ndarray) else None
    for name, data, dtype, op in [
        ("nan-sum", [1, "NaN", 2], "float64", "sum"),
        ("nan-min", [1, "NaN", 2], "float64", "min"),
        ("signed-zero", ["-0", 0], "float64", "negative"),
        ("empty-sum", [], "int32", "sum"),
        ("empty-min", [], "int32", "min"),
    ]:
        a = np.array(data, dtype=dtype)
        add(f"edge/{name}", op, [array(data, dtype)], lambda: getattr(np, op)(a))
    mutation_cases = []
    for kind, selection in [("basic", (slice(None), 1)), ("advanced", ([1, 0], 1))]:
        base = np.array([[1, 2, 3], [4, 5, 6]], dtype="int32")
        selected = base[selection]
        selected[0] = 99
        mutation_cases.append({"id": f"mutation/{kind}", "kind": kind,
            "baseAfter": result(base), "selectedAfter": result(selected), "criterion": "exact-contract"})
    write_json(args.output / "mutation.json", {"schemaVersion": 1, "versions": VERSIONS,
        "environment": environment(), "device": "cpu", "cases": mutation_cases})
    # Exact PCG64 output and state progression are part of the random contract.
    random_cases = []
    for seed in [0, 42, 12345, 2**64 + 1]:
        rng = np.random.default_rng(seed)
        calls = []
        for shape in [[3], [0], [2, 2], [1]]:
            before = rng.bit_generator.state
            values = rng.random(shape)
            calls.append({"shape": shape, "before": before, "expected": result(values), "after": rng.bit_generator.state})
        random_cases.append({"id": f"random/pcg64/{seed}", "seed": str(seed), "calls": calls, "criterion": "random-exact-contract"})
    write_json(args.output / "primitive.json", {"schemaVersion": 1, "versions": VERSIONS,
        "sources": SOURCES, "environment": environment(), "device": "cpu", "cases": cases})
    write_json(args.output / "random.json", {"schemaVersion": 1, "versions": VERSIONS,
        "sources": SOURCES, "environment": environment(), "device": "cpu", "cases": random_cases})
    print(f"Generated {len(cases)} primitive cases and {len(random_cases)} random sequences")


if __name__ == "__main__":
    main()
