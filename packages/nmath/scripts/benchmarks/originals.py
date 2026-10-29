# SPDX-License-Identifier: Apache-2.0
"""Development-only fixed-original oracle and benchmark. Not a runtime dependency."""
import argparse
import hashlib
import json
import os
import platform
import signal
try:
    import resource
except ImportError:
    resource = None
import statistics
import subprocess
import sys
import time
from pathlib import Path

VERSIONS = {"numpy": "2.5.4", "pandas": "3.1.0", "polars": "2.1.0"}
WORKLOAD = {"seed": 20261029, "matrixSize": 128, "rows": 32768, "groups": 64, "algorithmSize": 8192}

def values(length):
    state = WORKLOAD["seed"]
    result = []
    for _ in range(length):
        state = (state * 1664525 + 1013904223) & 0xffffffff
        result.append((state >> 24) % 16 - 8)
    return result

def digest(items):
    if any(int(x) != x for x in items):
        raise AssertionError("Expected exact integer-valued outputs")
    return hashlib.sha256(",".join(str(int(x)) for x in items).encode()).hexdigest()

def stats(samples):
    median = statistics.median(samples)
    ordered = sorted(samples)
    return {"count": len(samples), "min": min(samples), "median": median,
            "p95": ordered[__import__('math').ceil(len(samples) * .95) - 1],
            "max": max(samples), "mad": statistics.median(abs(x - median) for x in samples)}

def publish(path, report):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(report, indent=2) + "\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(signal.Signals(signum).name)
    signal.signal(signal.SIGTERM, interrupted)
    if not 3 <= args.samples <= 1000 or not 1 <= args.warmup <= 100:
        parser.error("samples must be 3..1000; warmup must be 1..100")
    report = {"schemaVersion": 1, "status": "running", "versions": VERSIONS, "workload": WORKLOAD,
              "command": sys.argv, "python": sys.version, "platform": platform.platform(),
              "machine": platform.machine(), "processor": platform.processor(), "cpuCount": os.cpu_count(),
              "sourceRevision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "scriptSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "workingTree": subprocess.check_output(["git", "status", "--porcelain"], text=True),
              "threadEnvironment": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "VECLIB_MAXIMUM_THREADS"]},
              "results": [], "limits": ["RSS is process lifetime high-water mark, not per-kernel allocation.", "No speedup or broad numerical acceptance claim.", "Imports and fixture generation are outside compute samples."]}
    try:
        started = time.perf_counter()
        import numpy as np
        import pandas as pd
        import polars as pl
        report["importMs"] = (time.perf_counter() - started) * 1000
        for name, mod in [("numpy", np), ("pandas", pd), ("polars", pl)]:
            if mod.__version__ != VERSIONS[name]:
                raise RuntimeError(f"{name}: expected {VERSIONS[name]}, got {mod.__version__}")
        report["numpyConfiguration"] = np.show_config(mode="dicts")
        setup_start = time.perf_counter()
        n, rows, groups = WORKLOAD["matrixSize"], WORKLOAD["rows"], WORKLOAD["groups"]
        a = np.array(values(n*n), dtype=np.float32).reshape(n, n)
        b = a.T.copy()
        x = np.array(values(WORKLOAD["algorithmSize"]), dtype=np.float32)
        keys = np.arange(rows, dtype=np.int64) % groups
        data = np.array(values(rows), dtype=np.int64)
        pdf = pd.DataFrame({"k": keys, "v": data})
        pldf = pl.DataFrame({"k": keys, "v": data})
        expected_matrix = [sum(int(a[i,k])*int(b[k,j]) for k in range(n)) for i in range(n) for j in range(n)]
        sums = [sum(int(data[i]) for i in range(g, rows, groups)) for g in range(groups)]
        expected_table = [v for pair in zip(range(groups), sums) for v in pair]
        expected_algorithm = [(v*2+3)*v+1 for v in values(len(x))]
        cases = [
            ("matrix", lambda: a @ b, lambda r: r.ravel().tolist(), expected_matrix, [n,n], "float32", "numpy"),
            ("tabular-pandas", lambda: pdf.groupby("k", as_index=False, sort=True)["v"].sum(), lambda r: r.to_numpy().ravel().tolist(), expected_table, [groups,2], "int64", "pandas"),
            ("tabular-polars", lambda: pldf.lazy().group_by("k").agg(pl.col("v").sum()).sort("k").collect(), lambda r: [v for row in r.rows() for v in row], expected_table, [groups,2], "int64", "polars"),
            ("algorithm-nts", lambda: (x*2+3)*x+1, lambda r: r.tolist(), expected_algorithm, [len(x)], "float32", "numpy")]
        report["setupAndIndependentOracleMs"] = (time.perf_counter() - setup_start) * 1000
        fixture = {"schemaVersion": 1, "versions": VERSIONS, "workload": WORKLOAD,
                   "criterion": "Exact full-output equality for bounded integer-valued inputs; no general floating-point tolerance claim.", "provenance": {"generatorSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "python": platform.python_version(), "platform": platform.platform()}, "cases": {}}
        for name, run, extract, expected, shape, dtype, original in cases:
            first_start = time.perf_counter_ns()
            result = run()
            first_ms = (time.perf_counter_ns() - first_start) / 1e6
            if extract(result) != expected:
                raise AssertionError(f"Correctness failed: {name}")
            actual_shape = list(result.shape)
            actual_dtypes = ([str(result.dtype)] if original == "numpy" else
                             [str(d).lower() for d in result.dtypes])
            if actual_shape != shape or any(d != dtype for d in actual_dtypes):
                raise AssertionError(f"Schema failed: {name}: {actual_shape}, {actual_dtypes}")
            fixture["cases"][name] = {"digest": digest(expected), "shape": shape, "dtype": dtype, "original": original, "missingValues": 0}
            for _ in range(args.warmup):
                result = run()
                del result
            samples = []
            for _ in range(args.samples):
                start = time.perf_counter_ns()
                result = run()
                samples.append((time.perf_counter_ns() - start) / 1e6)
                if digest(extract(result)) != fixture["cases"][name]["digest"]:
                    raise AssertionError(f"Timed output failed: {name}")
                del result
            report["results"].append({"name": name, "firstExecutionMs": first_ms, "correctness": "passed", "samplesMs": samples, "summaryMs": stats(samples), "maxRssBytes": (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024) if resource else None)})
        if args.fixture:
            publish(args.fixture, fixture)
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        publish(args.output, report)

if __name__ == "__main__":
    main()
