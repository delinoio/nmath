# SPDX-License-Identifier: Apache-2.0
"""Compare two fresh generations with each other and the committed artifacts."""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from oracle import ROOT


def files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def main():
    parser = argparse.ArgumentParser()
    for name in ["numpy", "pandas", "polars"]:
        parser.add_argument("--" + name + "-source", required=True)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    sources = [part for name in ["numpy", "pandas", "polars"] for part in ("--" + name + "-source", getattr(args, name + "_source"))]
    with tempfile.TemporaryDirectory(prefix="nmath-oracle-repro-") as temporary:
        outputs = []
        for index in range(2):
            output = Path(temporary) / str(index)
            subprocess.run([sys.executable, str(scripts / "generate-fixtures.py"), "--output", str(output / "fixtures")], check=True)
            subprocess.run([sys.executable, str(scripts / "generate-inventory.py"), *sources, "--output", str(output / "inventory")], check=True)
            outputs.append(files(output))
        if outputs[0] != outputs[1]:
            raise SystemExit("Fresh generations differ")
        expected = {**{"fixtures/" + k: v for k, v in files(ROOT / "fixtures").items()},
                    **{"inventory/" + k: v for k, v in files(ROOT / "inventory").items()}}
        differences = sorted(k for k in set(expected) | set(outputs[0]) if expected.get(k) != outputs[0].get(k))
        if differences:
            raise SystemExit("Committed artifacts differ: " + ", ".join(differences))
        print(f"Byte-identical: two fresh generations and {len(expected)} committed artifacts")


if __name__ == "__main__":
    main()
