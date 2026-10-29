# SPDX-License-Identifier: Apache-2.0
"""Compare two fresh generations with each other and the committed artifacts."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from oracle import ROOT

FIXTURES = {"fixtures/mutation.json", "fixtures/primitive.json", "fixtures/random.json"}
PROVENANCE_FIELDS = ("python", "system", "machine")


def files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def fixture_content(path, artifact, label):
    try:
        value = json.loads(artifact)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} {path}: invalid fixture JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} {path}: fixture must be a JSON object")
    provenance = value.get("environment")
    if not isinstance(provenance, dict):
        raise ValueError(f"{label} {path}: environment must be an object")
    for field in PROVENANCE_FIELDS:
        if not isinstance(provenance.get(field), str) or not provenance[field].strip():
            raise ValueError(f"{label} {path}: environment.{field} must be a nonempty string")
    if provenance.get("byteorder") not in ("little", "big"):
        raise ValueError(f"{label} {path}: environment.byteorder must be little or big")
    print(f"{label} {path} environment: {json.dumps(provenance, sort_keys=True)}")
    return {key: item for key, item in value.items() if key != "environment"}


def committed_differences(expected, fresh):
    differences = []
    for path in sorted(set(expected) | set(fresh)):
        if path in FIXTURES and path in expected and path in fresh:
            committed = fixture_content(path, expected[path], "Committed")
            generated = fixture_content(path, fresh[path], "Fresh")
            if json.dumps(committed, sort_keys=True) != json.dumps(generated, sort_keys=True):
                differences.append(path)
        elif expected.get(path) != fresh.get(path):
            differences.append(path)
    return differences


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
        try:
            differences = committed_differences(expected, outputs[0])
        except ValueError as error:
            raise SystemExit(str(error)) from error
        if differences:
            raise SystemExit("Committed artifacts differ: " + ", ".join(differences))
        print(f"Two fresh generations are byte-identical; {len(expected)} committed artifacts match after fixture provenance separation")


if __name__ == "__main__":
    main()
