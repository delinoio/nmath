# SPDX-License-Identifier: Apache-2.0
"""Fetch exact upstream sources with authenticated gh; keep them outside the repo."""
import argparse
import base64
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from oracle import SOURCES


def gh(path):
    return subprocess.check_output(["gh", "api", path])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    for library, spec in SOURCES.items():
        dest = args.destination / library
        if dest.exists():
            raise SystemExit(f"Destination already exists: {dest}; use a fresh directory")
        api = f"repos/{spec['repository']}"
        tree = json.loads(gh(f"{api}/git/trees/{spec['commit']}?recursive=1"))
        if tree.get("truncated"):
            raise SystemExit(f"Truncated source tree for {library}")
        data = gh(f"{api}/tarball/{spec['commit']}")
        dest.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for item in archive:
                # Strip the generated archive root, preserving source-relative paths.
                relative = Path(*Path(item.name).parts[1:])
                if not relative.parts:
                    continue
                item.name = relative.as_posix()
                archive.extract(item, path=dest, filter="data")
        # Some release archives omit documentation through export-ignore rules.
        # Fetch reference sources from the exact Git tree, never a moving website.
        reference = "py-polars/docs/source/reference/" if library == "polars" else "doc/source/reference/"
        for item in tree["tree"]:
            if item["type"] != "blob" or not item["path"].startswith(reference) or not item["path"].endswith(".rst"):
                continue
            target = dest / item["path"]
            if not target.exists():
                blob = json.loads(gh(f"{api}/git/blobs/{item['sha']}"))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(base64.b64decode(blob["content"]))
            content = target.read_bytes()
            actual = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            if actual != item["sha"]:
                raise SystemExit(f"Source blob mismatch: {library}/{item['path']}")
        (dest / ".nmath-source-tree.json").write_text(json.dumps({"commit": spec["commit"], "tree": tree["tree"]}, sort_keys=True))
        print(f"Fetched {library} at {spec['commit']}", flush=True)


if __name__ == "__main__":
    main()
