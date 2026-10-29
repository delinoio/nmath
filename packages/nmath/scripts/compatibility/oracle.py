# SPDX-License-Identifier: Apache-2.0
"""Development-only oracle helpers. Never imported by the production package."""
import importlib
import json
import platform
import sys
from pathlib import Path

VERSIONS = {"numpy": "2.5.4", "pandas": "3.1.0", "polars": "2.1.0"}
SOURCES = {
    "numpy": {"repository": "numpy/numpy", "tag": "v2.5.4", "commit": "a3d0ef1df045b858e68115001fdabc930bb3b68c"},
    "pandas": {"repository": "pandas-dev/pandas", "tag": "v3.1.0", "commit": "c6f46769b808eb60218189d089dd67841b44f1b9"},
    "polars": {"repository": "pola-rs/polars", "tag": "py-2.1.0", "commit": "7f902bd3ffd27ff82d14278a09a295ff2f7aed92"},
}
ROOT = Path(__file__).resolve().parents[2] / "compatibility"


def libraries():
    result = {}
    for name, version in VERSIONS.items():
        module = importlib.import_module(name)
        if module.__version__ != version:
            raise SystemExit(f"Oracle version mismatch: {name} requires {version}, got {module.__version__}")
        result[name] = module
    return result


def environment():
    return {"python": platform.python_version(), "system": platform.system(),
            "machine": platform.machine(), "byteorder": sys.byteorder}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def source(library, path, anchor=""):
    spec = SOURCES[library]
    return f"https://github.com/{spec['repository']}/blob/{spec['commit']}/{path}{anchor}"
