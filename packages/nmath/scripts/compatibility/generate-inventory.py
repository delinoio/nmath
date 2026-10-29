# SPDX-License-Identifier: Apache-2.0
"""Reconcile pinned reference directives, public exports, classes and accessors.

This tool does not infer support from the existence of an upstream symbol. Pending
row fields are release gaps. It never imports a Python library in NMath runtime.
"""
import argparse
import collections
import hashlib
import importlib
import inspect
import importlib.metadata
import json
import re
import types
import typing
import warnings
from pathlib import Path
from oracle import ROOT, SOURCES, VERSIONS, libraries, source, write_json

LIBRARIES = libraries()
MISSING = object()
MODULES = {
    "numpy": "numpy numpy.exceptions numpy.fft numpy.linalg numpy.polynomial numpy.random numpy.strings numpy.testing numpy.typing numpy.ctypeslib numpy.dtypes numpy.emath numpy.lib numpy.rec numpy.version numpy.char numpy.f2py numpy.ma numpy.matlib".split(),
    "pandas": "pandas pandas.errors pandas.plotting pandas.testing pandas.api.extensions pandas.api.indexers pandas.api.interchange pandas.api.types pandas.api.typing".split(),
    "polars": "polars polars.selectors polars.datatypes polars.exceptions polars.testing polars.api polars.plugins polars.catalog".split(),
}
REFERENCE = {"numpy": "doc/source/reference", "pandas": "doc/source/reference", "polars": "py-polars/docs/source/reference"}
PACKAGE = {"numpy": "numpy", "pandas": "pandas", "polars": "py-polars/src/polars"}
# These runtime implementation paths are exposed under documented public accessors.
# Resolve property return annotations rather than guessing accessor method names.
# Keep callable/data-model protocols, including reverse and in-place operators.
# Interpreter bookkeeping is not a numerical API entry.
INTROSPECTION = {"__class__", "__dict__", "__doc__", "__module__", "__weakref__", "__annotations__", "__slots__", "__abstractmethods__", "__classcell__", "__firstlineno__", "__static_attributes__", "__parameters__", "__orig_bases__", "__match_args__", "__annotate__"}

EXCLUSIONS = {
    "numpy.object_": "Python or JS object-valued dtypes",
    "numpy.dtypes.ObjectDType": "Python or JS object-valued dtypes",
    "numpy.ndarray.dump": "Python-only extension/serialization APIs",
    "numpy.ndarray.dumps": "Python-only extension/serialization APIs",
    "numpy.vectorize": "Python/JS callbacks and UDFs",
    "numpy.frompyfunc": "Python/JS callbacks and UDFs",
    "numpy.apply_along_axis": "Python/JS callbacks and UDFs",
    "numpy.apply_over_axes": "Python/JS callbacks and UDFs",
    "numpy.dtypes.register_dlpack_dtype": "Python-only extension/serialization APIs",
    "pandas.api.extensions.register_dataframe_accessor": "Python-only extension/serialization APIs",
    "pandas.api.extensions.register_series_accessor": "Python-only extension/serialization APIs",
    "pandas.api.extensions.register_index_accessor": "Python-only extension/serialization APIs",
    "pandas.api.extensions.register_extension_dtype": "Python-only extension/serialization APIs",
    "polars.api.register_dataframe_namespace": "Python-only extension/serialization APIs",
    "polars.api.register_series_namespace": "Python-only extension/serialization APIs",
    "polars.api.register_expr_namespace": "Python-only extension/serialization APIs",
    "polars.api.register_lazyframe_namespace": "Python-only extension/serialization APIs",
    "pandas.read_excel": "Excel",
    "pandas.ExcelFile": "Excel",
    "pandas.ExcelWriter": "Excel",
    "pandas.DataFrame.to_excel": "Excel",
    "pandas.Series.to_excel": "Excel",
    "polars.read_excel": "Excel",
    "polars.DataFrame.write_excel": "Excel",
    "pandas.read_pickle": "Python-only extension/serialization APIs",
    "pandas.DataFrame.to_pickle": "Python-only extension/serialization APIs",
    "pandas.Series.to_pickle": "Python-only extension/serialization APIs",
    "pandas.read_sql": "Remote storage, database connectors",
    "pandas.read_sql_query": "Remote storage, database connectors",
    "pandas.read_sql_table": "Remote storage, database connectors",
    "pandas.io.formats.style.Styler.to_excel": "Excel",
    "pandas.to_pickle": "Python-only extension/serialization APIs",
    "pandas.DataFrame.to_sql": "Remote storage, database connectors",
    "polars.read_database": "Remote storage, database connectors",
    "polars.read_database_uri": "Remote storage, database connectors",
    "polars.DataFrame.write_database": "Remote storage, database connectors",
}
COUNTERPARTS = {name: f"numpy.{target}" for name, target in {
    "array": "array", "zeros": "zeros", "ones": "ones", "full": "full", "arange": "arange",
    "add": "add", "subtract": "subtract", "multiply": "multiply", "divide": "divide", "true_divide": "trueDivide",
    "remainder": "remainder", "mod": "mod", "power": "power", "positive": "positive", "negative": "negative",
    "sum": "sum", "mean": "mean", "min": "min", "max": "max", "matmul": "matmul", "save": "save", "load": "load",
    "random.default_rng": "random.defaultRng", "random.Generator.random": "NMathGenerator.random",
}.items()}
COUNTERPARTS = {"numpy." + key: value for key, value in COUNTERPARTS.items()}
COUNTERPARTS.update({"numpy.random.Generator.random": "core.NMathGenerator.random", "numpy.ndarray": "core.NMathArray", "numpy.ndarray.__getitem__": "core.NMathArray.get",
    "numpy.ndarray.__setitem__": "core.NMathArray.set", "pandas.Series": "pandas.Series", "pandas.DataFrame": "pandas.DataFrame",
    "polars.Series": "polars.Series", "polars.DataFrame": "polars.DataFrame", "polars.LazyFrame": "polars.LazyFrame"})


def digest(data):
    return hashlib.sha256(data).hexdigest()


def accessor(obj):
    if isinstance(obj, property) and obj.fget:
        try:
            result = typing.get_type_hints(obj.fget).get("return")
            if inspect.isclass(result) and (result.__name__.endswith("NameSpace") or "accessor" in result.__module__):
                return result
        except (NameError, TypeError, AttributeError):
            pass
    return obj


def resolve(name):
    parts = name.split(".")
    for length in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:length]))
        except (ImportError, AttributeError, ValueError):
            continue
        try:
            for part in parts[length:]:
                parent = accessor(obj)
                if part in getattr(parent, "__annotations__", {}) and not hasattr(parent, part):
                    obj = parent.__annotations__[part]
                else:
                    obj = getattr(parent, part)
            return obj
        except (AttributeError, TypeError):
            continue
    return MISSING


def documentation(root, library):
    found = collections.defaultdict(list)
    modules = set()
    if not (root / REFERENCE[library] / "index.rst").is_file():
        raise SystemExit(f"Missing reference sources for {library}")
    for path in sorted((root / REFERENCE[library]).rglob("*.rst")):
        relative = path.relative_to(root).as_posix()
        # C API and Fortran wrapper documentation has a distinct extension boundary.
        if "/c-api/" in relative:
            continue
        current = library
        active_class = None
        class_indent = -1
        summary_indent = None
        continuation = None
        lines = path.read_text().splitlines()
        for number, line in enumerate(lines, 1):
            stripped = line.strip()
            if continuation:
                if stripped and re.fullmatch(r"[\w.]+", stripped) and len(line) - len(line.lstrip()) > continuation[0]:
                    name = stripped if stripped.startswith(library + ".") else current + "." + stripped
                    found[name].append({"path": relative, "line": number, "kind": continuation[1]})
                    continue
                continuation = None
            directive = re.match(r"\s*\.\.\s+(?:py:)?(\w+)::\s*(.*)", line)
            if directive:
                kind, value = directive.groups()
                indent = len(line) - len(line.lstrip())
                if indent <= class_indent:
                    active_class = None
                summary_indent = None
                if kind in {"attribute", "data"}:
                    continuation = (indent, kind)
                if kind in {"currentmodule", "module"}:
                    current = value
                elif kind == "autosummary":
                    summary_indent = len(line) - len(line.lstrip())
                elif kind in {"automodule", "autoclass", "autofunction", "automethod", "autoattribute", "autodata", "function", "class", "method", "attribute", "data"}:
                    name = value.split("(")[0].strip().lstrip("~")
                    if kind == "automodule":
                        current = name
                    if kind in {"automethod", "autoattribute", "method", "attribute"} and "." not in name and active_class:
                        name = active_class + "." + name
                    elif not name.startswith(library + ".") and name != library:
                        name = current + "." + name
                    name = name.replace("numpy.class.", "numpy.ndarray.").replace("numpy.object.__array_", "numpy.ndarray.__array_")
                    if re.fullmatch(r"[\w.]+", name) and name.startswith(library + "."):
                        found[name].append({"path": relative, "line": number, "kind": kind})
                        if kind == "automodule":
                            modules.add(name)
                        if kind in {"class", "autoclass"}:
                            active_class = name
                            class_indent = indent
                continue
            if summary_indent is not None and stripped and not stripped.startswith(":"):
                indent = len(line) - len(line.lstrip())
                if indent <= summary_indent:
                    summary_indent = None
                elif re.fullmatch(r"~?[\w.]+", stripped):
                    name = stripped.lstrip("~")
                    if not name.startswith(library + "."):
                        name = current + "." + name
                    found[name].append({"path": relative, "line": number, "kind": "autosummary"})
    return found, modules


def main():
    parser = argparse.ArgumentParser()
    for name in VERSIONS:
        parser.add_argument("--" + name + "-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "inventory")
    args = parser.parse_args()
    for line in (ROOT / "oracle-requirements.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            package, expected = line.split("==")
            if importlib.metadata.version(package) != expected:
                raise SystemExit(f"Inventory dependency version mismatch: {package} requires {expected}")
    roots = {name: getattr(args, name + "_source").resolve() for name in VERSIONS}
    trees = {}
    for library, root in roots.items():
        metadata = json.loads((root / ".nmath-source-tree.json").read_text())
        if metadata["commit"] != SOURCES[library]["commit"]:
            raise SystemExit(f"Source revision mismatch: {library}")
        trees[library] = {item["path"]: item["sha"] for item in metadata["tree"] if item["type"] == "blob"}

    def checked_source(library, path):
        content = (roots[library] / path).read_bytes()
        actual = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if trees[library].get(path) != actual:
            raise SystemExit(f"Source blob mismatch: {library}/{path}")
        return digest(content)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    entries = {}
    identities = collections.defaultdict(list)
    # Retain descriptors/bound methods: CPython can otherwise recycle their ids
    # and create false alias groups between unrelated API entries.
    identity_objects = {}
    module_audit = []
    unresolved = []
    source_files = {}
    expanded = set()

    def add(name, obj, evidence, ancestors=()):
        if name in entries:
            if evidence not in entries[name]["boundaryEvidence"]:
                entries[name]["boundaryEvidence"].append(evidence)
            if evidence.get("url") and entries[name]["definition"] is None:
                entries[name]["semantics"]["reference"] = evidence["url"]
            return
        lib = name.split(".")[0]
        doc = inspect.getdoc(obj) if obj is not MISSING else None
        target = accessor(obj)
        definition = None
        try:
            filename = inspect.getsourcefile(target.fget if isinstance(target, property) else target)
            if filename:
                installed_root = Path(LIBRARIES[lib].__file__).parent
                relative = Path(filename).resolve().relative_to(installed_root)
                path = PACKAGE[lib] + "/" + relative.as_posix()
                upstream = roots[lib] / path
                if upstream.is_file():
                    source_files[lib + ":" + path] = checked_source(lib, path)
                    definition = {"path": path, "url": source(lib, path), "sha256": source_files[lib + ":" + path]}
        except (TypeError, ValueError, OSError):
            pass
        try:
            signature = str(inspect.signature(obj))
            signature = re.sub(r"0x[0-9a-fA-F]+", "<address>", signature)
        except (TypeError, ValueError):
            signature = None
        reference = definition["url"] if definition else evidence.get("url", source(lib, REFERENCE[lib] + "/index.rst"))
        exclusion = next((rationale for prefix, rationale in EXCLUSIONS.items() if name == prefix or name.startswith(prefix + ".")), None)
        if name.startswith(("numpy.ctypeslib.", "numpy.f2py.")):
            exclusion = "Python-only extension/serialization APIs"
        status = "excluded" if exclusion else "partial-counterpart" if name in COUNTERPARTS else "pending"
        row = {"name": name, "library": lib, "version": VERSIONS[lib], "kind": "unresolved" if obj is MISSING else "module" if inspect.ismodule(obj) else "accessor" if target is not obj else "class" if inspect.isclass(target) else "callable" if callable(obj) else "attribute",
            "boundaryEvidence": [evidence], "aliasOf": None, "definition": definition,
            "docstringSha256": digest(doc.encode()) if doc else None,
            "counterpart": COUNTERPARTS.get(name), "status": status,
            "supportTarget": exclusion is None, "exclusion": exclusion,
            "semantics": {"status": "pending-review", "reference": reference},
            "arguments": {"signature": signature, "status": "pending-adaptation"},
            "dtypePlatform": {"status": "pending", "reference": "semantics.reference"},
            "viewCopyMutation": {"status": "pending", "reference": "semantics.reference"},
            "missingValues": {"status": "pending", "reference": "semantics.reference"},
            "verification": {"status": "pending", "fixtureIds": [], "cpuCriterion": None, "gpuCriterion": None}}
        entries[name] = row
        if obj is MISSING:
            if name in {"numpy.float96", "numpy.float128", "numpy.complex192", "numpy.complex256"}:
                row["kind"] = "platform-conditional-alias"
                row["aliasOf"] = "numpy.longdouble" if name.startswith("numpy.float") else "numpy.clongdouble"
                row["dtypePlatform"] = {"status": "source-defined", "condition": "Alias exists only when the platform long-double storage width matches the name; current oracle host does not export it.", "reference": source("numpy", "doc/source/reference/arrays.scalars.rst", "#L411")}
            else:
                unresolved.append(name)
            return
        if callable(obj) or inspect.isclass(obj) or isinstance(obj, property):
            identity_objects[id(obj)] = obj
            identities[(lib, id(obj))].append(name)
        # Each public alias is retained; ancestry prevents cycles through nested types.
        if inspect.isclass(target) or isinstance(target, LIBRARIES["numpy"].ufunc):
            if id(target) in ancestors or name in expanded:
                return
            expanded.add(name)
            for member in sorted(set(dir(target))):
                if member in INTROSPECTION or (member.startswith("_") and not (member.startswith("__") and member.endswith("__"))):
                    continue
                try:
                    child = getattr(target, member)
                except (AttributeError, TypeError):
                    continue
                # Builtin object methods do not become library APIs by inheritance.
                if member.startswith("__") and hasattr(object, member) and getattr(object, member) is child:
                    continue
                add(name + "." + member, child, {"kind": "public-class-member", "owner": name}, ancestors + (id(target),))

    for lib in VERSIONS:
        for path in sorted((roots[lib] / REFERENCE[lib]).rglob("*.rst")):
            relative = path.relative_to(roots[lib]).as_posix()
            source_files[lib + ":" + relative] = checked_source(lib, relative)
        documented, extra_modules = documentation(roots[lib], lib)
        queue = sorted(set(MODULES[lib]) | extra_modules)
        seen = set()
        while queue:
            name = queue.pop(0)
            if name in seen:
                continue
            seen.add(name)
            module = resolve(name)
            if not inspect.ismodule(module):
                continue
            declared = getattr(module, "__all__", None)
            # Without __all__, retain owned definitions and named reexports only.
            names = sorted(set(declared or []) | {n for n, v in vars(module).items() if not n.startswith("_") and (getattr(v, "__module__", "") or "").startswith(lib)})
            included, rejected = [], []
            for member in names:
                child_name = name + "." + member
                try:
                    obj = getattr(module, member)
                except (AttributeError, ImportError):
                    add(child_name, MISSING, {"kind": "declared-export", "owner": name})
                    continue
                owner = getattr(obj, "__module__", "") or ""
                if declared is None and (inspect.ismodule(obj) or (owner and not owner.startswith(lib))):
                    rejected.append({"name": child_name, "reason": "imported-dependency-not-declared-export"})
                    continue
                # NumPy exposes core for backward compatibility but defines it as private.
                if member in {"core", "compat", "internals"} and inspect.ismodule(obj):
                    rejected.append({"name": child_name, "reason": "private-namespace; numpy module_structure lists public namespaces"})
                    continue
                included.append(member)
                add(child_name, obj, {"kind": "declared-export" if declared is not None else "owned-public-definition", "owner": name})
                if inspect.ismodule(obj) and obj.__name__ == child_name and not any(p.startswith("_") for p in child_name.split(".")) and (lib == "numpy" or child_name in MODULES[lib]):
                    queue.append(child_name)
            module_audit.append({"name": name, "exportRule": "__all__" if declared is not None else "owned-public-definitions", "included": included, "rejected": rejected})
            add(name, module, {"kind": "public-module", "reference": source(lib, REFERENCE[lib] + "/index.rst")})
        for name, evidence in sorted(documented.items()):
            obj = resolve(name)
            for item in evidence:
                path = roots[lib] / item["path"]
                source_files[lib + ":" + item["path"]] = checked_source(lib, item["path"])
                add(name, obj, {**item, "url": source(lib, item["path"], f"#L{item['line']}")})

    for group in identities.values():
        canonical = min(group, key=lambda x: (x.count("."), len(x), x))
        for name in group:
            if name != canonical:
                entries[name]["aliasOf"] = canonical
        # A dtype-specific exclusion on an inherited method must not spread to
        # the same descriptor exposed by an in-scope numeric dtype. Only explicit
        # entry exclusions propagate through object aliases.
        exclusion = next((EXCLUSIONS[name] for name in group if name in EXCLUSIONS), None)
        if exclusion:
            for name in group:
                entries[name].update(status="excluded", supportTarget=False, exclusion=exclusion)
    for name in sorted(entries, key=lambda value: (value.count("."), value)):
        parent = entries.get(name.rsplit(".", 1)[0])
        if parent and parent["exclusion"] and parent["kind"] in {"class", "accessor"}:
            entries[name].update(status="excluded", supportTarget=False, exclusion=parent["exclusion"])
    primitive = json.loads((ROOT / "fixtures/primitive.json").read_text())
    for case in primitive["cases"]:
        name = "numpy.ndarray.__getitem__" if case["operation"] == "index" else "numpy." + case["operation"]
        if name in entries:
            verification = entries[name]["verification"]
            verification["fixtureIds"].append(case["id"])
            verification["status"] = "fixtures-available-native-result-not-certified"
            verification["cpuCriterion"] = "per-fixture:compatibility/criteria.json"
    entries["numpy.ndarray.__setitem__"]["verification"].update(status="fixtures-available-native-result-not-certified", fixtureIds=[x["id"] for x in json.loads((ROOT / "fixtures/mutation.json").read_text())["cases"]], cpuCriterion="exact-contract")
    if "numpy.random.Generator.random" in entries:
        entries["numpy.random.Generator.random"]["verification"].update(status="fixtures-available-native-result-not-certified", fixtureIds=[x["id"] for x in json.loads((ROOT / "fixtures/random.json").read_text())["cases"]], cpuCriterion="random-exact-contract")
    # JSON Lines shards are deterministic and remain below the Git LFS threshold.
    shards = []
    for lib in VERSIONS:
        rows = [entries[n] for n in sorted(entries) if entries[n]["library"] == lib]
        batch, size, number = [], 0, 0
        def flush():
            nonlocal batch, size, number
            if not batch:
                return
            filename = f"{lib}-{number:03d}.jsonl"
            data = "".join(batch).encode()
            (out / filename).write_bytes(data)
            shards.append({"path": filename, "rows": len(batch), "sha256": digest(data)})
            batch, size, number = [], 0, number + 1
        for row in rows:
            line = json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
            if size + len(line.encode()) > 240_000:
                flush()
            batch.append(line)
            size += len(line.encode())
        flush()
    expected = {x["path"] for x in shards}
    for path in out.glob("*.jsonl"):
        if path.name not in expected:
            path.unlink()
    write_json(out / "manifest.generated.json", {"schemaVersion": 1, "versions": VERSIONS, "sources": SOURCES,
        "boundaryStatus": "pending-reconciliation" if unresolved else "enumerated-pending-semantic-review",
        "acceptanceStatus": "incomplete", "counterpartSourceRevision": "576712dc5d8d471eb0744d236850ad8d08ee4fc7", "counts": dict(collections.Counter(row["library"] for row in entries.values())),
        "unresolvedNames": sorted(unresolved), "shards": shards})
    write_json(out / "modules.generated.json", sorted(module_audit, key=lambda x: x["name"]))
    write_json(out / "source-hashes.generated.json", dict(sorted(source_files.items())))
    print(json.dumps({"rows": len(entries), "unresolved": len(unresolved), "shards": len(shards)}))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
