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
# Pin the inspected implementation, not the generator commit or execution host.
COUNTERPART_REVISION = "8bd03f1ed61605ca8651a15057c8ae7579a38364"
COUNTERPARTS = {}
SCOPES = {}
NUMERIC = "bool int8 uint8 int16 uint16 int32 uint32 int64 uint64 float16 float32 float64 complex64 complex128".split()
REAL = [dtype for dtype in NUMERIC if dtype not in {"float16", "complex64", "complex128"}]


def counterparts(names, family, arguments, dtypes, limits, implementation, tests, fixtures=()):
    """Register only inspected public counterparts; never propagate through aliases."""
    for name, target in names.items():
        if name in COUNTERPARTS:
            raise ValueError(f"Duplicate counterpart: {name}")
        COUNTERPARTS[name] = target
        SCOPES[name] = {
            "family": family, "status": "partial", "arguments": arguments,
            "dtypes": list(dtypes), "conditions": limits,
            "implementation": list(implementation), "tests": list(tests),
            "fixtures": list(fixtures),
        }


CORE = ["packages/nmath/src/core.ts", "packages/nmath/src/numpy.ts", "packages/nmath/src/types.ts"]
DTYPE_TESTS = [f"packages/nmath/test/{name}-dtypes.test.mjs" for name in ["integer", "complex", "float16"]]
DTYPE_FIXTURES = [f"packages/nmath/test/fixtures/{name}-dtypes.json" for name in ["integer", "complex", "float16"]]
NUMERIC_LIMITS = [
    "Explicit context and dtype; lazy CPU computation with independent compute calls.",
    "int64/uint64 use bigint; complex values use Complex or interleaved typed components; float16 has explicit bit transport.",
    "GPU only supports its existing float32/int32/uint32 operation subset; float16, complex and narrow integer GPU jobs reject.",
    "Fixture criteria apply only to recorded inputs; arbitrary power and long reductions remain pending. GPU power has 84 unresolved one-ULP differences.",
]
for op in ["array", "zeros", "ones", "full", "ndarray"]:
    counterparts({"numpy." + op: "core.NMathArray" if op == "ndarray" else "numpy." + op},
        "array-construction", {"array": ["context", "data", "options: dtype, optional shape"], "ndarray": ["context.array factory; no public ndarray constructor parity"], "zeros": ["context", "shape", "dtype"], "ones": ["context", "shape", "dtype"], "full": ["context", "shape", "value", "dtype"]}[op], NUMERIC,
        NUMERIC_LIMITS + ["No scalar-class constructor parity, structured/extended/string/temporal dtypes, order/like/subok or full copy-policy parity."], CORE, DTYPE_TESTS, DTYPE_FIXTURES)
counterparts({"numpy.arange": "numpy.arange"}, "array-construction", ["context", "start", "stop", "step", "dtype"],
    ["int32", "uint32", "float32", "float64"], ["JS number start/stop/step; default dtype float64; no 64-bit integer or full upstream dtype support."], CORE, ["packages/nmath/test/frames.test.mjs"])
for op, target in {"add":"add", "subtract":"subtract", "multiply":"multiply", "divide":"divide", "true_divide":"trueDivide", "remainder":"remainder", "mod":"mod", "power":"power", "positive":"positive", "negative":"negative"}.items():
    dtypes = [d for d in NUMERIC if not (op in {"remainder", "mod"} and d.startswith("complex")) and not (op in {"subtract", "positive", "negative"} and d == "bool")]
    counterparts({"numpy." + op: "numpy." + target}, "elementwise", ["contextual operands", "broadcasting", "weak scalars"], dtypes,
        NUMERIC_LIMITS + ["No out, where, casting/order keyword surface or ufunc method parity; boolean power/remainder produce int8."], CORE, DTYPE_TESTS, DTYPE_FIXTURES)
for op in ["sum", "mean", "min", "max"]:
    counterparts({"numpy." + op: "numpy." + op, "numpy.ndarray." + op: "core.Lazy." + op},
        "reductions", ["array", "axis: one integer or omitted"], NUMERIC,
        NUMERIC_LIMITS + ["No axis tuple, keepdims, out, dtype, initial or where options on the public wrapper; integer mean/division return float64; narrow integer sums widen."], CORE, DTYPE_TESTS, DTYPE_FIXTURES)
for op in ["var", "std"]:
    counterparts({"numpy." + op: "numpy." + op, "numpy.ndarray." + op: "core.Lazy." + op},
        "reductions", ["array", "axis: one integer or omitted", "ddof: finite number"], NUMERIC,
        NUMERIC_LIMITS + ["Two-pass mean and squared deviations. Result dtype keeps float16, maps float32/complex64 to float32 and other dtypes to float64; complex variance is real. The divisor clamps at zero, so zero divisors return NaN or infinity. Strings and temporal dtypes are rejected. GPU rejects variance before admission."],
        CORE + ["crates/nmath-core/src/kernels.rs"],
        DTYPE_TESTS + ["packages/nmath/test/variance.test.mjs", "packages/nmath/test/types/variance.ts", "crates/nmath-core/tests/variance.rs"],
        DTYPE_FIXTURES + ["crates/nmath-core/tests/fixtures/variance/oracle.json"])
for op in ["cumsum", "cumprod"]:
    counterparts({"numpy." + op: "numpy." + op, "numpy.ndarray." + op: "core.Lazy." + op},
        "reductions", ["array", "axis: one integer or omitted"], NUMERIC,
        NUMERIC_LIMITS + ["Without an axis the result flattens to one dimension. Result dtype uses the sum promotion. Strings and temporal dtypes are rejected. The first element of each run is copied without an identity. GPU rejects scans before admission."],
        CORE + ["crates/nmath-core/src/kernels.rs"],
        DTYPE_TESTS + ["packages/nmath/test/scans.test.mjs", "packages/nmath/test/types/scans.ts", "crates/nmath-core/tests/scans.rs"],
        DTYPE_FIXTURES + ["crates/nmath-core/tests/fixtures/scans/oracle.json"])
for op in ["prod", "argmin", "argmax"]:
    counterparts({"numpy." + op: "numpy." + op, "numpy.ndarray." + op: "core.Lazy." + op},
        "reductions", ["array", "axis: one integer or omitted"], NUMERIC,
        NUMERIC_LIMITS + ["At most one axis; strings and temporal dtypes are rejected. Products reuse the sum dtype promotion and empty-axis identity. Index reductions return int64 bigint positions, keep the first equal or NaN position, compare complex values lexicographically, and reject a zero-length reduced axis with NMATH_SHAPE_MISMATCH. GPU rejects all three before admission."],
        CORE + ["crates/nmath-core/src/kernels.rs"],
        DTYPE_TESTS + ["packages/nmath/test/reductions.test.mjs", "packages/nmath/test/types/reductions.ts", "crates/nmath-core/tests/reductions.rs"],
        DTYPE_FIXTURES + ["crates/nmath-core/tests/fixtures/reductions/oracle.json"])
counterparts({"numpy.matmul":"numpy.matmul"}, "matrix-product", ["a", "b"], NUMERIC,
    NUMERIC_LIMITS + ["Broadcast batch axes and NumPy vector promotion; full ufunc options remain targets. Exact small-value fixtures are not arbitrary floating-point or platform acceptance."],
    CORE + ["crates/nmath-core/src/shape.rs", "crates/nmath-core/src/kernels.rs"],
    DTYPE_TESTS + ["packages/nmath/test/batched-matmul.test.mjs", "packages/nmath/test/types/batched-matmul.ts", "packages/nmath/test/compiler.test.mjs", "crates/nmath-core/tests/gpu.rs"],
    DTYPE_FIXTURES + ["packages/nmath/test/fixtures/batched-matmul.json"])
counterparts({"numpy.reshape": "numpy.reshape", "numpy.ndarray.reshape": "core.Lazy.reshape"},
    "shape-transform", ["array", "shape: dimension list with one optional -1", "options: order C, copy true/false/null"],
    NUMERIC + ["str", "datetime64", "timedelta64"],
    ["Lazy CPU C-order reshape; compatible signed strides share storage, copy true is independent and copy false rejects copy-required layouts.",
     "Preserves dtype descriptors and context; copy-required execution uses actual-start snapshots.",
     "F/A traversal, alternate scalar shapes and wider upstream signatures remain targets; GPU rejects before admission."],
    CORE + ["packages/nmath/src/layout-types.ts", "crates/nmath-core/src/layout.rs", "crates/nmath-core/src/expr.rs", "crates/nmath-core/src/array.rs"],
    ["packages/nmath/test/layout.test.mjs", "packages/nmath/test/types/layout.ts", "crates/nmath-core/tests/layout.rs", "packages/nmath/test/compiler.test.mjs"],
    ["packages/nmath/test/fixtures/layout.json"])
counterparts({"numpy.transpose": "numpy.transpose", "numpy.ndarray.transpose": "core.Lazy.transpose"},
    "shape-transform", ["array", "axes: complete permutation or omitted"],
    NUMERIC + ["str", "datetime64", "timedelta64"],
    ["Lazy CPU transpose; omitted axes reverse all axes, negative axes normalize to a complete permutation.",
     "Pure view chains share storage, preserving dtype descriptors, context, writes and lifetime; calculations read actual-start snapshots.",
     "JS takes an axes list; upstream varargs signatures remain targets; GPU rejects before admission."],
    CORE + ["packages/nmath/src/layout-types.ts", "crates/nmath-core/src/layout.rs", "crates/nmath-core/src/expr.rs", "crates/nmath-core/src/array.rs"],
    ["packages/nmath/test/layout.test.mjs", "packages/nmath/test/types/layout.ts", "crates/nmath-core/tests/layout.rs", "packages/nmath/test/compiler.test.mjs"],
    ["packages/nmath/test/fixtures/layout.json"])
for op, target, args, limit in [
    ("__getitem__", "get", ["integer", "slice", "indices", "mask"], "Basic indexing shares storage; advanced selections copy. Full upstream mixed indexing remains pending."),
    ("__setitem__", "set", ["selectors", "value", "signal"], "Asynchronous atomic writes; complex-to-real writes require explicit casts; failed writes preserve storage."),
    ("astype", "astype", ["dtype"], "Lazy cast; explicit complex-to-real cast can discard imaginary components; casting/order/copy/subok options remain targets."),
]:
    counterparts({"numpy.ndarray." + op: "core.NMathArray." + target}, "indexing-casting", args, NUMERIC,
        NUMERIC_LIMITS + [limit], CORE, DTYPE_TESTS + ["packages/nmath/test/core.test.mjs"], DTYPE_FIXTURES)
for op in ["save", "load", "savez", "savez_compressed"]:
    counterparts({"numpy." + op: "numpy." + op}, "numpy-io", ["local path", "explicit context/owned arrays", "signal", "overwrite for writes"], NUMERIC,
        ["Writes reject existing files unless overwrite is explicit; scheduled atomic publication.",
         "load is NPY-only; NPZ loading uses the additional numpy.loadNpz(context, path) adaptation returning a completed Map, not a lazy NpzFile.",
         "savez/savez_compressed accept positional arrays plus a named Map; append .npz; stored/DEFLATE only; safe member names and bounded archive/header sizes.",
         "Complex and float16 NPY accept both byte orders and C/Fortran order; other dtype layout restrictions remain. No mmap or file-object interface."],
        CORE, DTYPE_TESTS + ["packages/nmath/test/npz.test.mjs"], DTYPE_FIXTURES + ["crates/nmath-core/tests/fixtures/io-npz/manifest.json", "crates/nmath-core/tests/fixtures/io-npz/complex-expected.json"])
for op in ["default_rng", "Generator", "Generator.random", "Generator.integers", "Generator.uniform", "Generator.standard_normal", "Generator.normal", "Generator.exponential"]:
    method = op.split(".")[-1]
    dtype = REAL[:9] if method == "integers" else ["float32", "float64"] if method in {"random", "standard_normal"} else ["float64"] if method in {"uniform", "normal", "exponential"} else REAL
    target = "numpy.random.defaultRng" if op == "default_rng" else "core.NMathGenerator" + ("." + method if "." in op else "")
    counterparts({"numpy.random." + op: target}, "random", {
            "default_rng": ["context", "seed: nonnegative integer <=128 bits"],
            "Generator": ["context.generator(seed) factory"],
            "random": ["shape (scalar: [])", "dtype: float32/float64, default float64"],
            "integers": ["low: safe integer/bigint", "high: safe integer/bigint/undefined", "shape", "dtype: default int64", "endpoint: default false"],
            "uniform": ["low: scalar", "high: scalar", "shape"],
            "standard_normal": ["shape", "dtype: float32/float64, default float64"],
            "normal": ["loc: scalar", "scale: scalar", "shape"],
            "exponential": ["scale: scalar", "shape"],
        }[method], dtype,
        ["PCG64 only; Generator construction is through context.generator/defaultRng, not an arbitrary bit generator.",
         "Integer bounds require safe JS integers or bigint; undefined high means [0, low); uint64 exclusive high can be 2**64.",
         "Each successful compute advances shared state in request order; failures/cancellation preserve state.",
         "No parameter broadcasting, out, entropy default, SeedSequence spawning or state import/export. Other distributions and bit generators remain targets.",
         "Exact recorded byte sequences do not certify cross-platform transcendental behavior."],
        ["packages/nmath/src/random.ts", "packages/nmath/src/numpy.ts"], ["packages/nmath/test/random-distributions.test.mjs"], ["packages/nmath/test/fixtures/random/provenance.json", "packages/nmath/test/fixtures/random/mixed.txt"])
for family, ops in [("linalg", ["solve", "inv", "det"]), ("fft", ["fft", "ifft", "rfft", "irfft"])]:
    for op in ops:
        dtype = [d for d in NUMERIC if not (family == "linalg" and d == "float16") and not (op == "rfft" and d.startswith("complex"))]
        limits = ["CPU only; lazy scheduled results; other family members remain pending.", "Non-finite/extreme-scale criteria and cross-platform acceptance remain pending.", "Linalg fixtures use source-matched residual/eigenvalue criteria; FFT identity fixtures check bounded round trips. FFT adaptation fixtures certify shape/dtype only, not forward-output error."]
        limits += (["Final two axes must be square; batch axes broadcast; only rank-one solve RHS is a vector.", "Integer/bool compute as float64; float32/complex64 retained unless operands require double precision; singular solve/inv reject, det returns zero."] if family == "linalg" else ["One axis with crop/zero-pad; positive uint32 n; norm backward/forward/ortho; no out.", "float16/float32/complex64 produce single precision except irfft(float16) returns float16; other numeric inputs use double precision.", "rfft rejects complex input; irfft defaults to 2*(axis length-1), discards DC and even-length Nyquist imaginary parts."])
        counterparts({"numpy." + family + "." + op: "numpy." + family + "." + op}, family,
            ["a", "b (solve only)"] if family == "linalg" else ["a", "n", "axis", "norm"], dtype, limits,
            ["packages/nmath/src/" + family + ".ts", "crates/nmath-core/src/numerical.rs"],
            ["packages/nmath/test/numerical.test.mjs", "crates/nmath-core/tests/numerical.rs"],
            ["crates/nmath-core/tests/fixtures/numerical/" + file for file in (["linalg.json"] if family == "linalg" else ["fft_adaptation.json", "fft_identity.json"])])

for name, target, arguments in [
    ("numpy.polyval", "numpy.polyval", ["coefficients: descending", "points"]),
    ("numpy.polynomial.polynomial.polyval", "numpy.polynomial.polynomial.polyval", ["points", "coefficients: ascending", "options: tensor boolean, default true"]),
]:
    counterparts({name: target}, "polynomial", arguments, NUMERIC,
        ["CPU-only lazy native Horner evaluation with explicit-context operands, actual-start snapshots and independent results.",
         "Ascending integer/bool coefficients become float64; boolean points promote through int64. Descending coefficients retain array dtype promotion and integer wrapping.",
         "Ascending tensor evaluation concatenates polynomial-batch and point shapes; tensor false and descending evaluation broadcast them.",
         "Ascending empty coefficients reject; descending empty coefficients return zeros with the point dtype and shape. Scalar operands require context arrays; descending coefficients require an axis.",
         "Exact recorded float bits with NaN equivalence are diagnostic, not general numerical acceptance. Other polynomial APIs/classes/bases and platform acceptance remain pending."],
        ["packages/nmath/src/polynomial.ts", "packages/nmath/src/numpy.ts", "crates/nmath-core/src/polynomial.rs", "crates/nmath-core/src/kernels.rs", "crates/nmath-core/src/expr.rs"],
        ["packages/nmath/test/polynomial.test.mjs", "packages/nmath/test/types/polynomial.ts", "crates/nmath-core/tests/polynomial.rs"],
        ["crates/nmath-core/tests/fixtures/polynomial/oracle.json"])

for name, arguments in [
    ("numpy.polyder", ["coefficients: rank-one descending", "m: nonnegative safe integer, default 1"]),
    ("numpy.polynomial.polynomial.polyder", ["coefficients: ascending", "options: m, axis, scl"]),
]:
    counterparts({name: name}, "polynomial", arguments, NUMERIC,
        ["CPU-only lazy native differentiation with execution-start snapshots, cancellation and managed memory budgets.",
         "Ascending integer/bool coefficients become float64; floating/complex precision is preserved with separate scale/degree rounding at each stage.",
         "Ascending supports scalar coefficients and negative/multivariate axes; scalar scale is real or complex. Active complex scaling of real coefficients requires an explicit cast.",
         "Ascending high orders retain the first coefficient slice multiplied by zero (including non-finite effects); empty axes stay empty. Ascending zero order copies.",
         "Descending rank-one positive orders promote against int64 degree arrays, preserve integer wrapping and may return empty coefficients; zero order preserves input storage.",
         "Fractional/string legacy order coercion, array-valued scales, multidimensional legacy coefficients, polynomial classes and integration remain pending.",
         "Exact recorded floating bits with NaN equivalence are diagnostic fixture evidence, not all-platform numerical acceptance."],
        ["packages/nmath/src/polynomial.ts", "packages/nmath/src/layout-types.ts", "packages/nmath/src/numpy.ts", "crates/nmath-node/src/lib.rs", "crates/nmath-core/src/polynomial.rs", "crates/nmath-core/src/kernels.rs", "crates/nmath-core/src/expr.rs"],
        ["packages/nmath/test/polynomial-derivative.test.mjs", "packages/nmath/test/types/polynomial-derivative.ts", "crates/nmath-core/tests/polynomial.rs"],
        ["crates/nmath-core/tests/fixtures/polynomial/derivative-oracle.json"])

TABLE_LIMITS = [
    "CPU numeric/bool columns with explicit validity masks; no string/temporal/complex/nested columns at the inspected revision.",
    "Completed Promise results; internal bounded row-local batches do not establish full streaming. Snapshots/results and sort/join/group boundaries materialize.",
    "Single-key stable sort, inner/left equijoin and sum/mean/min/max/non-null-count grouping only. Full argument sets remain targets.",
]
PANDAS_LIMITS = [
    "String/number labels only; arithmetic aligns equal unique label sets or identical duplicate sequences, not full outer alignment.",
    "Nullable Series arithmetic/reductions and nullable scalar iloc reject. Boolean-column row filtering uses DataFrame.filter, unlike upstream label-filter semantics.",
    "merge matches null keys; groupBy is an immediate single-key/single-value aggregate with as_index=False behavior, sorted/dropna defaults and optional sort/dropna flags.",
    "Suffix adaptation uses one right suffix; no full pandas suffixes/index join interface.",
]
POLARS_LIMITS = [
    "Positional rows; joins default to unmatched null keys, nullsEqual opts in; groupBy retains null groups and first-seen key order.",
    "Computed projections require non-nullable input and must precede row operations; named selection supports validity masks.",
    "collect has ComputeOptions, not upstream engine/optimizer/background options; no complete streaming acceptance.",
]
for lib, mappings, limits in [
    ("pandas", {"Series":"Series", "DataFrame":"DataFrame", "Series.iloc":"Series.iloc", "Series.loc":"Series.loc", "Series.rename":"Series.rename", "DataFrame.sort_values":"DataFrame.sortValues", "DataFrame.merge":"DataFrame.merge", "DataFrame.groupby":"DataFrame.groupBy", "DataFrame.assign":"DataFrame.assign", "DataFrame.sum":"DataFrame.sum", **{"Series."+op:"Series."+op for op in ["add", "sub", "mul", "div", "mod", "pow", "sum", "mean", "min", "max"]}}, PANDAS_LIMITS),
    ("polars", {"Series":"Series", "DataFrame":"DataFrame", "LazyFrame":"LazyFrame", "col":"col", "Series.len":"Series.len", "Series.alias":"Series.alias", "Series.to_pandas":"Series.toPandas", "DataFrame.to_pandas":"DataFrame.toPandas", "DataFrame.get_column":"DataFrame.getColumn", "DataFrame.lazy":"DataFrame.lazy", **{"LazyFrame."+op:"LazyFrame."+target for op,target in {"select":"select", "with_columns":"withColumns", "filter":"filter", "sort":"sort", "join":"join", "group_by":"groupBy", "collect":"collect"}.items()}, **{"Expr."+op:"ColumnExpr."+op for op in ["add", "sub", "mul", "truediv", "mod", "pow", "alias"]}}, POLARS_LIMITS),
]:
    counterparts({lib+"."+name:lib+"."+target for name,target in mappings.items()}, "tables", ["named numeric columns", "single key/value/operation where applicable", "explicit validity/index metadata"], REAL,
        TABLE_LIMITS + limits, ["packages/nmath/src/"+lib+".ts", "packages/nmath/src/frame.ts"], ["packages/nmath/test/tables.test.mjs", "packages/nmath/test/frames.test.mjs"], ["packages/nmath/test/fixtures/tables.json"])
for lib in ["pandas", "polars"]:
    for format in ["csv", "json", "parquet", "feather" if lib == "pandas" else "ipc"]:
        text_format = format in {"csv", "json"}
        write = ("to_" if lib == "pandas" else "write_") + format
        write_target = (("to" if lib == "pandas" else "write") + format.title()) if text_format else write
        limits = ["Local paths; explicit context; scheduled CPU Promise; signal and explicit overwrite for writes; no full upstream options."]
        limits += (["CSV header without index; JSON records orientation only (differs from pandas default columns). Explicit complete schema required for narrow/all-null dtype preservation.", "Parser buffers whole input; no full text streaming. No custom delimiter/encoding/compressed text/NDJSON; missing-token policies differ.", "JSON nonfinite values write as null; pandas custom indexes require dropIndex:true."] if text_format else ["Flat numeric/bool schema only; no float16/complex/string/temporal/nested/dictionary/extension columns or custom schema/field metadata.", "Arrow IPC file/Feather V2 only, uncompressed; no IPC stream or Feather V1. Parquet reads uncompressed/Snappy, writes Snappy without dictionary encoding.", "Serialized input buffered; decoding/writing batches still return a complete table. No projection/engine/compression options.", "pandas non-default indexes reject; default index reconstructed; nullable numeric preservation corresponds to dtype_backend=numpy_nullable, not default pandas inference."])
        counterparts({lib+".read_"+format:lib+".read_"+format, lib+".DataFrame."+write:lib+".DataFrame."+write_target}, "text-io" if text_format else "columnar-io", ["context", "path", "schema (text reads)", "signal", "overwrite (writes)"], REAL, limits,
            ["packages/nmath/src/"+lib+".ts", "packages/nmath/src/"+("io-text" if text_format else "columnar")+".ts"],
            ["packages/nmath/test/"+("io-text" if text_format else "columnar")+".test.mjs"],
            ["crates/nmath-core/tests/fixtures/"+("io-text" if text_format else "io-columnar")+"/manifest.json"])


# Storage support does not implement NumPy scalar constructors or namespaces.
for name in ["numpy.array", "numpy.full", "numpy.ndarray", "numpy.ndarray.__getitem__",
             "numpy.ndarray.__setitem__", "numpy.ndarray.astype", "numpy.save", "numpy.load",
             "numpy.savez", "numpy.savez_compressed"]:
    scope = SCOPES[name]
    scope["dtypes"] += ["str", "datetime64", "timedelta64"]
    scope["conditions"] = [line.replace("structured/extended/string/temporal dtypes", "structured/extended dtypes") for line in scope["conditions"]]
    scope["conditions"].append("Fixed UTF-32 width and temporal kind/unit metadata only. Temporal scalars preserve bigint ticks/NaT; explicit casts change units/width. No byte/variable strings, generic temporal units, parsing or full scalar-class parity. NPY/NPZ preserve supported descriptors; other formats reject these dtypes.")
    scope["implementation"] += ["crates/nmath-core/src/temporal.rs"]
    scope["tests"] += ["packages/nmath/test/text-temporal.test.mjs"]
    scope["fixtures"] += ["packages/nmath/test/fixtures/text-temporal.json"]


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
    missing_counterparts = sorted(set(COUNTERPARTS) - set(entries))
    if missing_counterparts:
        raise SystemExit(f"Counterparts outside the inventory: {missing_counterparts}")
    repository = ROOT.parents[2]
    for name, scope in SCOPES.items():
        row = entries[name]
        row["supportedScope"] = scope
        if not row["supportTarget"]:
            raise SystemExit(f"Counterpart conflicts with exclusion: {name}")
        evidence = []
        for kind in ["implementation", "tests", "fixtures"]:
            for path in scope[kind]:
                evidence.append({"kind": kind, "path": path, "sha256": digest((repository / path).read_bytes())})
        row["verification"].update(
            evidence=evidence, sourceRevision=COUNTERPART_REVISION,
            status="evidence-available-native-acceptance-pending",
            crossPlatformStatus="pending", gpuStatus="pending",
        )
        row["arguments"]["supported"] = scope["arguments"]
        row["dtypePlatform"]["supportedDtypes"] = scope["dtypes"]
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
        "acceptanceStatus": "incomplete", "counterpartSourceRevision": COUNTERPART_REVISION, "counts": dict(collections.Counter(row["library"] for row in entries.values())),
        "unresolvedNames": sorted(unresolved), "shards": shards})
    write_json(out / "support-summary.generated.json", {
        "schemaVersion": 1, "sourceRevision": COUNTERPART_REVISION,
        "acceptanceStatus": "incomplete",
        "statuses": dict(sorted(collections.Counter(row["status"] for row in entries.values()).items())),
        "partialFamilies": dict(sorted(collections.Counter(scope["family"] for scope in SCOPES.values()).items())),
        "unperformed": ["cross-platform-native-acceptance", "gpu-release-matrix-acceptance", "full-api-semantic-review"],
        "unresolved": ["GPU power: 84 one-ULP differences; no tolerance waiver", "strings/temporal worker integration requires follow-up reconciliation"],
    })
    write_json(out / "modules.generated.json", sorted(module_audit, key=lambda x: x["name"]))
    write_json(out / "source-hashes.generated.json", dict(sorted(source_files.items())))
    print(json.dumps({"rows": len(entries), "unresolved": len(unresolved), "shards": len(shards)}))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
