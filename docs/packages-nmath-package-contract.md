# NMath npm Package Contract

## Scope

This contract owns the `nmath-package` component of the [`nmath` project](project-nmath.md): the public npm package `@delinoio/nmath` at `packages/nmath/`.

## Runtime and Language

- ESM only. CommonJS consumers use dynamic `import()`. No CommonJS build is published.
- Node.js 24 only. Bun, Deno and browsers are out of scope.
- TypeScript sources with declaration output. Ordinary consumer type tests use TypeScript 7.0.2.
- The package loads the `nmath-node` native binary. If the prebuilt binary cannot load, import fails with `NMATH_NATIVE_LOAD`, the platform triple, the cause and manual source-build instructions. There is no automatic source build and no JavaScript computation fallback.

## Users and Operators

- JavaScript and TypeScript developers who use NMath from Node.js 24.
- `.nts` output produced by `nmath-compiler`, which imports `@delinoio/nmath/nts-runtime`.

When the `.nts` transform's extension-rewriting option is enabled, source-relative `.nts` specifiers in quoted imports and static template dynamic imports become `.js` specifiers. The [compiler contract](crates-nmath-compiler-contract.md) owns this behavior and its span mapping.

## Interfaces and Contracts

### Entry points

| Specifier                     | Entry point ID | Contents                                                                                                    |
| ----------------------------- | -------------- | ----------------------------------------------------------------------------------------------------------- |
| `@delinoio/nmath`             | `core`         | `createContext`, `NMathArray`, `Expr`, `slice`, `indices`, `mask`, dtypes, errors and arithmetic functions. |
| `@delinoio/nmath/numpy`       | `numpy`        | NumPy-style array creation, arithmetic, reductions, `matmul`, `random`.                                     |
| `@delinoio/nmath/pandas`      | `pandas`       | pandas-style `Series` and `DataFrame`.                                                                      |
| `@delinoio/nmath/polars`      | `polars`       | Polars-style `Series`, `DataFrame` and lazy frames.                                                         |
| `@delinoio/nmath/compiler`    | `compiler`     | `check` and `build` project APIs.                                                                           |
| `@delinoio/nmath/nts-runtime` | (internal)     | Helpers that compiled `.nts` output imports. Not documented for direct use.                                 |

The CLI executable is `nmath-compiler`. No separate compiler package is published. The package `bin` entry is a small Node.js launcher that runs the platform's prebuilt `nmath-compiler` executable from `native/` or the platform package; `@delinoio/nmath/compiler` runs the same executable with `--json`.
The compiler entry point, CLI launcher and compiler build script resolve or build the compiler without loading the numerical addon. A missing compiler reports compiler-specific recovery guidance. Numerical entry points still load the addon when imported.

### Contexts

```ts
const ctx = createContext({
  maxConcurrentJobs: 4,
  maxQueueLength: 64,
  cpuMemoryBudget: 1n << 30n,
  gpuMemoryBudget: 0n,
});
```

All four options are required. `ctx.close(): Promise<void>` follows the runtime close contract. `ctx.metrics()` returns measurement snapshots. Contexts and arrays implement `dispose()` and `Symbol.dispose` for `using`.
Native jobs release their execution slot before they settle their result Promise. Promise handlers can close the context or submit another job under the normal admission rules.

Array creation helpers (`ctx.array`, `ctx.full`, `ctx.zeros` and the NumPy `ones`, `full` and `arange` forms) check the CPU budget before they build JS or native input buffers and fail with `NMATH_MEMORY_BUDGET_EXCEEDED` when the input cannot fit. The context still makes the authoritative reservation when it creates the array.

Explicit and inferred dimensions in `ctx.array`, `ctx.full`, `ctx.zeros` and `ctx.generator(seed).random(shape)` must be finite integers from `0` through `4294967295`, inclusive. Their NumPy wrappers use the same rule. Invalid dimensions fail with `NMATH_SHAPE_MISMATCH` before budget arithmetic, input allocation or native shape conversion. Rank-zero shape `[]` and zero dimensions remain valid. A valid shape whose data length differs from its element count also fails with `NMATH_SHAPE_MISMATCH`.

### Expressions and compute

- Functions such as `add(a, b)` and methods such as `a.add(b)` return `Expr` without computing.
- `expr.compute({ device?: 'cpu' | 'gpu', signal?: AbortSignal }): Promise<Result>` runs one independent job.
- Results with rank 0 are JS scalars: `number` for floating and 32-bit-or-smaller integer dtypes, `bigint` for 64-bit integers, `boolean` and `string`. Complex values use the lossless `Complex` class. Other results are `NMathArray` values.
- Explicit lossy conversions: `toNumber()`, `toDate()`.
- Weak operand scalars follow NumPy NEP 50 with a fixed JS mapping: `number` acts as a Python `float`, `bigint` as a Python `int` and `boolean` as a Python `bool`. `Complex` acts as a Python `complex` weak operand.

Float16 arrays use `Float16Array` for numeric typed-array exchange when the Node.js 24 runtime provides it. Scalar reads return `number`. `ctx.arrayFromFloat16Bits(bits: Uint16Array, { shape? })` and `array.toFloat16Bits()` exchange exact IEEE-754 binary16 encodings, including NaN payloads. Both methods copy storage and preserve the existing budget and disposal rules. Numeric creation and `toTypedArray()` require `Float16Array`; they report an explicit error when it is unavailable. There is no float32 storage substitution. Public dtype promotion, reductions, casts and compiler helper signatures retain float16 where the runtime rules require it.

Small integer arrays exchange data through `Int8Array`, `Int16Array`, `Uint8Array` and `Uint16Array`. Scalar reads return `number`; signed and unsigned small-integer sums return `bigint`. Array-to-array promotion follows NumPy, including `int8` with `uint8` becoming `int16`, and `float32` with any 8-bit or 16-bit integer remaining `float32`.

`new Complex(real, imag = 0)` is the dedicated immutable complex scalar. `real` and `imag` are JS numbers and preserve both IEEE-754 components, including signed zeros and non-finite values. `item()`, nested `toArray()` output and rank-0 `compute()` results return `Complex` for complex dtypes. Implicit numeric conversion throws; `Complex.toNumber()` explicitly returns the real component and discards the imaginary component.

Complex array creation accepts nested `Complex` values or interleaved typed-array components in real/imaginary order. `complex64` requires `Float32Array`; `complex128` requires `Float64Array`. Typed input must contain an even number of components. Shape counts complex elements, not components. `toTypedArray()` returns an independent interleaved copy with the corresponding floating type. `full()` takes a `Complex` scalar and `zeros()` creates complex zero values.

Complex arithmetic, unary operations, reductions, indexing, writes, copies, explicit casts and 1-D/2-D matrix multiplication follow the runtime contract. Complex remainder is rejected by ordinary TypeScript declarations and by runtime expression validation. Complex GPU computation remains outside the GPU dtype subset.

### Typing

`NMathArray<D extends DType, S extends Shape>` and `Expr<D, S>` carry dtype and shape type parameters. `Shape` is a readonly tuple of `number` literals or `number` for a dynamic dimension. Broadcasting is computed at the type level for tuple shapes. Incompatible static shapes produce a type error; dynamic dimensions fall back to runtime validation without erasing dtype or rank.

The NumPy `sum`, `mean`, `min` and `max` wrappers preserve the core methods' dtype and shape inference. Omitting the axis, or passing `undefined`, produces a scalar result type. A literal axis removes its dimension from a tuple shape. A dynamic axis retains the result dtype and uses a dynamic shape because its result rank is not known at type-check time.

### Errors

`NMathError` has a `code` property whose values are the error codes in the [runtime crates contract](crates-nmath-runtime-contract.md#error-codes). Messages are English.

## Storage

The package stores nothing. File I/O follows the [runtime storage contract](crates-nmath-runtime-contract.md#storage).

## Security

The package has no network access and sends no telemetry. Native binaries are loaded only from the package's own platform-specific optional dependencies or the local build output.

## Logging

The package does not log by default. `NMATH_LOG` enables native `tracing` output on stderr.

## Build and Test

- `pnpm --filter @delinoio/nmath build` compiles TypeScript to `dist/`. `build:native` builds `nmath-node` with the napi-rs CLI. `build:compiler` builds `nmath-compiler` and copies it to `native/`.
- `pnpm --filter @delinoio/nmath test` runs Node.js tests against the built native binary.
- `dist/` is ignored generated output and is removed after validation.
- Platform acceptance for the eight CPU binary combinations and the GPU matrix is recorded in pull requests, issues and CI artifacts, not in this document.
- Native package staging, installed-archive smoke tests, license bundles and release evidence follow the [distribution contract](packages-nmath-distribution-contract.md). The staged public manifest selects exact-version platform optional dependencies; development installs do not resolve unpublished platform packages.

### Compatibility inventory and reference fixtures

`packages/nmath/compatibility/inventory/` records the fixed originals' public API boundaries, aliases, per-entry conditions, verification and agreed exclusions. Pending semantics, argument adaptations, dtype/platform, view/mutation and missing-value fields remain explicit support obligations. Source references use pinned revisions and hashes; a row count or successful generation does not establish behavioral or cross-platform acceptance.

`packages/nmath/scripts/compatibility/` generates and checks fixtures from the exact development-only versions in `compatibility/oracle-requirements.txt`. Runtime builds and native comparisons do not need Python. `oracle:generate`, `oracle:inventory` and `oracle:check` are uncached package tasks run through root Turbo; inventory and reproducibility tasks take explicit pinned upstream source-directory arguments. Run them with the pinned oracle environment on PATH. Do not regenerate fixtures from unpinned originals or silently change numerical criteria.

`oracle:check` compares two fresh generations byte for byte. It then checks the committed mutation, primitive and random fixture JSON files. The checker validates and reports the committed and fresh `environment` records separately, and compares every other fixture field. Each record requires nonempty `python`, `system` and `machine` strings and a `byteorder` of `little` or `big`. Inventory artifacts and the artifact file set must still match exactly. Host provenance differences alone can pass this comparison, but platform-dependent inventory aliases or numerical results can still fail it. Keep both provenance records in the artifacts and check output; do not replace them with a reference host. This check does not establish cross-platform acceptance.

`test:compatibility` depends on the native package build, runs through root Turbo without caching, and returns failure for native mismatches. It accepts an optional report output path and records actual source revision and environment. Fixture equality alone is not full numerical acceptance; operation/dtype-specific criteria and unperformed scenarios remain recorded.

The published package includes `compatibility.json.gz`, a gzip-compressed JSON object with fixed `versions`, `acceptanceStatus` and API `rows`. This is the public inventory artifact. It preserves pending and excluded states and ships no Python tooling, oracle environment or input fixtures.

## Dependencies and Integrations

- Upstream: [runtime crates contract](crates-nmath-runtime-contract.md) and [compiler crates contract](crates-nmath-compiler-contract.md).
- The package includes the Apache-2.0 license and third-party notices of bundled native dependencies.

## Change Triggers

- Changes to entry points or public types update this contract, the [project index](project-nmath.md) and the package README.
- Update `AGENTS.md` only when development procedures, directory ownership or repository/domain development rules change; follow the [instruction-update policy](README.md#instruction-update-policy).

## References

- [NMath project index](project-nmath.md)
- [Repository defaults](repository-defaults.md)
- [Repository workflow](repository-workflow-contract.md)

Compiler helpers retain public arithmetic signatures, indexing result types and assignment value types. Only checker-proven NMath operations call these helpers; ordinary JavaScript syntax keeps its native semantics. Bracket access used as a constructor target, including through parentheses and type-only wrappers, keeps its JavaScript grouping.

## Numeric tables

The pandas and Polars numeric table entry points use the same native table executor. A completed result is returned by a Promise; no batch iterator or AsyncIterable is exposed. Table work supports CPU only and rejects explicit GPU selection. Row-local arithmetic, casts, selection and filtering can execute in bounded internal batches. Snapshots and completed results still require their full memory budget; sort, joins, groups and non-row-local expressions retain materialization boundaries. This bounded subset does not establish full Polars streaming compatibility.

A pandas Series accepts optional `index` labels and `validity` column arguments after its values and name. Labels are strings or numbers. `loc(label)` selects matching labels; `iloc` preserves selected labels. Integer `iloc` on a nullable Series is rejected because the current scalar expression cannot carry a validity mask. Arithmetic aligns Series with the same unique label set; identical duplicate-label sequences also work. Unmatched or differently ordered duplicate labels require explicit alignment and are rejected. Full outer label alignment remains a target. Nullable Series arithmetic and reductions are rejected rather than silently discarding nulls.

A pandas DataFrame accepts `{index, validity}` as optional constructor metadata. Columns supplied as Series must already have matching label sequences. `filter(booleanColumn)` and `sortValues(column, options)` return completed frames and preserve labels. `merge(other, key, options)` supports inner and left joins and matches null keys. `groupBy(key, value, operation, options)` returns a two-column result with the group key as a column, equivalent to the supported `as_index=False` workflow. It defaults to sorted keys and dropped null keys; `sort:false` and `dropna:false` are supported.

Polars Series accept an optional validity column. LazyFrame supports named `select`, `filter`, `sort`, `join` and `groupBy`, followed by `collect`. Polars has positional rows, retains null groups and defaults to unmatched null join keys. `nullsEqual:true` enables null-key matching. Group results retain first-seen key order. Arithmetic projections are available before row operations on non-nullable columns. Computed projections after row operations or on nullable inputs remain targets and fail explicitly. Named selection on nullable inputs is supported.

Joins use a single key and `how:'inner'|'left'`. `rightKey` chooses a differently named right key. An equal-name join key is coalesced to the left key by removing only the right key column. Other overlapping right names receive `suffix`, defaulting to `_right`; collisions among retained output names fail with a stable error. This suffix option is an explicit JS adaptation, not the full pandas suffix API. Group aggregate dtype selection differs between entry points: pandas retains narrow integer sums when they fit and widens on overflow; Polars widens 8/16-bit sums to int64 and returns uint32 counts. Polars float32 mean remains float32, while pandas mean is float64.

`Series.toPolars`, `DataFrame.toPolars`, `Series.toPandas` and `DataFrame.toPandas` preserve names, numeric dtypes and validity expressions. These synchronous conversions wrap the same lazy column handles; they do not copy buffers or move data between contexts. Calling compute/collect makes a native execution snapshot and returns independent result storage. pandas-to-Polars conversion rejects a non-default index unless `dropIndex:true` explicitly authorizes label loss. pandas `Series.toColumn()` returns the name, values, validity and labels together. `toArray()` explicitly drops the name and rejects nullable Series; use `toColumn()` to preserve the mask. Cross-context copies remain explicit array operations and require copying both values and validity columns.

Errors use the existing stable codes and English recovery guidance. Table schema errors do not include input values or column names. Names are resolved to positions before native execution.

### NPZ API

`ctx.saveNpz(path, arrays, options?)` writes a `ReadonlyMap<string, NMathArray<any, any>>` in insertion order. Arrays may have different dtypes and shapes, but must belong to `ctx`. Options are `{ compressed?: boolean, overwrite?: boolean, signal?: AbortSignal }`. `compressed` and `overwrite` default to `false`. The core method uses the exact path without adding a suffix and returns `Promise<void>`.

`ctx.loadNpz(path, { signal }?)` returns `Promise<Map<string, NMathArray<ComputableDType, number[]>>>`. Member names remain Map keys, including names such as `__proto__`; they are never assigned as object properties. Rank-zero members remain arrays and expose their scalar through `item()`. Callers dispose each returned array when finished. Empty archives return an empty Map.

The NumPy entry point exports these explicit JS adaptations:

```ts
interface NpzInputs {
  positional?: readonly NMathArray<any, any>[];
  named?: ReadonlyMap<string, NMathArray<any, any>>;
}

savez(context, path, inputs?: NpzInputs, options?: { overwrite?: boolean; signal?: AbortSignal }): Promise<void>;
savez_compressed(context, path, inputs?: NpzInputs, options?: { overwrite?: boolean; signal?: AbortSignal }): Promise<void>;
loadNpz(context, path, options?: { signal?: AbortSignal }): Promise<Map<string, NMathArray<ComputableDType, number[]>>>;
```

`savez` uses stored members; `savez_compressed` uses DEFLATE. Both append `.npz` when the path does not already end with that exact suffix. They insert named entries in Map order, followed by positional entries named `arr_0`, `arr_1`, and so on. A positional name that collides with a named entry rejects with `TypeError` before native submission. Omitted inputs produce an empty archive. There is no pickle option. Existing `load` remains NPY-only; NPZ uses the explicit `loadNpz` method.

All NPZ operations use native scheduled file jobs and explicit contexts. One save freezes all input arrays at a consistent execution-start snapshot boundary, including arrays that share storage. Snapshot offsets and copied payloads count against the context CPU memory budget; a save fails before publication if a required reservation does not fit. Pre-publication cancellation or failure preserves existing output, and writes reject existing destinations unless `overwrite: true`. Successful atomic publication is terminal success even if the signal aborts before Promise completion. Reads retain cancellation checks through scheduler completion. Name, compression, size, dtype and layout limits follow the runtime NPZ contract and remain visible compatibility constraints.

## Random generator API

`context.generator(seed)` and `numpy.random.defaultRng(context, seed)` return the same lazy PCG64 generator. The explicit seed is a nonnegative integer of at most 128 bits. Use bigint for seeds outside the safe JS integer range.

Methods use an explicit shape, with `[]` for a scalar:

- `random(shape, dtype = "float64")`, with float32 or float64.
- `integers(low, high, shape, { dtype = "int64", endpoint = false } = {})`, with bool or signed/unsigned 8/16/32/64-bit dtypes. Passing `undefined` for high uses `[0, low)`. Bounds are safe JS integers or bigint; endpoint includes the upper bound. Full-width uint64 exclusive high accepts `2n ** 64n`.
- `uniform(low, high, shape)`, returning float64.
- `standard_normal(shape, dtype = "float64")`, with float32 or float64.
- `normal(loc, scale, shape)` and `exponential(scale, shape)`, returning float64.

Distribution parameters are scalars. Each `compute()` is a new scheduled native job. Jobs sharing a generator run in request order and commit state only on success. Invalid bounds, allocation errors and cancellation before completion preserve generator state, including cached uint32 values. No JS random kernel or Python runtime is used.

Parameter broadcasting, `out`, other distributions and bit generators, optional entropy seeding, state import/export and SeedSequence spawning remain targets. Integer float coercion is not supported: callers must supply safe integers or bigint explicitly. These methods do not establish full NumPy random compatibility or cross-platform numerical acceptance.

### CSV and JSON JS adaptations

`ctx.readText(path, 'csv' | 'json', {schema?, signal?}?)` returns a Promise of `{columns, validity}` Maps. `schema` is a complete ordered `ReadonlyMap<string, TextDType>`; `TextDType` includes the current real numeric and boolean dtypes. Every returned column has a bool validity array. Dispose both values and masks. `ctx.writeText(path, format, columns, validity?, {overwrite?, signal?}?)` returns `Promise<void>` and accepts numeric lazy columns owned by the context. Paths are used exactly as supplied. Overwrite defaults to false. Both methods use one native scheduled CPU file job.

The pandas and Polars entry points export `read_csv(context, path, options?)` and `read_json(context, path, options?)` returning their respective DataFrame in a Promise. pandas frames expose `toCsv(context, path, options?)` and `toJson`; Polars frames expose `writeCsv` and `writeJson`. These names, explicit contexts, Maps, AbortSignals and Promise results are JS adaptations. JSON uses records orientation for both entry points; it does not implement pandas' default columns orientation. CSV writes a header without an index. Non-default pandas indexes require `dropIndex:true` to authorize loss; reads create a default positional index. Use an explicit schema to preserve narrow/all-null dtypes and empty JSON column schemas. Exact int64/uint64 values return as bigint arrays.

The numeric token, validity, memory, cancellation, atomic publication and unsupported-option limits are defined in the runtime text-file contract. Unsupported orientations/options, richer dtypes and upstream default-policy differences remain open compatibility obligations. No Python runtime is required.

## Workflow benchmark protocol

The development-only harness in `packages/nmath/scripts/benchmarks/` runs three synthetic workflows: a 128 by 128 float32 matrix product, a 32,768-row int64 group sum with 64 groups through both pandas and Polars APIs, and an 8,192-element float32 Horner polynomial authored in `.nts`. The seed is 20261029. Each stream starts from that seed and uses the unsigned 32-bit recurrence `state = 1664525 * state + 1013904223`; values are `(state >>> 24) % 16 - 8`. The matrix right operand is the transpose of the left. Table keys are row index modulo 64. The algorithm is `(x * 2 + 3) * x + 1`.

Small integer-valued inputs keep every float32 intermediate exact. Full-output SHA-256 digests cover comma-separated decimal integer values, with schema and missing-value checks performed separately. Exact equality is a workload-specific criterion, not a tolerance decision for general floating-point operations. An independent integer reference checks generated oracle results. The checked-in fixture records NumPy 2.5.4, pandas 3.1.0 and Polars 2.1.0; regeneration rejects other versions. No Python dependency is needed for native benchmark execution or its correctness checks.

Run with Node.js 24 and the pinned pnpm from the repository root:

```sh
pnpm exec turbo run benchmark --filter=@delinoio/nmath -- --output /tmp/nmath-cpu.json
pnpm exec turbo run benchmark --filter=@delinoio/nmath -- --device gpu --output /tmp/nmath-gpu.json
pnpm exec turbo run test:benchmarks --filter=@delinoio/nmath
```

The output parent directory must exist. Each run replaces the requested report with atomic complete JSON snapshots. Use a separate output path for each experiment. On POSIX hosts, SIGINT/SIGTERM abort native work or compilation and preserve completed measurements and an interrupted status. Windows forced termination does not provide this guarantee; its SIGTERM test is skipped. Errors produce a failed report and a nonzero exit. SIGKILL or a power loss cannot produce a final status; a running report is incomplete evidence.

`benchmark` and `test:benchmarks` depend on the package and compiler builds and are uncached. `benchmark:originals` is uncached and development-only. Put a dedicated environment containing the exact versions in `compatibility/oracle-requirements.txt` on PATH before running it:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1 pnpm exec turbo run benchmark:originals --filter=@delinoio/nmath -- --output /tmp/nmath-originals.json
```

To regenerate the fixture, append `--fixture test/fixtures/benchmarks/originals.json`; that path is relative to the owning package. Turbo passes the five listed thread controls explicitly. The report records their values and NumPy build configuration. Missing original environments or mismatched versions fail rather than substitute other releases. Python RSS high-water measurement is unavailable on platforms without the `resource` module and is reported as null.

Default runs use two warmups and nine measured executions, configurable with `--warmup` and `--samples`. Each native workload checks all output values, shape, dtype and validity before timing, and checks every measured result outside the timed interval. The harness reports raw samples, median, nearest-rank p95, median absolute deviation and extrema without deleting outliers. Run performance experiments separately from test suites and other workloads; repeat independent process runs when studying variance. These small representative inputs do not establish scaling behavior for all dataset sizes.

Native reports separate import, compilation, setup/input construction, first execution, compute wall time and output extraction. Native execution metrics include snapshots and scheduler work within their existing boundary. Wall minus queue and execution metrics is a signed residual estimate, not an isolated N-API timer. A one-element probe exposes a small dispatch workload. RSS samples, process-lifetime maximum RSS and cumulative context CPU/GPU allocation peaks have different accounting scopes; none is advertised as a kernel-only memory measurement. Original benchmarks exclude imports, input generation, correctness checks and output extraction from timed compute, and expose first execution separately.

GPU requests apply only to numerical arrays. Tabular measurements remain CPU. A full-array identity probe reports transfer-inclusive time and logical input/output bytes. Isolated upload/download times are null because public metrics provide no transfer timestamps; that measurement gap remains explicit. Device initialization and pipeline setup may occur in first execution. No automatic fallback, retry, official GPU allowlist approval or speedup claim follows from a completed report. Known unaccepted GPU power differences remain outside this workload and are not waived.

Reports record source revision, dirty-tree state, script/fixture/algorithm/native-binary hashes, exact software and hardware metadata, commands, versions and seed. Original-library reports obtain Linux CPU models from `/proc/cpuinfo` and macOS models from `sysctl`. They record identification status and source; an unavailable model is null with an explicit reason, never an architecture substituted as a model. Keep local reports and logs in external validation artifacts, not source-of-truth contracts. Cross-platform execution and wider numerical acceptance remain separate release requirements.

### Columnar file API

`ctx.readColumnar(path, 'arrow' | 'parquet', {signal}?)` returns a Promise of `{columns, validity, rows}`. Columns are context-owned materialized `NMathArray` instances keyed by name; validity masks are materialized boolean arrays. Their public types expose scalar reads, TypedArray conversion and disposal. `ctx.writeColumnar(path, format, columns, validity?, {overwrite, signal}?)` returns `Promise<void>`. The exact local path is used; suffixes are not added. `overwrite` defaults to false. Dispose returned value and validity arrays when finished.

pandas exports `read_parquet(context, path, options?)` and `read_feather(context, path, options?)`, each returning `Promise<DataFrame>`. DataFrame methods `to_parquet(context, path, options?)` and `to_feather(context, path, options?)` return `Promise<void>` and reject non-default index labels before submission. This explicit-context argument is a JS adaptation. The default index is reconstructed on read. The file adapter preserves numeric dtype and null masks instead of applying pandas' default nullable-integer-to-float inference; compare this representation with pandas `dtype_backend="numpy_nullable"`. Unsupported pandas metadata files fail explicitly.

Polars exports `read_parquet(context, path, options?)` and `read_ipc(context, path, options?)`; DataFrame methods are `write_parquet(context, path, options?)` and `write_ipc(context, path, options?)`. They preserve supported column names, dtypes, nullability and positional rows. File options currently contain only `signal` and, for writes, `overwrite`. Engine, compression, column projection and original-library options not listed here remain targets rather than silently ignored options. All file adapters use scheduled native CPU work and completed Promise results. Format, codec, metadata and dtype limits follow the runtime contract.

DataFrame columnar read adapters reject nonempty zero-column tables with `NMATH_DTYPE_UNSUPPORTED` because their constructors cannot preserve an explicit row count. `ctx.readColumnar` retains that row count for direct callers.

### Linear algebra and FFT adaptation

The NumPy entry point exports `linalg.solve(a, b)`, `linalg.inv(a)`, `linalg.det(a)` and `fft.fft(a, options)`, `fft.ifft`, `fft.rfft`, `fft.irfft`. Arguments are contextual lazy arrays or expressions; results are lazy expressions and use normal `compute()` and complex transport. FFT options are `{n?, axis?, norm?: "backward" | "forward" | "ortho"}`. Unknown options, including `out`, fail explicitly. The native runtime contract owns batch, dtype, normalization and resource behavior. Singular linear systems use `NMATH_LINALG_SINGULAR`.

Linalg declarations preserve batch/result shape and reject known nonsquare matrices and incompatible solve core dimensions. FFT declarations preserve unchanged dimensions for omitted or positive literal axes and axis `-1`. Real-transform axis lengths and other negative/dynamic axis inference remain conservative; exact inference for those forms remains a target. These APIs do not establish full linear-algebra or FFT family coverage.

### Partial counterpart evidence

An optional `supportedScope` object records an explicitly inspected partial counterpart: family, adapted arguments, supported dtypes, conditions, and implementation/test/fixture paths. It does not claim full semantics, argument coverage, scalar-class support, or alias support. All non-excluded forms remain targets. `verification.evidence` records content hashes and the inspected implementation revision; evidence availability is separate from execution results and acceptance. GPU and cross-platform statuses remain pending until separately accepted. The generated support summary counts public entry paths, including aliases, not independent supported features.
