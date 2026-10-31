# @delinoio/nmath

Native numerical computing for Node.js 24 without Python. NMath runs heavy work in Rust on native worker threads, and offers NumPy, pandas and Polars style APIs with TypeScript dtype and shape types.

> **Status: 0.x prerelease.** Only part of the planned API exists. See the [compatibility inventory](#compatibility-inventory) before you depend on a feature. Breaking changes can occur in any 0.x release.

## Install

```sh
npm install @delinoio/nmath
```

Requirements:

- Node.js 24. Other runtimes (Bun, Deno, browsers) are not supported.
- ESM. From CommonJS, use `await import("@delinoio/nmath")`.
- A prebuilt native binary for your platform. Planned targets: macOS 14+ (x64, arm64), Windows 11 (x64, arm64), Linux glibc 2.36+ and musl 1.2.5+ (x64, arm64). If the binary cannot load, `import` fails with `NMATH_NATIVE_LOAD`. NMath does not build from source or fall back to JavaScript automatically.

## Quick start

```ts
import { createContext, slice } from "@delinoio/nmath";

// Every computation belongs to an explicit context with explicit limits.
await using ctx = createContext({
  maxConcurrentJobs: 4,
  maxQueueLength: 64,
  cpuMemoryBudget: 1n << 30n, // bytes
  gpuMemoryBudget: 0n, // CPU-only
});

const a = ctx.array([[1, 2], [3, 4]], { dtype: "float64" });
const b = ctx.array([10, 20], { dtype: "float64" });

// Operations are lazy. Nothing runs until compute().
const expr = a.mul(b).add(1).sum(0);
const result = await expr.compute(); // NMathArray<"float64", [2]>
console.log(result.toArray()); // [42, 122]

// Rank-0 results are JS scalars.
const total: number = await a.sum().compute();

// Reads are lazy too. Basic indexing returns a view that shares storage.
const firstColumn = await a.get(slice(), 0).compute();
await a.set([0, 0], 100); // atomic, asynchronous write
```

## Execution model

- `expr.compute({ device, signal })` returns a Promise. `device` is `"cpu"` (default) or `"gpu"`. `signal` is an `AbortSignal`.
- Each call runs an independent job. NMath never caches results.
- A job reads its inputs when it starts running, not when you call `compute()`. Changes made while a job is queued are visible to it; changes after it starts are not.
- A context runs up to `maxConcurrentJobs` jobs and queues the rest in FIFO order. When the queue is full, `compute()` rejects with `NMATH_SCHED_QUEUE_FULL`.
- Every allocation (inputs, snapshots, intermediates, results) is charged to the context's CPU or GPU budget. An allocation that does not fit fails with `NMATH_MEMORY_BUDGET_EXCEEDED` before it is made and is not retried.
- Aborting a queued job removes it. Aborting a running job stops it at the next check point; a cancelled job never returns a result.
- Writes to the same storage run in request order and commit completely or not at all.
- `ctx.close()` rejects new work, cancels queued and running work and resolves after cleanup. Contexts support `await using`; arrays support `using` and `dispose()`.
- Arrays from different contexts cannot be combined directly (`NMATH_CONTEXT_MISMATCH`). Use `await target.copy(array)`, which charges the copy to the target context.

## Scalars

| Dtype                                                                       | JS value  |
| --------------------------------------------------------------------------- | --------- |
| `float32`, `float64`, `int8`, `int16`, `int32`, `uint8`, `uint16`, `uint32` | `number`  |
| `int64`, `uint64`                                                           | `bigint`  |
| `bool`                                                                      | `boolean` |
| `complex64`, `complex128`                                                   | `Complex` |

A `Complex` operand acts like a Python `complex`. Adding it to `float32` produces `complex64`, while adding it to an integer array produces `complex128`.

As operands, a JS `number` acts like a Python `float`, a `bigint` like a Python `int` and a `boolean` like a Python `bool` (NumPy NEP 50 rules). For example, `int32Array.add(1n)` stays `int32`, and `int32Array.add(1)` becomes `float64`.

```ts
import { Complex } from "@delinoio/nmath";

const z = ctx.array([new Complex(1, 2), new Complex(3, -4)], {
  dtype: "complex64",
});
const total: Complex = await z.sum().compute();
console.log(total.real, total.imag); // 4, -2
const parts: Float32Array = z.toTypedArray(); // [1, 2, 3, -4]
const restored = ctx.array(parts, { dtype: "complex64" });
```

`Complex` stores separate `real` and `imag` numbers. Complex scalar reads and rank-0 results retain both components. `toTypedArray()` returns interleaved real/imaginary components: `Float32Array` for `complex64`, `Float64Array` for `complex128`. Array shapes count complex elements. Nested array input uses `Complex` instances.

Numeric conversion is explicit: `value.toNumber()` returns only the real component, and `array.astype("float64")` discards imaginary components. Assigning complex values to a real numeric array requires that explicit cast first. Complex-to-boolean casts test both components. Complex remainder is undefined and is rejected; the other listed arithmetic operations support complex arrays. Complex min/max compare real components first, then imaginary components, and propagate NaNs. Complex arrays execute on the CPU.

## Types

`NMathArray<D, S>` and `Expr<D, S>` carry the dtype `D` and shape `S`. TypeScript checks dtype promotion, broadcasting, reductions, `matmul` and indexing when shapes are literal tuples:

```ts
const m = ctx.zeros([2, 3], "float32");
m.add(ctx.zeros([3], "int32")); // Expr<"float64", [2, 3]>
m.add(ctx.zeros([4], "float32")); // type error: shapes do not broadcast
```

A dimension typed as `number` is dynamic. The runtime validates it, and the dtype and rank stay known.

Array, `full`, `zeros` and generator `random` shapes, including their NumPy forms, accept only finite integer dimensions from `0` through `4294967295`. Invalid dimensions throw `NMathError` with code `NMATH_SHAPE_MISMATCH` before allocation. Empty dimensions and scalar shape `[]` are valid. Array data must still match the requested shape.

## Entry points

| Import                     | Purpose                                                                                                              |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `@delinoio/nmath`          | Contexts, arrays, expressions, selectors, errors.                                                                    |
| `@delinoio/nmath/numpy`    | NumPy-style functions (`array`, `zeros`, `ones`, `arange`, `add`, `matmul`, `random.defaultRng`, `save`, `load`, …). |
| `@delinoio/nmath/pandas`   | pandas-style `Series` and `DataFrame` (subset).                                                                      |
| `@delinoio/nmath/polars`   | Polars-style `Series`, `DataFrame`, `col` and `LazyFrame` (subset).                                                  |
| `@delinoio/nmath/compiler` | Project `check` and `build` APIs for `.nts` files.                                                                   |

## `.nts` compiler

Use `.nts` files when you want arithmetic operators and bracket indexing on NMath values. Ordinary `.ts` files use the functions and methods shown above.

```ts
// main.nts
import { createContext } from "@delinoio/nmath";

const ctx = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 4,
  cpuMemoryBudget: 1n << 20n,
  gpuMemoryBudget: 0n,
});

try {
  const a = ctx.array([1, 2, 3], { dtype: "float64" });
  const result = await (a * 2 + 1).compute();
  console.log(result.toArray()); // [3, 5, 7]
} finally {
  await ctx.close();
}
```

For a project with `main.nts` next to `tsconfig.json`, use:

```json
{
  "compilerOptions": {
    "target": "es2024",
    "module": "nodenext",
    "moduleResolution": "nodenext",
    "strict": true,
    "outDir": "out",
    "types": ["node"],
    "lib": ["es2024", "esnext.disposable"]
  },
  "include": ["*.nts"]
}
```

The consuming project must use ESM (`"type": "module"` in `package.json`), resolve `@delinoio/nmath`, and have Node.js type declarations installed. After installing or building the compiler binary, run:

```sh
nmath-compiler check -p "tsconfig.json"
nmath-compiler build -p "tsconfig.json"
node "out/main.js"
```

The CLI is included in this package. When using a local installation with pnpm, invoke it with `pnpm exec nmath-compiler`. `check` reports diagnostics without emitting files. `build` emits ESM JavaScript, declarations and source maps; type or transform errors leave existing output unchanged. Add `--json` for structured diagnostics. Exit codes are 0 for success, 1 for diagnostics and 2 for usage or configuration errors.

The compiler API and CLI can run when the numerical addon is unavailable, provided the compiler executable is installed. If the compiler executable is missing, reinstall the package or build the compiler from source with `pnpm --filter @delinoio/nmath build:compiler`.

The programmatic API runs the same compiler:

```ts
import { check, build } from "@delinoio/nmath/compiler";

const checked = await check({ tsconfig: "tsconfig.json" });
if (checked.ok) {
  const built = await build({ tsconfig: "tsconfig.json" });
  console.log(built.outputFiles);
} else {
  console.error(checked.diagnostics);
}
```

Arithmetic operators `+`, `-`, `*`, `/`, `%`, `**`, unary `+`/`-` and bracket reads lower to lazy NMath operations. Bracket writes require an async function body or module top level. Comparisons, bitwise operators and compound assignments are not NMath operator syntax. Project references and `-b` builds are not supported. Emitted source maps and diagnostics refer to original `.nts` positions.

The NMath VS Code extension associates `.nts` files and starts the compiler language server in trusted workspaces. Set `nmath.compilerPath` to a compiler executable, or let the extension find `node_modules/.bin/nmath-compiler` in the workspace folder and then `nmath-compiler` on `PATH`. Use **NMath: Restart Language Server** after changing the executable. Server output appears in **NMath Language Server**.

## `.nts` projects and editor support

Use `nmath-compiler check -p tsconfig.json` to check a project, or `nmath-compiler build -p tsconfig.json` to emit ESM JavaScript, declarations and source maps. Keep source-relative imports ending in `.nts`; emitted imports use `.js`. Type or transformation errors preserve existing output.

The `@delinoio/nmath/compiler` entry point exposes asynchronous `check({ tsconfig })` and `build({ tsconfig })` functions. Results contain `ok`, structured `diagnostics`, and `outputFiles`.

`nmath-compiler lsp` serves type diagnostics, completion and definitions over standard input/output. Locations refer to original `.nts` sources, including unsaved editor changes. The NMath VS Code extension starts this server and uses standard editor UI. Automatic import suggestions are not available. Project references and `-b` builds are not supported.

## GPU

GPU execution is limited to `float32`, `int32` and `uint32` elementwise arithmetic, broadcasting, `sum`, `mean`, `min`, `max` and `matmul` on macOS arm64 (Metal), Linux x64 glibc (Vulkan) and Windows x64 (DX12). NMath never falls back to the CPU, never lowers precision and never retries. Unsupported dtypes or operations reject with `NMATH_GPU_UNSUPPORTED_OPERATION` before the job is queued.

> GPU acceptance remains incomplete. Native execution rejects devices that cannot preserve required float32 subnormal arithmetic with `NMATH_GPU_UNAVAILABLE`. No fallback or relaxed precision is used. NMath makes no official GPU support claim until the complete tested hardware matrix is published.

## Files

`array.save(path, { overwrite })` writes NPY files and `ctx.load(path)` reads them. NMath never replaces an existing file unless you pass `overwrite: true`. Output goes to a temporary file that is renamed into place, so cancellation or failure leaves the original file unchanged. Keep input files unchanged while a job reads them; NMath reports changes it detects with `NMATH_IO_MODIFIED`, but it cannot detect every change.

Complex NPY output uses little-endian components in C order. Complex NPY input accepts either byte order and C or Fortran order. Loading preserves dtype, logical shape and both components.

## Errors

Every failure is an `NMathError` with a stable `code` (for example `NMATH_SCHED_QUEUE_FULL`) and an English message with recovery guidance. Error messages and measurements never include input data, file paths or file contents. NMath sends no telemetry. `ctx.metrics()` returns per-execution status and timing and memory use, so you can collect measurements yourself. Set `NMATH_LOG=debug` to write structured native logs to stderr (`NO_COLOR` disables color).

## Compatibility inventory

The installed package includes `compatibility.json.gz`. Decompress this JSON inventory to inspect each original API, its counterpart and verification state. A `pending` row or field is an unresolved support target; `excluded` records an agreed exclusion. The inventory does not certify full compatibility. Python is needed only by maintainers generating reference fixtures, never by the runtime.

Reference versions: NumPy 2.5.4, pandas 3.1.0 and Polars 2.1.0. This prerelease covers a small part of each library. Every API not listed as available is a planned support target unless a published exclusion says otherwise.

| Area                 | Available now                                                                                                                                                           | Not yet available                                                                                                                                                |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dtypes               | bool, signed/unsigned 8/16/32/64-bit integers, float16/32/64, complex64/128, fixed-width str, unit-bearing datetime64/timedelta64                                       | extended precision, byte/variable strings, categorical, structured and full scalar-class APIs                                                                    |
| NumPy arithmetic     | `+ - * / **`, unary `+ -`, broadcasting, NEP 50 promotion; `%` for supported real dtypes                                                                                | comparisons, bitwise, ufunc methods, `where`, math functions                                                                                                     |
| NumPy reductions     | `sum`, `mean`, `min`, `max`, `prod`, `argmin`, `argmax` (all axes or one axis)                                                                                          | `std`, `var`, `cumsum`, `cumprod`, multi-axis lists, `keepdims`                                                                                                  |
| NumPy indexing       | integers, slices, one integer-array or mask selector per operation, views for basic indexing, atomic assignment                                                         | several advanced selectors combined, `newaxis`, ellipsis                                                                                                         |
| NumPy linear algebra | `matmul` with broadcast batch axes and vector promotion; ascending/descending polynomial evaluation and derivatives                                                     | full `linalg`/`fft` coverage, polynomial operations beyond evaluation/derivatives, masked arrays                                                                 |
| NumPy random         | PCG64 `random`, `integers`, `uniform`, `standard_normal`, `normal`, and `exponential` with scalar parameters                                                            | parameter broadcasting, `out`, other distributions and bit generators, state import/export and SeedSequence spawning                                             |
| NumPy I/O            | NPY and stored/deflated NPZ; float16/complex input supports both byte orders and C/Fortran layout                                                                       | Automatic NPY/NPZ load dispatch, richer dtypes and remaining noncomplex endian/layout support                                                                    |
| pandas               | Numeric Series/DataFrame, basic labels, nullable filtering/sorting, single-key inner/left joins and grouped aggregates; CSV, records JSON, Parquet and Feather adapters | Full label alignment, nullable arithmetic, richer join/group options, pivots, windows, time series, string/categorical APIs and full file option/metadata parity |
| Polars               | Numeric Series/DataFrame, lazy arithmetic/select/filter, single-key joins/groups, bounded internal pipelines and CSV/JSON/Parquet/IPC adapters                          | Full streaming, richer projections/joins/groups, windows, string/temporal namespaces and full file option/metadata parity                                        |
| `.nts` compiler      | Project checking, ESM/declarations/original-source maps, semantic LSP                                                                                                   | Marketplace publication and editor acceptance                                                                                                                    |
| GPU                  | Native backend with strict device and precision rejection                                                                                                               | Complete numerical and hardware acceptance                                                                                                                       |

## Build from source

In a clone of the repository, install Node.js 24, Rust (the toolchain in `rust-toolchain`) and pnpm 10.26.2. Run from the repository root:

```sh
pnpm install --frozen-lockfile
pnpm exec turbo run build --filter=@delinoio/nmath
pnpm exec turbo run build:compiler --filter=@delinoio/nmath
```

`build:native` and `build:compiler` write the binaries for your platform to `native/`.

## Support

Report problems in [GitHub Issues](https://github.com/delinoio/nmath/issues). There is no guaranteed response time.

## License

Apache-2.0. See `LICENSE` and `NOTICE`. Native dependencies keep their own licenses.

## Numeric tables

The numeric table APIs are a bounded native subset of pandas 3.1.0 and Polars 2.1.0. They do not implement the complete pandas or Polars API. Columns can use bool, signed/unsigned 8-, 16-, 32- or 64-bit integers, float32 and float64. Nulls use explicit boolean validity columns; true means valid. Fixed-width strings and temporal columns support materialization, selection, filtering and access; their sorting/join/group operations remain targets. Categorical, complex and extended-precision table columns remain targets. NaN/inf and floating-point summation behavior still require broader original-derived numerical acceptance; explicit-null integer fixtures do not establish complete missing-value parity.

```ts
import { createContext } from '@delinoio/nmath';
import { DataFrame, Series } from '@delinoio/nmath/polars';

const ctx = createContext({
  maxConcurrentJobs: 2,
  maxQueueLength: 8,
  cpuMemoryBudget: 64n << 20n,
  gpuMemoryBudget: 0n,
});
const frame = new DataFrame([
  new Series('key', ctx.array([1n, 1n, 2n], {dtype: 'int64'})),
  new Series('value', ctx.array([10n, 0n, 30n], {dtype: 'int64'}),
    ctx.array([true, false, true], {dtype: 'bool'})),
]);
const result = await frame.lazy().groupBy('key', 'value', 'sum').collect();
// key: [1n, 2n], value: [10n, 30n]
await ctx.close();
```

Supported table operations: native multi-column compute, boolean-column filtering, stable single-key sorting, single-key inner/left equijoins, sum/mean/min/max/count grouping, and basic materialized lazy plans. pandas preserves explicit labels and matches null join keys; Polars uses positional rows and requires `nullsEqual:true` to match null join keys. Full label alignment, nullable expression arithmetic, arbitrary post-row projections, multiple keys, additional join modes, windows, pivots, richer string/temporal/categorical operations, query optimization and full internal streaming remain targets.

Pandas `groupBy` drops null and floating NaN keys by default. Pass `{ dropna: false }` to retain those groups. Polars `groupBy` retains NaN keys and null groups.

### NPZ files

Use an explicit context and await NPZ operations:

```ts
import { savez, savez_compressed, loadNpz } from '@delinoio/nmath/numpy';

await savez(ctx, 'arrays', {
  named: new Map([['weights', weights]]),
  positional: [samples],
});
const arrays = await loadNpz(ctx, 'arrays.npz');
const weightsAgain = arrays.get('weights');
const samplesAgain = arrays.get('arr_0');

await savez_compressed(ctx, 'arrays.npz', { named: arrays }, { overwrite: true });
for (const array of arrays.values()) array.dispose();
```

The save helpers append `.npz` if needed. Named Map entries precede positional `arr_0`, `arr_1`, and subsequent entries; a name collision rejects before the write starts. Empty archives are supported. Loaded values are always arrays, including rank-zero members. Use `item()` for a scalar. `loadNpz` returns a Map, so arbitrary supported names do not become object properties. Existing `load` reads NPY files only.

For exact paths, use `ctx.saveNpz(path, map, { compressed, overwrite, signal })` and `ctx.loadNpz(path, { signal })`. Compression and overwrite default to false. All saved arrays must belong to the context. Cancellation before publication preserves existing output. A write that has already published completes successfully even if its signal aborts before the Promise settles.

NPZ supports stored and deflated numeric/boolean NPY members, including complex64 and complex128. Complex input supports both byte orders and C/Fortran layout; output is little-endian C order. Pickle/object data, byte/variable string, generic temporal, structured and extended dtypes, unsupported noncomplex byte orders/layouts, encrypted members and other compression methods are unsupported. Names must be nonempty UTF-8 components other than `.` or `..`, without slashes, backslashes, colons or control characters, and at most 65,531 UTF-8 bytes. Duplicate names and non-NPY members are rejected. ZIP64 central directories, multi-disk archives, 65,535 or more members and output capacity bounds at or above 4 GiB are unsupported. Conservative memory and output reservations can reject an archive earlier. These limits do not establish full NumPy compatibility.

### Float16 arrays

Use `dtype: "float16"` for two-byte IEEE-754 binary16 storage. Arithmetic, reductions and 1-D/2-D matrix multiplication run on CPU. Scalars are JS numbers. Promotion follows NumPy: float16 with an 8-bit integer stays float16, with a 16-bit integer becomes float32, and with a 32-bit/64-bit integer becomes float64. Weak real scalars preserve float16. GPU execution rejects float16 before queue admission.

`toTypedArray()` returns a `Float16Array` copy. Numeric creation requires a Node.js 24 runtime with `Float16Array`. For exact bit transport, including NaN payloads, use `ctx.arrayFromFloat16Bits(new Uint16Array([0x3c00, 0x8000]))` and `array.toFloat16Bits()`. These bit methods also work when `Float16Array` is unavailable. No method substitutes float32 storage.

Float16 NPY and NPZ input supports both byte orders and C/Fortran layout. Output uses little-endian C order. Recorded numerical cases compare exact finite bits and NaN classification. Broader power, long-reduction and matrix-accumulation criteria remain incomplete; these tests do not establish full numerical or platform acceptance.

### Numeric CSV and JSON files

```ts
import { createContext } from '@delinoio/nmath';
const ctx = createContext({ maxConcurrentJobs: 2, maxQueueLength: 8,
  cpuMemoryBudget: 1n << 30n, gpuMemoryBudget: 0n });
const table = await ctx.readText('input.csv', 'csv', {
  schema: new Map([['id', 'uint64'], ['value', 'float64']]),
});
await ctx.writeText('output.json', 'json', table.columns, table.validity);
for (const value of table.columns.values()) value.dispose();
for (const mask of table.validity.values()) mask.dispose();
await ctx.close();
```

Both pandas and Polars export `read_csv(context, path, options?)` and `read_json`.
CSV is UTF-8 with comma-separated fields and a header. JSON is an array of numeric
record objects. pandas writes with `frame.toCsv(context, path, options?)` and
`toJson`; Polars uses `writeCsv` and `writeJson`. These are explicit JS adaptations,
including JSON records orientation instead of pandas' default columns orientation.
Files retain numeric values and nulls. Supply a complete ordered schema Map to
preserve narrow dtypes, all-null columns and empty JSON schemas. int64 and uint64
remain exact bigint values. A custom pandas index requires `dropIndex:true` on
write; reads create a positional index.

Use `{overwrite:true}` to authorize atomic replacement and `{signal}` to cancel.
Writes preserve the original destination on failure before publication. A completed
publication remains successful after late cancellation. All file work runs in the
native CPU scheduler. Readers currently materialize files with conservative memory
reservations, so a large file can exceed the budget before parsing. Keep source
files unchanged during reads; modification detection is best effort.

The core methods `ctx.readText(path, format, options?)` and
`ctx.writeText(path, format, columns, validity?, options?)` use `format: 'csv'|'json'`
and Maps of columns/masks. Dispose both Maps returned by a core read.
Custom delimiters, compressed/alternate encodings, other JSON orientations/JSONL,
string/datetime columns and full upstream read/write options remain unsupported.
CSV empty cells and JSON null become invalid values. JSON non-finite floats become
null. CSV non-finite numeric tokens remain valid floating values; full pandas
missing-token and blank-line defaults are not yet implemented.

## Third-party notices

Distribution archives include Apache-2.0 terms for NMath and the original license terms and notices for included third-party components in `THIRD_PARTY_LICENSES.txt`. Third-party material retains its original licenses. The accompanying JSON receipt records the notice bundle digest.

## Benchmark interpretation

Performance evidence uses synthetic matrix multiplication, pandas/Polars group aggregation and a numerical algorithm compiled from `.nts`. Comparisons use fixed NumPy, pandas and Polars versions. Results must identify the hardware, software, seed, workloads and source revision, and separate startup/compilation from repeated execution. Compute time, memory and native-boundary overhead have different measurement boundaries. GPU timings include transfers unless a report explicitly provides isolated transfer measurements.

These representative workloads are not a promise of a speedup or full numerical/platform acceptance. An exact result on bounded integer-valued float32 inputs does not establish correctness for arbitrary float32 inputs. GPU support and numerical acceptance require their separate validation evidence.

### Arrow IPC and Parquet local files

Use an explicit context for asynchronous file operations:

```ts
const table = await ctx.readColumnar("input.parquet", "parquet");
await ctx.writeColumnar("output.arrow", "arrow", table.columns, table.validity);
for (const value of [...table.columns.values(), ...table.validity.values()]) {
  value.dispose();
}
```

Writes reject existing destinations unless `{ overwrite: true }` is provided. Reads and writes accept `{ signal }`. Failed or cancelled writes preserve the destination until atomic publication; a published file is a successful write. Column names, numeric/boolean dtypes and null masks are preserved. 64-bit integer scalars use `bigint`. File operations use CPU workers and context scheduling and memory budgets.

pandas provides `read_parquet(ctx, path)` and `read_feather(ctx, path)`, plus DataFrame `to_parquet(ctx, path)` and `to_feather(ctx, path)`. A non-default pandas index is rejected. Polars provides `read_parquet(ctx, path)` and `read_ipc(ctx, path)`, plus DataFrame `write_parquet(ctx, path)` and `write_ipc(ctx, path)`. Each returns a Promise. The pandas adapter retains nullable numeric widths, corresponding to pandas' `dtype_backend="numpy_nullable"` representation.

Current format limits: uncompressed Arrow IPC files/Feather V2 and uncompressed or Snappy Parquet. Output Parquet uses Snappy. IPC stream containers, compressed IPC, other Parquet codecs, pandas/custom metadata, duplicate names, dictionaries, and nonnumeric/nonboolean schemas fail explicitly. IPC supports at most 1,024 flat fields and 4,096 UTF-8 bytes per field name. These are implementation limits, not full format compatibility. Reads currently buffer serialized input before batched decoding, and conservative codec reservations may reject files whose final table alone would fit the budget.

Columnar DataFrame readers reject nonempty tables with no columns. Use `ctx.readColumnar` to retain their explicit `rows` count.

### Random distributions

Random expressions use an explicit context and seed. They run natively when computed. Each computation generates new values; successful jobs sharing a generator advance its state in request order. Errors and cancellation before completion preserve its state.

```ts
const rng = ctx.generator(12345n);
const samples = await rng.random([100], "float32").compute();
const dice = await rng.integers(1, 6, [100], { dtype: "int32", endpoint: true }).compute();
const gaussian = await rng.normal(0, 1, [100]).compute();
const waitingTimes = await rng.exponential(2, [100]).compute();
```

`random(shape, dtype?)` and `standard_normal(shape, dtype?)` accept float32 or float64 and default to float64. `uniform(low, high, shape)`, `normal(loc, scale, shape)` and `exponential(scale, shape)` return float64. All distribution parameters are scalars. Use `[]` as the shape for a scalar result.

`integers(low, high, shape, options?)` supports bool and signed/unsigned 8/16/32/64-bit integers. It defaults to int64 and excludes high unless `endpoint: true`. Pass `undefined` for high to use zero as the lower bound. Bounds must be safe JS integers or bigint. Use bigint for exact 64-bit bounds; uint64 supports an exclusive high of `2n ** 64n`. Int64 and uint64 scalar results are bigint.

Parameter broadcasting, `out`, other distributions, other bit generators, optional entropy seeding, state import/export and SeedSequence spawning remain unsupported. Exact numerical acceptance on other platforms remains pending; this is not full NumPy random compatibility.

### Product and index reductions

`prod` multiplies over all axes or one axis and widens small integers like `sum`. `argmin` and `argmax` return `int64` positions: a flat C-order position without an axis, or a position inside the selected axis. Rank-0 index results are `bigint`.

```ts
const a = ctx.array(
  [
    [3, 1],
    [2, 4],
  ],
  { dtype: "float64" },
);
console.log(await a.prod().compute()); // 24
console.log((await a.argmin(0).compute()).toArray()); // [1n, 0n]
console.log((await a.argmax(1).compute()).toArray()); // [0n, 1n]
```

Equal values keep their first position, and NaNs are sticky: the first NaN position wins. Complex values compare by real component, then imaginary component. A zero-length reduced axis fails with `NMATH_SHAPE_MISMATCH`, except for `prod`, which returns the multiplicative identity. String and temporal arrays reject products and index reductions at expression construction. GPU execution rejects these reductions before admission. `std`, `var`, cumulative scans and multi-axis lists remain targets.

### Linear algebra and FFT

The NumPy entry point provides lazy `linalg.solve(a, b)`, `linalg.inv(a)`, and `linalg.det(a)` for square matrices, including broadcast batches. Only a one-dimensional solve right-hand side is a vector. Float16 linear algebra is unsupported; cast explicitly. Singular solve/inverse rejects with `NMATH_LINALG_SINGULAR`.

`fft.fft(a, options)`, `fft.ifft`, `fft.rfft`, and `fft.irfft` accept `{ n?, axis?, norm? }`. The default axis is `-1`; normalization is `"backward"`, `"forward"`, or `"ortho"`. Length crops or zero-pads input. `rfft` requires real input. `irfft` defaults to `2 * (inputLength - 1)`; pass `n` to recover an odd original length. Call `.compute()` to execute on CPU. GPU execution and `out` buffers are unsupported. Complex output uses `Complex` values and interleaved typed arrays. Other linear-algebra and FFT functions remain compatibility targets.

### Strings and exact temporal values

```ts
import { Temporal } from '@delinoio/nmath';

const words = context.array(['hello', '한글', '😀'], { dtype: 'str' });
const greetings = await words.add('!').compute();
console.log(greetings.toArray());

const dates = context.array(new BigInt64Array([-1n, 9007199254740993n]), {
  dtype: 'datetime64',
  unit: 'ns',
});
const first = dates.item(0); // Temporal<'datetime64'>
console.log(first.ticks, first.unit); // -1n, 'ns'
console.log(first.toDate()); // explicit conversion to millisecond precision

const duration = new Temporal(1n, 'ns', 'timedelta64');
const shifted = await dates.add(duration).compute();
```

Strings use JS strings. Width counts Unicode scalar values; you can select it with `stringWidth`, or let creation infer it. Creation and assignment reject truncation. Use `astype('str', {stringWidth})` when truncation is intentional. Strings export through `item()` and `toArray()`.

Temporal values preserve exact bigint ticks, units and NaT. Their unit defaults to `ns` unless selected explicitly. Use `toTypedArray()` to copy raw ticks, and retain the array's `unit` and `dtype` when transferring them. `toDate()` is explicit, supports datetime values only, and reduces precision to milliseconds. Implicit number conversion is rejected. NPY/NPZ files preserve string widths and temporal metadata.

Current string operations include concatenation, indexing, copies and explicit width casts. Temporal operations include date/duration arithmetic, compatible-unit conversion and supported reductions. Unsupported operators and casts report errors. Byte-string and variable-width string dtypes, string processing APIs, date parsing/formatting, business calendars and richer time-series/table operations remain compatibility work; they are not new exclusions.

### Batched matrix products

`numpy.matmul(a, b)` and `a.matmul(b)` accept numeric or boolean lazy arrays with rank at least one. The last two axes are matrix axes; leading axes broadcast. A vector on either side is temporarily promoted and its added axis is removed from the result. Two vectors produce a scalar. Static shapes retain known batch and matrix dimensions; incompatible static dimensions are type errors. Dynamic dimensions are checked when constructing the expression.

```ts
import { matmul } from "@delinoio/nmath/numpy";

const a = ctx.full([2, 1, 3, 4] as const, 1, "float32");
const b = ctx.full([5, 4, 6] as const, 1, "float32");
const result = await matmul(a, b).compute(); // shape [2, 5, 3, 6]
result.dispose();
```

Matrix products execute through the context scheduler. CPU supports all current numeric and boolean dtypes; GPU retains its float32/int32/uint32 and platform restrictions. An empty contraction yields zeros. Incompatible batch axes fail with `NMATH_SHAPE_BROADCAST`; scalars or incompatible contraction axes fail with `NMATH_SHAPE_MISMATCH`. Ufunc keyword options, including `out`, remain compatibility targets.

### Reshape and transpose

Shape transformations create lazy expressions and run through `compute()`:

```ts
import { reshape, transpose } from "@delinoio/nmath/numpy";

const a = ctx.array([1, 2, 3, 4, 5, 6], { dtype: "float64", shape: [2, 3] });
const transposed = await transpose(a).compute(); // shape [3, 2], shared view
const flat = await reshape(a, [-1]).compute(); // shape [6], shared view
const copied = await a.transpose().reshape([6], { copy: true }).compute();
// copied values: [1, 4, 2, 5, 3, 6]

transposed.dispose();
flat.dispose();
copied.dispose();
a.dispose();
```

`transpose(axes?)` reverses all axes by default. An explicit list must name each axis once; negative axes count from the end. `reshape(shape, {order?: "C", copy?: boolean | null})` preserves the logical C-order element sequence and supports one inferred `-1` dimension. C is the supported traversal order. Other orders and unknown options fail explicitly. The element count must match; `[0, -1]` is ambiguous and fails, while reshaping an empty input to `[-1]` produces `[0]`.

With omitted or null `copy`, compatible layouts share storage. This includes regular stepped and reversed views. `copy: true` produces independent storage; `copy: false` throws if a copy would be required. Shared results reflect later writes to the source and remain usable after the source handle is disposed. Computed operands and copied results use execution-start snapshots. These shape transforms currently execute on CPU; a GPU request rejects before queue admission.

Literal TypeScript shapes infer permutations and inferred reshape dimensions and reject incompatible counts or repeated/out-of-range axes. Dynamic inputs retain known dtype, dimensions and rank and validate unknown sizes at runtime.

## Evaluate polynomials

```ts
import { polynomial, polyval } from "@delinoio/nmath/numpy";

const coefficients = ctx.array([1, 2, 3], { dtype: "float32" });
const points = ctx.array([0, 1, 2], { dtype: "float32" });
const ascending = await polynomial.polynomial.polyval(points, coefficients).compute();
// [1, 6, 17]: 1 + 2*x + 3*x**2
const descending = await polyval(coefficients, points).compute();
// [3, 6, 11]: x**2 + 2*x + 3
ascending.dispose();
descending.dispose();
```

Coefficients occupy the first axis. Ascending evaluation defaults to `{ tensor: true }`: it evaluates every polynomial against every point and returns `coefficients.shape.slice(1)` followed by `points.shape`. Pass `{ tensor: false }` to broadcast those shapes instead. Descending evaluation always broadcasts. Both APIs accept arrays and lazy expressions from one context, run on CPU, and take fresh input snapshots at actual execution start.

Ascending evaluation converts integer and boolean coefficients to float64. Descending evaluation preserves ordinary array promotion, including exact bigint transport and integer wrapping. Empty descending coefficients return zeros with the points' shape and dtype; empty ascending coefficients fail. Wrap scalar inputs as rank-zero context arrays; descending coefficients require at least one axis. Results use independent storage. GPU requests reject without fallback. Polynomial classes, fitting, roots, other bases and coefficient arithmetic remain planned support targets.

## Differentiate polynomials

```ts
import { polynomial, polyder } from "@delinoio/nmath/numpy";

const coefficients = ctx.array([1, 2, 3], { dtype: "float32" });
const ascending = await polynomial.polynomial.polyder(coefficients).compute();
// [2, 6]: derivative of 1 + 2*x + 3*x**2, dtype float32
const descending = await polyder(coefficients).compute();
// [2, 2]: derivative of x**2 + 2*x + 3, dtype float64
ascending.dispose();
descending.dispose();
```

Ascending derivatives accept `{ m, axis, scl }`, defaulting to order one, axis zero and scale one. Negative axes count from the end. Every derivative stage multiplies by `scl` before multiplying by the degree and preserves the coefficient precision. Boolean and integer coefficients convert to float64. Scalar coefficients become length-one arrays. Empty axes stay empty. Orders at least as large as a nonempty axis return one coefficient multiplied by zero, so an infinite constant produces NaN. Complex scaling requires complex coefficients for active differentiation.

Descending `polyder(coefficients, m)` accepts rank-one coefficients. Positive orders promote against int64 degree factors, so float32 becomes float64 and complex64 becomes complex128. The result is empty when the order reaches the coefficient count. Order zero preserves the original shared storage; ascending order zero returns an independent copy. Both APIs require nonnegative safe integer orders, run lazily on CPU, and support cancellation and context memory limits. Unknown options and invalid axes fail. Derivative nodes reject GPU requests; the zero-order legacy identity retains the original expression and its device support. Array-valued scales, multidimensional legacy derivatives, integration and polynomial classes remain support targets.
