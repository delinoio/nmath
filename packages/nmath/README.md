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

## Entry points

| Import                   | Purpose                                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `@delinoio/nmath`        | Contexts, arrays, expressions, selectors, errors.                                                                    |
| `@delinoio/nmath/numpy`  | NumPy-style functions (`array`, `zeros`, `ones`, `arange`, `add`, `matmul`, `random.defaultRng`, `save`, `load`, …). |
| `@delinoio/nmath/pandas` | pandas-style `Series` and `DataFrame` (subset).                                                                      |
| `@delinoio/nmath/polars` | Polars-style `Series`, `DataFrame`, `col` and `LazyFrame` (subset).                                                  |

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

| Area                 | Available now                                                                                                                     | Not yet available                                                                                                                             |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Dtypes               | `bool`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float32`, `float64`, `complex64`, `complex128` | float16, extended precision, string, datetime, timedelta, categorical, structured                                                             |
| NumPy arithmetic     | `+ - * / **`, unary `+ -`, broadcasting, NEP 50 promotion; `%` for supported real dtypes                                          | comparisons, bitwise, ufunc methods, `where`, math functions                                                                                  |
| NumPy reductions     | `sum`, `mean`, `min`, `max` (all axes or one axis)                                                                                | `prod`, `std`, `var`, `argmin`, `argmax`, `cumsum`, multi-axis lists, `keepdims`                                                              |
| NumPy indexing       | integers, slices, one integer-array or mask selector per operation, views for basic indexing, atomic assignment                   | several advanced selectors combined, `newaxis`, ellipsis                                                                                      |
| NumPy linear algebra | `matmul` for 1-D and 2-D                                                                                                          | batched `matmul`, `linalg`, `fft`, polynomials, masked arrays                                                                                 |
| NumPy random         | `default_rng(seed).random(shape)` with NumPy-identical PCG64 streams                                                              | other distributions and bit generators                                                                                                        |
| NumPy I/O            | NPY (C order, little endian; complex input also supports big endian and Fortran order)                                            | NPZ; Fortran-order input for non-complex dtypes                                                                                               |
| pandas               | `Series` arithmetic and reductions, `iloc`, `DataFrame` construction, `get`, `assign`, `select`, `sum`                            | indexes and labels, missing values, joins, group-by, pivots, windows, time series, string and categorical methods, CSV/JSON/Parquet/Arrow I/O |
| Polars               | `Series`, `DataFrame`, `col` arithmetic, `alias`, `LazyFrame.select`, `withColumns`, `collect`                                    | filters, joins, group-by, windows, streaming execution, string and temporal namespaces, file I/O                                              |
| `.nts` compiler      | Project checking, ESM/declarations/original-source maps, semantic LSP                                                             | Marketplace publication and editor acceptance                                                                                                 |
| GPU                  | Native backend with strict device and precision rejection                                                                         | Complete numerical and hardware acceptance                                                                                                    |

## Build from source

Install Rust (the toolchain in `rust-toolchain`) and pnpm 10, then run:

```sh
pnpm install
pnpm --filter @delinoio/nmath build:native
pnpm --filter @delinoio/nmath build
pnpm --filter @delinoio/nmath build:compiler
```

`build:native` and `build:compiler` write the binaries for your platform to `native/`.

## Support

Report problems in [GitHub Issues](https://github.com/delinoio/nmath/issues). There is no guaranteed response time.

## License

Apache-2.0. See `LICENSE` and `NOTICE`. Native dependencies keep their own licenses.

## Numeric tables

The numeric table APIs are a bounded native subset of pandas 3.1.0 and Polars 2.1.0. They do not implement the complete pandas or Polars API. Columns can use bool, signed/unsigned 8-, 16-, 32- or 64-bit integers, float32 and float64. Nulls use explicit boolean validity columns; true means valid. String, time, categorical, complex and extended-precision families remain support targets. NaN/inf and floating-point summation behavior still require broader original-derived numerical acceptance; explicit-null integer fixtures do not establish complete missing-value parity.

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

Supported table operations: native multi-column compute, boolean-column filtering, stable single-key sorting, single-key inner/left equijoins, sum/mean/min/max/count grouping, and basic materialized lazy plans. pandas preserves explicit labels and matches null join keys; Polars uses positional rows and requires `nullsEqual:true` to match null join keys. Full label alignment, nullable expression arithmetic, arbitrary post-row projections, multiple keys, additional join modes, windows, pivots, strings, temporal/categorical operations, query optimization and internal streaming remain targets.

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

NPZ supports stored and deflated numeric/boolean NPY members, including complex64 and complex128. Complex input supports both byte orders and C/Fortran layout; output is little-endian C order. Pickle/object data, string/date/structured/extended dtypes, unsupported noncomplex byte orders/layouts, encrypted members and other compression methods are unsupported. Names must be nonempty UTF-8 components other than `.` or `..`, without slashes, backslashes, colons or control characters, and at most 65,531 UTF-8 bytes. Duplicate names and non-NPY members are rejected. ZIP64 central directories, multi-disk archives, 65,535 or more members and output capacity bounds at or above 4 GiB are unsupported. Conservative memory and output reservations can reject an archive earlier. These limits do not establish full NumPy compatibility.
