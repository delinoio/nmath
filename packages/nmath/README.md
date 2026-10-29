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

As operands, a JS `number` acts like a Python `float`, a `bigint` like a Python `int` and a `boolean` like a Python `bool` (NumPy NEP 50 rules). For example, `int32Array.add(1n)` stays `int32`, and `int32Array.add(1)` becomes `float64`.

## Types

`NMathArray<D, S>` and `Expr<D, S>` carry the dtype `D` and shape `S`. TypeScript checks dtype promotion, broadcasting, reductions, `matmul` and indexing when shapes are literal tuples:

```ts
const m = ctx.zeros([2, 3], "float32");
m.add(ctx.zeros([3], "int32")); // Expr<"float64", [2, 3]>
m.add(ctx.zeros([4], "float32")); // type error: shapes do not broadcast
```

A dimension typed as `number` is dynamic. The runtime validates it, and the dtype and rank stay known.

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

Arithmetic operators `+`, `-`, `*`, `/`, `%`, `**`, unary `+`/`-` and bracket reads lower to lazy NMath operations. Bracket writes require an async function body or module top level. Comparisons, bitwise operators and compound assignments are not NMath operator syntax. Project references and `-b` builds are not supported. Emitted source maps refer to lowered TypeScript; diagnostics refer to original `.nts` positions.

The NMath VS Code extension associates `.nts` files and starts the compiler language server in trusted workspaces. Set `nmath.compilerPath` to a compiler executable, or let the extension find `node_modules/.bin/nmath-compiler` in the workspace folder and then `nmath-compiler` on `PATH`. Use **NMath: Restart Language Server** after changing the executable. Server output appears in **NMath Language Server**.

## GPU

GPU execution is limited to `float32`, `int32` and `uint32` elementwise arithmetic, broadcasting, `sum`, `mean`, `min`, `max` and `matmul` on macOS arm64 (Metal), Linux x64 glibc (Vulkan) and Windows x64 (DX12). NMath never falls back to the CPU, never lowers precision and never retries. Unsupported dtypes or operations reject with `NMATH_GPU_UNSUPPORTED_OPERATION` before the job is queued.

> The GPU backend is not in this prerelease. GPU requests in the supported subset reject with `NMATH_GPU_UNAVAILABLE`. NMath makes no official GPU support claim until tested hardware combinations are published.

## Files

`array.save(path, { overwrite })` writes NPY files and `ctx.load(path)` reads them. NMath never replaces an existing file unless you pass `overwrite: true`. Output goes to a temporary file that is renamed into place, so cancellation or failure leaves the original file unchanged. Keep input files unchanged while a job reads them; NMath reports changes it detects with `NMATH_IO_MODIFIED`, but it cannot detect every change.

## Errors

Every failure is an `NMathError` with a stable `code` (for example `NMATH_SCHED_QUEUE_FULL`) and an English message with recovery guidance. Error messages and measurements never include input data, file paths or file contents. NMath sends no telemetry. `ctx.metrics()` returns per-execution status and timing and memory use, so you can collect measurements yourself. Set `NMATH_LOG=debug` to write structured native logs to stderr (`NO_COLOR` disables color).

## Compatibility inventory

Reference versions: NumPy 2.5.4, pandas 3.1.0 and Polars 2.1.0. This prerelease covers a small part of each library. Every API not listed as available is a planned support target unless a published exclusion says otherwise.

| Area                 | Available now                                                                                                   | Not yet available                                                                                                                             |
| -------------------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Dtypes               | `bool`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float32`, `float64`          | complex, string, datetime, timedelta, categorical, structured                                                                                 |
| NumPy arithmetic     | `+ - * / % **`, unary `+ -`, broadcasting, NEP 50 promotion                                                     | comparisons, bitwise, ufunc methods, `where`, math functions                                                                                  |
| NumPy reductions     | `sum`, `mean`, `min`, `max` (all axes or one axis)                                                              | `prod`, `std`, `var`, `argmin`, `argmax`, `cumsum`, multi-axis lists, `keepdims`                                                              |
| NumPy indexing       | integers, slices, one integer-array or mask selector per operation, views for basic indexing, atomic assignment | several advanced selectors combined, `newaxis`, ellipsis                                                                                      |
| NumPy linear algebra | `matmul` for 1-D and 2-D                                                                                        | batched `matmul`, `linalg`, `fft`, polynomials, masked arrays                                                                                 |
| NumPy random         | `default_rng(seed).random(shape)` with NumPy-identical PCG64 streams                                            | other distributions and bit generators                                                                                                        |
| NumPy I/O            | NPY (C order, little endian)                                                                                    | NPZ, Fortran order                                                                                                                            |
| pandas               | `Series` arithmetic and reductions, `iloc`, `DataFrame` construction, `get`, `assign`, `select`, `sum`          | indexes and labels, missing values, joins, group-by, pivots, windows, time series, string and categorical methods, CSV/JSON/Parquet/Arrow I/O |
| Polars               | `Series`, `DataFrame`, `col` arithmetic, `alias`, `LazyFrame.select`, `withColumns`, `collect`                  | filters, joins, group-by, windows, streaming execution, string and temporal namespaces, file I/O                                              |
| `.nts` compiler      | `nmath-compiler check`, `build`, `lsp`, operator and indexing transforms, project APIs, VS Code integration     | project references, `-b` builds, source maps to original `.nts` text                                                                          |
| GPU                  | —                                                                                                               | GPU backend                                                                                                                                   |

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
