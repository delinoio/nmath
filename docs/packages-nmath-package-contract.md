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

Array creation helpers (`ctx.array`, `ctx.full`, `ctx.zeros` and the NumPy `ones`, `full` and `arange` forms) check the CPU budget before they build JS or native input buffers and fail with `NMATH_MEMORY_BUDGET_EXCEEDED` when the input cannot fit. The context still makes the authoritative reservation when it creates the array.

### Expressions and compute

- Functions such as `add(a, b)` and methods such as `a.add(b)` return `Expr` without computing.
- `expr.compute({ device?: 'cpu' | 'gpu', signal?: AbortSignal }): Promise<Result>` runs one independent job.
- Results with rank 0 are JS scalars: `number` for floating and 32-bit-or-smaller integer dtypes, `bigint` for 64-bit integers, `boolean` and `string`. Complex values use the lossless `Complex` class. Other results are `NMathArray` values.
- Explicit lossy conversions: `toNumber()`, `toDate()`.
- Weak operand scalars follow NumPy NEP 50 with a fixed JS mapping: `number` acts as a Python `float`, `bigint` as a Python `int` and `boolean` as a Python `bool`. `Complex` acts as a Python `complex` weak operand.

Small integer arrays exchange data through `Int8Array`, `Int16Array`, `Uint8Array` and `Uint16Array`. Scalar reads return `number`; signed and unsigned small-integer sums return `bigint`. Array-to-array promotion follows NumPy, including `int8` with `uint8` becoming `int16`, and `float32` with any 8-bit or 16-bit integer remaining `float32`.

`new Complex(real, imag = 0)` is the dedicated immutable complex scalar. `real` and `imag` are JS numbers and preserve both IEEE-754 components, including signed zeros and non-finite values. `item()`, nested `toArray()` output and rank-0 `compute()` results return `Complex` for complex dtypes. Implicit numeric conversion throws; `Complex.toNumber()` explicitly returns the real component and discards the imaginary component.

Complex array creation accepts nested `Complex` values or interleaved typed-array components in real/imaginary order. `complex64` requires `Float32Array`; `complex128` requires `Float64Array`. Typed input must contain an even number of components. Shape counts complex elements, not components. `toTypedArray()` returns an independent interleaved copy with the corresponding floating type. `full()` takes a `Complex` scalar and `zeros()` creates complex zero values.

Complex arithmetic, unary operations, reductions, indexing, writes, copies, explicit casts and 1-D/2-D matrix multiplication follow the runtime contract. Complex remainder is rejected by ordinary TypeScript declarations and by runtime expression validation. Complex GPU computation remains outside the GPU dtype subset.

### Typing

`NMathArray<D extends DType, S extends Shape>` and `Expr<D, S>` carry dtype and shape type parameters. `Shape` is a readonly tuple of `number` literals or `number` for a dynamic dimension. Broadcasting is computed at the type level for tuple shapes. Incompatible static shapes produce a type error; dynamic dimensions fall back to runtime validation without erasing dtype or rank.

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
