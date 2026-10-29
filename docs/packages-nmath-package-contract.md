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

The CLI executable is `nmath-compiler`. No separate compiler package is published.

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

### Expressions and compute

- Functions such as `add(a, b)` and methods such as `a.add(b)` return `Expr` without computing.
- `expr.compute({ device?: 'cpu' | 'gpu', signal?: AbortSignal }): Promise<Result>` runs one independent job.
- Results with rank 0 are JS scalars: `number` for floating and 32-bit-or-smaller integer dtypes, `bigint` for 64-bit integers, `boolean` and `string`. Complex values use the lossless `Complex` class. Other results are `NMathArray` values.
- Explicit lossy conversions: `toNumber()`, `toDate()`.

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

- `pnpm --filter @delinoio/nmath build` compiles TypeScript to `dist/`. `build:native` builds `nmath-node` with the napi-rs CLI.
- `pnpm --filter @delinoio/nmath test` runs Node.js tests against the built native binary.
- `dist/` is ignored generated output and is removed after validation.
- Platform acceptance for the eight CPU binary combinations and the GPU matrix is recorded in pull requests, issues and CI artifacts, not in this document.

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
