# NMath Project Index

## Goal

NMath is a Python-free native computation library for Node.js 24. It serves three workflows:

- JavaScript and TypeScript developers who run large matrix calculations in Node.js.
- TypeScript developers who analyze tabular data.
- Developers who write numerical algorithms in `.nts` files with arithmetic operators on NMath values.

Heavy computation runs in native Rust code outside the calling Node.js thread. Standard JS/TS APIs and `.nts` syntax build the same lazy expressions and execute through the same asynchronous `compute()` contract. NumPy 2.5.4, pandas 3.1.0 and Polars 2.1.0 are fixed behavioral references for compatibility entry points. They are test oracles only and are never runtime dependencies.

## Project ID

`nmath`

Component IDs:

| Component ID     | Kind                          | Summary                                                                                                                          |
| ---------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `nmath-core`     | Rust library crate            | Arrays, dtypes, lazy expressions, kernels, contexts, scheduling, memory accounting, file I/O and random state.                   |
| `nmath-node`     | Rust napi-rs `cdylib` crate   | Node.js binding over `nmath-core`. Runs `compute()` jobs on native worker threads.                                               |
| `nmath-nts`      | Rust library crate            | `.nts` content-mapping transform: operator and indexing rewrites with a position map back to the original source.                |
| `nmath-compiler` | Rust binary and library crate | `nmath-compiler check`, `build`, `lsp` and the content-mapper process that integrates `nmath-nts` with the pinned tsrs compiler. |
| `nmath-package`  | npm package `@delinoio/nmath` | Single public ESM package with the core, `numpy`, `pandas`, `polars` and `compiler` entry points.                                |
| `nmath-vscode`   | VS Code extension             | Associates `.nts` files and starts `nmath-compiler lsp`.                                                                         |

## Domain Ownership Map

| Domain     | Canonical path           | Component ID     | Status  |
| ---------- | ------------------------ | ---------------- | ------- |
| `crates`   | `crates/nmath-core/`     | `nmath-core`     | active  |
| `crates`   | `crates/nmath-node/`     | `nmath-node`     | active  |
| `crates`   | `crates/nmath-nts/`      | `nmath-nts`      | planned |
| `crates`   | `crates/nmath-compiler/` | `nmath-compiler` | planned |
| `packages` | `packages/nmath/`        | `nmath-package`  | active  |
| `apps`     | `apps/nmath-vscode/`     | `nmath-vscode`   | planned |

Paths marked `planned` do not exist yet. A path becomes a Cargo workspace member or pnpm workspace package only in the change that creates its skeleton. Root `Cargo.toml` lists every active crate explicitly.

## Domain Contract Documents

- [Runtime crates contract](crates-nmath-runtime-contract.md): `nmath-core` and `nmath-node`.
- [Compiler crates contract](crates-nmath-compiler-contract.md): `nmath-nts` and `nmath-compiler`.
- [npm package contract](packages-nmath-package-contract.md): `@delinoio/nmath` and its entry points.
- [VS Code extension contract](apps-nmath-vscode-contract.md): `nmath-vscode`.

## Cross-Domain Invariants

### Stable identifiers

- Device IDs: `cpu`, `gpu`. `cpu` is the default.
- Dtype IDs follow NumPy names: `bool`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float32`, `float64`, `complex64`, `complex128`, `str`, `datetime64`, `timedelta64`. Every layer uses these identifiers.
- Entry point IDs: `core`, `numpy`, `pandas`, `polars`, `compiler`.
- Error codes use the form `NMATH_<CATEGORY>_<NAME>` (for example `NMATH_SCHED_QUEUE_FULL`). Codes are stable public API. Rust, the binding, the package and the compiler share one registry, owned by the runtime crates contract.
- Execution status IDs: `queued`, `running`, `succeeded`, `failed`, `cancelled`.

### Execution model

- Every array and expression belongs to exactly one explicit execution context. No implicit default context exists. Direct combination of values from different contexts fails with `NMATH_CONTEXT_MISMATCH`. Explicit copies are charged to the target context.
- `expr.compute({ device, signal })` returns a Promise. Each call is an independent job without automatic result caching.
- A job freezes its in-memory inputs when it obtains an execution slot and starts. Changes made while the job is queued are visible to it.
- Jobs run up to the context concurrency limit and then queue in FIFO order. A full queue rejects new work with `NMATH_SCHED_QUEUE_FULL`.
- A required allocation that does not fit the CPU or GPU budget fails before allocation with `NMATH_MEMORY_BUDGET_EXCEEDED` and is not retried.
- Writes to the same storage serialize in request order and commit atomically.
- GPU execution never falls back to CPU, never lowers precision and never retries automatically.

### `.nts` and ordinary TypeScript

- `.nts` operator and indexing syntax lowers to the same public functions that ordinary `.ts` code calls. The transform does not extend the TypeScript parser or lexer.
- Ordinary `.ts` usage is validated with TypeScript 7.0.2. `.nts` is validated with the pinned tsrs integration. The two baselines are separate.

### Deviations from repository defaults

- Primary language: Rust for native runtime and compiler components, TypeScript for the npm package surface and the VS Code extension. The repository's Go default does not apply because the product is a napi-rs native addon and a tsrs-based compiler.
- Build tooling: `nmath-package` uses the TypeScript compiler and napi-rs tooling, not Rspack, because it publishes a Node.js library with native binaries and no web bundle. `nmath-vscode` uses the TypeScript compiler for the same reason.
- Storage: NMath reads and writes local files through the calling process's permissions. The repository's Cloudflare R2 default does not apply to a local library. Remote storage is out of scope.
- Identifiers: execution identifiers are process-local monotonically increasing integers, not UUID v7, because they are never persisted.
- `apps/nmath-vscode` is an editor extension, not a static site. The Rsbuild/Cloudflare Pages default does not apply.

## Change Policy

- Changes to dtype IDs, error codes, device IDs or execution semantics update the runtime crates contract and the npm package contract together.
- Changes to `.nts` syntax or transform output update the compiler crates contract and the npm package contract together.
- Changes to the language server command or protocol surface update the compiler crates contract and the VS Code extension contract together.
- Path or component changes update this index and `docs/README.md` in the same change.
- Update `AGENTS.md` only when development procedures, directory ownership or repository/domain development rules change; follow the [instruction-update policy](README.md#instruction-update-policy).
- Record implementation status, validation results, platform evidence and unresolved limits in pull requests, issues and CI logs or artifacts.

## References

- [Documentation catalog](README.md)
- [Repository defaults](repository-defaults.md)
- [Repository workflow](repository-workflow-contract.md)
- [Repository license](repository-license-contract.md)
- [Repository environment](repository-environment-contract.md)
- [Dependency security](repository-dependency-security-contract.md)
- [Project template](project-template.md)
