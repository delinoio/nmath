# NMath

Native numerical computing for Node.js 24 without Python. NMath runs heavy computation in Rust on native worker threads and exposes JavaScript and TypeScript APIs for arrays, lazy expressions and tabular data.

The `@delinoio/nmath` ESM package provides a core API and NumPy-, pandas- and Polars-style entry points. The `nmath-compiler` CLI adds arithmetic operators and indexing on NMath values in `.nts` files. A VS Code extension provides `.nts` diagnostics, completion and go-to-definition.

NMath is a 0.x prerelease with a partial API. Read the [package guide and compatibility inventory](packages/nmath/README.md) for supported operations and limitations. GPU execution is not yet available. Platform targets do not establish tested binary availability; source builds require Rust.

## Quick start

After [building from source](#build-from-source), run this ESM example in a project that resolves `@delinoio/nmath`:

```js
import { createContext } from "@delinoio/nmath";

const ctx = createContext({
  maxConcurrentJobs: 4,
  maxQueueLength: 64,
  cpuMemoryBudget: 1n << 30n,
  gpuMemoryBudget: 0n,
});

try {
  const a = ctx.array([1, 2, 3], { dtype: "float64" });
  const result = await a.mul(2).add(1).compute();
  console.log(result.toArray()); // [3, 5, 7]
} finally {
  await ctx.close();
}
```

Each context has explicit concurrency, queue and memory limits. Operations build lazy expressions; `compute()` returns a Promise. Python libraries are compatibility references, not runtime dependencies.

## Build from source

Use Node.js 24 from `.nvmrc`, pnpm 10.26.2 and the Rust toolchain in `rust-toolchain`.

```sh
git clone https://github.com/delinoio/nmath.git
cd "nmath"
pnpm install --frozen-lockfile
pnpm build
pnpm exec turbo run build:compiler --filter=@delinoio/nmath
```

`pnpm build` builds the native Node.js addon, the ESM package and the VS Code extension. The separate `build:compiler` task builds and copies the compiler executable for the current platform. Build output does not install or publish the npm package or VS Code extension.

See the [development workflow](docs/repository-workflow-contract.md) for task dependencies and generated-output cleanup.

## Validation

Run from the repository root:

```sh
cargo test
pnpm test
pnpm typecheck
pnpm format:check
```

`cargo test` covers the Rust workspace. `pnpm test` builds the required native and TypeScript outputs, then runs package, extension and shared tooling tests. The shared CI workflow selects formatting and tooling-related validation; it does not establish native platform, GPU or release acceptance. See the [workflow contract](docs/repository-workflow-contract.md#validation) for details.

## Documentation

- [Package guide](packages/nmath/README.md): API examples, execution model, compatibility inventory and compiler usage.
- [Documentation catalog](docs/README.md): repository contracts and NMath domain contracts.
- [NMath project index](docs/project-nmath.md): component ownership, canonical paths and integration boundaries.
- [Repository instructions](AGENTS.md): contribution and development rules. New projects still require a project index and domain contracts before implementation.

## License

Repository-owned content is licensed under [Apache-2.0](LICENSE). Preserve the applicable [notices](NOTICE) and original licenses of imported skills and native dependencies.
