# Repository Workflow Contract

## Scope

The repository contains the [NMath project](project-nmath.md), its Rust workspace, the `@delinoio/nmath` npm package and a VS Code extension, alongside shared instructions, skills, documentation templates and formatting infrastructure. The NMath project and distribution contracts own runtime build and release preparation boundaries. Release credentials remain outside source control.

## Installation

Use Node.js 24 from `.nvmrc` and pnpm 10.26.2 from `package.json`. Run `pnpm install --frozen-lockfile`. The prepare script installs Lefthook through `scripts/install-git-hooks.sh`. Archives without Git metadata skip hook installation. Linked worktrees may use their shared Git hooks directory; unrelated custom hook paths remain protected.

## Task orchestration

The pinned `turbo` dependency and root `turbo.json` orchestrate packages selected by `pnpm-workspace.yaml`: `apps/*`, `packages/*` and `servers/*`. Commands run the tasks declared by the active workspace packages. Rust crates are explicit members of the root Cargo workspace, not pnpm packages.

Root `pnpm build`, `pnpm dev`, `pnpm lint` and `pnpm typecheck` delegate to `turbo run`. Root `pnpm test` runs workspace tests and the repository-wide `test:tooling` task. The latter runs the existing formatter and hook fixtures directly and is registered as `//#test:tooling`. It is uncached so each invocation checks the current Git and hook environment. `prepare`, `format` and `format:check` remain direct commands: hook installation has side effects, and formatting depends on Git revisions, staged changes and untracked files rather than package inputs.

Root `pnpm dev` starts all workspace development servers together through the unfiltered `turbo run dev` command. Each workspace package that owns a development server exposes it through its `dev` script. Packages without a development server do not need a `dev` script. Package-specific development is available through an explicit filtered command; the root command always selects all packages. The NMath package and VS Code extension currently expose no `dev` scripts, so there are no development servers to start.

Define each workspace task in its owning `package.json`. Declare workspace dependencies so Turbo can order tasks: builds depend on dependency builds, tests depend on dependency builds, and lint and typecheck tasks depend on their corresponding dependency tasks. Build output defaults to `dist/**`; packages with other outputs must override them in a package-local `turbo.json` that extends `//`. Declare any additional test reports or incremental compiler outputs there. Development tasks are persistent and uncached.

Turbo uses strict environment mode by default. Package contracts and package-local task configuration must declare environment variables, ignored configuration inputs and shared root inputs that affect results before enabling their caching. Keep side-effecting or environment-dependent tasks uncached when their inputs cannot be fully declared. Do not route Git-sensitive formatting through cached tasks or pass credentials through global environment wildcards. Remote caching is not configured. `.turbo/` is ignored local state; generated `dist` directories remain subject to the cleanup rule below.

For a package subset, use `pnpm exec turbo run <task> --filter=<package>`. Use `pnpm exec turbo run <task> --dry=json` to inspect the graph without running tasks. Root-only fixture execution is also available as `pnpm test:tooling`. New runtime projects must add their own contracts and package scripts before these commands can establish runtime acceptance.

### NMath build and test tasks

Install the Rust toolchain selected by `rust-toolchain` as well as Node.js and pnpm. `pnpm build` runs the package and extension builds. The package build first runs its uncached `build:native` task through napi-rs, then emits TypeScript output to `packages/nmath/dist/`. The extension emits to `apps/nmath-vscode/dist/`.

Run `pnpm exec turbo run build:compiler --filter=@delinoio/nmath` to build the Rust compiler in release mode and copy its executable to `packages/nmath/native/`. This task depends on the package build because its copy script imports the generated compiler-path helper. Native addon and compiler binaries are local generated output.

`pnpm test` includes package runtime and compiler integration tests, extension tests and `test:tooling`. Package tests depend on `build` and `build:compiler` and are uncached; extension tests depend on their build. `pnpm typecheck` runs package declarations, consumer type tests and extension type checks; the package task also requires the native build. `cargo test` from the root independently runs the Rust workspace tests.

Use the root Turbo task runner for package subsets, for example `pnpm exec turbo run test --filter=@delinoio/nmath` or `pnpm exec turbo run test --filter=nmath-vscode`. Direct package test scripts do not build their prerequisites. See the [runtime](crates-nmath-runtime-contract.md), [compiler](crates-nmath-compiler-contract.md), [package](packages-nmath-package-contract.md) and [extension](apps-nmath-vscode-contract.md) contracts for component coverage.

## Formatting

Root `pnpm format` writes fixes and `pnpm format:check` checks without rewriting. Both use pinned Rstack and `rstack.config.ts`. Default local selection compares the merge base of `origin/main` and HEAD, including working changes and untracked files. Explicit `--base <sha> --head <sha>` arguments select that committed comparison. Local `--all` selects existing tracked files. Before the first commit, use `--staged`.

Lefthook runs `node scripts/ci/format-changed.mjs --staged --write` with `stage_fixed: true`. It preserves unstaged edits, supports initial commits and rejects parser failures. Select files with NUL-separated Git paths, skip deletions and symlinks, and preserve imported skills, generated files and lockfiles through shared exclusions.

CI requires an explicit comparison and rejects local `--staged` and `--all` modes. An unset, empty or case-insensitive `false` CI value means local execution; every other nonempty value enables CI restrictions.

## Validation

Run `pnpm test:tooling` for formatter, hook and change-planning fixtures without native builds. Root `pnpm test` also runs the NMath workspace tests described above. CI plans affected fixtures and `pnpm format:check --base <sha> --head <sha>` on Ubuntu for pull requests, main pushes and manual dispatch. PR comparisons use the pull request base and head. Push comparisons use the prior and current commits. Manual comparisons use the merge base with `origin/main`; an optional `base` input selects another existing revision. A root push has no previous commit, so CI creates a local empty-tree comparison commit without pushing it. No full manual mode or macOS runners are configured.

The `changes` job resolves that comparison and runs `scripts/ci/plan.mjs` without installing dependencies. It publishes `base`, `head`, `format`, `fixtures` and `quality` outputs. Formatting uses the existing formatter's changed-file selection; Rstack still owns exclusions. Fixtures run for changes, deletions or renames affecting `scripts/ci/`, `scripts/install-git-hooks.sh`, `.github/workflows/CI.yml`, `.nvmrc`, `package.json`, `pnpm-lock.yaml`, `pnpm-workspace.yaml`, `rstack.config.ts` or `lefthook.yml`. Documentation-only changes select formatting without fixtures. The `quality` job runs when either check is selected and passes the exact comparison to formatting. Empty comparisons or changes with no validation inputs skip `quality`. The planner records the checkout revision, comparison and selections in logs and the job summary. New shared validation dependencies must update the planner inputs.

Each CI run records the validation revision, comparison revisions, commands and results in its logs and job summary. The CI workflow currently plans only formatting and shared-tooling inputs; it has no dedicated Rust, package, compiler or extension validation jobs. Its fixtures step invokes root `pnpm test`, which now includes NMath tasks and native build prerequisites, rather than only shared fixtures. Changes limited to runtime source do not select a runtime test job through this planner. Passing formatting or shared fixtures does not establish runtime, native platform, GPU, editor accessibility or release acceptance. Record actual component and platform validation separately under the owning contracts.

The `ci-result` job aggregates validation through `needs`, including `changes` and `quality`. It runs with `always()` so dependency failures do not skip the aggregate check. Failed or cancelled dependencies fail `ci-result`. The `changes` job must succeed; a skipped `quality` job passes only when the planner selected no validation. Successful dependencies and other skipped dependencies pass. Every new PR validation job must be included in `ci-result.needs`. The aggregate job runs on Ubuntu for the same PR, main-push and manual-dispatch events.

## NMath distribution validation

The [NMath distribution contract](packages-nmath-distribution-contract.md) owns native tarball staging, VSIX packaging and evidence validation. The affected planner also selects fixtures for NMath package and extension paths, Rust crate paths, Cargo manifests/lockfile, the Rust toolchain selector, packaging targets and NMath release workflows. Affected quality validation runs an installed-package smoke through root Turbo in addition to existing checks.

Dedicated prerelease artifact builds use the eight labeled native release environments; they are separate from PRs, main pushes and affected manual CI. The manual distribution evidence gate runs on Ubuntu against the selected source revision and actual tested artifact hashes. Neither workflow publishes packages or substitutes build results for native/GPU/editor acceptance.

## Assets and licensing

Every repository-owned `dist` directory is ignored generated output and must be removed after validation. Assets at least 512 KiB require exact-path Git LFS attributes and hydration before use. Hydrate any LFS-managed assets before builds, tests or packaging that use them.

## Dependency updates

Renovate extends `github>delinoio/renovate-config`. Ordinary updates and lockfile maintenance use Monday 00:00–04:00 in `Asia/Seoul`; vulnerability updates retain the inherited immediate security policy. Follow the dependency security contract and retain unresolved findings in GitHub or CI records.
