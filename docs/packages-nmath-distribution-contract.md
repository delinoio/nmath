# NMath Distribution Contract

## Ownership

The `nmath-package` component owns `packaging/nmath/`, the distribution scripts in `packages/nmath/scripts/`, and the release preparation workflow. The VS Code component owns its VSIX packaging task. These produce artifacts without publishing npm packages or Marketplace extensions.

## Packages and build boundaries

`packaging/nmath/targets.json` defines eight addon/compiler pairs: macOS x64 and arm64; Windows MSVC x64 and arm64; Linux x64 and arm64 with glibc or musl. Minimum targets are macOS 14, Windows 11, Linux kernel 6.1 with glibc 2.36 or musl 1.2.5. Vendor-supported OS versions are required. Build hosts must use compatible SDKs and libc versions; cross-compilation does not establish these minimums.

`distribution.mjs` builds both Rust artifacts with the locked dependency graph and an explicit Rust target. It stages platform packages and the public ESM package under ignored `packaging/nmath/dist/`. The public staged manifest has exact-version optional dependencies on all eight platform packages. The source manifest does not resolve unpublished platform versions during development. The public package also includes the gzip-compressed API ledger `compatibility.json.gz`; its content preserves explicit pending acceptance. Python generators and oracle environments are excluded. Platform manifests declare OS, CPU and Linux libc restrictions. No install scripts compile native code or download executables. The compiler remains part of the same distribution, not a separate public compiler package.

Run distribution tasks through root Turbo with `NMATH_TARGET` and `NMATH_LICENSE_BUNDLE`. The latter names a reviewed UTF-8 file containing complete third-party license terms and notices for the actual native dependency graph. Packaging fails without that file. Its existence is not a license-completeness audit; release evidence must record that review. Repository Apache-2.0 terms and notices are also included. Release evidence and generated binaries remain outside tracked source.

## Release environments and credentials

Release preparation uses dedicated self-hosted runners labeled `nmath-release` and the exact platform triple, behind the `nmath-release` environment. Only a prerelease event starts these builds. PRs, main pushes and affected manual validation retain non-macOS runners. Runners supply Node.js 24, pnpm, the pinned Rust toolchain, matching linker/SDK/libc and the reviewed license bundle. Linux musl addon builds require a dynamic musl toolchain; configure `RUSTFLAGS=-C target-feature=-crt-static`. Build each target on its native architecture for subsequent execution acceptance.

The workflow has read-only repository permissions and uploads CI artifacts. It has no npm or Marketplace publishing credentials. Publication remains a separate authorized operation after evidence review. Build artifacts are candidates, not an announcement of official CPU or GPU support.

## Release evidence gate

The gate validates an external JSON evidence document against an exact source SHA and the actual artifact bytes. Each CPU target needs actual installed-package execution, Node.js 24, hardware/OS/libc details and successful ESM, CommonJS dynamic import, CPU-without-GPU, compiler and native-load-error scenarios. Each GPU target needs actual hardware execution, exact OS/model/driver/backend, numerical comparison and tolerance evidence, cancellation, memory exhaustion, device loss and no-fallback checks. Required GPU targets are macOS arm64/Metal, Linux x64 glibc/Vulkan and Windows x64/DX12. The extension needs actual VSIX installation, activation, LSP behavior, keyboard and screen-reader evidence and Marketplace distribution verification.

Evidence includes commands, immutable result references, artifact SHA-256 digests, and successful scenario outcomes. Fixture, build-only, cross-compile and capability-only records are rejected. A schema validator cannot independently prove a submitted hardware observation; maintainers must review the linked records. Missing or failed evidence blocks full release. No fabricated positive evidence or default GPU allowlist is shipped. Other product-wide acceptance requirements remain separate release prerequisites.

### Evidence format and operation

Run `node scripts/ci/release-gate.mjs <evidence-json> <40-character-source-sha> <artifact-directory>`. The manual distribution-gate workflow downloads those files from the selected candidate release; run it at the exact source revision being evaluated. It does not use the affected-CI comparison or run a platform test matrix.

The JSON root has `schemaVersion: 1`, `revision`, `artifacts`, `cpu`, `gpu`, and `extension`. `artifacts` maps `nmath`, `nmath-vscode` and each of the eight platform suffixes to `{file, sha256}`. Files are relative to the artifact directory and cannot escape it through traversal or symlinks. Hashes refer to the tested archives, not unpacked executables.

Every execution record has `kind: "actual-execution"`, `revision`, `status: "passed"`, an empty `unperformed` array, an `environment` description, exact `command`, ISO `executedAt`, an HTTPS `results` link to retained logs, and a `checks` object. Every required check must equal `"passed"`. The exported `cpuChecks`, `gpuChecks`, and `extensionChecks` arrays in the gate define the canonical scenario IDs.

CPU records additionally have `target`, exact `node` version, `nativeExecution: true`, `emulated: false`, `packageSha256`, and `nativeSha256`. Describe architecture, kernel/OS, libc and vendor support in `environment`. GPU records additionally have `target`, `backend`, `hardwareExecution: true`, `softwareAdapter: false`, exact `model`, `driver`, `osVersion`, comparison `tolerances`, `packageSha256`, and `nativeSha256`. Extension evidence additionally has `vsixSha256`, `vscodeVersion`, and `screenReader`.

The reviewed GPU records become the output allowlist. This output is evidence for the recorded artifacts only; it does not configure runtime authorization automatically. Capability probes or successful validator fixtures must never be promoted to hardware records.

The VSIX task bundles `vscode-languageclient` and its runtime dependencies, keeps `vscode` external, and packages only the bundle, manifest, language configuration, README and license files. `NMATH_EXTENSION_LICENSE_BUNDLE` supplies the runner's reviewed extension terms file. No publisher token is required to build a VSIX. Marketplace verification requires a separate authorized publication and actual distribution evidence; this workflow does not perform it.

## Local packaging checks

`pnpm exec turbo run test:package --filter=@delinoio/nmath` packs the local addon and compiler with a clearly marked synthetic license fixture, installs the archives offline into a temporary consumer, executes ESM and CommonJS dynamic-import CPU calculations, starts the compiler, and checks addon-load failure independently of compiler startup. It removes the temporary archives afterward. This smoke test cannot establish minimum-OS support, absence of GPU drivers, license completeness or full platform acceptance.

The platform manifest's `publishConfig.executableFiles` preserves compiler execution permissions in pnpm archives without adding a second CLI command. The public launcher reports executable-start failures with native-load recovery guidance. Compiler resolution uses platform detection without eagerly loading the computation addon.
