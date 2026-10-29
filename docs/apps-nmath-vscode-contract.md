# NMath VS Code Extension Contract

## Scope

This contract owns the `nmath-vscode` component of the [`nmath` project](project-nmath.md) at `apps/nmath-vscode/`.

## Runtime and Language

- TypeScript compiled with the TypeScript compiler to a CommonJS extension entry, as VS Code requires.
- Uses `vscode-languageclient` to start `nmath-compiler lsp` over stdio.
- The Rsbuild/Cloudflare Pages default for `apps/` does not apply because this is an editor extension, not a static site.

## Users and Operators

- `.nts` authors who use VS Code.
- Maintainers who publish the extension to the VS Code Marketplace.

## Interfaces and Contracts

- Language ID: `nts`. File extension: `.nts`.
- Setting `nmath.compilerPath` (string, default empty): path to `nmath-compiler`. When empty, the extension resolves `node_modules/.bin/nmath-compiler` in the workspace folder and then `nmath-compiler` on `PATH`.
- Command `nmath.restartLanguageServer` restarts the server.
- The extension uses standard VS Code diagnostics, completion, definition and command palette UI. Semantic results use original `.nts` positions and include unsaved dependency changes from the language server. It adds no custom views or webviews.
- All extension text is English.
- If the compiler cannot start, the extension shows a standard error notification with the resolved path and recovery guidance.

## Storage

The extension stores no data.

## Security

The extension runs the configured compiler executable with the workspace's permissions. In untrusted workspaces it does not start the server.

## Logging

Server output goes to the standard `NMath Language Server` output channel.

## Build and Test

- `pnpm --filter nmath-vscode build` compiles to `dist/`.
- `pnpm --filter nmath-vscode test` runs unit tests for path resolution and manifest checks.
- Keyboard and screen-reader accessibility are verified manually through standard VS Code UI and recorded in pull requests.
- `pnpm exec turbo run package:vsix --filter=nmath-vscode` bundles runtime dependencies and builds a VSIX with reviewed third-party terms. Packaging, Marketplace publication and actual editor/accessibility acceptance remain separate under the [distribution contract](packages-nmath-distribution-contract.md).

## Dependencies and Integrations

- Upstream: [compiler crates contract](crates-nmath-compiler-contract.md) for `nmath-compiler lsp`.
- Marketplace publishing credentials stay outside source control under the [environment contract](repository-environment-contract.md).

## Change Triggers

- Changes to the language server command line update the [compiler crates contract](crates-nmath-compiler-contract.md).
- Update `AGENTS.md` only when development procedures, directory ownership or repository/domain development rules change; follow the [instruction-update policy](README.md#instruction-update-policy).

## References

- [NMath project index](project-nmath.md)
- [Repository defaults](repository-defaults.md)
