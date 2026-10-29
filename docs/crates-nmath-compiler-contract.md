# NMath Compiler Crates Contract

## Scope

This contract owns the `nmath-nts` and `nmath-compiler` components of the [`nmath` project](project-nmath.md):

- `crates/nmath-nts/`: the `.nts` source transform and its position map.
- `crates/nmath-compiler/`: the `nmath-compiler` executable, its programmatic project API and the content-mapper process.

## Runtime and Language

- Rust with the repository toolchain.
- The compiler depends on tsrs crates from Rust, pinned to commit `77d3cddb44540734abc33a42ccca4d3f33bee527` (workspace version `0.20.1`, TypeScript `7.1.0-beta` source baseline). The tsrs npm executable is not part of the architecture.
- The tsrs TypeScript 7.1 beta baseline is separate from the TypeScript 7.0.2 baseline that validates ordinary `.ts` consumers of the npm package.

## Users and Operators

- `.nts` authors who run `nmath-compiler check` and `nmath-compiler build`.
- Editors that start `nmath-compiler lsp`.
- Programmatic callers of `@delinoio/nmath/compiler`.

## Interfaces and Contracts

### `.nts` transform

The transform parses `.nts` text with the standard tsrs TypeScript parser and produces TypeScript text plus a span map. It does not extend the parser or lexer. Only checker-proven NMath operations are lowered. Ordinary JavaScript operators and bracket accesses remain unchanged. Project analysis repeats until lowered expression types propagate through local inference and imports; the language server uses the same analysis with unsaved overlays.

An interface or class that inherits a package-owned NMath declaration retains NMath operator and numeric-index behavior. This also applies through transitive bases and generic constraints. A matching local class name or property name alone does not establish NMath identity.

| `.nts` source                                         | Lowered TypeScript                                                                                                           |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `a + b`, `a - b`, `a * b`, `a / b`, `a % b`, `a ** b` | `__nmath.add(a, b)`, `__nmath.sub(a, b)`, `__nmath.mul(a, b)`, `__nmath.div(a, b)`, `__nmath.mod(a, b)`, `__nmath.pow(a, b)` |
| `+a`, `-a`                                            | `__nmath.pos(a)`, `__nmath.neg(a)`                                                                                           |
| `a[i]`                                                | `__nmath.get(a, i)`                                                                                                          |
| `a[[i, j]]`                                           | `__nmath.get(a, [i, j])`                                                                                                     |
| `a[i] = value`                                        | `(await __nmath.set(a, i, value))`                                                                                           |

- The transform inserts `import * as __nmath from "@delinoio/nmath/nts-runtime";` only when its lowering plan contains NMath operations, at the top of the module, after any shebang line and leading triple-slash directives (`/// <reference ... />`, `/// <amd-module ... />`).
- The runtime namespace is named `__nmath` unless the file already uses that identifier anywhere (as a binding or a reference); then the transform uses the first unused name of `__nmath_1`, `__nmath_2`, ... for the import and every lowered helper call, so no local binding captures generated calls.
- Compiler helpers retain public arithmetic signatures, indexing result types, and assignment value types. They call NMath directly without speculative runtime dispatch. Types that may represent either NMath or ordinary JavaScript require narrowing.
- When a lowered assignment starts an expression statement that follows a statement not ending in `;`, the transform prefixes it with `;` so semicolonless code keeps its statement boundaries.
- A comma-expression index stays one helper argument: `a[f(), g()]` lowers to `__nmath.get(a, (f(), g()))`, and the same applies to `a[f(), g()] = value`.
- Literal-only arithmetic (both operands numeric or string literals) is not rewritten.
- NMath `a[i] = value` returns the assigned value and is rewritten only inside async function bodies or module top level. Elsewhere, including parameter initializers of async functions, the transform reports `NMATH_NTS_ASSIGN_REQUIRES_ASYNC`.
- Compound assignment (`+=` and others), `++` and `--` on bracket expressions are left unchanged. Comparison and bitwise operators are never rewritten.
- These forms are also left unchanged: optional element access (`a?.[i]`), `super[k]`, element access used as a call target (`a[i](x)`, `(a[i])(x)`), `delete a[i]` (including parenthesized or type-asserted operands such as `delete (a[i])`), destructuring and `for-of` assignment targets, ordinary property access (`a.b`) and string-literal keys (`obj["key"]`). Nothing inside type positions or enums is rewritten. Unary operators applied to numeric literals (for example `-1`) count as literals.
- Source-relative imports ending in `.nts` are rewritten to `.js` in emitted output. With the transform's extension-rewriting option enabled, this includes quoted specifiers and dynamic imports with a template literal that has no substitutions. The transform preserves the original specifier when this option is disabled. Interpolated templates are unchanged.

Unsupported NMath syntax reports a diagnostic instead of silently executing ordinary JavaScript semantics.

### Diagnostics

Compiler diagnostics use stable codes:

| Code                              | Meaning                                                       |
| --------------------------------- | ------------------------------------------------------------- |
| `NMATH_NTS_AMBIGUOUS_TYPE`        | An operand requires narrowing before lowering.                |
| `NMATH_NTS_UNSUPPORTED_SYNTAX`    | An NMath operation requires an explicit API.                  |
| `NMATH_NTS_PARSE`                 | The `.nts` file has a syntax error.                           |
| `NMATH_NTS_ASSIGN_REQUIRES_ASYNC` | An NMath bracket assignment appears outside an async context. |
| `NMATH_COMPILER_CONFIG`           | The `tsconfig.json` path is missing or invalid.               |
| `NMATH_COMPILER_TYPE_ERROR`       | Type checking reported errors.                                |
| `NMATH_COMPILER_EMIT`             | Output could not be written.                                  |

Every diagnostic reports the original `.nts` file and 1-based line and column, an English message and recovery guidance.

### CLI

- `nmath-compiler check -p <tsconfig>`: type-check the project. Exit code 0 on success, 1 on diagnostics, 2 on usage or configuration errors.
- `nmath-compiler build -p <tsconfig>`: emit ESM JavaScript, declaration files and source maps. On any type or transform error, write nothing and keep existing outputs.
- `nmath-compiler lsp`: run a generic Language Server Protocol server over stdio with semantic type diagnostics, completion and cross-file go-to-definition for `.nts` files. Locations use original-source UTF-16 coordinates. Full-document synchronization overlays unsaved buffers on disk content; changes refresh dependent diagnostics, and closing a document restores its disk content. Stale document versions do not replace newer buffers.
- Each language-server semantic snapshot keeps the effective project roots from the selected `tsconfig.json`, including inherited declarations and include/exclude rules. It adds the queried document once if that document is outside the root set. Unsaved document text takes precedence over disk text for all roots.
- `nmath-compiler content-mapper`: internal tsrs content-mapper process. Not a public interface.
- `--json` prints the structured result `{ ok, diagnostics: [{ code, file, line, column, message, recovery }], outputFiles: [] }`.

### Programmatic API

`@delinoio/nmath/compiler` exposes `check({ tsconfig })` and `build({ tsconfig })`. Each returns `Promise<CompilerResult>` with the structured result above. An in-memory source API is out of scope.

## Storage

Build output is written to the `outDir` from `tsconfig.json`. Each output file is written to a temporary file and renamed only after the whole build succeeds.

## Security

The compiler reads and writes local files with the calling user's permissions. It does not execute project code. The content-mapper process accepts requests only from its parent over stdio.

## Logging

Use `tracing` with structured fields. `NMATH_LOG` enables logs on stderr; ANSI color is on by default and `NO_COLOR` disables it. Logs do not contain source text.

## Build and Test

- `cargo test` from the repository root runs transform and CLI tests.
- Transform tests cover every rewrite in the table above, preservation of ordinary JS, span-map round trips and the async-assignment diagnostic.

## Dependencies and Integrations

- tsrs crates pinned to `77d3cddb44540734abc33a42ccca4d3f33bee527`. A pinned fork or patch is allowed only when an upstream change is essential; document its scope and removal condition next to the patch.
- The pinned tsrs LSP server lacks the mapper process hook required by NMath. `nmath-compiler lsp` uses the tsrs compiler and language-service crates directly with an in-process NMath mapper and an adapter for original protocol positions. Each request builds an immutable semantic snapshot; automatic import discovery is disabled. Remove this adapter when a pinned tsrs server supports the required content mapping end to end.
- Downstream: [npm package contract](packages-nmath-package-contract.md) for `/compiler` and `nts-runtime`; [VS Code extension contract](apps-nmath-vscode-contract.md) for `lsp`.

### Integration workarounds

- `check` and `build` read the user's `tsconfig.json` through an in-memory layer that adds a config extending it and registering the NMath content mapper, which runs `nmath-compiler content-mapper`. The user's files are never edited. The reserved virtual names are `__nmath_compiler__.tsconfig.json` and `__nmath_compiler_emit__.tsconfig.json` next to the user's config.
- tsrs, like TypeScript 7.1, does not emit JavaScript for content-mapped files. `build` therefore runs a second tsrs pass with `--noCheck` over the lowered TypeScript. NMath composes the JavaScript source maps with its transform span maps before publishing output. The maps name the original `.nts` files and embed their original text. Generated helper imports have no original source location. Source map coordinates use UTF-16 columns; transform spans use UTF-8 byte offsets. The JavaScript pass emits external maps beside its output with an empty source root, so virtual `.ts` paths do not escape into debugging metadata. Declaration maps are emitted by the checked pass through the pinned tsrs content-map support. Remove this workaround when the pinned tsrs emits content-mapped files.
- `--rewriteRelativeImportExtensions` is enabled in both compiler passes, even when the user's config omits or disables it. This maps source-relative `.nts` and `.ts` imports to emitted `.js` files. `build` also enables `--declaration`, `--declarationMap` and `--sourceMap`.
- Project references and `-b` builds are not supported.

## Change Triggers

- Changes to rewrites or helper names update the [npm package contract](packages-nmath-package-contract.md).
- Changes to the CLI or LSP surface update the [VS Code extension contract](apps-nmath-vscode-contract.md).
- Update `AGENTS.md` only when development procedures, directory ownership or repository/domain development rules change; follow the [instruction-update policy](README.md#instruction-update-policy).

## References

- [NMath project index](project-nmath.md)
- [Runtime crates contract](crates-nmath-runtime-contract.md)
- [Repository defaults](repository-defaults.md)
- [tsrs content-mapper notes at the pinned commit](https://github.com/oxc-project/tsrs/blob/77d3cddb44540734abc33a42ccca4d3f33bee527/notes/contentmappers.md)
