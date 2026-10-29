# Documentation Catalog

`docs/` is the source of truth for repository contracts.

## Repository contracts

- [Technology defaults](repository-defaults.md)
- [Development and CI](repository-workflow-contract.md)
- [Configuration and secrets](repository-environment-contract.md)
- [Licensing](repository-license-contract.md)
- [Dependency security](repository-dependency-security-contract.md)

## Templates

- [Project index](project-template.md)
- [Domain contract](domain-template.md)

## Project catalog

Before implementing a new project, add `project-<id>.md`, at least one `<domain>-<project-or-component>-<contract>.md`, and their links here. Use lowercase kebab-case IDs and domain prefixes `apps`, `cmds`, `servers`, `crates`, `protos`, or `packages`.

### NMath

- [Project index](project-nmath.md)
- [Runtime crates contract](crates-nmath-runtime-contract.md)
- [Compiler crates contract](crates-nmath-compiler-contract.md)
- [npm package contract](packages-nmath-package-contract.md)
- [VS Code extension contract](apps-nmath-vscode-contract.md)
- [Native distribution and release evidence](packages-nmath-distribution-contract.md)

## Instruction update policy

Keep feature behavior in its owning contract. Update `AGENTS.md` only when development procedures, directory ownership or development rules change. Keep validation evidence and issue or pull request traceability in GitHub records and CI logs, rather than instruction files or documentation. See [root instructions](../AGENTS.md).
