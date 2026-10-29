# NMath for VS Code

Language support for NMath `.nts` files, including diagnostics, completion and go to definition. Install `@delinoio/nmath` in your workspace with Node.js 24 before opening an `.nts` file. The extension uses the package's `nmath-compiler` language server.

The extension starts only in trusted workspaces. To use another compiler installation, set `nmath.compilerPath` to its executable path. If the server cannot start, check that path and the **NMath Language Server** output channel, then run **NMath: Restart Language Server** from the Command Palette.

Diagnostics appear in the editor and Problems panel. Completion and go to definition use standard VS Code commands and keyboard controls. This extension adds no custom screens.

NMath is prerelease software. Packaging an extension does not establish Marketplace availability or completion of platform and accessibility acceptance.

Repository-owned code is licensed under Apache-2.0. Bundled dependency terms and notices accompany the extension.
