// SPDX-License-Identifier: Apache-2.0
import * as path from "node:path";

export interface ResolveInput {
  /** The `nmath.compilerPath` setting. */
  setting: string;
  /** Workspace folder paths in order. */
  workspaceFolders: readonly string[];
  platform: NodeJS.Platform;
  exists: (file: string) => boolean;
}

/**
 * Chooses the compiler command: the explicit setting, then the first workspace-local
 * `node_modules/.bin/nmath-compiler`, then `nmath-compiler` on PATH.
 */
export function resolveCompilerPath(input: ResolveInput): string {
  const setting = input.setting.trim();
  if (setting !== "") {
    return setting;
  }
  const binary =
    input.platform === "win32" ? "nmath-compiler.cmd" : "nmath-compiler";
  for (const folder of input.workspaceFolders) {
    const candidate = path.join(folder, "node_modules", ".bin", binary);
    if (input.exists(candidate)) {
      return candidate;
    }
  }
  return "nmath-compiler";
}

/** English error text for a server that failed to start. */
export function startFailureMessage(command: string, cause: string): string {
  return (
    `The NMath language server could not start with '${command}': ${cause}. ` +
    "Install @delinoio/nmath in the workspace or set 'nmath.compilerPath' to the nmath-compiler executable, then run 'NMath: Restart Language Server'."
  );
}
