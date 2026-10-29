// SPDX-License-Identifier: Apache-2.0
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { NMathError } from "./errors.js";
import { platformTriple } from "./platform.js";

/** File name of the compiler executable for this platform inside `native/`. */
export function compilerFileName(): string {
  const suffix = process.platform === "win32" ? ".exe" : "";
  return `nmath-compiler.${platformTriple()}${suffix}`;
}

/** Absolute path of the prebuilt `nmath-compiler` executable. */
export function compilerPath(): string {
  const local = fileURLToPath(
    new URL(`../native/${compilerFileName()}`, import.meta.url),
  );
  if (existsSync(local)) {
    return local;
  }
  try {
    const require = createRequire(import.meta.url);
    const manifest = require.resolve(
      `@delinoio/nmath-${platformTriple()}/package.json`,
    );
    const candidate = join(dirname(manifest), compilerFileName());
    if (existsSync(candidate)) {
      return candidate;
    }
  } catch {
    // Fall through to the error below.
  }
  throw new NMathError(
    "NMATH_NATIVE_LOAD",
    `NMATH_NATIVE_LOAD: The nmath-compiler executable is missing for platform '${platformTriple()}'. ` +
      "Reinstall @delinoio/nmath on a supported platform, or build it from source with `pnpm --filter @delinoio/nmath build:compiler`.",
  );
}
