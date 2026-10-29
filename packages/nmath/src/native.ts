// SPDX-License-Identifier: Apache-2.0
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

import { NMathError } from "./errors.js";
import type * as Binding from "./native-binding.generated.js";

const require = createRequire(import.meta.url);

/** The napi-rs platform suffix for this process, for example `linux-x64-gnu`. */
export function platformTriple(): string {
  const { platform, arch } = process;
  if (platform === "darwin") {
    return `darwin-${arch}`;
  }
  if (platform === "win32") {
    return `win32-${arch}-msvc`;
  }
  if (platform === "linux") {
    const report = process.report?.getReport() as
      { header?: { glibcVersionRuntime?: string } } | undefined;
    const libc = report?.header?.glibcVersionRuntime ? "gnu" : "musl";
    return `linux-${arch}-${libc}`;
  }
  return `${platform}-${arch}`;
}

const SUPPORTED = new Set([
  "darwin-x64",
  "darwin-arm64",
  "win32-x64-msvc",
  "win32-arm64-msvc",
  "linux-x64-gnu",
  "linux-arm64-gnu",
  "linux-x64-musl",
  "linux-arm64-musl",
]);

const BUILD_GUIDANCE =
  "Reinstall @delinoio/nmath on a supported platform, or build the native module from source: " +
  "install Rust, clone the repository and run `pnpm --filter @delinoio/nmath build:native`.";

function load(): typeof Binding {
  const triple = platformTriple();
  const local = fileURLToPath(
    new URL(`../native/nmath.${triple}.node`, import.meta.url),
  );
  const candidates = [local, `@delinoio/nmath-${triple}`];
  const failures: string[] = [];
  for (const candidate of candidates) {
    if (candidate === local && !existsSync(local)) {
      failures.push("no local build");
      continue;
    }
    try {
      return require(candidate) as typeof Binding;
    } catch (error) {
      failures.push(error instanceof Error ? error.message : String(error));
    }
  }
  const support = SUPPORTED.has(triple)
    ? "This platform is supported but its prebuilt binary could not load."
    : "This platform has no prebuilt binary.";
  // No source build or JavaScript fallback is attempted.
  throw new NMathError(
    "NMATH_NATIVE_LOAD",
    `NMATH_NATIVE_LOAD: The NMath native module could not load for platform '${triple}'. ` +
      `${support} Cause: ${failures.join("; ")}. ${BUILD_GUIDANCE}`,
  );
}

export const binding: typeof Binding = load();
