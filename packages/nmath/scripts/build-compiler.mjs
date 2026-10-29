// SPDX-License-Identifier: Apache-2.0
// Builds nmath-compiler for this platform and copies it to native/.
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { compilerFileName } from "../dist/compiler-binary.js";

const root = fileURLToPath(new URL("../../../", import.meta.url));
execFileSync("cargo", ["build", "--release", "-p", "nmath-compiler"], {
  cwd: root,
  stdio: "inherit",
});
const exe =
  process.platform === "win32" ? "nmath-compiler.exe" : "nmath-compiler";
const native = fileURLToPath(new URL("../native/", import.meta.url));
mkdirSync(native, { recursive: true });
copyFileSync(`${root}target/release/${exe}`, `${native}${compilerFileName()}`);
