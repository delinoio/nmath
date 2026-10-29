#!/usr/bin/env node
// SPDX-License-Identifier: Apache-2.0
import { spawn } from "node:child_process";

import { compilerPath } from "../dist/compiler-binary.js";

let binary;
try {
  binary = compilerPath();
} catch (error) {
  console.error(error.message);
  process.exit(2);
}
const child = spawn(binary, process.argv.slice(2), { stdio: "inherit" });
child.once("error", (error) => {
  console.error(
    `NMATH_NATIVE_LOAD: The nmath-compiler executable could not start. Cause: ${error.message}. Reinstall @delinoio/nmath for this platform or build the compiler from source.`,
  );
  process.exitCode = 2;
});
child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
  } else {
    process.exit(code ?? 1);
  }
});
