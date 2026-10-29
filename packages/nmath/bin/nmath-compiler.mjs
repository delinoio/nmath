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
child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
  } else {
    process.exit(code ?? 1);
  }
});
