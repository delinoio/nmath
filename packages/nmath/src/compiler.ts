// SPDX-License-Identifier: Apache-2.0
// Programmatic project API for `.nts` projects. Runs the `nmath-compiler` executable.

import { execFile } from "node:child_process";

import { compilerPath } from "./compiler-binary.js";

export type CompilerCode =
  | "NMATH_NTS_PARSE"
  | "NMATH_NTS_ASSIGN_REQUIRES_ASYNC"
  | "NMATH_NTS_AMBIGUOUS_TYPE"
  | "NMATH_NTS_UNSUPPORTED_SYNTAX"
  | "NMATH_COMPILER_CONFIG"
  | "NMATH_COMPILER_TYPE_ERROR"
  | "NMATH_COMPILER_EMIT";

export interface CompilerDiagnostic {
  code: CompilerCode;
  /** Original source file, when the diagnostic has a location. */
  file?: string;
  /** 1-based. */
  line?: number;
  /** 1-based. */
  column?: number;
  message: string;
  recovery: string;
}

export interface CompilerResult {
  ok: boolean;
  diagnostics: CompilerDiagnostic[];
  /** Files written by `build`. Empty for `check` and for failed builds. */
  outputFiles: string[];
}

export interface ProjectOptions {
  /** Path to `tsconfig.json`. */
  tsconfig: string;
  signal?: AbortSignal;
}

function run(
  command: "check" | "build",
  options: ProjectOptions,
): Promise<CompilerResult> {
  const binary = compilerPath();
  return new Promise((resolve, reject) => {
    execFile(
      binary,
      [command, "-p", options.tsconfig, "--json"],
      {
        signal: options.signal,
        maxBuffer: 64 * 1024 * 1024,
        windowsHide: true,
      },
      (error, stdout, stderr) => {
        if (error && error.name === "AbortError") {
          reject(error);
          return;
        }
        try {
          resolve(JSON.parse(stdout) as CompilerResult);
        } catch {
          reject(
            new Error(
              `nmath-compiler ${command} did not return a result: ${(stderr || String(error)).trim()}`,
            ),
          );
        }
      },
    );
  });
}

/** Type-checks a project. */
export function check(options: ProjectOptions): Promise<CompilerResult> {
  return run("check", options);
}

/**
 * Emits ESM JavaScript, declarations and source maps. On any error nothing is written and
 * existing outputs are kept.
 */
export function build(options: ProjectOptions): Promise<CompilerResult> {
  return run("build", options);
}
