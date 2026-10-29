// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import {
  chmodSync,
  copyFileSync,
  cpSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import { compilerFileName } from "../dist/compiler-binary.js";
import { platformTriple } from "../dist/platform.js";

const source = fileURLToPath(new URL("../", import.meta.url));
const compilerName = compilerFileName();

function fixture() {
  const root = mkdtempSync(join(tmpdir(), "nmath-compiler-only-"));
  const pkg = join(root, "node_modules", "@delinoio", "nmath");
  mkdirSync(pkg, { recursive: true });
  cpSync(join(source, "dist"), join(pkg, "dist"), { recursive: true });
  copyFileSync(join(source, "package.json"), join(pkg, "package.json"));
  mkdirSync(join(pkg, "bin"));
  copyFileSync(
    join(source, "bin", "nmath-compiler.mjs"),
    join(pkg, "bin", "nmath-compiler.mjs"),
  );
  mkdirSync(join(pkg, "native"));
  return { root, pkg, log: join(root, "compiler-args.json") };
}

function addCompiler(path) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(
    path,
    `#!/bin/sh\nexec "${process.execPath}" "${path}.mjs" "$@"\n`,
  );
  writeFileSync(
    `${path}.mjs`,
    `
import fs from "node:fs";
fs.appendFileSync(process.env.NMATH_COMPILER_TEST_LOG, JSON.stringify(process.argv.slice(2)) + "\\n");
if (process.argv.includes("--json")) {
  console.log(JSON.stringify({ ok: true, diagnostics: [], outputFiles: ["emitted.js"] }));
}
process.exit(Number(process.env.NMATH_COMPILER_TEST_EXIT || 0));
`,
  );
  chmodSync(path, 0o755);
}

function runNode(root, code, env = {}) {
  return spawnSync(process.execPath, ["--input-type=module", "-e", code], {
    cwd: root,
    encoding: "utf8",
    env: { ...process.env, ...env },
  });
}

test(
  "compiler API and CLI use the local executable without a numerical addon",
  {
    skip: process.platform === "win32",
  },
  () => {
    const { root, pkg, log } = fixture();
    try {
      addCompiler(join(pkg, "native", compilerName));
      const env = { NMATH_COMPILER_TEST_LOG: log };
      const api = runNode(
        root,
        `import assert from "node:assert/strict";
       import { check, build } from "@delinoio/nmath/compiler";
       for (const [name, fn] of [["check", check], ["build", build]]) {
         assert.deepEqual(await fn({ tsconfig: "project.json" }),
           { ok: true, diagnostics: [], outputFiles: ["emitted.js"] });
       }`,
        env,
      );
      assert.equal(api.status, 0, api.stderr);
      assert.deepEqual(
        readFileSync(log, "utf8").trim().split("\n").map(JSON.parse),
        [
          ["check", "-p", "project.json", "--json"],
          ["build", "-p", "project.json", "--json"],
        ],
      );
      const cli = spawnSync(
        process.execPath,
        [join(pkg, "bin", "nmath-compiler.mjs"), "check", "--flag", "value"],
        {
          encoding: "utf8",
          env: { ...process.env, ...env, NMATH_COMPILER_TEST_EXIT: "7" },
        },
      );
      assert.equal(cli.status, 7, cli.stderr);
      assert.deepEqual(
        readFileSync(log, "utf8").trim().split("\n").map(JSON.parse),
        [
          ["check", "-p", "project.json", "--json"],
          ["build", "-p", "project.json", "--json"],
          ["check", "--flag", "value"],
        ],
      );

      const core = runNode(root, `await import("@delinoio/nmath");`);
      assert.notEqual(core.status, 0);
      assert.match(core.stderr, /NMATH_NATIVE_LOAD/);
      assert.match(core.stderr, /build:native/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  },
);

test(
  "an unloadable addon does not block compiler API or CLI",
  {
    skip: process.platform === "win32",
  },
  () => {
    const { root, pkg, log } = fixture();
    try {
      addCompiler(join(pkg, "native", compilerName));
      writeFileSync(
        join(pkg, "native", `nmath.${platformTriple()}.node`),
        "invalid addon",
      );
      const env = { NMATH_COMPILER_TEST_LOG: log };
      const api = runNode(
        root,
        `import { check, build } from "@delinoio/nmath/compiler";
       if (!(await check({ tsconfig: "project.json" })).ok) process.exit(1);
       if (!(await build({ tsconfig: "project.json" })).ok) process.exit(1);`,
        env,
      );
      assert.equal(api.status, 0, api.stderr);
      const cli = spawnSync(
        process.execPath,
        [join(pkg, "bin", "nmath-compiler.mjs"), "build"],
        {
          encoding: "utf8",
          env: { ...process.env, ...env },
        },
      );
      assert.equal(cli.status, 0, cli.stderr);
      assert.deepEqual(
        readFileSync(log, "utf8").trim().split("\n").map(JSON.parse),
        [
          ["check", "-p", "project.json", "--json"],
          ["build", "-p", "project.json", "--json"],
          ["build"],
        ],
      );
      const core = runNode(root, `await import("@delinoio/nmath");`);
      assert.notEqual(core.status, 0);
      assert.match(core.stderr, /NMATH_NATIVE_LOAD/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  },
);

test(
  "compiler lookup prefers local then platform package and keeps missing guidance",
  {
    skip: process.platform === "win32",
  },
  () => {
    const { root, pkg } = fixture();
    try {
      const platform = join(
        root,
        "node_modules",
        "@delinoio",
        `nmath-${platformTriple()}`,
      );
      mkdirSync(platform, { recursive: true });
      writeFileSync(
        join(platform, "package.json"),
        JSON.stringify({ name: `@delinoio/nmath-${platformTriple()}` }),
      );
      addCompiler(join(platform, compilerName));
      const local = join(pkg, "native", compilerName);
      addCompiler(local);
      const code = `import { compilerPath } from ${JSON.stringify(pathToFileURL(join(pkg, "dist", "compiler-binary.js")).href)};
      console.log(compilerPath());`;
      assert.equal(runNode(root, code).stdout.trim(), realpathSync(local));
      rmSync(local);
      assert.equal(
        runNode(root, code).stdout.trim(),
        realpathSync(join(platform, compilerName)),
      );
      rmSync(platform, { recursive: true });
      const missing = runNode(
        root,
        `import { check } from "@delinoio/nmath/compiler";
      await check({ tsconfig: "project.json" });`,
      );
      assert.notEqual(missing.status, 0);
      assert.match(missing.stderr, /nmath-compiler executable is missing/);
      assert.match(missing.stderr, /build:compiler/);
      assert.doesNotMatch(missing.stderr, /native module could not load/);
      const cli = spawnSync(
        process.execPath,
        [join(pkg, "bin", "nmath-compiler.mjs")],
        { encoding: "utf8" },
      );
      assert.equal(cli.status, 2);
      assert.match(cli.stderr, /build:compiler/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  },
);

test(
  "compiler build invokes Cargo without a numerical addon",
  {
    skip: process.platform === "win32",
  },
  () => {
    const root = mkdtempSync(join(tmpdir(), "nmath-build-compiler-"));
    try {
      const pkg = join(root, "packages", "nmath");
      mkdirSync(join(pkg, "scripts"), { recursive: true });
      cpSync(join(source, "dist"), join(pkg, "dist"), { recursive: true });
      copyFileSync(
        join(source, "scripts", "build-compiler.mjs"),
        join(pkg, "scripts", "build-compiler.mjs"),
      );
      writeFileSync(
        join(pkg, "package.json"),
        JSON.stringify({ type: "module" }),
      );
      const bin = join(root, "bin");
      mkdirSync(bin);
      const cargoLog = join(root, "cargo-args.json");
      writeFileSync(
        join(bin, "cargo"),
        `#!/usr/bin/env node
const fs = require("node:fs");
const path = require("node:path");
fs.writeFileSync(process.env.NMATH_CARGO_TEST_LOG, JSON.stringify(process.argv.slice(2)));
const output = path.join(process.cwd(), "target", "release", "nmath-compiler");
fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, "compiler fixture");
`,
      );
      chmodSync(join(bin, "cargo"), 0o755);
      execFileSync(
        process.execPath,
        [join(pkg, "scripts", "build-compiler.mjs")],
        {
          env: {
            ...process.env,
            PATH: `${bin}:${process.env.PATH}`,
            NMATH_CARGO_TEST_LOG: cargoLog,
          },
        },
      );
      assert.deepEqual(JSON.parse(readFileSync(cargoLog, "utf8")), [
        "build",
        "--locked",
        "--release",
        "-p",
        "nmath-compiler",
      ]);
      assert.equal(
        readFileSync(join(pkg, "native", compilerName), "utf8"),
        "compiler fixture",
      );
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  },
);
