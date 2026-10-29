// SPDX-License-Identifier: Apache-2.0
// Uses real local binaries and synthetic license text. These archives are fixtures,
// never release candidates or platform acceptance records.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import {
  copyFileSync,
  mkdtempSync,
  mkdirSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  root,
  stagePlatform,
  stagePublic,
  targetFor,
} from "./distribution.mjs";
import { platformTriple } from "../dist/platform.js";
const temporary = mkdtempSync(join(tmpdir(), "nmath-installed-package-"));
const pnpm = (...args) =>
  execFileSync(process.execPath, [process.env.npm_execpath, ...args], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
try {
  const licenseBundle = join(temporary, "LICENSE-FIXTURE.txt");
  writeFileSync(
    licenseBundle,
    "Synthetic license fixture. Do not distribute these archives.",
  );
  const triple = platformTriple();
  const artifacts = join(temporary, "artifacts");
  mkdirSync(artifacts);
  const target = targetFor(triple);
  const exe = target.os === "win32" ? ".exe" : "";
  copyFileSync(
    join(root, "packages/nmath/native", `nmath.${triple}.node`),
    join(artifacts, target.library),
  );
  copyFileSync(
    join(root, "packages/nmath/native", `nmath-compiler.${triple}${exe}`),
    join(artifacts, `nmath-compiler${exe}`),
  );
  const platform = stagePlatform({
    triple,
    artifacts,
    output: temporary,
    licenseBundle,
  });
  const main = stagePublic(temporary, licenseBundle);
  for (const path of [platform, main])
    pnpm("--dir", path, "pack", "--pack-destination", temporary);
  const archives = readdirSync(temporary)
    .filter((name) => name.endsWith(".tgz"))
    .map((name) => join(temporary, name));
  assert.equal(archives.length, 2);
  const consumer = join(temporary, "consumer");
  mkdirSync(consumer);
  writeFileSync(
    join(consumer, "package.json"),
    JSON.stringify({ private: true, type: "module" }),
  );
  pnpm("--dir", consumer, "add", "--ignore-scripts", "--offline", ...archives);
  const program = `
    const assert = (await import('node:assert/strict')).default;
    const {createContext} = await import('@delinoio/nmath');
    const {check} = await import('@delinoio/nmath/compiler');
    assert.equal(typeof check, 'function');
    const ctx = createContext({maxConcurrentJobs:1,maxQueueLength:1,cpuMemoryBudget:1048576n,gpuMemoryBudget:0n});
    assert.deepEqual((await ctx.array([1,2],{dtype:'float64'}).add(2).compute()).toArray(),[3,4]);
    await ctx.close();
  `;
  execFileSync(process.execPath, ["--input-type=module", "-e", program], {
    cwd: consumer,
    stdio: "inherit",
  });
  // This is an actual .cjs entry, not an ESM test that happens to import dynamically.
  writeFileSync(
    join(consumer, "consumer.cjs"),
    `(async()=>{${program}})().catch(e=>{console.error(e);process.exitCode=1});`,
  );
  execFileSync(process.execPath, [join(consumer, "consumer.cjs")], {
    cwd: consumer,
    stdio: "inherit",
  });
  const cli = join(
    consumer,
    "node_modules/@delinoio/nmath/bin/nmath-compiler.mjs",
  );
  execFileSync(process.execPath, [cli, "--help"], {
    cwd: consumer,
    stdio: "inherit",
  });
  rmSync(
    join(
      consumer,
      "node_modules",
      `@delinoio/nmath-${triple}`,
      `nmath.${triple}.node`,
    ),
  );
  execFileSync(process.execPath, [cli, "--help"], {
    cwd: consumer,
    stdio: "inherit",
  });
  execFileSync(
    process.execPath,
    [
      "--input-type=module",
      "-e",
      `
    import assert from 'node:assert/strict';
    await assert.rejects(import('@delinoio/nmath'), e => e.code === 'NMATH_NATIVE_LOAD' && e.message.includes('${triple}') && e.message.includes('build:native'));
  `,
    ],
    { cwd: consumer, stdio: "inherit" },
  );
  console.log(
    JSON.stringify({
      event: "installed_package_smoke",
      triple,
      node: process.versions.node,
      status: "passed",
      scope: "local-smoke-only",
    }),
  );
} finally {
  rmSync(temporary, { recursive: true, force: true });
}
