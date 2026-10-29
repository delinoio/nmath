// SPDX-License-Identifier: Apache-2.0
// Actual local artifacts and source notices; pending review remains explicit.
import { createHash } from "node:crypto";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import {
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
  root,
  stagePlatform,
  stagePublic,
  targetFor,
} from "./distribution.mjs";
import { verifyBundle } from "./license-bundle.mjs";
import { platformTriple } from "../dist/platform.js";

const bundle = resolve(
  process.argv[2] ?? process.env.NMATH_LICENSE_BUNDLE ?? "",
);
const receipt = verifyBundle(bundle, { allowPending: true });
const temporary = mkdtempSync(join(tmpdir(), "nmath-real-license-smoke-"));
const triple = platformTriple();
const target = targetFor(triple);
const pnpm = (...args) =>
  execFileSync(process.execPath, [process.env.npm_execpath, ...args], {
    encoding: "utf8",
  });
try {
  const artifacts = join(temporary, "artifacts");
  mkdirSync(artifacts);
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
    licenseBundle: bundle,
  });
  const main = stagePublic(temporary, bundle);
  for (const directory of [platform, main])
    pnpm("--dir", directory, "pack", "--pack-destination", temporary);
  const archives = readdirSync(temporary)
    .filter((n) => n.endsWith(".tgz"))
    .map((n) => join(temporary, n));
  assert.equal(archives.length, 2);
  const consumer = join(temporary, "consumer");
  mkdirSync(consumer);
  writeFileSync(
    join(consumer, "package.json"),
    JSON.stringify({ private: true, type: "module" }),
  );
  pnpm("--dir", consumer, "add", "--ignore-scripts", "--offline", ...archives);
  for (const name of ["nmath", `nmath-${triple}`]) {
    const installed = join(
      consumer,
      "node_modules/@delinoio",
      name,
      "THIRD_PARTY_LICENSES.txt",
    );
    assert.deepEqual(readFileSync(installed), readFileSync(bundle));
    assert.equal(
      verifyBundle(installed, { allowPending: true }).bundleSha256,
      receipt.bundleSha256,
    );
  }
  const program = `const assert = (await import('node:assert/strict')).default;
    const { createContext } = await import('@delinoio/nmath');
    const { check } = await import('@delinoio/nmath/compiler');
    assert.equal(typeof check, 'function');
    const ctx = createContext({maxConcurrentJobs:1,maxQueueLength:1,cpuMemoryBudget:1048576n,gpuMemoryBudget:0n});
    assert.deepEqual((await ctx.array([1,2],{dtype:'float64'}).add(2).compute()).toArray(), [3,4]);
    await ctx.close();`;
  execFileSync(process.execPath, ["--input-type=module", "-e", program], {
    cwd: consumer,
    stdio: "inherit",
  });
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
      "node_modules/@delinoio",
      `nmath-${triple}`,
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
      `import assert from 'node:assert/strict'; await assert.rejects(import('@delinoio/nmath'), e=>e.code==='NMATH_NATIVE_LOAD');`,
    ],
    { cwd: consumer, stdio: "inherit" },
  );
  console.log(
    JSON.stringify({
      event: "actual_license_installed_package_smoke",
      revision: execFileSync("git", ["rev-parse", "HEAD"], {
        cwd: root,
        encoding: "utf8",
      }).trim(),
      triple,
      node: process.versions.node,
      status: "passed",
      bundle: {
        sha256: receipt.bundleSha256,
        inventorySha256: receipt.inventorySha256,
        reviewStatus: receipt.reviewStatus,
        components: receipt.components,
      },
      artifacts: archives.map((file) => ({
        file: file.split("/").at(-1),
        sha256: createHash("sha256").update(readFileSync(file)).digest("hex"),
      })),
      acceptance:
        "local-smoke-only; no license-review or platform-release acceptance",
    }),
  );
} finally {
  rmSync(temporary, { recursive: true, force: true });
}
