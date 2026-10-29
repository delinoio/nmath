// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  mkdtempSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
  platformManifest,
  stagePlatform,
  targets,
  targetFor,
} from "../../packages/nmath/scripts/distribution.mjs";
import {
  cpuChecks,
  extensionChecks,
  gpuChecks,
  gpuTargets,
  validateEvidence,
} from "./release-gate.mjs";
function fixture(t) {
  const dir = mkdtempSync(join(tmpdir(), "nmath-packaging-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  return dir;
}
test("all eight packages contain addon, executable and licenses with platform filters", (t) => {
  const dir = fixture(t);
  const licenseBundle = join(dir, "licenses.txt");
  writeFileSync(licenseBundle, "Fixture license terms, not release evidence.");
  assert.equal(new Set(targets.map((t) => t.triple)).size, 8);
  for (const target of targets) {
    writeFileSync(join(dir, target.library), "fixture addon");
    writeFileSync(
      join(dir, `nmath-compiler${target.os === "win32" ? ".exe" : ""}`),
      "fixture compiler",
    );
    const output = stagePlatform({
      triple: target.triple,
      artifacts: dir,
      output: dir,
      licenseBundle,
    });
    const manifest = JSON.parse(readFileSync(join(output, "package.json")));
    assert.deepEqual(manifest, platformManifest(target));
    assert.equal(
      readFileSync(join(output, manifest.main), "utf8"),
      "fixture addon",
    );
    assert.match(
      readFileSync(join(output, "LICENSE"), "utf8"),
      /Apache License/,
    );
    const exe = join(
      output,
      `nmath-compiler.${target.triple}${target.os === "win32" ? ".exe" : ""}`,
    );
    if (process.platform !== "win32") assert.ok(statSync(exe).mode & 0o111);
    assert.deepEqual(manifest.libc, target.libc ? [target.libc] : undefined);
  }
  assert.throws(() => targetFor("linux-riscv64-gnu"), /Unsupported/);
  assert.throws(
    () =>
      stagePlatform({ triple: targets[0].triple, artifacts: dir, output: dir }),
    /NMATH_LICENSE_BUNDLE/,
  );
});
function evidenceFixture(dir) {
  const revision = "a".repeat(40);
  const digest = createHash("sha256").update("fixture").digest("hex");
  writeFileSync(join(dir, "artifact"), "fixture");
  const record = (checks) => ({
    kind: "actual-execution",
    revision,
    status: "passed",
    unperformed: [],
    environment: "synthetic validator fixture only",
    command: "fixture",
    executedAt: "2026-10-29T00:00:00Z",
    results: "https://example.invalid/fixture",
    checks: Object.fromEntries(checks.map((k) => [k, "passed"])),
  });
  return {
    schemaVersion: 1,
    revision,
    artifacts: Object.fromEntries(
      ["nmath", "nmath-vscode", ...targets.map((t) => t.triple)].map((k) => [
        k,
        { file: "artifact", sha256: digest },
      ]),
    ),
    cpu: targets.map(({ triple }) => ({
      ...record(cpuChecks),
      target: triple,
      node: "24.0.0",
      nativeExecution: true,
      emulated: false,
      packageSha256: digest,
      nativeSha256: digest,
    })),
    gpu: Object.entries(gpuTargets).map(([target, backend]) => ({
      ...record(gpuChecks),
      target,
      backend,
      hardwareExecution: true,
      softwareAdapter: false,
      model: "fixture",
      driver: "fixture",
      osVersion: "fixture",
      tolerances: "fixture",
      packageSha256: digest,
      nativeSha256: digest,
    })),
    extension: {
      ...record(extensionChecks),
      vsixSha256: digest,
      vscodeVersion: "fixture",
      screenReader: "fixture",
    },
  };
}
test("gate validates structure but does not attest fixture hardware", (t) => {
  const dir = fixture(t);
  const evidence = evidenceFixture(dir);
  assert.equal(
    validateEvidence(evidence, evidence.revision, dir).gpuAllowlist.length,
    3,
  );
  for (const mutate of [
    (e) => e.cpu.pop(),
    (e) => e.gpu.pop(),
    (e) => (e.cpu[0].kind = "cross-build"),
    (e) => (e.cpu[0].emulated = true),
    (e) => (e.cpu[0].node = "26.0.0"),
    (e) => (e.cpu[0].revision = "b".repeat(40)),
    (e) => (e.cpu[0].nativeSha256 = "0".repeat(64)),
    (e) => (e.gpu[0].softwareAdapter = true),
    (e) => (e.gpu[0].driver = ""),
    (e) => (e.gpu[0].checks["device-loss"] = "skipped"),
    (e) => (e.extension.checks["screen-reader"] = "skipped"),
    (e) => (e.extension.checks["marketplace-distribution"] = "skipped"),
    (e) => (e.artifacts.nmath.sha256 = "0".repeat(64)),
    (e) => (e.artifacts.nmath.file = "/etc/passwd"),
  ]) {
    const bad = structuredClone(evidence);
    mutate(bad);
    assert.throws(
      () => validateEvidence(bad, evidence.revision, dir),
      /Release blocked/,
    );
  }
});
