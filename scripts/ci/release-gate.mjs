// SPDX-License-Identifier: Apache-2.0
import { createHash } from "node:crypto";
import { readFileSync, realpathSync } from "node:fs";
import { isAbsolute, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { targets } from "../../packages/nmath/scripts/distribution.mjs";

export const cpuChecks = [
  "esm",
  "commonjs-dynamic-import",
  "cpu-without-gpu",
  "compiler",
  "native-load-error",
  "minimum-platform",
  "licenses",
];
export const gpuChecks = [
  "numerical-comparison",
  "dtype-operation-matrix",
  "broadcast",
  "reductions",
  "matmul",
  "unsupported-before-queue",
  "no-adapter",
  "device-loss",
  "memory-exhaustion",
  "cancellation",
  "no-fallback-no-retry",
];
export const extensionChecks = [
  "vsix-install",
  "activation",
  "diagnostics",
  "completion",
  "definition",
  "keyboard",
  "screen-reader",
  "marketplace-distribution",
];
export const gpuTargets = {
  "darwin-arm64": "Metal",
  "linux-x64-gnu": "Vulkan",
  "win32-x64-msvc": "DX12",
};
const sha = /^[a-f0-9]{64}$/;
function requireValue(condition, message) {
  if (!condition) throw new Error(`Release blocked: ${message}`);
}
function text(value) {
  return typeof value === "string" && value.trim().length > 0;
}
function record(value, checks, revision) {
  requireValue(
    value?.kind === "actual-execution" && value.revision === revision,
    "actual execution at the release revision is required",
  );
  requireValue(
    value.status === "passed" && value.unperformed?.length === 0,
    "failed or unperformed checks",
  );
  requireValue(
    text(value.environment) &&
      text(value.command) &&
      text(value.executedAt) &&
      Number.isFinite(Date.parse(value.executedAt)),
    "missing environment, command or execution date",
  );
  requireValue(
    /^https:\/\//.test(value.results ?? ""),
    "missing durable result URL",
  );
  for (const check of checks)
    requireValue(value.checks?.[check] === "passed", `missing check: ${check}`);
}
export function validateEvidence(evidence, revision, artifactRoot) {
  requireValue(
    /^[a-f0-9]{40}$/.test(revision) &&
      evidence.schemaVersion === 1 &&
      evidence.revision === revision,
    "revision/schema mismatch",
  );
  const artifacts = evidence.artifacts;
  const required = ["nmath", "nmath-vscode", ...targets.map((t) => t.triple)];
  requireValue(
    artifacts && Object.keys(artifacts).length === required.length,
    "incomplete artifact set",
  );
  const root = realpathSync(artifactRoot);
  for (const id of required) {
    const entry = artifacts[id];
    requireValue(
      entry &&
        text(entry.file) &&
        !isAbsolute(entry.file) &&
        sha.test(entry.sha256),
      `missing artifact: ${id}`,
    );
    const path = realpathSync(resolve(root, entry.file));
    const rel = relative(root, path);
    requireValue(
      rel !== ".." && !rel.startsWith(`..${sep}`) && !isAbsolute(rel),
      "artifact escapes its root",
    );
    requireValue(
      createHash("sha256").update(readFileSync(path)).digest("hex") ===
        entry.sha256,
      `artifact digest mismatch: ${id}`,
    );
  }
  requireValue(
    Array.isArray(evidence.cpu) && evidence.cpu.length === targets.length,
    "eight CPU records required",
  );
  for (const { triple } of targets) {
    const matches = evidence.cpu.filter((r) => r.target === triple);
    requireValue(
      matches.length === 1,
      `missing or duplicate CPU target: ${triple}`,
    );
    const value = matches[0];
    record(value, cpuChecks, revision);
    requireValue(
      /^24\./.test(value.node ?? "") &&
        value.nativeExecution === true &&
        value.emulated === false,
      `native Node.js 24 execution required: ${triple}`,
    );
    requireValue(
      value.packageSha256 === artifacts.nmath.sha256 &&
        value.nativeSha256 === artifacts[triple].sha256,
      `CPU artifact mismatch: ${triple}`,
    );
  }
  requireValue(
    Array.isArray(evidence.gpu) && evidence.gpu.length >= 3,
    "three GPU platforms required",
  );
  for (const [target, backend] of Object.entries(gpuTargets)) {
    requireValue(
      evidence.gpu.some((r) => r.target === target),
      `missing GPU target: ${target}`,
    );
    for (const value of evidence.gpu.filter((r) => r.target === target)) {
      record(value, gpuChecks, revision);
      requireValue(
        value.backend === backend &&
          value.hardwareExecution === true &&
          value.softwareAdapter === false &&
          text(value.model) &&
          text(value.driver) &&
          text(value.osVersion) &&
          text(value.tolerances),
        `exact GPU environment and tolerances required: ${target}`,
      );
      requireValue(
        value.packageSha256 === artifacts.nmath.sha256 &&
          value.nativeSha256 === artifacts[target].sha256,
        `GPU artifact mismatch: ${target}`,
      );
    }
  }
  requireValue(
    evidence.gpu.every((r) => Object.hasOwn(gpuTargets, r.target)),
    "unknown GPU target",
  );
  record(evidence.extension, extensionChecks, revision);
  requireValue(
    evidence.extension.vsixSha256 === artifacts["nmath-vscode"].sha256 &&
      text(evidence.extension.vscodeVersion) &&
      text(evidence.extension.screenReader),
    "extension artifact/environment mismatch",
  );
  return {
    revision,
    status: "passed",
    scope: "distribution-evidence",
    gpuAllowlist: evidence.gpu.map(
      ({ target, backend, model, driver, osVersion }) => ({
        target,
        backend,
        model,
        driver,
        osVersion,
      }),
    ),
  };
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  const [file, revision, artifacts] = process.argv.slice(2);
  console.log(
    JSON.stringify(
      validateEvidence(
        JSON.parse(readFileSync(file, "utf8")),
        revision,
        artifacts,
      ),
      null,
      2,
    ),
  );
}
