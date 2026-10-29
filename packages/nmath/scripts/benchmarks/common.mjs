// SPDX-License-Identifier: Apache-2.0
import { createHash } from "node:crypto";
import { readFileSync, renameSync, writeFileSync, rmSync } from "node:fs";

export const versions = { numpy: "2.5.4", pandas: "3.1.0", polars: "2.1.0" };
export const workload = Object.freeze({
  seed: 20261029,
  matrixSize: 128,
  rows: 32768,
  groups: 64,
  algorithmSize: 8192,
});
export function values(length, seed = workload.seed) {
  let state = seed >>> 0;
  return Array.from({ length }, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return ((state >>> 24) % 16) - 8;
  });
}
export function digest(values) {
  return createHash("sha256")
    .update(values.map(String).join(","))
    .digest("hex");
}
export function hashFile(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}
export function aggregate(samples) {
  if (!samples.length || samples.some((x) => !Number.isFinite(x) || x < 0))
    throw new Error("Expected nonempty finite nonnegative samples");
  const sorted = [...samples].sort((a, b) => a - b);
  const median =
    sorted.length % 2
      ? sorted[(sorted.length - 1) / 2]
      : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
  const deviations = sorted
    .map((x) => Math.abs(x - median))
    .sort((a, b) => a - b);
  const middle = Math.floor(deviations.length / 2);
  return {
    count: samples.length,
    min: sorted[0],
    median,
    p95: sorted[Math.ceil(sorted.length * 0.95) - 1],
    max: sorted.at(-1),
    mad:
      deviations.length % 2
        ? deviations[middle]
        : (deviations[middle - 1] + deviations[middle]) / 2,
  };
}
export function atomicReport(path, report) {
  const temp = `${path}.${process.pid}.tmp`;
  try {
    writeFileSync(
      temp,
      JSON.stringify(
        report,
        (_, v) => (typeof v === "bigint" ? v.toString() : v),
        2,
      ) + "\n",
      { flag: "wx" },
    );
    renameSync(temp, path);
  } finally {
    rmSync(temp, { force: true });
  }
}
export function parseArgs(args) {
  const options = { output: undefined, device: "cpu", samples: 9, warmup: 2 };
  for (let i = 0; i < args.length; i += 2) {
    if (!args[i].startsWith("--"))
      throw new Error(`Invalid option: ${args[i]}`);
    const key = args[i].replace(/^--/, "");
    if (!Object.hasOwn(options, key) || args[i + 1] === undefined)
      throw new Error(`Invalid option: ${args[i]}`);
    options[key] = ["samples", "warmup"].includes(key)
      ? Number(args[i + 1])
      : args[i + 1];
  }
  if (
    !options.output ||
    !["cpu", "gpu"].includes(options.device) ||
    !Number.isInteger(options.samples) ||
    options.samples < 3 ||
    options.samples > 1000 ||
    !Number.isInteger(options.warmup) ||
    options.warmup < 1 ||
    options.warmup > 100
  )
    throw new Error(
      "Require --output PATH, --device cpu|gpu, --samples 3..1000 and --warmup 1..100",
    );
  return options;
}

export function gpuIdentity(platform, probe) {
  if (platform !== "darwin")
    return {
      status: "unavailable",
      platform,
      model: null,
      driverVersion: null,
      reason:
        "GPU model and driver identification is not implemented on this platform. Record them separately before comparing GPU timings.",
    };
  const displays = probe();
  return typeof displays === "string"
    ? JSON.parse(displays).SPDisplaysDataType.map((device) => ({
        model: device.sppci_model,
        vendor: device.sppci_vendor,
        metal: device.spdisplays_metal,
        driverVersion:
          "Bundled with the recorded macOS version; separate driver version unavailable",
      }))
    : displays;
}
