// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import {
  aggregate,
  gpuIdentity,
  atomicReport,
  digest,
  parseArgs,
  values,
  workload,
  versions,
} from "../scripts/benchmarks/common.mjs";

test("aggregation retains outliers and uses median, nearest-rank p95 and MAD", () => {
  assert.deepEqual(aggregate([100, 1, 3, 2, 4]), {
    count: 5,
    min: 1,
    median: 3,
    p95: 100,
    max: 100,
    mad: 1,
  });
  assert.equal(aggregate([1, 2, 3, 4]).median, 2.5);
  for (const invalid of [[], [NaN], [Infinity], [-1]])
    assert.throws(() => aggregate(invalid));
});
test("report publication replaces complete JSON and preserves existing report on serialization error", () => {
  const directory = mkdtempSync(join(tmpdir(), "nmath-benchmark-report-"));
  const path = join(directory, "report.json");
  try {
    atomicReport(path, { status: "running", memory: 42n });
    const circular = {};
    circular.self = circular;
    assert.throws(() => atomicReport(path, circular));
    assert.deepEqual(JSON.parse(readFileSync(path, "utf8")), {
      status: "running",
      memory: "42",
    });
    atomicReport(path, { status: "interrupted", samples: [1, 2] });
    assert.deepEqual(readdirSync(directory), ["report.json"]);
    assert.equal(JSON.parse(readFileSync(path, "utf8")).status, "interrupted");
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
test("CLI rejects invalid samples, devices, missing output and unknown flags", () => {
  for (const args of [
    [],
    ["--output"],
    ["--unknown", "1"],
    ["--output", "a", "--samples", "2"],
    ["--output", "a", "--warmup", "0"],
    ["--output", "a", "--device", "metal"],
  ])
    assert.throws(() => parseArgs(args));
  assert.equal(parseArgs(["--output", "a"]).samples, 9);
});
test("synthetic stream and exact independent arithmetic match fixed-original fixture", () => {
  const fixture = JSON.parse(
    readFileSync(
      new URL("./fixtures/benchmarks/originals.json", import.meta.url),
      "utf8",
    ),
  );
  assert.deepEqual(fixture.versions, versions);
  assert.deepEqual(fixture.workload, workload);
  const n = workload.matrixSize,
    input = values(n * n);
  const matrix = [];
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) sum += input[i * n + k] * input[j * n + k];
      matrix.push(sum);
    }
  assert.equal(digest(matrix), fixture.cases.matrix.digest);
  const sums = Array(workload.groups).fill(0);
  values(workload.rows).forEach((v, i) => {
    sums[i % workload.groups] += v;
  });
  const table = sums.flatMap((v, k) => [k, v]);
  for (const name of ["tabular-pandas", "tabular-polars"])
    assert.equal(digest(table), fixture.cases[name].digest);
  assert.equal(
    digest(values(workload.algorithmSize).map((x) => 2 * x * x + 3 * x + 1)),
    fixture.cases["algorithm-nts"].digest,
  );
});

test(
  "SIGTERM preserves an interrupted machine-readable native benchmark report",
  {
    timeout: 30000,
    skip:
      process.platform === "win32"
        ? "Windows process termination does not deliver POSIX SIGTERM"
        : false,
  },
  async () => {
    const { spawn } = await import("node:child_process");
    const { watch } = await import("node:fs");
    const directory = mkdtempSync(join(tmpdir(), "nmath-benchmark-interrupt-"));
    const output = join(directory, "report.json");
    let child;
    let sent = false;
    const watcher = watch(directory, (_, filename) => {
      if (filename === "report.json" && child && !sent) {
        sent = true;
        child.kill("SIGTERM");
      }
    });
    try {
      child = spawn(
        process.execPath,
        [
          fileURLToPath(
            new URL("../scripts/benchmarks/run.mjs", import.meta.url),
          ),
          "--output",
          output,
        ],
        { stdio: ["ignore", "pipe", "pipe"] },
      );
      let stderr = "";
      child.stderr.on("data", (data) => {
        stderr += data;
      });
      child.stdout.resume();
      const [code, signal] = await new Promise((resolve, reject) => {
        child.once("error", reject);
        child.once("exit", (code, signal) => resolve([code, signal]));
      });
      assert.equal(signal, null, stderr);
      assert.equal(code, 130, stderr);
      const result = JSON.parse(readFileSync(output, "utf8"));
      assert.equal(result.status, "interrupted");
      assert.ok(result.finishedAt);
      assert.deepEqual(readdirSync(directory), ["report.json"]);
    } finally {
      watcher.close();
      if (child && child.exitCode === null) child.kill("SIGKILL");
      rmSync(directory, { recursive: true, force: true });
    }
  },
);

test("GPU identity reports unavailable platforms explicitly without probing macOS tools", () => {
  for (const platform of ["linux", "win32"]) {
    const identity = gpuIdentity(platform, () => {
      throw new Error("macOS probe invoked");
    });
    assert.equal(identity.status, "unavailable");
    assert.equal(identity.platform, platform);
    assert.equal(identity.model, null);
    assert.equal(identity.driverVersion, null);
    assert.ok(identity.reason);
  }
  const identity = gpuIdentity("darwin", () =>
    JSON.stringify({ SPDisplaysDataType: [{ sppci_model: "fixture GPU" }] }),
  );
  assert.equal(identity[0].model, "fixture GPU");
});
