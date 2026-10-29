// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { cpus, release, totalmem, tmpdir } from "node:os";
import {
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { performance } from "node:perf_hooks";
import {
  aggregate,
  gpuIdentity,
  atomicReport,
  digest,
  hashFile,
  parseArgs,
  values,
  versions,
  workload,
} from "./common.mjs";

const options = parseArgs(process.argv.slice(2));
const packageRoot = fileURLToPath(new URL("../../", import.meta.url));
const root = fileURLToPath(new URL("../../../../", import.meta.url));
const fixturePath = join(
  packageRoot,
  "test/fixtures/benchmarks/originals.json",
);
const fixture = JSON.parse(readFileSync(fixturePath, "utf8"));
assert.deepEqual(fixture.versions, versions);
assert.deepEqual(fixture.workload, workload);
assert.equal(
  process.versions.node.split(".")[0],
  "24",
  "Benchmarks require Node.js 24",
);
const command = (program, args) => {
  try {
    return execFileSync(program, args, { cwd: root, encoding: "utf8" }).trim();
  } catch (error) {
    return { unavailable: true, error: error.message };
  }
};
const report = {
  schemaVersion: 1,
  status: "running",
  startedAt: new Date().toISOString(),
  command: [process.execPath, ...process.argv.slice(1)],
  options,
  versions,
  workload,
  sourceRevision: command("git", ["rev-parse", "HEAD"]),
  workingTree: command("git", ["status", "--porcelain"]),
  sourceHashes: Object.fromEntries(
    ["run.mjs", "common.mjs", "originals.py"].map((name) => [
      name,
      hashFile(new URL(name, import.meta.url)),
    ]),
  ),
  fixtureSha256: hashFile(fixturePath),
  algorithmSha256: hashFile(
    join(packageRoot, "test/fixtures/benchmarks/horner.nts"),
  ),
  environment: {
    node: process.version,
    v8: process.versions.v8,
    platform: process.platform,
    arch: process.arch,
    osRelease: release(),
    cpuModel: cpus()[0]?.model,
    logicalCpus: cpus().length,
    totalMemoryBytes: totalmem(),
    rust: command("rustc", ["--version"]),
    pnpm: command("pnpm", ["--version"]),
    osVersion:
      process.platform === "darwin" ? command("sw_vers", []) : release(),
  },
  criterion: fixture.criterion,
  results: [],
  limits: [
    "Synthetic bounded integer-valued float32 and int64 data only; no general numerical or platform acceptance.",
    "Known GPU NumPy power discrepancies (84 one-ULP differences) remain unaccepted; this harness does not test power.",
    "Compute wall time includes JS/native dispatch, queue, snapshots and result wrapping. Native execution metrics are not pure kernel timings.",
    "Boundary residual is wall minus reported queue/execution time; it is not an isolated N-API timer and can include scheduler/clock overhead.",
    "RSS is sampled at phase boundaries; maxRSS is process lifetime high-water mark. Managed peaks are context cumulative peaks, not process peaks.",
    "No isolated GPU upload/download timer is exposed. GPU samples include transfers; identity probes also include scheduling, allocation and kernel work.",
    "Original timings are separate development-only reports. No speedup is inferred automatically.",
    "Build and process launch are outside compute samples. Import, fixture setup, compilation and first execution are reported separately.",
  ],
};
const controller = new AbortController();
for (const signal of ["SIGINT", "SIGTERM"])
  process.once(signal, () => controller.abort(new Error(signal)));
const checkAbort = () => controller.signal.throwIfAborted();
let temporary;
let ctx;
const memory = () => ({
  process: process.memoryUsage(),
  maxRssBytes: process.resourceUsage().maxRSS * 1024,
  managed: ctx?.metrics(),
});
function disposeFrame(frame) {
  for (const name of frame.columns) {
    const column = frame.getColumn ? frame.getColumn(name) : frame.get(name);
    column.values.dispose();
    column.validity?.dispose();
  }
}
function extractFrame(frame) {
  assert.deepEqual(frame.columns, ["k", "v"]);
  const columns = frame.columns.map((name) =>
    frame.getColumn ? frame.getColumn(name) : frame.get(name),
  );
  for (const col of columns) {
    assert.equal(col.values.dtype, "int64");
    if (col.validity)
      assert.ok(Array.from(col.validity.toTypedArray()).every(Boolean));
  }
  const arrays = columns.map((c) => Array.from(c.values.toTypedArray()));
  const rows = arrays[0].map((k, i) => [k, arrays[1][i]]);
  return { shape: [rows.length, 2], dtype: "int64", values: rows.flat() };
}
async function measure(name, run, extract, dispose, expected) {
  checkAbort();
  const entry = {
    name,
    device: name.startsWith("tabular") ? "cpu" : options.device,
    status: "checking",
    samples: [],
    memoryBefore: memory(),
  };
  report.results.push(entry);
  atomicReport(options.output, report);
  async function once() {
    checkAbort();
    const start = performance.now();
    const result = await run();
    const wallMs = performance.now() - start;
    try {
      const native = ctx.metrics().executions.at(-1);
      assert.equal(native?.status, "succeeded");
      const exportStart = performance.now();
      const output = extract(result);
      const exportMs = performance.now() - exportStart;
      assert.deepEqual(output.shape, expected.shape);
      assert.equal(output.dtype, expected.dtype);
      assert.equal(
        digest(output.values),
        expected.digest,
        `${name}: full-output digest mismatch`,
      );
      return {
        wallMs,
        exportMs,
        queueMs: native.queueTimeMs,
        nativeExecutionMs: native.executionTimeMs,
        boundaryResidualMs:
          wallMs - native.queueTimeMs - native.executionTimeMs,
      };
    } finally {
      dispose(result);
    }
  }
  entry.firstExecution = await once();
  entry.correctness = "passed-exact-full-output";
  entry.status = "warming";
  for (let i = 0; i < options.warmup; i++) await once();
  entry.status = "measuring";
  for (let i = 0; i < options.samples; i++) {
    entry.samples.push(await once());
    atomicReport(options.output, report);
  }
  entry.summaryMs = Object.fromEntries(
    ["wallMs", "exportMs", "queueMs", "nativeExecutionMs"].map((key) => [
      key,
      aggregate(entry.samples.map((s) => s[key])),
    ]),
  );
  // Preserve signed residual samples: clock resolution can produce a negative residual.
  entry.memoryAfter = memory();
  entry.status = "completed";
  atomicReport(options.output, report);
}
try {
  atomicReport(options.output, report);
  report.binaryHashes = Object.fromEntries(
    readdirSync(join(packageRoot, "native"))
      .filter(
        (name) => name.endsWith(".node") || name.startsWith("nmath-compiler"),
      )
      .map((name) => [name, hashFile(join(packageRoot, "native", name))]),
  );
  if (options.device === "gpu") {
    report.environment.gpu = gpuIdentity(process.platform, () =>
      command("system_profiler", ["SPDisplaysDataType", "-json"]),
    );
  }
  const importStart = performance.now();
  const [{ createContext }, pl, compiler] = await Promise.all([
    import("../../dist/index.js"),
    import("../../dist/polars.js"),
    import("../../dist/compiler.js"),
  ]);
  report.importMs = performance.now() - importStart;
  temporary = mkdtempSync(join(tmpdir(), "nmath-benchmark-"));
  mkdirSync(join(temporary, "node_modules/@delinoio"), { recursive: true });
  symlinkSync(
    packageRoot,
    join(temporary, "node_modules/@delinoio/nmath"),
    process.platform === "win32" ? "junction" : "dir",
  );
  mkdirSync(join(temporary, "node_modules/@types"));
  symlinkSync(
    join(packageRoot, "node_modules/@types/node"),
    join(temporary, "node_modules/@types/node"),
    process.platform === "win32" ? "junction" : "dir",
  );
  writeFileSync(join(temporary, "package.json"), '{"type":"module"}');
  copyFileSync(
    join(packageRoot, "test/fixtures/benchmarks/horner.nts"),
    join(temporary, "horner.nts"),
  );
  writeFileSync(
    join(temporary, "tsconfig.json"),
    JSON.stringify({
      compilerOptions: {
        strict: true,
        target: "es2024",
        module: "nodenext",
        moduleResolution: "nodenext",
        outDir: "out",
        types: ["node"],
        lib: ["es2024", "esnext.disposable"],
      },
      include: ["*.nts"],
    }),
  );
  const compileStart = performance.now();
  const built = await compiler.build({
    tsconfig: join(temporary, "tsconfig.json"),
    signal: controller.signal,
  });
  report.compilationMs = performance.now() - compileStart;
  assert.equal(built.ok, true, JSON.stringify(built.diagnostics));
  const emitted = readFileSync(join(temporary, "out/horner.js"), "utf8");
  assert.match(
    emitted,
    /nts-runtime/,
    "Algorithm must execute compiler-lowered NMath operators",
  );
  report.emittedAlgorithmSha256 = hashFile(join(temporary, "out/horner.js"));
  const { polynomial } = await import(
    pathToFileURL(join(temporary, "out/horner.js")).href
  );
  checkAbort();
  const setupStart = performance.now();
  ctx = createContext({
    maxConcurrentJobs: 1,
    maxQueueLength: 8,
    cpuMemoryBudget: 1n << 30n,
    gpuMemoryBudget: options.device === "gpu" ? 1n << 30n : 0n,
  });
  const { matrixSize: n, rows, groups, algorithmSize } = workload;
  const input = values(n * n);
  const transposed = input.map(
    (_, i) => input[(i % n) * n + Math.floor(i / n)],
  );
  const ingress = performance.now();
  const a = ctx.array(new Float32Array(input), {
    dtype: "float32",
    shape: [n, n],
  });
  const b = ctx.array(new Float32Array(transposed), {
    dtype: "float32",
    shape: [n, n],
  });
  const x = ctx.array(new Float32Array(values(algorithmSize)), {
    dtype: "float32",
  });
  const k = ctx.array(
    Array.from({ length: rows }, (_, i) => BigInt(i % groups)),
    { dtype: "int64" },
  );
  const v = ctx.array(values(rows).map(BigInt), { dtype: "int64" });
  report.inputConstructionMs = performance.now() - ingress;
  const frame = new pl.DataFrame([
    new pl.Series("k", k),
    new pl.Series("v", v),
  ]);
  const pandas = frame.toPandas();
  const matrix = a.matmul(b),
    algorithm = polynomial(x);
  const lazy = frame.lazy().groupBy("k", "v", "sum").sort("k");
  const computeOptions = { device: options.device, signal: controller.signal };
  report.setupMs = performance.now() - setupStart;
  const extract = (result) => ({
    shape: result.shape,
    dtype: result.dtype,
    values: Array.from(result.toTypedArray()),
  });
  const dispose = (result) => result.dispose();
  const scalar = ctx.array(new Float32Array([1]), { dtype: "float32" });
  const identity = scalar.pos();
  await measure(
    "boundary-probe",
    () => identity.compute(computeOptions),
    extract,
    dispose,
    { shape: [1], dtype: "float32", digest: digest([1]) },
  );
  if (options.device === "gpu") {
    report.gpuTransfer = {
      isolatedUploadMs: null,
      isolatedDownloadMs: null,
      reason:
        "Public metrics expose no transfer timestamps; identity probe is transfer-inclusive, not a pure transfer measurement.",
      inputBytes: algorithmSize * 4,
      outputBytes: algorithmSize * 4,
    };
    await measure(
      "gpu-transfer-inclusive-probe",
      () => x.pos().compute(computeOptions),
      extract,
      dispose,
      {
        shape: [algorithmSize],
        dtype: "float32",
        digest: digest(values(algorithmSize)),
      },
    );
  }
  await measure(
    "matrix",
    () => matrix.compute(computeOptions),
    extract,
    dispose,
    fixture.cases.matrix,
  );
  await measure(
    "tabular-pandas",
    () => pandas.groupBy("k", "v", "sum", { signal: controller.signal }),
    extractFrame,
    disposeFrame,
    fixture.cases["tabular-pandas"],
  );
  await measure(
    "tabular-polars",
    () => lazy.collect({ signal: controller.signal }),
    extractFrame,
    disposeFrame,
    fixture.cases["tabular-polars"],
  );
  await measure(
    "algorithm-nts",
    () => algorithm.compute(computeOptions),
    extract,
    dispose,
    fixture.cases["algorithm-nts"],
  );
  for (const array of [a, b, x, k, v, scalar]) array.dispose();
  report.status = "completed";
} catch (error) {
  report.status = controller.signal.aborted ? "interrupted" : "failed";
  report.error = {
    name: error.name,
    code: error.code,
    message: error.message,
    stack: error.stack,
  };
  for (const entry of report.results)
    if (entry.status !== "completed") {
      entry.status = report.status;
      entry.error = report.error;
    }
  process.exitCode = controller.signal.aborted ? 130 : 1;
} finally {
  try {
    if (ctx) {
      report.finalMemory = memory();
      await ctx.close();
    }
  } catch (error) {
    report.cleanupError = { name: error.name, message: error.message };
    report.status = "failed";
    process.exitCode = 1;
  } finally {
    if (temporary) rmSync(temporary, { recursive: true, force: true });
  }
  report.finishedAt = new Date().toISOString();
  atomicReport(options.output, report);
  console.log(
    JSON.stringify({
      status: report.status,
      output: options.output,
      completed: report.results
        .filter((r) => r.status === "completed")
        .map((r) => r.name),
      error: report.error,
    }),
  );
}
