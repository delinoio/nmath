// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  add,
  createContext,
  ERROR_CODES,
  indices,
  isNMath,
  mask,
  NMathArray,
  NMathError,
  slice,
  sub,
} from "../dist/index.js";
import { context } from "./helpers.mjs";

async function rejectsWith(promise, code) {
  await assert.rejects(promise, (error) => {
    assert.ok(error instanceof NMathError, `expected NMathError, got ${error}`);
    assert.equal(error.code, code);
    return true;
  });
}

function throwsWith(fn, code) {
  assert.throws(
    fn,
    (error) => error instanceof NMathError && error.code === code,
  );
}

test("contexts require every limit", () => {
  throwsWith(() => createContext(), "NMATH_CONTEXT_INVALID_OPTIONS");
  throwsWith(
    () =>
      createContext({
        maxConcurrentJobs: 1,
        maxQueueLength: 1,
        cpuMemoryBudget: 1,
      }),
    "NMATH_CONTEXT_INVALID_OPTIONS",
  );
  throwsWith(
    () => context({ maxConcurrentJobs: 0 }),
    "NMATH_CONTEXT_INVALID_OPTIONS",
  );
});

test("lazy arithmetic computes off the calling thread", async () => {
  const ctx = context();
  const a = ctx.array(
    [
      [1, 2],
      [3, 4],
    ],
    { dtype: "float64" },
  );
  const b = ctx.array([10, 20], { dtype: "float64" });
  const expr = a.mul(b).add(1).neg();
  assert.ok(isNMath(expr));
  assert.deepEqual(expr.shape, [2, 2]);
  const result = await expr.compute();
  assert.ok(result instanceof NMathArray);
  assert.deepEqual(result.toArray(), [
    [-11, -41],
    [-31, -81],
  ]);
  assert.equal(await a.sum().compute(), 10);
  assert.deepEqual((await a.mean(0).compute()).toArray(), [2, 3]);
  assert.deepEqual((await a.matmul(b).compute()).toArray(), [50, 110]);
  await ctx.close();
});

test("functional forms accept scalars on either side", async () => {
  const ctx = context();
  const a = ctx.array([1, 2], { dtype: "float32" });
  assert.deepEqual((await sub(10, a).compute()).toArray(), [9, 8]);
  assert.deepEqual((await add(a, 0.5).compute()).toArray(), [1.5, 2.5]);
  assert.equal(sub(10, a).dtype, "float32");
  await ctx.close();
});

test("64-bit integers stay exact as bigint", async () => {
  const ctx = context();
  const big = 2n ** 62n + 1n;
  const a = ctx.array([big, 1n], { dtype: "int64" });
  assert.equal(a.item(0), big);
  assert.equal(await a.sum().compute(), big + 1n);
  assert.equal(a.add(1n).dtype, "int64");
  assert.equal(a.add(1.5).dtype, "float64");
  assert.equal(a.div(2n).dtype, "float64");
  await ctx.close();
});

test("bigint scalars outside the signed 64-bit range are rejected", async () => {
  const ctx = context();
  const a = ctx.array([0n], { dtype: "int64" });
  assert.deepEqual((await a.add(-1n).compute()).toArray(), [-1n]);
  assert.deepEqual((await a.add(-(2n ** 63n)).compute()).toArray(), [
    -(2n ** 63n),
  ]);
  throwsWith(() => a.add(-(2n ** 63n) - 1n), "NMATH_DTYPE_UNSUPPORTED");
  throwsWith(() => a.add(-(2n ** 64n) + 1n), "NMATH_DTYPE_UNSUPPORTED");
  await ctx.close();
});

test("dtype promotion and shape errors are reported at construction", () => {
  const ctx = context();
  const i32 = ctx.array([1, 2, 3], { dtype: "int32" });
  const f32 = ctx.array([1, 2, 3], { dtype: "float32" });
  assert.equal(i32.add(f32).dtype, "float64");
  throwsWith(
    () => i32.add(ctx.array([1, 2], { dtype: "int32" })),
    "NMATH_SHAPE_BROADCAST",
  );
  throwsWith(() => i32.get(5), "NMATH_INDEX_OUT_OF_BOUNDS");
});

test("indexing reads are lazy; basic reads return shared views", async () => {
  const ctx = context();
  const a = ctx.array(
    [
      [1, 2, 3],
      [4, 5, 6],
    ],
    { dtype: "int32" },
  );
  assert.equal(await a.get(1, -1).compute(), 6);
  const row = await a.get(1).compute();
  assert.ok(row.sharesStorage(a));
  assert.deepEqual(
    (await a.get(slice(undefined, undefined, -1), 0).compute()).toArray(),
    [4, 1],
  );
  const picked = await a.get(indices([1, 0]), 2).compute();
  assert.deepEqual(picked.toArray(), [6, 3]);
  assert.equal(picked.sharesStorage(a), false);
  assert.deepEqual((await a.get(mask([false, true])).compute()).toArray(), [
    [4, 5, 6],
  ]);
  await a.set([1, 1], 50);
  assert.deepEqual(row.toArray(), [4, 50, 6]);
  await ctx.close();
});

test("writes are atomic and failed writes keep previous values", async () => {
  const ctx = context();
  const a = ctx.array([1, 2, 3], { dtype: "float64" });
  await rejectsWith(
    a.set([], ctx.array([1, 2], { dtype: "float64" })),
    "NMATH_SHAPE_BROADCAST",
  );
  assert.deepEqual(a.toArray(), [1, 2, 3]);
  const writes = [a.set([], 7), a.set([0], 8), a.set([], 9)];
  await Promise.all(writes);
  assert.deepEqual(a.toArray(), [9, 9, 9]);
  await ctx.close();
});

test("inputs freeze when the job starts and results are never cached", async () => {
  const ctx = context({ maxConcurrentJobs: 1 });
  const a = ctx.array([1, 1], { dtype: "float64" });
  const expr = a.mul(2);
  const first = await expr.compute();
  await a.set([0], 5);
  const second = await expr.compute();
  assert.deepEqual(first.toArray(), [2, 2]);
  assert.deepEqual(second.toArray(), [10, 2]);
  assert.notEqual(first, second);
  await ctx.close();
});

test("abort signals cancel queued work and full queues reject", async () => {
  const ctx = context({ maxConcurrentJobs: 1, maxQueueLength: 1 });
  const big = ctx.zeros([512, 512], "float64");
  const running = big.matmul(big).compute();
  const controller = new AbortController();
  const queued = big.add(1).compute({ signal: controller.signal });
  await rejectsWith(big.add(2).compute(), "NMATH_SCHED_QUEUE_FULL");
  controller.abort();
  await rejectsWith(queued, "NMATH_CANCELLED");
  await running;
  const aborted = new AbortController();
  aborted.abort();
  await rejectsWith(
    big.add(1).compute({ signal: aborted.signal }),
    "NMATH_CANCELLED",
  );
  const metrics = ctx.metrics();
  assert.equal(metrics.totalRejected, 1);
  assert.ok(metrics.executions.some((e) => e.status === "cancelled"));
  assert.ok(
    metrics.executions.some(
      (e) => e.operation === "matmul" && e.status === "succeeded",
    ),
  );
  await ctx.close();
});

test("close cancels pending work and rejects new work", async () => {
  const ctx = context({ maxConcurrentJobs: 1 });
  const big = ctx.zeros([400, 400], "float64");
  const pending = [big.matmul(big).compute(), big.matmul(big).compute()];
  const outcomes = Promise.allSettled(pending);
  await ctx.close();
  const settled = await outcomes;
  assert.ok(
    settled.some(
      (s) => s.status === "rejected" && s.reason.code === "NMATH_CANCELLED",
    ),
  );
  assert.equal(ctx.closed, true);
  await rejectsWith(big.add(1).compute(), "NMATH_CONTEXT_CLOSED");
});

test("memory budgets reject before allocation", async () => {
  const ctx = context({ cpuMemoryBudget: 64 });
  const a = ctx.array([1, 2, 3, 4], { dtype: "float64" });
  throwsWith(() => ctx.zeros([16], "float64"), "NMATH_MEMORY_BUDGET_EXCEEDED");
  await rejectsWith(a.add(a).compute(), "NMATH_MEMORY_BUDGET_EXCEEDED");
  assert.equal(ctx.metrics().cpuMemory.used, 32n);
  await ctx.close();
});

test("contexts do not mix; copies are explicit", async () => {
  const one = context();
  const two = context();
  const a = one.array([1, 2], { dtype: "int32" });
  const b = two.array([1, 2], { dtype: "int32" });
  throwsWith(() => a.add(b), "NMATH_CONTEXT_MISMATCH");
  const copy = await two.copy(a);
  assert.equal(copy.context, two);
  assert.deepEqual((await copy.add(b).compute()).toArray(), [2, 4]);
  await Promise.all([one.close(), two.close()]);
});

test("disposed arrays reject access but keep views alive", async () => {
  const ctx = context();
  const a = ctx.array([1, 2, 3], { dtype: "float64" });
  const view = await a.get(slice(1)).compute();
  {
    using b = ctx.array([1], { dtype: "float64" });
    assert.equal(b.disposed, false);
  }
  a.dispose();
  throwsWith(() => a.shape, "NMATH_DISPOSED");
  assert.deepEqual(view.toArray(), [2, 3]);
  await ctx.close();
});

test("random generation matches NumPy 2.5.4", async () => {
  const ctx = context();
  const rng = ctx.generator(42);
  const values = await rng.random([3]).compute();
  // numpy.random.default_rng(42).random(3)
  assert.deepEqual(
    values.toArray(),
    [0.7739560485559633, 0.4388784397520523, 0.8585979199113825],
  );
  const next = await rng.random([1]).compute();
  assert.notEqual(next.item(0), 0.7739560485559633);
  await ctx.close();
});

test("GPU execution fails without CPU fallback", async () => {
  const ctx = context();
  const f64 = ctx.array([1], { dtype: "float64" });
  await rejectsWith(
    f64.add(1).compute({ device: "gpu" }),
    "NMATH_GPU_UNSUPPORTED_OPERATION",
  );
  const f32 = ctx.array([1], { dtype: "float32" });
  await rejectsWith(
    f32.add(1).compute({ device: "gpu" }),
    "NMATH_GPU_UNAVAILABLE",
  );
  await ctx.close();
});

test("NPY files round-trip atomically without overwriting", async () => {
  const dir = mkdtempSync(join(tmpdir(), "nmath-test-"));
  try {
    const ctx = context();
    const path = join(dir, "a.npy");
    const a = ctx.array(
      [
        [1, 2],
        [3, 4],
      ],
      { dtype: "int32" },
    );
    await a.save(path);
    await rejectsWith(a.save(path), "NMATH_IO_EXISTS");
    await a
      .mul(2n)
      .compute()
      .then((b) => b.save(path, { overwrite: true }));
    const loaded = await ctx.load(path);
    assert.deepEqual(loaded.toArray(), [
      [2, 4],
      [6, 8],
    ]);
    assert.deepEqual(readdirSync(dir), ["a.npy"]);
    await rejectsWith(ctx.load(join(dir, "missing.npy")), "NMATH_IO_FAILED");
    await ctx.close();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("error codes are stable and messages carry recovery guidance", async () => {
  assert.ok(ERROR_CODES.includes("NMATH_SCHED_QUEUE_FULL"));
  const ctx = context();
  try {
    ctx.array([1], { dtype: "float64" }).get(3);
  } catch (error) {
    assert.match(
      error.message,
      /^NMATH_INDEX_OUT_OF_BOUNDS: .*Use an index inside/,
    );
  }
  await ctx.close();
});

test("CommonJS consumers can load the package with dynamic import", async () => {
  const { createRequire } = await import("node:module");
  const require = createRequire(import.meta.url);
  const vm = require("node:vm");
  assert.ok(vm);
  const mod = await import("../dist/index.js");
  assert.equal(typeof mod.createContext, "function");
});
