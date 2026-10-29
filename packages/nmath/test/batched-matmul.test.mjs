// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { createContext, Complex, slice } from "../dist/index.js";
import { matmul } from "../dist/numpy.js";
const fixtures = JSON.parse(
  readFileSync(
    new URL("fixtures/batched-matmul.json", import.meta.url),
    "utf8",
  ),
);
const context = (budget = 1n << 24n) =>
  createContext({
    maxConcurrentJobs: 1,
    maxQueueLength: 4,
    cpuMemoryBudget: budget,
    gpuMemoryBudget: 0n,
  });
function unpack(c, value) {
  let data = value.values;
  if (["int64", "uint64"].includes(value.dtype)) data = data.map(BigInt);
  if (value.dtype === "complex64") data = new Float32Array(data);
  if (value.dtype === "complex128") data = new Float64Array(data);
  return c.array(data, { dtype: value.dtype, shape: value.shape });
}
function values(result, dtype) {
  let data =
    result && typeof result.dispose === "function"
      ? [...result.toTypedArray()]
      : result instanceof Complex
        ? [result.real, result.imag]
        : [result];
  if (dtype === "bool") data = data.map(Boolean);
  if (["int64", "uint64"].includes(dtype)) data = data.map(String);
  return data;
}
test("batched matmul agrees exactly with fixed NumPy small-value fixtures", async () => {
  assert.equal(fixtures.numpy, "2.5.4");
  assert.equal(fixtures.criterion, "exact-small-values");
  const c = context();
  try {
    for (const fixture of fixtures.cases) {
      const a = unpack(c, fixture.a),
        b = unpack(c, fixture.b);
      const expr = matmul(a, b);
      assert.deepEqual(expr.shape, fixture.expected.shape, fixture.id);
      assert.equal(expr.dtype, fixture.expected.dtype, fixture.id);
      const result = await expr.compute();
      assert.deepEqual(
        values(result, expr.dtype),
        fixture.expected.values,
        fixture.id,
      );
      result?.dispose?.();
      a.dispose();
      b.dispose();
    }
  } finally {
    await c.close();
  }
});

test("batched matmul snapshots strided views for each independent execution", async () => {
  const c = context();
  try {
    const a = c.array([1, 2, 3, 4, 5, 6, 7, 8], {
      dtype: "int32",
      shape: [2, 2, 2],
    });
    const b = c.array([1, 2], { dtype: "int32" });
    const expr = matmul(a.get(slice(undefined, undefined, -1)), b);
    const before = await expr.compute();
    assert.deepEqual(before.toArray(), [
      [17, 23],
      [5, 11],
    ]);
    await a.set([1, 0, 0], 10);
    const [first, second] = await Promise.all([expr.compute(), expr.compute()]);
    assert.deepEqual(first.toArray(), [
      [22, 23],
      [5, 11],
    ]);
    assert.deepEqual(second.toArray(), first.toArray());
    assert.equal(first.sharesStorage(second), false);
    assert.deepEqual(before.toArray(), [
      [17, 23],
      [5, 11],
    ]);
    for (const result of [before, first, second]) result.dispose();
    const cancel = new AbortController();
    cancel.abort();
    await assert.rejects(expr.compute({ signal: cancel.signal }), {
      code: "NMATH_CANCELLED",
    });
  } finally {
    await c.close();
  }
});

test("matmul rejects invalid dimensions and cross-context operands before execution", async () => {
  const c = context(),
    other = context();
  try {
    const a = c.zeros([2, 3, 4], "float32");
    assert.throws(() => matmul(a, c.zeros([3, 4, 5], "float32")), {
      code: "NMATH_SHAPE_BROADCAST",
    });
    assert.throws(() => matmul(a, c.zeros([2, 5, 6], "float32")), {
      code: "NMATH_SHAPE_MISMATCH",
    });
    assert.throws(() => matmul(a, c.zeros([], "float32")), {
      code: "NMATH_SHAPE_MISMATCH",
    });
    assert.throws(() => matmul(a, other.zeros([4, 5], "float32")), {
      code: "NMATH_CONTEXT_MISMATCH",
    });
    assert.throws(
      () => matmul(c.array(["x"], { dtype: "str" }), c.zeros([1], "float32")),
      { code: "NMATH_DTYPE_UNSUPPORTED" },
    );
  } finally {
    await c.close();
    await other.close();
  }
});

test("matmul rejects output over budget and retains input storage", async () => {
  const c = context(96n);
  try {
    const a = c.zeros([2, 1, 2, 1], "float32"),
      b = c.zeros([8, 1, 2], "float32");
    await assert.rejects(matmul(a, b).compute(), {
      code: "NMATH_MEMORY_BUDGET_EXCEEDED",
    });
    assert.deepEqual(a.toArray(), [[[[0], [0]]], [[[0], [0]]]]);
  } finally {
    await c.close();
  }
});
