// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { createContext, Complex, slice } from "../dist/index.js";
import { polynomial, polyder } from "../dist/numpy.js";
const series = polynomial.polynomial.polyder;
const context = (budget = 1n << 22n) =>
  createContext({
    maxConcurrentJobs: 1,
    maxQueueLength: 4,
    cpuMemoryBudget: budget,
    gpuMemoryBudget: 0n,
  });

function unpack(c, value) {
  const bytes = Uint8Array.from(Buffer.from(value.hex, "hex"));
  if (value.dtype === "float16")
    return c.arrayFromFloat16Bits(new Uint16Array(bytes.buffer), {
      shape: value.shape,
    });
  const constructors = {
    int8: Int8Array,
    uint8: Uint8Array,
    int16: Int16Array,
    uint16: Uint16Array,
    int32: Int32Array,
    uint32: Uint32Array,
    int64: BigInt64Array,
    uint64: BigUint64Array,
    float32: Float32Array,
    float64: Float64Array,
    complex64: Float32Array,
    complex128: Float64Array,
  };
  return c.array(
    value.dtype === "bool"
      ? [...bytes].map(Boolean)
      : new constructors[value.dtype](bytes.buffer),
    { dtype: value.dtype, shape: value.shape },
  );
}

test("derivative APIs match fixed NumPy dtype, shape and coefficient fixtures", async () => {
  const fixture = JSON.parse(
    readFileSync(
      new URL(
        "../../../crates/nmath-core/tests/fixtures/polynomial/derivative-oracle.json",
        import.meta.url,
      ),
    ),
  );
  assert.equal(fixture.numpy, "2.5.4");
  const c = context();
  try {
    for (const [i, item] of fixture.cases.entries()) {
      const coefficients = unpack(c, item.c),
        expected = unpack(c, item.expected);
      const expression = item.ascending
        ? series(coefficients, {
            m: item.m,
            axis: item.axis,
            scl: item.complexScale ? new Complex(...item.scale) : item.scale[0],
          })
        : polyder(coefficients, item.m);
      assert.equal(expression.dtype, expected.dtype, `case ${i}`);
      assert.deepEqual(expression.shape, expected.shape, `case ${i}`);
      const out = await expression.compute();
      assert.deepEqual(out.toArray(), expected.toArray(), `case ${i}`);
      out.dispose();
      expected.dispose();
      coefficients.dispose();
    }
  } finally {
    await c.close();
  }
});

test("zero-order legacy derivatives alias arrays while ascending derivatives copy", async () => {
  const c = context();
  try {
    const a = c.array([1, 2, 3, 4], { dtype: "float32" });
    const reverse = a.get(slice(undefined, undefined, -1));
    const legacy = await polyder(reverse, 0).compute();
    const ascending = await series(reverse, { m: 0 }).compute();
    assert.equal(legacy.sharesStorage(a), true);
    assert.equal(ascending.sharesStorage(a), false);
    await a.set([0], 10);
    assert.deepEqual(legacy.toArray(), [4, 3, 2, 10]);
    assert.deepEqual(ascending.toArray(), [4, 3, 2, 1]);
    const expression = series(a, { m: 2 });
    const first = await expression.compute();
    await a.set([2], 20);
    const second = await expression.compute();
    assert.deepEqual(first.toArray(), [6, 24]);
    assert.deepEqual(second.toArray(), [40, 24]);
    first.dispose();
    second.dispose();
    legacy.dispose();
    ascending.dispose();
    const transposed = c
      .array([1, 2, 3, 4, 5, 6], { dtype: "int32", shape: [2, 3] })
      .transpose();
    const axis = await series(transposed, { axis: -2 }).compute();
    assert.deepEqual(axis.shape, [2, 2]);
    assert.deepEqual(axis.toArray(), [
      [2, 5],
      [6, 12],
    ]);
    axis.dispose();
  } finally {
    await c.close();
  }
});

test("derivative inputs freeze at execution start rather than while queued", async () => {
  const c = context(1n << 26n);
  try {
    const blocker = c.zeros([1000000], "float64");
    const coefficients = c.array([1, 2, 3], { dtype: "float32" });
    // FIFO admission orders the write before the derivative. The derivative
    // must observe the committed coefficients when it obtains its slot.
    const running = blocker.mul(blocker).compute();
    const write = coefficients.set([2], 9);
    const pending = series(coefficients).compute();
    (await running).dispose();
    await write;
    const out = await pending;
    assert.deepEqual(out.toArray(), [2, 18]);
    out.dispose();
  } finally {
    await c.close();
  }
});

test("derivative errors preserve budgets and reject invalid options and GPU admission", async () => {
  const c = context();
  try {
    const a = c.array([1, 2, 3], { dtype: "float32" });
    for (const m of [-1, 0.1, NaN, Infinity, 2 ** 53, "1", null]) {
      assert.throws(() => series(a, { m }), TypeError);
      assert.throws(() => polyder(a, m), TypeError);
    }
    for (const options of [
      null,
      [],
      { out: a },
      { axis: 0.5 },
      { axis: null },
      { scl: null },
      { scl: {} },
    ])
      assert.throws(() => series(a, options), TypeError);
    assert.throws(() => series(a, { axis: -2 }), {
      code: "NMATH_INDEX_INVALID",
    });
    assert.throws(() => series(a, { scl: new Complex(1) }), {
      code: "NMATH_DTYPE_UNSUPPORTED",
    });
    assert.throws(() => polyder(c.zeros([2, 2], "float32")), {
      code: "NMATH_SHAPE_MISMATCH",
    });
    assert.throws(() => series(c.zeros([3], "str")), {
      code: "NMATH_DTYPE_UNSUPPORTED",
    });
    const before = c.metrics().executions.length;
    await assert.rejects(series(a).compute({ device: "gpu" }), {
      code: "NMATH_GPU_UNSUPPORTED_OPERATION",
    });
    assert.equal(c.metrics().executions.length, before);
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(series(a).compute({ signal: controller.signal }), {
      code: "NMATH_CANCELLED",
    });
    const out = await series(a, { m: Number.MAX_SAFE_INTEGER }).compute();
    assert.deepEqual(out.toArray(), [0]);
    out.dispose();
  } finally {
    await c.close();
  }
  const tiny = context(40n);
  try {
    const a = tiny.array([1, 2, 3], { dtype: "int32" });
    const used = tiny.metrics().cpuMemory.used;
    await assert.rejects(series(a).compute(), {
      code: "NMATH_MEMORY_BUDGET_EXCEEDED",
    });
    assert.equal(tiny.metrics().cpuMemory.used, used);
  } finally {
    await tiny.close();
  }
});
