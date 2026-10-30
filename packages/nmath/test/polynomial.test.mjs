// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { createContext, Complex } from "../dist/index.js";
import { polynomial, polyval } from "../dist/numpy.js";
const series = polynomial.polynomial.polyval;
const context = (budget = 1n << 22n) =>
  createContext({
    maxConcurrentJobs: 1,
    maxQueueLength: 4,
    cpuMemoryBudget: budget,
    gpuMemoryBudget: 0n,
  });
function unpack(c, value) {
  const bytes = Uint8Array.from(Buffer.from(value.hex, "hex"));
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
  if (value.dtype === "float16")
    return c.arrayFromFloat16Bits(new Uint16Array(bytes.buffer), {
      shape: value.shape,
    });
  const values =
    value.dtype === "bool"
      ? [...bytes].map(Boolean)
      : new constructors[value.dtype](bytes.buffer);
  return c.array(values, { dtype: value.dtype, shape: value.shape });
}
test("polynomial native API matches the fixed NumPy dtype and shape matrix", async () => {
  const c = context();
  const fixture = JSON.parse(
    readFileSync(
      new URL(
        "../../../crates/nmath-core/tests/fixtures/polynomial/oracle.json",
        import.meta.url,
      ),
    ),
  );
  assert.equal(fixture.numpy, "2.5.4");
  try {
    for (const [i, item] of fixture.cases.entries()) {
      const coefficients = unpack(c, item.c),
        points = unpack(c, item.x),
        expected = unpack(c, item.expected);
      const expression = item.ascending
        ? series(points, coefficients, { tensor: item.tensor })
        : polyval(coefficients, points);
      assert.equal(expression.dtype, expected.dtype, `case ${i}`);
      assert.deepEqual(expression.shape, expected.shape, `case ${i}`);
      const result = await expression.compute();
      assert.deepEqual(
        result?.toArray ? result.toArray() : result,
        expected.shape.length ? expected.toArray() : expected.item(),
        `case ${i}`,
      );
      if (result?.dispose) result.dispose();
      coefficients.dispose();
      points.dispose();
      expected.dispose();
    }
  } finally {
    await c.close();
  }
});
test("polynomial expressions execute independently and preserve integer precision", async () => {
  const c = context();
  try {
    const coefficients = c.array([2n ** 60n + 1n, 3n], { dtype: "int64" });
    const points = c.array([1n, 2n], { dtype: "int64" });
    const expression = polyval(coefficients, points);
    const first = await expression.compute();
    assert.deepEqual(first.toArray(), [2n ** 60n + 4n, 2n ** 61n + 5n]);
    assert.equal(first.sharesStorage(points), false);
    await points.set([0], 3n);
    const second = await expression.compute();
    assert.deepEqual(second.toArray(), [
      3n * (2n ** 60n + 1n) + 3n,
      2n ** 61n + 5n,
    ]);
    assert.deepEqual(first.toArray(), [2n ** 60n + 4n, 2n ** 61n + 5n]);
    first.dispose();
    second.dispose();
    const complex = c.array([new Complex(1, 2)], { dtype: "complex128" });
    const infinity = c.array([Infinity], { dtype: "float64" });
    const out = await series(infinity, complex).compute();
    assert.ok(Number.isNaN(out.item(0).real));
    assert.equal(out.item(0).imag, 2);
    out.dispose();
  } finally {
    await c.close();
  }
});
test("polynomial errors reject invalid shapes, options, contexts and GPU work", async () => {
  const c = context(),
    foreign = context();
  try {
    const x = c.zeros([2], "float32"),
      coefficients = c.zeros([3, 2], "float32");
    assert.throws(
      () => series(x, coefficients, { tensor: "false" }),
      TypeError,
    );
    assert.throws(() => series(x, coefficients, { out: x }), TypeError);
    assert.throws(() => series(x, c.zeros([0], "float32")), {
      code: "NMATH_INDEX_OUT_OF_BOUNDS",
    });
    assert.throws(() => polyval(c.zeros([], "float32"), x), {
      code: "NMATH_SHAPE_MISMATCH",
    });
    assert.throws(
      () => series(c.zeros([4], "float32"), coefficients, { tensor: false }),
      { code: "NMATH_SHAPE_BROADCAST" },
    );
    assert.throws(() => polyval(coefficients, foreign.zeros([2], "float32")), {
      code: "NMATH_CONTEXT_MISMATCH",
    });
    const count = c.metrics().executions.length;
    await assert.rejects(series(x, coefficients).compute({ device: "gpu" }), {
      code: "NMATH_GPU_UNSUPPORTED_OPERATION",
    });
    assert.equal(c.metrics().executions.length, count);
    const abort = new AbortController();
    abort.abort();
    await assert.rejects(
      series(x, coefficients).compute({ signal: abort.signal }),
      { code: "NMATH_CANCELLED" },
    );
  } finally {
    await c.close();
    await foreign.close();
  }
  const tiny = context(32n);
  try {
    const coefficients = tiny.zeros([3], "float32"),
      x = tiny.zeros([2], "float32");
    const used = tiny.metrics().cpuMemory.used;
    await assert.rejects(polyval(coefficients, x).compute(), {
      code: "NMATH_MEMORY_BUDGET_EXCEEDED",
    });
    assert.equal(tiny.metrics().cpuMemory.used, used);
  } finally {
    await tiny.close();
  }
});
