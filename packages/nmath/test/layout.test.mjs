// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { Complex, Temporal, slice } from "../dist/index.js";
import { reshape, transpose } from "../dist/numpy.js";
import { context } from "./helpers.mjs";
const fixtures = JSON.parse(
  readFileSync(new URL("fixtures/layout.json", import.meta.url), "utf8"),
);
function unpack(c, input) {
  let values = input.values;
  if (["int64", "uint64"].includes(input.dtype)) values = values.map(BigInt);
  if (input.dtype === "complex64") values = new Float32Array(values);
  if (input.dtype === "complex128") values = new Float64Array(values);
  return c.array(values, { dtype: input.dtype, shape: input.shape });
}
function packed(result, dtype) {
  let values = result?.toTypedArray
    ? [...result.toTypedArray()]
    : result instanceof Complex
      ? [result.real, result.imag]
      : [result];
  if (dtype === "bool") values = values.map(Boolean);
  if (["int64", "uint64"].includes(dtype)) values = values.map(String);
  return values;
}
function expression(a, operations) {
  return operations.reduce(
    (expr, op) =>
      op.kind === "transpose"
        ? transpose(expr, op.axes)
        : op.kind === "slice"
          ? expr.get(
              ...op.selectors.map((s) =>
                slice(...s.map((v) => v ?? undefined)),
              ),
            )
          : reshape(expr, op.shape, { copy: op.copy }),
    a,
  );
}
test("shape transforms match exact NumPy values, dtype, shape and storage decisions", async () => {
  assert.equal(fixtures.numpy, "2.5.4");
  const c = context();
  try {
    for (const fixture of fixtures.cases) {
      const a = unpack(c, fixture.input);
      const expr = expression(a, fixture.operations);
      assert.deepEqual(expr.shape, fixture.expected.shape, fixture.id);
      assert.equal(expr.dtype, fixture.expected.dtype, fixture.id);
      const result = await expr.compute();
      assert.deepEqual(
        packed(result, expr.dtype),
        fixture.expected.values,
        fixture.id,
      );
      if (fixture.sharesMemory !== null && result?.sharesStorage)
        assert.equal(result.sharesStorage(a), fixture.sharesMemory, fixture.id);
      result?.dispose?.();
      a.dispose();
    }
  } finally {
    await c.close();
  }
});

test("shared views propagate atomic writes and survive source disposal", async () => {
  const c = context();
  try {
    const a = c.array([0, 1, 2, 3, 4, 5], { dtype: "int32", shape: [2, 3] });
    const view = await a.transpose().compute();
    await view.set([1, 0], 19);
    assert.equal(a.item(0, 1), 19);
    const flat = await a.reshape([6], { copy: false }).compute();
    await a.set([1, 2], 25);
    assert.equal(flat.item(5), 25);
    a.dispose();
    assert.equal(view.item(2, 1), 25);
    await flat.set([0], 17);
    assert.equal(view.item(0, 0), 17);
    view.dispose();
    flat.dispose();
  } finally {
    await c.close();
  }
});

test("computed layout chains preserve copy rules and independent results", async () => {
  const c = context();
  try {
    const a = c.array([1, 2, 3, 4, 5, 6], { dtype: "int32", shape: [2, 3] });
    assert.throws(() => a.add(1).transpose().reshape([6], { copy: false }), {
      code: "NMATH_SHAPE_MISMATCH",
    });
    const expr = a.add(1).transpose().reshape([6]);
    const first = await expr.compute();
    await a.set([0, 0], 10);
    const second = await expr.compute();
    assert.deepEqual(first.toArray(), [2, 5, 3, 6, 4, 7]);
    assert.deepEqual(second.toArray(), [11, 5, 3, 6, 4, 7]);
    assert.equal(first.sharesStorage(a), false);
    assert.equal(first.sharesStorage(second), false);
    const forced = await a
      .transpose()
      .reshape([3, 2], { copy: true })
      .compute();
    const flat = await forced.reshape([6], { copy: false }).compute();
    assert.equal(flat.sharesStorage(forced), true);
    for (const value of [a, first, second, forced, flat]) value.dispose();
  } finally {
    await c.close();
  }
});

test("shape transforms preserve string widths, temporal units and exact complex components", async () => {
  const c = context();
  try {
    for (const a of [
      c.array(["a", "🌍", "가", "xx"], {
        dtype: "str",
        shape: [2, 2],
        stringWidth: 4,
      }),
      c.array(new BigInt64Array([2n ** 60n, -7n, -(2n ** 63n), 99n]), {
        dtype: "datetime64",
        unit: "3ns",
        shape: [2, 2],
      }),
      c.array(new BigInt64Array([1n, -7n, 8n, 99n]), {
        dtype: "timedelta64",
        unit: "us",
        shape: [2, 2],
      }),
      c.array(new Float64Array([1, -0, 2, 3, 4, 5, 6, 7]), {
        dtype: "complex128",
        shape: [2, 2],
      }),
    ]) {
      const transformed = await a
        .transpose()
        .reshape([4], { copy: true })
        .compute();
      assert.equal(transformed.dtypeDescriptor, a.dtypeDescriptor);
      const expected = [a.item(0, 0), a.item(1, 0), a.item(0, 1), a.item(1, 1)];
      assert.deepEqual(transformed.toArray(), expected);
      assert.equal(transformed.sharesStorage(a), false);
      if (a.dtype === "datetime64")
        assert.ok(transformed.item(0) instanceof Temporal);
      transformed.dispose();
      a.dispose();
    }
  } finally {
    await c.close();
  }
});

test("invalid shapes, axes and options fail explicitly before scheduling", async () => {
  const c = context();
  try {
    const a = c.zeros([2, 3], "float64");
    for (const shape of [
      [5],
      [-1, -1],
      [-2],
      [NaN],
      [Infinity],
      [1.5, 4],
      [4294967296],
      [Number.MAX_SAFE_INTEGER],
    ]) {
      assert.throws(() => reshape(a, shape), { code: "NMATH_SHAPE_MISMATCH" });
    }
    for (const axes of [
      [0],
      [0, 0],
      [0, 2],
      [-3, 0],
      [0.5, 1],
      [NaN, 1],
      [Infinity, 0],
    ]) {
      assert.throws(() => transpose(a, axes), { code: "NMATH_SHAPE_MISMATCH" });
    }
    const empty = c.zeros([0, 3], "int32");
    assert.throws(() => empty.reshape([0, -1]), {
      code: "NMATH_SHAPE_MISMATCH",
    });
    assert.throws(() => a.transpose().reshape([-1], { copy: false }), {
      code: "NMATH_SHAPE_MISMATCH",
    });
    assert.throws(() => a.reshape([6], { order: "F" }), /order/);
    assert.throws(() => a.reshape([6], { out: a }), /Unsupported/);
    assert.throws(() => a.reshape([6], { copy: "false" }), /copy/);
    const abort = new AbortController();
    abort.abort();
    await assert.rejects(a.transpose().compute({ signal: abort.signal }), {
      code: "NMATH_CANCELLED",
    });
    await assert.rejects(a.transpose().compute({ device: "gpu" }), {
      code: "NMATH_GPU_UNSUPPORTED_OPERATION",
    });
    assert.equal(c.metrics().executions.length, 0);
    a.dispose();
    empty.dispose();
  } finally {
    await c.close();
  }
});
