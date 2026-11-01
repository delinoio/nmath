// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { Complex, where as coreWhere } from "../dist/index.js";
import { less, where } from "../dist/numpy.js";
import { context } from "./helpers.mjs";

test("where picks branches with NumPy broadcasting and promotion", async () => {
  const ctx = context();
  try {
    assert.equal(coreWhere, where);
    const a = ctx.array([1, 2, 3], { dtype: "int16" });
    const b = ctx.array([10, 20, 30], { dtype: "int16" });
    const picked = where(less(a, 2.5), a, b);
    assert.equal(picked.dtype, "int16");
    assert.deepEqual(picked.shape, [3]);
    assert.deepEqual((await picked.compute()).toArray(), [1, 2, 30]);

    const rows = ctx.array([[true], [false]], { dtype: "bool" });
    const x = ctx.array([1, 2, 3], { dtype: "float32" });
    const y = ctx.array([[10], [20]], { dtype: "int32" });
    const broadcast = where(rows, x, y);
    assert.equal(broadcast.dtype, "float64");
    assert.deepEqual((await broadcast.compute()).toArray(), [
      [1, 2, 3],
      [20, 20, 20],
    ]);

    // Non-bool conditions use nonzero truthiness; NaN counts as true.
    const condition = ctx.array([0, Number.NaN, -0, 0.5], { dtype: "float64" });
    const values = ctx.array([1, 2, 3, 4], { dtype: "int32" });
    assert.deepEqual(
      (await where(condition, values, -1n).compute()).toArray(),
      [-1, 2, -1, 4],
    );
  } finally {
    await ctx.close();
  }
});

test("where follows NEP 50 weak scalar rules", async () => {
  const ctx = context();
  try {
    const mask = ctx.array([true, false, true], { dtype: "bool" });
    const small = ctx.array([1, 2, 3], { dtype: "int8" });
    const intWeak = where(mask, small, 7n);
    assert.equal(intWeak.dtype, "int8");
    assert.deepEqual((await intWeak.compute()).toArray(), [1, 7, 3]);
    assert.equal(where(mask, 7n, small).dtype, "int8");
    // JS numbers are weak floats.
    assert.equal(where(mask, small, 7).dtype, "float64");
    const f32 = ctx.array([1, 2, 3], { dtype: "float32" });
    assert.equal(where(mask, f32, 2.5).dtype, "float32");
    const complex = where(mask, f32, new Complex(0, 1));
    assert.equal(complex.dtype, "complex64");
    const [first, second] = (await complex.compute()).toArray();
    assert.equal(first.real, 1);
    assert.equal(second.imag, 1);

    const scalars = where(mask, 1n, 2n);
    assert.equal(scalars.dtype, "int64");
    assert.deepEqual((await scalars.compute()).toArray(), [1n, 2n, 1n]);
    assert.equal(where(mask, true, false).dtype, "bool");

    const unsigned = ctx.array([1, 2, 3], { dtype: "uint8" });
    assert.throws(() => where(mask, unsigned, -1n), {
      code: "NMATH_DTYPE_UNSUPPORTED",
    });
  } finally {
    await ctx.close();
  }
});

test("where rejections, snapshots and GPU admission are stable", async () => {
  const ctx = context();
  const other = context();
  try {
    const mask = ctx.array([true, false], { dtype: "bool" });
    const x = ctx.array([1, 2], { dtype: "float32" });
    assert.throws(() => where(mask, x, ctx.zeros([3], "float32")), {
      code: "NMATH_SHAPE_BROADCAST",
    });
    const strings = ctx.array(["a", "b"], { dtype: "str" });
    assert.throws(() => where(mask, strings, strings), {
      code: "NMATH_DTYPE_UNSUPPORTED",
    });
    const times = ctx.array([1n, 2n], { dtype: "datetime64", unit: "D" });
    assert.throws(() => where(times, x, x), {
      code: "NMATH_DTYPE_UNSUPPORTED",
    });
    assert.throws(() => where(mask, x, other.zeros([2], "float32")), {
      code: "NMATH_CONTEXT_MISMATCH",
    });
    assert.throws(() => where(true, x, x), TypeError);

    // Changes before compute() starts are visible; the returned array is new
    // storage.
    const expr = where(mask, x, 0);
    await mask.set([0], false);
    const result = await expr.compute();
    assert.deepEqual(result.toArray(), [0, 0]);
    assert.equal(result.sharesStorage(x), false);

    await assert.rejects(where(mask, x, x).compute({ device: "gpu" }), {
      code: "NMATH_GPU_UNSUPPORTED_OPERATION",
    });
  } finally {
    await ctx.close();
    await other.close();
  }
});
