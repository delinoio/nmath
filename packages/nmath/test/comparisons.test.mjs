// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { Complex } from "../dist/index.js";
import {
  equal,
  greater,
  greater_equal,
  less,
  less_equal,
  not_equal,
} from "../dist/numpy.js";
import { context } from "./helpers.mjs";

const values = [1, 2, 3];
const other = [2, 2, 2];

test("comparison ufuncs return bool arrays with NumPy broadcasting", async () => {
  const ctx = context();
  try {
    const a = ctx.array(values, { dtype: "int32" });
    const b = ctx.array(other, { dtype: "int32" });
    for (const [fn, expected] of [
      [equal, [false, true, false]],
      [not_equal, [true, false, true]],
      [less, [true, false, false]],
      [less_equal, [true, true, false]],
      [greater, [false, false, true]],
      [greater_equal, [false, true, true]],
    ]) {
      const expr = fn(a, b);
      assert.equal(expr.dtype, "bool");
      assert.deepEqual(expr.shape, [3]);
      const result = await expr.compute();
      assert.deepEqual(result.toArray(), expected);
      result.dispose();
    }

    const rows = ctx.array([[2], [5]], { dtype: "float64" });
    const columns = ctx.array([1, 3, 5], { dtype: "float64" });
    assert.deepEqual((await equal(rows, columns).compute()).toArray(), [
      [false, false, false],
      [false, false, true],
    ]);
    assert.deepEqual((await greater(rows, 1.5).compute()).toArray(), [
      [true],
      [true],
    ]);
    assert.deepEqual((await less_equal(rows, 2n).compute()).toArray(), [
      [true],
      [false],
    ]);

    const flags = ctx.array([true, false, true], { dtype: "bool" });
    assert.deepEqual(
      (
        await equal(flags, ctx.array([1, 0, 1], { dtype: "int16" })).compute()
      ).toArray(),
      [true, true, true],
    );
  } finally {
    await ctx.close();
  }
});

test("NaN, complex and mixed dtypes follow NumPy comparison rules", async () => {
  const ctx = context();
  try {
    const nan = ctx.array([Number.NaN, 1, 2], { dtype: "float64" });
    const ones = ctx.array([1, 1, 1], { dtype: "float64" });
    assert.deepEqual((await equal(nan, ones).compute()).toArray(), [
      false,
      true,
      false,
    ]);
    assert.deepEqual((await not_equal(nan, ones).compute()).toArray(), [
      true,
      false,
      true,
    ]);
    for (const [fn, expected] of [
      [less, [false, false, false]],
      [less_equal, [false, true, false]],
      [greater, [false, false, true]],
      [greater_equal, [false, true, true]],
    ]) {
      assert.deepEqual((await fn(nan, ones).compute()).toArray(), expected);
    }

    const complex = ctx.array([new Complex(1, 1), new Complex(0, 1)], {
      dtype: "complex64",
    });
    const complexOther = ctx.array([new Complex(1, 1), new Complex(1, 0)], {
      dtype: "complex128",
    });
    assert.deepEqual((await equal(complex, complexOther).compute()).toArray(), [
      true,
      false,
    ]);
    assert.deepEqual(
      (await not_equal(complex, new Complex(0, 1)).compute()).toArray(),
      [true, false],
    );
    assert.throws(() => less(complex, complexOther), {
      code: "NMATH_DTYPE_UNSUPPORTED",
    });

    const mixed = ctx.array([1, 2, 3], { dtype: "int8" });
    const wide = ctx.array([1, 2, 3], { dtype: "float64" });
    assert.deepEqual((await equal(mixed, wide).compute()).toArray(), [
      true,
      true,
      true,
    ]);
  } finally {
    await ctx.close();
  }
});

test("comparison rejections and GPU admission are stable", async () => {
  const ctx = context();
  try {
    const a = ctx.array(values, { dtype: "int32" });
    assert.throws(() => equal(a, ctx.zeros([4], "int32")), {
      code: "NMATH_SHAPE_BROADCAST",
    });
    const strings = ctx.array(["a", "b"], { dtype: "str" });
    assert.throws(() => equal(strings, strings), {
      code: "NMATH_DTYPE_UNSUPPORTED",
    });
    const times = ctx.array([1n, 2n], { dtype: "datetime64", unit: "D" });
    assert.throws(() => less(times, times), {
      code: "NMATH_DTYPE_UNSUPPORTED",
    });

    await assert.rejects(
      equal(a, ctx.array(other, { dtype: "int32" })).compute({ device: "gpu" }),
      { code: "NMATH_GPU_UNSUPPORTED_OPERATION" },
    );
  } finally {
    await ctx.close();
  }
});
