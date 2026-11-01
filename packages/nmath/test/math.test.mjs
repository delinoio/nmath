// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { Complex } from "../dist/index.js";
import * as np from "../dist/numpy.js";
import { context } from "./helpers.mjs";

test("math ufuncs follow NumPy result dtypes and values", async () => {
  const ctx = context();
  try {
    const ints = ctx.array([1, 4, 9], { dtype: "int16" });
    const roots = np.sqrt(ints);
    assert.equal(roots.dtype, "float32");
    assert.deepEqual((await roots.compute()).toArray(), [1, 2, 3]);
    assert.equal(np.sqrt(ctx.array([4], { dtype: "int8" })).dtype, "float16");
    assert.equal(np.exp(ctx.array([0], { dtype: "int64" })).dtype, "float64");

    const x = ctx.array([0, 0.5, -1, Number.NaN], { dtype: "float64" });
    const [s0, s1] = (await np.sin(x).compute()).toArray();
    assert.equal(s0, 0);
    assert.ok(Math.abs(s1 - Math.sin(0.5)) <= Number.EPSILON);
    assert.deepEqual((await np.isnan(x).compute()).toArray(), [
      false,
      false,
      false,
      true,
    ]);
    assert.deepEqual((await np.signbit(x).compute()).toArray(), [
      false,
      false,
      true,
      false,
    ]);
    const logs = (await np.log(x).compute()).toArray();
    assert.ok(Number.isNaN(logs[2]));
    assert.equal(logs[0], Number.NEGATIVE_INFINITY);

    const signed = ctx.array([-128, -3, 0, 5], { dtype: "int8" });
    assert.deepEqual(
      (await np.abs(signed).compute()).toArray(),
      [-128, 3, 0, 5],
    );
    assert.deepEqual(
      (await np.sign(signed).compute()).toArray(),
      [-1, -1, 0, 1],
    );
    assert.deepEqual(
      (await np.square(signed).compute()).toArray(),
      [0, 9, 0, 25],
    );
    const halves = ctx.array([-2.5, -0.5, 0.5, 1.5, 2.5], { dtype: "float32" });
    assert.deepEqual(
      (await np.rint(halves).compute()).toArray(),
      [-2, -0, 0, 2, 2],
    );
    assert.deepEqual(
      (await np.floor(halves).compute()).toArray(),
      [-3, -1, 0, 1, 2],
    );
    assert.deepEqual(
      (
        await np.reciprocal(ctx.array([2, 0], { dtype: "float64" })).compute()
      ).toArray(),
      [0.5, Number.POSITIVE_INFINITY],
    );

    const z = ctx.array([new Complex(1, 2), new Complex(0, 0)], {
      dtype: "complex64",
    });
    const [conj] = (await np.conj(z).compute()).toArray();
    assert.equal(conj.imag, -2);
    assert.deepEqual((await np.logical_not(z).compute()).toArray(), [
      false,
      true,
    ]);
    assert.equal(np.asin, np.arcsin);
  } finally {
    await ctx.close();
  }
});

test("math ufunc rejections and GPU admission are stable", async () => {
  const ctx = context();
  try {
    const z = ctx.array([new Complex(1, 2)], { dtype: "complex128" });
    for (const fn of [np.sqrt, np.exp, np.abs, np.signbit, np.floor]) {
      assert.throws(() => fn(z), { code: "NMATH_DTYPE_UNSUPPORTED" });
    }
    const ints = ctx.array([1, 0], { dtype: "int32" });
    assert.throws(() => np.reciprocal(ints), {
      code: "NMATH_DTYPE_UNSUPPORTED",
    });
    assert.throws(() => np.sign(ctx.array([true], { dtype: "bool" })), {
      code: "NMATH_DTYPE_UNSUPPORTED",
    });
    assert.throws(() => np.sqrt(ctx.array(["a"], { dtype: "str" })), {
      code: "NMATH_DTYPE_UNSUPPORTED",
    });
    assert.throws(() => np.sqrt(4), TypeError);
    await assert.rejects(
      np.sqrt(ctx.array([4], { dtype: "float32" })).compute({ device: "gpu" }),
      { code: "NMATH_GPU_UNSUPPORTED_OPERATION" },
    );
  } finally {
    await ctx.close();
  }
});
