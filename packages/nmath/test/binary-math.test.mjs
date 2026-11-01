// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import * as np from "../dist/numpy.js";
import { context } from "./helpers.mjs";

test("binary math ufuncs follow NumPy", async () => {
  const ctx = context();
  try {
    const a = ctx.array([1, Number.NaN, -0, 7], { dtype: "float64" });
    const b = ctx.array([2, 1, 0, Number.NaN], { dtype: "float64" });
    const most = (await np.maximum(a, b).compute()).toArray();
    assert.equal(most[0], 2);
    assert.ok(Number.isNaN(most[1]) && Number.isNaN(most[3]));
    assert.deepEqual(
      (await np.fmax(a, b).compute()).toArray().slice(0, 2),
      [2, 1],
    );
    const ints = ctx.array([7, -7, 7, -7, 5], { dtype: "int8" });
    const divisors = ctx.array([2, 2, -2, -2, 0], { dtype: "int8" });
    assert.deepEqual(
      (await np.floor_divide(ints, divisors).compute()).toArray(),
      [3, -4, -4, 3, 0],
    );
    assert.deepEqual(
      (await np.fmod(ints, divisors).compute()).toArray(),
      [1, -1, 1, -1, 0],
    );
    const floats = ctx.array([7, -7, 1], { dtype: "float32" });
    assert.deepEqual(
      (await np.floor_divide(floats, 2).compute()).toArray(),
      [3, -4, 0],
    );
    const angle = np.arctan2(ctx.array([1], { dtype: "int16" }), 1n);
    assert.equal(angle.dtype, "float32");
    assert.ok(
      Math.abs((await angle.compute()).toArray()[0] - Math.PI / 4) < 1e-6,
    );
    assert.equal(
      np.copysign(ints, ctx.array([1], { dtype: "uint8" })).dtype,
      "float16",
    );
    assert.deepEqual(
      (
        await np
          .heaviside(ctx.array([-1, 0, 2], { dtype: "float64" }), 0.5)
          .compute()
      ).toArray(),
      [0, 0.5, 1],
    );
    assert.deepEqual(
      (await np.hypot(floats, floats).compute()).toArray().length,
      3,
    );
    assert.equal(np.atan2, np.arctan2);
  } finally {
    await ctx.close();
  }
});

test("binary math rejections and GPU admission are stable", async () => {
  const ctx = context();
  try {
    const z = ctx.zeros([2], "complex64");
    assert.throws(() => np.arctan2(z, z), { code: "NMATH_DTYPE_UNSUPPORTED" });
    assert.throws(() => np.fmod(z, z), { code: "NMATH_DTYPE_UNSUPPORTED" });
    const s = ctx.array(["a"], { dtype: "str" });
    assert.throws(() => np.maximum(s, s), { code: "NMATH_DTYPE_UNSUPPORTED" });
    assert.throws(() => np.maximum(1, 2), TypeError);
    const f = ctx.array([1, 2], { dtype: "float32" });
    await assert.rejects(np.maximum(f, f).compute({ device: "gpu" }), {
      code: "NMATH_GPU_UNSUPPORTED_OPERATION",
    });
  } finally {
    await ctx.close();
  }
});
