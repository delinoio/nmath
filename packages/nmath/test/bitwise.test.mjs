// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import * as np from "../dist/numpy.js";
import { context } from "./helpers.mjs";

test("bitwise, shift and logical ufuncs follow NumPy", async () => {
  const ctx = context();
  try {
    const a = ctx.array([12, -1, 5], { dtype: "int8" });
    const b = ctx.array([10, 3, 1], { dtype: "uint8" });
    const and = np.bitwise_and(a, b);
    assert.equal(and.dtype, "int16");
    assert.deepEqual((await and.compute()).toArray(), [8, 3, 1]);
    assert.deepEqual(
      (await np.bitwise_or(a, 1n).compute()).toArray(),
      [13, -1, 5],
    );
    assert.deepEqual(
      (await np.bitwise_xor(3n, a).compute()).toArray(),
      [15, -4, 6],
    );
    const counts = ctx.array([1, 9, -1], { dtype: "int8" });
    assert.deepEqual(
      (await np.left_shift(a, counts).compute()).toArray(),
      [24, 0, 0],
    );
    assert.deepEqual(
      (await np.right_shift(a, counts).compute()).toArray(),
      [6, -1, 0],
    );
    assert.deepEqual((await np.invert(a).compute()).toArray(), [-13, 0, -6]);
    const flags = ctx.array([true, false], { dtype: "bool" });
    assert.equal(np.left_shift(flags, flags).dtype, "int8");
    assert.deepEqual((await np.invert(flags).compute()).toArray(), [
      false,
      true,
    ]);

    const x = ctx.array([0, Number.NaN, 2.5], { dtype: "float64" });
    assert.deepEqual((await np.logical_and(x, 1).compute()).toArray(), [
      false,
      true,
      true,
    ]);
    assert.deepEqual((await np.logical_xor(x, x).compute()).toArray(), [
      false,
      false,
      false,
    ]);
    assert.equal(np.bitwise_not, np.invert);
  } finally {
    await ctx.close();
  }
});

test("bitwise rejections and GPU admission are stable", async () => {
  const ctx = context();
  try {
    const ints = ctx.array([1, 2], { dtype: "int32" });
    const floats = ctx.array([1, 2], { dtype: "float32" });
    assert.throws(() => np.bitwise_and(floats, floats), {
      code: "NMATH_DTYPE_UNSUPPORTED",
    });
    assert.throws(() => np.bitwise_and(ints, 1), {
      code: "NMATH_DTYPE_UNSUPPORTED",
    });
    assert.throws(
      () =>
        np.bitwise_or(
          ctx.array([1n], { dtype: "int64" }),
          ctx.array([1n], { dtype: "uint64" }),
        ),
      { code: "NMATH_DTYPE_UNSUPPORTED" },
    );
    assert.throws(() => np.invert(floats), {
      code: "NMATH_DTYPE_UNSUPPORTED",
    });
    assert.throws(() => np.logical_and(1, 2), TypeError);
    await assert.rejects(
      np.bitwise_and(ints, ints).compute({ device: "gpu" }),
      { code: "NMATH_GPU_UNSUPPORTED_OPERATION" },
    );
  } finally {
    await ctx.close();
  }
});
