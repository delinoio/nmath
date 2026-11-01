// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import * as np from "../dist/numpy.js";
import { context } from "./helpers.mjs";

// Expected values come from NumPy 2.5.4 for np.arange(24).reshape(2, 3, 4).
test("reductions accept axis lists and keepdims like NumPy", async () => {
  const ctx = context();
  try {
    const values = Array.from({ length: 2 }, (_, i) =>
      Array.from({ length: 3 }, (_, j) =>
        Array.from({ length: 4 }, (_, k) => i * 12 + j * 4 + k),
      ),
    );
    const a = ctx.array(values, { dtype: "int32" });
    const listed = np.sum(a, [0, 2]);
    assert.deepEqual(listed.shape, [3]);
    assert.deepEqual((await listed.compute()).toArray(), [60n, 92n, 124n]);
    const kept = np.sum(a, [0, 2], { keepdims: true });
    assert.deepEqual(kept.shape, [1, 3, 1]);
    assert.deepEqual((await kept.compute()).toArray(), [
      [[60n], [92n], [124n]],
    ]);
    assert.deepEqual((await np.max(a, [1, -1]).compute()).toArray(), [11, 23]);
    assert.equal(await np.mean(a, [0, 1, 2]).compute(), 11.5);
    assert.deepEqual(np.sum(a, []).shape, [2, 3, 4]);
    assert.deepEqual(np.sum(a, undefined, { keepdims: true }).shape, [1, 1, 1]);
    assert.deepEqual(
      (await a.min(undefined, { keepdims: true }).compute()).toArray(),
      [[[0]]],
    );
    const index = np.argmax(a, 1, { keepdims: true });
    assert.deepEqual(index.shape, [2, 1, 4]);
    assert.deepEqual((await index.compute()).toArray()[0][0], [2n, 2n, 2n, 2n]);
    assert.deepEqual(np.var(a, 2, 0, { keepdims: true }).shape, [2, 3, 1]);
    assert.deepEqual(np.std(a, -1, 1, { keepdims: true }).shape, [2, 3, 1]);
    const product = np.prod(
      ctx.array(
        [
          [1, 2],
          [3, 4],
        ],
        { dtype: "int8" },
      ),
      [0, 1],
    );
    assert.equal(product.dtype, "int64");
    assert.equal(await product.compute(), 24n);
  } finally {
    await ctx.close();
  }
});

test("invalid axis lists and keepdims values are rejected", async () => {
  const ctx = context();
  try {
    const a = ctx.zeros([2, 3], "float64");
    assert.throws(() => np.sum(a, [0, 0]), { code: "NMATH_INDEX_INVALID" });
    assert.throws(() => np.sum(a, [0, -2]), { code: "NMATH_INDEX_INVALID" });
    assert.throws(() => np.sum(a, [2]), { code: "NMATH_INDEX_OUT_OF_BOUNDS" });
    assert.throws(() => np.argmax(a, [0, 1]), { code: "NMATH_INDEX_INVALID" });
    assert.throws(() => np.sum(a, 0, { keepdims: "yes" }), {
      code: "NMATH_INDEX_INVALID",
    });
    assert.throws(() => np.sum(a, [0.5]), { code: "NMATH_INDEX_INVALID" });
  } finally {
    await ctx.close();
  }
});
