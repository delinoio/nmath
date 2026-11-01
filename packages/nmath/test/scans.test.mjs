// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { Complex } from "../dist/index.js";
import { cumsum, cumprod } from "../dist/numpy.js";
import { context } from "./helpers.mjs";

const values = [
  [1, 2, 3],
  [4, 5, 6],
];

test("cumulative scans promote like NumPy sums", async () => {
  const ctx = context();
  try {
    const int16 = ctx.array(values, { dtype: "int16" });
    assert.equal(int16.cumsum().dtype, "int64");
    assert.deepEqual((await int16.cumsum().compute()).toArray(), [
      1n,
      3n,
      6n,
      10n,
      15n,
      21n,
    ]);
    assert.deepEqual((await int16.cumsum(0).compute()).toArray(), [
      [1n, 2n, 3n],
      [5n, 7n, 9n],
    ]);
    assert.deepEqual((await int16.cumsum(-1).compute()).toArray(), [
      [1n, 3n, 6n],
      [4n, 9n, 15n],
    ]);
    assert.deepEqual((await int16.cumprod(1).compute()).toArray(), [
      [1n, 2n, 6n],
      [4n, 20n, 120n],
    ]);

    const uint8 = ctx.array(values, { dtype: "uint8" });
    assert.equal(uint8.cumsum(0).dtype, "uint64");

    const half = ctx.array([1.5, 2], { dtype: "float16" });
    assert.equal(half.cumprod().dtype, "float16");
    assert.deepEqual((await half.cumprod().compute()).toArray(), [1.5, 3]);

    const bool = ctx.array([true, false, true], { dtype: "bool" });
    assert.equal(bool.cumsum().dtype, "int64");
    assert.deepEqual((await bool.cumsum().compute()).toArray(), [1n, 1n, 2n]);
    assert.deepEqual((await bool.cumprod().compute()).toArray(), [1n, 0n, 0n]);

    const complex = ctx.array(
      [new Complex(1, 1), new Complex(2, 0), new Complex(0, 1)],
      { dtype: "complex64" },
    );
    assert.equal(complex.cumsum().dtype, "complex64");
    const total = await complex.cumsum().compute();
    assert.deepEqual(total.toArray(), [
      new Complex(1, 1),
      new Complex(3, 1),
      new Complex(3, 2),
    ]);

    const large = ctx.array([2n ** 40n, 3n, -1n], { dtype: "int64" });
    assert.deepEqual((await large.cumsum().compute()).toArray(), [
      2n ** 40n,
      2n ** 40n + 3n,
      2n ** 40n + 2n,
    ]);
  } finally {
    await ctx.close();
  }
});

test("empty and rank-0 scans keep NumPy shapes", async () => {
  const ctx = context();
  try {
    assert.deepEqual(
      (await ctx.zeros([0], "float32").cumsum().compute()).toArray(),
      [],
    );
    assert.deepEqual(
      (await ctx.zeros([3, 0], "int32").cumsum(0).compute()).toArray(),
      [[], [], []],
    );
    assert.deepEqual(
      (await ctx.zeros([0, 3], "int32").cumprod(1).compute()).toArray(),
      [],
    );
    const scalar = ctx.array(5, { dtype: "int32", shape: [] });
    assert.deepEqual((await scalar.cumsum().compute()).toArray(), [5n]);
  } finally {
    await ctx.close();
  }
});

test("scan rejections are stable and precede execution", async () => {
  const ctx = context();
  try {
    const ints = ctx.array(values, { dtype: "int32" });
    assert.throws(() => ints.cumsum(2), { code: "NMATH_INDEX_OUT_OF_BOUNDS" });

    const strings = ctx.array(["a", "b"], { dtype: "str" });
    assert.throws(() => strings.cumsum(), { code: "NMATH_DTYPE_UNSUPPORTED" });
    const times = ctx.array([1n, 2n], { dtype: "datetime64", unit: "D" });
    assert.throws(() => times.cumprod(), { code: "NMATH_DTYPE_UNSUPPORTED" });

    for (const expr of [ints.cumsum(0), ints.cumprod()]) {
      await assert.rejects(expr.compute({ device: "gpu" }), {
        code: "NMATH_GPU_UNSUPPORTED_OPERATION",
      });
    }
  } finally {
    await ctx.close();
  }
});

test("NumPy scan helpers share the lazy expressions", async () => {
  const ctx = context();
  try {
    const a = ctx.array(values, { dtype: "int32" });
    assert.equal(cumsum(a).dtype, "int64");
    assert.deepEqual((await cumsum(a).compute()).toArray(), [
      1n,
      3n,
      6n,
      10n,
      15n,
      21n,
    ]);
    assert.deepEqual((await cumsum(a, 0).compute()).toArray(), [
      [1n, 2n, 3n],
      [5n, 7n, 9n],
    ]);
    assert.deepEqual((await cumprod(a).compute()).toArray(), [
      1n,
      2n,
      6n,
      24n,
      120n,
      720n,
    ]);
  } finally {
    await ctx.close();
  }
});
