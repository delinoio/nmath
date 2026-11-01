// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { Complex } from "../dist/index.js";
import { std, var as variance } from "../dist/numpy.js";
import { context } from "./helpers.mjs";

const values = [
  [1, 3, 1, 3],
  [1, 3, 1, 3],
];

test("variance and standard deviation use NumPy result dtypes", async () => {
  const ctx = context();
  try {
    const int16 = ctx.array(values, { dtype: "int16" });
    assert.equal(int16.var().dtype, "float64");
    assert.equal(await int16.var().compute(), 1);
    assert.equal(await int16.std().compute(), 1);
    assert.deepEqual((await int16.var(0).compute()).toArray(), [0, 0, 0, 0]);
    assert.deepEqual((await int16.var(1).compute()).toArray(), [1, 1]);
    assert.deepEqual((await int16.std(-1).compute()).toArray(), [1, 1]);

    assert.equal(ctx.zeros([4], "float16").var().dtype, "float16");
    assert.equal(ctx.zeros([4], "float32").var().dtype, "float32");
    assert.equal(ctx.zeros([4], "float64").var().dtype, "float64");
    assert.equal(ctx.zeros([4], "complex64").var().dtype, "float32");
    assert.equal(ctx.zeros([4], "complex128").std().dtype, "float64");
    assert.equal(ctx.zeros([4], "bool").var().dtype, "float64");

    const bool = ctx.array([true, true, false, false], { dtype: "bool" });
    assert.equal(await bool.var().compute(), 0.25);

    const complex = ctx.array(
      [
        [
          new Complex(1, 1),
          new Complex(3, 3),
          new Complex(1, 1),
          new Complex(3, 3),
        ],
        [
          new Complex(1, 1),
          new Complex(3, 3),
          new Complex(1, 1),
          new Complex(3, 3),
        ],
      ],
      { dtype: "complex64" },
    );
    assert.equal(await complex.var().compute(), 2);
    assert.deepEqual(
      (await complex.std(1).compute()).toArray(),
      [1.4142135381698608, 1.4142135381698608],
    );
  } finally {
    await ctx.close();
  }
});

test("degrees of freedom follow NumPy boundary behavior", async () => {
  const ctx = context();
  try {
    const pair = ctx.array([1, -1], { dtype: "float64" });
    assert.equal(await pair.var().compute(), 1);
    assert.equal(await pair.var(undefined, 1).compute(), 2);
    assert.equal(await pair.var(undefined, 2).compute(), Infinity);
    assert.equal(await pair.var(undefined, 3).compute(), Infinity);
    assert.equal(await pair.std(undefined, 2).compute(), Infinity);
    const equal = ctx.array([1, 1], { dtype: "float64" });
    assert.ok(Number.isNaN(await equal.var(undefined, 2).compute()));
  } finally {
    await ctx.close();
  }
});

test("empty inputs and empty axes produce NumPy results", async () => {
  const ctx = context();
  try {
    assert.ok(Number.isNaN(await ctx.zeros([0], "float64").var().compute()));
    assert.deepEqual(
      (await ctx.zeros([2, 0], "float64").var(1).compute()).toArray(),
      [Number.NaN, Number.NaN],
    );
    assert.deepEqual(
      (await ctx.zeros([0, 3], "float64").std(0).compute()).toArray(),
      [Number.NaN, Number.NaN, Number.NaN],
    );
    assert.equal(
      await ctx.array(5, { dtype: "int32", shape: [] }).var().compute(),
      0,
    );
  } finally {
    await ctx.close();
  }
});

test("variance rejections are stable and precede execution", async () => {
  const ctx = context();
  try {
    const ints = ctx.array(values, { dtype: "int32" });
    assert.throws(() => ints.var(2), { code: "NMATH_INDEX_OUT_OF_BOUNDS" });
    assert.throws(() => ints.std(undefined, Number.NaN), {
      code: "NMATH_INDEX_INVALID",
    });

    const strings = ctx.array(["a", "b"], { dtype: "str" });
    assert.throws(() => strings.var(), { code: "NMATH_DTYPE_UNSUPPORTED" });
    const times = ctx.array([1n, 2n], { dtype: "datetime64", unit: "D" });
    assert.throws(() => times.std(), { code: "NMATH_DTYPE_UNSUPPORTED" });

    for (const expr of [ints.var(0), ints.std()]) {
      await assert.rejects(expr.compute({ device: "gpu" }), {
        code: "NMATH_GPU_UNSUPPORTED_OPERATION",
      });
    }
  } finally {
    await ctx.close();
  }
});

test("NumPy variance helpers share the lazy expressions", async () => {
  const ctx = context();
  try {
    const a = ctx.array(values, { dtype: "int32" });
    assert.equal(variance(a).dtype, "float64");
    assert.equal(await variance(a).compute(), 1);
    assert.deepEqual((await variance(a, 1).compute()).toArray(), [1, 1]);
    assert.equal(await std(a).compute(), 1);
  } finally {
    await ctx.close();
  }
});
