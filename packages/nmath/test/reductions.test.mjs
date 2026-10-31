// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { Complex } from "../dist/index.js";
import { argmax, argmin, prod } from "../dist/numpy.js";
import { context } from "./helpers.mjs";

const values = [
  [1, 2, 3],
  [4, 5, 6],
];

test("product reductions promote like NumPy sums", async () => {
  const ctx = context();
  try {
    const int16 = ctx.array(values, { dtype: "int16" });
    assert.equal(int16.prod().dtype, "int64");
    assert.equal(await int16.prod().compute(), 720n);
    assert.deepEqual((await int16.prod(0).compute()).toArray(), [4n, 10n, 18n]);
    assert.deepEqual((await int16.prod(1).compute()).toArray(), [6n, 120n]);

    const int8 = ctx.array(values, { dtype: "int8" });
    assert.equal(int8.prod(0).dtype, "int64");
    const uint8 = ctx.array(values, { dtype: "uint8" });
    assert.equal(uint8.prod(0).dtype, "uint64");

    const half = ctx.array([1.5, 2], { dtype: "float16" });
    assert.equal(half.prod().dtype, "float16");
    assert.equal(await half.prod().compute(), 3);

    const bool = ctx.array(
      [
        [true, true, false],
        [true, true, true],
      ],
      { dtype: "bool" },
    );
    assert.equal(bool.prod().dtype, "int64");
    assert.equal(await bool.prod().compute(), 0n);
    assert.deepEqual((await bool.prod(0).compute()).toArray(), [1n, 1n, 0n]);

    const complex = ctx.array([new Complex(1, 2), new Complex(3, -1)], {
      dtype: "complex64",
    });
    assert.equal(complex.prod().dtype, "complex64");
    const product = await complex.prod().compute();
    assert.equal(product.real, 5);
    assert.equal(product.imag, 5);

    const large = ctx.array([2n ** 40n, 3n], { dtype: "int64" });
    assert.equal(await large.prod().compute(), 2n ** 40n * 3n);
  } finally {
    await ctx.close();
  }
});

test("empty products use the multiplicative identity", async () => {
  const ctx = context();
  try {
    const empty = ctx.zeros([0], "float32");
    assert.equal(await empty.prod().compute(), 1);
    assert.deepEqual(
      (await ctx.zeros([0, 3], "float64").prod(0).compute()).toArray(),
      [1, 1, 1],
    );
    assert.deepEqual(
      (await ctx.zeros([2, 0], "int32").prod(1).compute()).toArray(),
      [1n, 1n],
    );
  } finally {
    await ctx.close();
  }
});

test("index reductions return bigint int64 positions", async () => {
  const ctx = context();
  try {
    const a = ctx.array(values, { dtype: "int16" });
    assert.equal(a.argmin().dtype, "int64");
    assert.equal(await a.argmin().compute(), 0n);
    assert.equal(await a.argmax().compute(), 5n);
    assert.deepEqual((await a.argmin(0).compute()).toArray(), [0n, 0n, 0n]);
    assert.deepEqual((await a.argmax(0).compute()).toArray(), [1n, 1n, 1n]);
    assert.deepEqual((await a.argmin(1).compute()).toArray(), [0n, 0n]);
    assert.deepEqual((await a.argmax(1).compute()).toArray(), [2n, 2n]);
    assert.equal(typeof (await a.argmax(1).compute()).toArray()[0], "bigint");

    const bool = ctx.array([true, false, true], { dtype: "bool" });
    assert.equal(await bool.argmin().compute(), 1n);
    assert.equal(await bool.argmax().compute(), 0n);
  } finally {
    await ctx.close();
  }
});

test("index reductions keep the first tie and the first NaN", async () => {
  const ctx = context();
  try {
    const tied = ctx.array([3, 1, 1, 2], { dtype: "float64" });
    assert.equal(await tied.argmin().compute(), 1n);
    assert.equal(await tied.argmax().compute(), 0n);

    const nan = ctx.array([1, Number.NaN, 0, Number.NaN], { dtype: "float64" });
    assert.equal(await nan.argmin().compute(), 1n);
    assert.equal(await nan.argmax().compute(), 1n);
    assert.deepEqual(
      (
        await ctx
          .array(
            [
              [1, Number.NaN],
              [0, 2],
            ],
            { dtype: "float64" },
          )
          .argmin(0)
          .compute()
      ).toArray(),
      [1n, 0n],
    );

    const complex = ctx.array(
      [new Complex(1, 2), new Complex(1, 1), new Complex(0, 5)],
      { dtype: "complex128" },
    );
    assert.equal(await complex.argmin().compute(), 2n);
    assert.equal(await complex.argmax().compute(), 0n);
  } finally {
    await ctx.close();
  }
});

test("rejections are stable and happen before execution", async () => {
  const ctx = context();
  try {
    const empty = ctx.zeros([0], "float64");
    await assert.rejects(empty.argmin().compute(), {
      code: "NMATH_SHAPE_MISMATCH",
    });
    await assert.rejects(ctx.zeros([2, 0], "int32").argmax(1).compute(), {
      code: "NMATH_SHAPE_MISMATCH",
    });
    // A nonempty reduced axis over an empty output shape is valid.
    assert.deepEqual(
      (await ctx.zeros([2, 0], "int32").argmax(0).compute()).toArray(),
      [],
    );

    const strings = ctx.array(["a", "b"], { dtype: "str" });
    assert.throws(() => strings.prod(), { code: "NMATH_DTYPE_UNSUPPORTED" });
    assert.throws(() => strings.argmin(), { code: "NMATH_DTYPE_UNSUPPORTED" });
    const times = ctx.array([2n, 1n], {
      dtype: "datetime64",
      unit: "D",
    });
    assert.throws(() => times.argmax(), { code: "NMATH_DTYPE_UNSUPPORTED" });
    const durations = ctx.array([1n, 2n], {
      dtype: "timedelta64",
      unit: "s",
    });
    assert.throws(() => durations.prod(), { code: "NMATH_DTYPE_UNSUPPORTED" });

    const integers = ctx.array(values, { dtype: "int32" });
    for (const expr of [integers.prod(0), integers.argmin(0)]) {
      await assert.rejects(expr.compute({ device: "gpu" }), {
        code: "NMATH_GPU_UNSUPPORTED_OPERATION",
      });
    }
  } finally {
    await ctx.close();
  }
});

test("NumPy entry points share the same lazy expressions", async () => {
  const ctx = context();
  try {
    const a = ctx.array(values, { dtype: "int32" });
    assert.equal(prod(a).dtype, "int64");
    assert.equal(await prod(a).compute(), 720n);
    assert.deepEqual((await prod(a, 0).compute()).toArray(), [4n, 10n, 18n]);
    assert.equal(argmin(a).dtype, "int64");
    assert.equal(await argmin(a).compute(), 0n);
    assert.deepEqual((await argmax(a, 1).compute()).toArray(), [2n, 2n]);
  } finally {
    await ctx.close();
  }
});
