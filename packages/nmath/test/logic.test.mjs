// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { Complex, NMathError } from "../dist/index.js";
import * as np from "../dist/numpy.js";
import { context } from "./helpers.mjs";

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/logic.json", import.meta.url), "utf8"),
);

function real(v) {
  if (v === "nan") return Number.NaN;
  if (v === "inf") return Number.POSITIVE_INFINITY;
  if (v === "-inf") return Number.NEGATIVE_INFINITY;
  return v;
}

function decodeValue(v, dtype) {
  if (dtype.startsWith("complex")) return new Complex(real(v[0]), real(v[1]));
  if (dtype === "int64" || dtype === "uint64") return BigInt(v);
  if (dtype.startsWith("int") || dtype.startsWith("uint")) return Number(v);
  if (dtype === "bool") return v;
  return real(v);
}

function decode(ctx, input) {
  if ("array" in input) {
    const { dtype, shape, values } = input.array;
    return ctx.array(
      values.map((v) => decodeValue(v, dtype)),
      { dtype, shape },
    );
  }
  switch (input.scalar) {
    case "bool":
      return input.value;
    case "int":
      return BigInt(input.value);
    case "complex":
      return new Complex(real(input.value[0]), real(input.value[1]));
    default:
      return real(input.value);
  }
}

function expectedValues(expected) {
  return expected.values.map((v) =>
    expected.dtype === "bool" ? v : BigInt(v),
  );
}

async function check(expr, expected, label) {
  assert.equal(expr.dtype, expected.dtype === "bool" ? "bool" : "int64", label);
  assert.deepEqual(expr.shape, expected.shape, label);
  const result = await expr.compute();
  if (expected.shape.length === 0) {
    assert.equal(
      typeof result,
      expected.dtype === "bool" ? "boolean" : "bigint",
      label,
    );
    assert.deepEqual([result], expectedValues(expected), label);
  } else {
    assert.deepEqual(
      [result.toArray()].flat(Infinity),
      expectedValues(expected),
      label,
    );
    result.dispose();
  }
}

function options(args) {
  const out = {};
  for (const [key, value] of Object.entries(args)) {
    out[key] = value;
  }
  return out;
}

test("any, all and count_nonzero match NumPy", async () => {
  assert.equal(fixture.numpy, "2.5.4");
  const ctx = context();
  try {
    let checked = 0;
    for (const [i, c] of fixture.cases.entries()) {
      if (c.fn !== "reductions") continue;
      const a = decode(ctx, c.inputs[0]);
      for (const r of c.results) {
        const axis = r.args.axis ?? undefined;
        const opts = { keepdims: r.args.keepdims };
        for (const fn of ["any", "all", "count_nonzero"]) {
          const label = `case ${i}: ${fn} ${JSON.stringify(r.args)} ${a.dtype} ${JSON.stringify(a.shape)}`;
          await check(np[fn](a, axis, opts), r[fn], label);
          checked++;
        }
      }
      a.dispose();
    }
    assert.ok(checked > 700, `${checked} reduction results`);
  } finally {
    await ctx.close();
  }
});

test("isclose, allclose, array_equal and array_equiv match NumPy", async () => {
  const ctx = context();
  try {
    let checked = 0;
    for (const [i, c] of fixture.cases.entries()) {
      if (c.fn === "reductions") continue;
      const [a, b] = c.inputs.map((input) => decode(ctx, input));
      const label = `case ${i}: ${c.fn} ${JSON.stringify(c.inputs.map((x) => x.array?.dtype ?? x.scalar))} ${JSON.stringify(c.args)}`;
      const args = options(c.args);
      const expr =
        c.fn === "array_equiv" ? np.array_equiv(a, b) : np[c.fn](a, b, args);
      await check(expr, c.expected, label);
      checked++;
    }
    assert.ok(checked > 150, `${checked} closeness results`);
  } finally {
    await ctx.close();
  }
});

test("truth reductions and closeness tests keep NumPy result types", async () => {
  const ctx = context();
  try {
    const x = ctx.array([1, 0, 3], { dtype: "int32", shape: [3] });
    const y = ctx.array([1, 2], { dtype: "float32", shape: [2] });
    assert.equal(await np.count_nonzero(x).compute(), 2n);
    assert.equal(await np.any(x).compute(), true);
    assert.equal(await np.all(x).compute(), false);
    // Different shapes are known before execution: the expression is a
    // rank-0 constant that computes to false.
    const different = np.array_equal(x, y);
    assert.equal(different.dtype, "bool");
    assert.deepEqual(different.shape, []);
    assert.equal(await different.compute(), false);
    assert.equal(await np.array_equiv(x, y).compute(), false);
    assert.equal(await np.allclose(x, x).compute(), true);
    // Rank-0 isclose computes to a JS boolean.
    const zero = ctx.array([0.5], { dtype: "float64", shape: [] });
    assert.equal(await np.isclose(zero, 0.5).compute(), true);
    // The expressions stay usable as lazy operands.
    const combined = np.logical_and(np.isclose(x, x), np.not_equal(x, 0));
    assert.deepEqual((await combined.compute()).toArray(), [true, false, true]);
  } finally {
    await ctx.close();
  }
});

test("truth and closeness rejections and GPU admission are stable", async () => {
  const ctx = context();
  try {
    const text = ctx.array(["a", ""], { dtype: "str", shape: [2] });
    const x = ctx.array([1, 2], { dtype: "float64", shape: [2] });
    const code = (code) => (error) =>
      error instanceof NMathError && error.code === code;
    for (const fn of ["any", "all", "count_nonzero"]) {
      assert.throws(() => np[fn](text), code("NMATH_DTYPE_UNSUPPORTED"));
    }
    assert.throws(() => np.isclose(x, text), code("NMATH_DTYPE_UNSUPPORTED"));
    assert.throws(
      () => np.any(ctx.zeros([2], "datetime64")),
      code("NMATH_DTYPE_UNSUPPORTED"),
    );
    assert.throws(
      () => np.any(x, undefined, { where: true }),
      code("NMATH_INDEX_INVALID"),
    );
    assert.throws(
      () => np.isclose(x, x, { out: x }),
      code("NMATH_INDEX_INVALID"),
    );
    assert.throws(
      () => np.isclose(x, x, { equal_nan: 1 }),
      code("NMATH_INDEX_INVALID"),
    );
    assert.throws(
      () => np.array_equal(x, x, { equal_nan: "yes" }),
      code("NMATH_INDEX_INVALID"),
    );
    assert.throws(() => np.any(x, 1), code("NMATH_INDEX_OUT_OF_BOUNDS"));
    assert.throws(() => np.isclose(1, 2), TypeError);
    assert.throws(() => np.any([1, 2]), TypeError);
    await assert.rejects(
      np.isclose(x, x).compute({ device: "gpu" }),
      code("NMATH_GPU_UNSUPPORTED_OPERATION"),
    );
  } finally {
    await ctx.close();
  }
});
