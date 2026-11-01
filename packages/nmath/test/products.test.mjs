// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { Complex } from "../dist/index.js";
import {
  clip,
  diagonal,
  diff,
  dot,
  inner,
  outer,
  trace,
  vdot,
} from "../dist/numpy.js";
import { context } from "./helpers.mjs";

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/products.json", import.meta.url), "utf8"),
);

function decode(v, dtype) {
  if (dtype.startsWith("complex")) return new Complex(v[0], v[1]);
  if (dtype === "int64" || dtype === "uint64") return BigInt(v);
  if (v === null) return Number.NaN;
  if (v === "inf") return Number.POSITIVE_INFINITY;
  if (v === "-inf") return Number.NEGATIVE_INFINITY;
  return v;
}

function unpack(ctx, packed) {
  const values = packed.values.map((v) => decode(v, packed.dtype));
  return ctx.array(packed.shape.length === 0 ? values[0] : values, {
    dtype: packed.dtype,
    shape: packed.shape,
  });
}

// Signed zeros are not compared: NumPy's 0-d dot path can return +0 where
// IEEE multiplication (np.multiply and NMath) gives -0.
const zero = (x) => (Object.is(x, -0) ? 0 : x);

function encode(v) {
  if (v instanceof Complex) return [zero(v.real), zero(v.imag)];
  if (typeof v === "bigint") return String(v);
  if (typeof v === "number" && Number.isNaN(v)) return null;
  if (v === Number.POSITIVE_INFINITY) return "inf";
  if (v === Number.NEGATIVE_INFINITY) return "-inf";
  return zero(v);
}

function flat(result) {
  const list =
    result !== null && typeof result === "object" && "toArray" in result
      ? [result.toArray()].flat(Infinity)
      : [result];
  return list.map(encode);
}

function bound(ctx, b) {
  switch (b.kind) {
    case "none":
      return undefined;
    case "int":
      return BigInt(b.value);
    case "float":
      return decode(b.value, "float64");
    case "complex":
      return new Complex(b.value[0], b.value[1]);
    case "array":
      return unpack(ctx, b);
    default:
      return b.value;
  }
}

function call(ctx, c, operands) {
  const k = c.kwargs;
  switch (c.fn) {
    case "dot":
      return dot(...operands);
    case "vdot":
      return vdot(...operands);
    case "inner":
      return inner(...operands);
    case "outer":
      return outer(...operands);
    case "diagonal":
      return diagonal(operands[0], k.offset, k.axis1, k.axis2);
    case "trace":
      return trace(operands[0], k.offset, k.axis1, k.axis2);
    case "clip":
      return clip(operands[0], bound(ctx, k.a_min), bound(ctx, k.a_max));
    default:
      return diff(operands[0], k.n, k.axis);
  }
}

test("products, diagonals, clip and diff agree exactly with fixed NumPy fixtures", async () => {
  assert.equal(fixture.numpy, "2.5.4");
  assert.equal(fixture.criterion, "exact-small-values");
  const ctx = context();
  try {
    for (const c of fixture.cases) {
      const label = `${c.id}: ${JSON.stringify(c.kwargs)} ${c.operands.map((o) => `${o.dtype}${JSON.stringify(o.shape)}`).join(" ")}`;
      const operands = c.operands.map((o) => unpack(ctx, o));
      const expr = call(ctx, c, operands);
      assert.deepEqual(expr.shape, c.expected.shape, label);
      assert.equal(expr.dtype, c.expected.dtype, label);
      const result = await expr.compute();
      assert.deepEqual(
        flat(result),
        c.expected.values.map((v) =>
          Array.isArray(v) ? v.map(zero) : zero(v),
        ),
        label,
      );
      result?.dispose?.();
      for (const o of operands) o.dispose();
    }
  } finally {
    await ctx.close();
  }
});

test("products reject invalid shapes, arguments and dtypes like NumPy", async () => {
  const ctx = context();
  try {
    const m = ctx.zeros([2, 3], "float64");
    const t = ctx.zeros([2, 2, 3], "float64");
    const mismatch = { code: "NMATH_SHAPE_MISMATCH" };
    assert.throws(() => dot(m, ctx.zeros([4, 2], "float64")), mismatch);
    assert.throws(() => dot(t, ctx.zeros([2, 4, 2], "float64")), mismatch);
    assert.throws(() => dot(t, ctx.zeros([4], "float64")), mismatch);
    assert.throws(() => inner(m, ctx.zeros([4], "float64")), mismatch);
    assert.throws(() => vdot(m, ctx.zeros([5], "float64")), mismatch);
    assert.throws(() => dot(m, [1, 2, 3]), TypeError);
    assert.throws(() => outer(1, m), TypeError);
    const text = ctx.array(["a", "b"], { dtype: "str" });
    assert.throws(() => dot(text, text), { code: "NMATH_DTYPE_UNSUPPORTED" });
    assert.throws(() => diff(text), { code: "NMATH_DTYPE_UNSUPPORTED" });

    const vector = ctx.zeros([3], "int32");
    assert.throws(() => diagonal(vector), mismatch);
    assert.throws(() => diagonal(m, 0, 1, 1), mismatch);
    assert.throws(() => diagonal(m, 0, -1, 1), mismatch);
    assert.throws(() => diagonal(m, 0, 0, 2), {
      code: "NMATH_INDEX_OUT_OF_BOUNDS",
    });
    assert.throws(() => diagonal(m, 0.5), { code: "NMATH_INDEX_INVALID" });
    assert.throws(() => trace(vector), mismatch);

    const flags = ctx.array([true, false], { dtype: "bool" });
    assert.throws(() => clip(flags, null, undefined), {
      code: "NMATH_DTYPE_UNSUPPORTED",
    });
    const small = ctx.array([1, 2], { dtype: "int8" });
    assert.throws(() => clip(small, 1000n, null), {
      code: "NMATH_DTYPE_UNSUPPORTED",
    });

    assert.throws(() => diff(vector, -1), { code: "NMATH_INDEX_INVALID" });
    assert.throws(() => diff(vector, 1.5), { code: "NMATH_INDEX_INVALID" });
    assert.throws(() => diff(vector, 1, 1), {
      code: "NMATH_INDEX_OUT_OF_BOUNDS",
    });
    const scalar = ctx.array(5, { dtype: "int32", shape: [] });
    assert.throws(() => diff(scalar), mismatch);
    assert.equal(await diff(scalar, 0).compute(), 5);
  } finally {
    await ctx.close();
  }
});

test("diff, diagonal and trace keep temporal and string semantics", async () => {
  const ctx = context();
  try {
    const days = ctx.array(new BigInt64Array([1n, 4n, 9n, 16n]), {
      dtype: "datetime64",
      unit: "D",
    });
    const first = diff(days);
    assert.equal(first.dtype, "timedelta64");
    assert.equal(first.unit, "D");
    const once = await first.compute();
    assert.deepEqual([...once.toTypedArray()], [3n, 5n, 7n]);
    const twice = await diff(days, 2).compute();
    assert.equal(twice.dtype, "timedelta64");
    assert.deepEqual([...twice.toTypedArray()], [2n, 2n]);

    const durations = ctx.array(new BigInt64Array([1n, 2n, 3n, 4n]), {
      dtype: "timedelta64",
      unit: "s",
      shape: [2, 2],
    });
    const total = await trace(durations).compute();
    assert.equal(total.ticks, 5n);
    assert.equal(total.dtype, "timedelta64");

    const text = ctx.array(["a", "bb", "c", "d"], {
      dtype: "str",
      shape: [2, 2],
    });
    const diagonalText = await diagonal(text).compute();
    assert.deepEqual(diagonalText.toArray(), ["a", "d"]);
    for (const r of [once, twice, diagonalText]) r.dispose();
  } finally {
    await ctx.close();
  }
});

test("diff evaluates each difference input once", async () => {
  const ctx = context();
  try {
    // Both slices must read one draw, and the generator advances once.
    const draw = await ctx.generator(5).random([6]).compute();
    const values = Array.from(draw.toTypedArray());
    const rng = ctx.generator(5);
    const result = await diff(rng.random([6])).compute();
    assert.deepEqual(
      Array.from(result.toTypedArray()),
      values.slice(1).map((v, i) => v - values[i]),
    );
    const next = await rng.random([6]).compute();
    const reference = ctx.generator(5);
    await reference.random([6]).compute();
    const expected = await reference.random([6]).compute();
    assert.deepEqual(
      Array.from(next.toTypedArray()),
      Array.from(expected.toTypedArray()),
    );
    // A high order reuses each difference instead of doubling the work.
    const deep = await diff(ctx.zeros([64], "float64"), 60).compute();
    assert.deepEqual(deep.toArray(), [0, 0, 0, 0]);
    for (const array of [draw, result, next, expected, deep]) array.dispose();
  } finally {
    await ctx.close();
  }
});

test("diagonal and trace of empty retained axes skip diagonal selectors", async () => {
  const ctx = context();
  try {
    const a = ctx.zeros([0, 100000000, 100000000], "float64");
    const d = diagonal(a, 0, 1, 2);
    assert.deepEqual(d.shape, [0, 100000000]);
    assert.deepEqual(diagonal(a, -3, 2, 1).shape, [0, 99999997]);
    const t = trace(a, 0, 1, 2);
    assert.deepEqual(t.shape, [0]);
    const [dv, tv] = await Promise.all([d.compute(), t.compute()]);
    assert.deepEqual(dv.shape, [0, 100000000]);
    assert.deepEqual(tv.toArray(), []);
    dv.dispose();
    tv.dispose();
  } finally {
    await ctx.close();
  }
});

test("products are lazy and snapshot inputs at execution start", async () => {
  const ctx = context();
  try {
    const a = ctx.array([1, 2, 3, 4, 5, 6], { dtype: "int32", shape: [2, 3] });
    const b = ctx.array([1, 1, 1], { dtype: "int32" });
    const product = dot(a, b);
    const tr = trace(a, 1);
    const clipped = clip(a, 2n, 4n);
    await a.set([0, 0], 10);
    const [p, t, c] = await Promise.all([
      product.compute(),
      tr.compute(),
      clipped.compute(),
    ]);
    assert.deepEqual(p.toArray(), [15, 15]);
    assert.equal(t, 8n);
    assert.deepEqual(c.toArray(), [
      [4, 2, 3],
      [4, 4, 4],
    ]);
    p.dispose();
    c.dispose();
  } finally {
    await ctx.close();
  }
});
