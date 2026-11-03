// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { Complex, Temporal } from "../dist/index.js";
import {
  array_split,
  diff,
  dsplit,
  hsplit,
  repeat,
  roll,
  split,
  tile,
  vsplit,
} from "../dist/numpy.js";
import { context } from "./helpers.mjs";

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/splitting.json", import.meta.url), "utf8"),
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
  const { dtype, shape } = packed;
  if (dtype === "datetime64" || dtype === "timedelta64") {
    return ctx.array(BigInt64Array.from(packed.values, BigInt), {
      dtype,
      unit: packed.unit,
      shape,
    });
  }
  const values = packed.values.map((v) => decode(v, dtype));
  return ctx.array(shape.length === 0 ? values[0] : values, { dtype, shape });
}

function edge(ctx, e) {
  switch (e?.kind) {
    case undefined:
      return undefined;
    case "int":
      return BigInt(e.value);
    case "complex":
      return new Complex(e.value[0], e.value[1]);
    case "array":
      return unpack(ctx, e);
    default:
      return e.value;
  }
}

function encode(v) {
  if (v instanceof Complex) return [v.real, v.imag];
  if (v instanceof Temporal) return String(v.ticks);
  if (typeof v === "bigint") return String(v);
  if (typeof v === "number" && Number.isNaN(v)) return null;
  return v;
}

function call(ctx, c, a) {
  const k = c.kwargs;
  switch (c.fn) {
    case "split":
      return split(a, k.sections, k.axis);
    case "array_split":
      return array_split(a, k.sections, k.axis);
    case "hsplit":
      return hsplit(a, k.sections);
    case "vsplit":
      return vsplit(a, k.sections);
    case "dsplit":
      return dsplit(a, k.sections);
    case "repeat":
      return repeat(a, k.repeats, k.axis);
    case "tile":
      return tile(a, k.reps);
    case "roll":
      return roll(a, k.shift, k.axis);
    default:
      return diff(a, k.n, k.axis, {
        prepend: edge(ctx, k.prepend),
        append: edge(ctx, k.append),
      });
  }
}

async function check(expr, expected, label) {
  assert.deepEqual(expr.shape, expected.shape, label);
  assert.equal(expr.dtype, expected.dtype, label);
  if (expected.width !== undefined) {
    assert.equal(expr.stringWidth, expected.width, label);
  }
  if (expected.unit !== undefined)
    assert.equal(expr.unit, expected.unit, label);
  const result = await expr.compute();
  const values =
    result === null || typeof result !== "object" || !("toArray" in result)
      ? [encode(result)]
      : [result.toArray()].flat(Infinity).map(encode);
  assert.deepEqual(values, expected.values, label);
  result?.dispose?.();
}

test("splitting, repetition, roll and diff edges agree exactly with fixed NumPy fixtures", async () => {
  assert.equal(fixture.numpy, "2.5.4");
  assert.equal(fixture.criterion, "exact");
  const ctx = context();
  try {
    for (const c of fixture.cases) {
      const label = `${c.id}: ${JSON.stringify(c.kwargs)} ${c.operand.dtype}${JSON.stringify(c.operand.shape)}`;
      const a = unpack(ctx, c.operand);
      const result = call(ctx, c, a);
      if (Array.isArray(c.expected)) {
        assert.ok(Object.isFrozen(result), label);
        assert.equal(result.length, c.expected.length, label);
        for (let i = 0; i < result.length; i++) {
          await check(result[i], c.expected[i], `${label} [${i}]`);
        }
      } else {
        await check(result, c.expected, label);
      }
      a.dispose();
    }
  } finally {
    await ctx.close();
  }
});

test("split parts are views and repeat, tile and roll return copies", async () => {
  const ctx = context();
  try {
    const a = ctx.array([1, 2, 3, 4], { dtype: "int32" });
    const [left, right] = split(a, 2);
    const repeated = await repeat(a, 2).compute();
    const tiled = await tile(a, 1).compute();
    const rolled = await roll(a, 0).compute();
    const first = await left.compute();
    await a.set([0], 10);
    // Like NumPy, split parts share the original storage.
    assert.deepEqual(first.toArray(), [10, 2]);
    assert.deepEqual((await right.compute()).toArray(), [3, 4]);
    assert.deepEqual(repeated.toArray(), [1, 1, 2, 2, 3, 3, 4, 4]);
    assert.deepEqual(tiled.toArray(), [1, 2, 3, 4]);
    assert.deepEqual(rolled.toArray(), [1, 2, 3, 4]);
  } finally {
    await ctx.close();
  }
});

test("splitting and repetition reject invalid arguments like NumPy", async () => {
  const ctx = context();
  try {
    const v = ctx.zeros([6], "float64");
    const m = ctx.zeros([2, 3], "float64");
    const s = ctx.zeros([], "float64");
    const invalid = { code: "NMATH_INDEX_INVALID" };
    const mismatch = { code: "NMATH_SHAPE_MISMATCH" };
    const bounds = { code: "NMATH_INDEX_OUT_OF_BOUNDS" };
    assert.throws(() => split(v, 4), mismatch);
    assert.throws(() => split(v, 0), invalid);
    assert.throws(() => array_split(v, -1), invalid);
    assert.throws(() => split(v, 1.5), invalid);
    assert.throws(() => split(v, [1.5]), invalid);
    assert.throws(() => split(v, 2, 1), bounds);
    assert.throws(() => split(s, 1), bounds);
    assert.throws(() => hsplit(s, 1), mismatch);
    assert.throws(() => vsplit(v, 2), mismatch);
    assert.throws(() => dsplit(m, 2), mismatch);
    assert.throws(() => repeat(v, -1), invalid);
    assert.throws(() => repeat(m, [1, 2], 1), {
      code: "NMATH_SHAPE_BROADCAST",
    });
    assert.throws(() => repeat(m, 1, 2), bounds);
    assert.throws(() => repeat(s, 1, 1), bounds);
    assert.throws(() => tile(v, -2), invalid);
    assert.throws(() => tile(v, [1, 0.5]), invalid);
    assert.throws(() => roll(m, 1, 2), bounds);
    assert.throws(() => roll(m, [1, 2, 3], [0, 1]), {
      code: "NMATH_SHAPE_BROADCAST",
    });
    assert.throws(() => roll(m, 0.5, 0), invalid);
    assert.throws(() => roll(m, [], [0, 1]), {
      code: "NMATH_SHAPE_BROADCAST",
    });
    assert.throws(() => diff(v, 1, -1, { prepend: "a" }), TypeError);
    assert.throws(() => diff(v, 1, -1, { prepended: 1 }), invalid);
    assert.throws(
      () => diff(m, 1, 1, { prepend: ctx.zeros([3, 1], "float64") }),
      mismatch,
    );
    assert.throws(() => split([1, 2], 2), TypeError);
    await assert.rejects(roll(v, 1).compute({ device: "gpu" }), {
      code: "NMATH_GPU_UNSUPPORTED_OPERATION",
    });
  } finally {
    await ctx.close();
  }
});
