// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { Complex, Temporal } from "../dist/index.js";
import {
  append,
  atleast_1d,
  atleast_2d,
  atleast_3d,
  column_stack,
  concatenate,
  dstack,
  hstack,
  stack,
  vstack,
} from "../dist/numpy.js";
import { context } from "./helpers.mjs";

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/joining.json", import.meta.url), "utf8"),
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

function encode(v) {
  if (v instanceof Complex) return [v.real, v.imag];
  if (v instanceof Temporal) return String(v.ticks);
  if (typeof v === "bigint") return String(v);
  if (typeof v === "number" && Number.isNaN(v)) return null;
  return v;
}

function flat(result) {
  return [result.toArray()].flat(Infinity).map(encode);
}

function call(c, operands) {
  const axis = c.kwargs.axis;
  switch (c.fn) {
    case "concatenate":
      return concatenate(operands, axis);
    case "stack":
      return stack(operands, axis);
    case "vstack":
      return vstack(operands);
    case "hstack":
      return hstack(operands);
    case "dstack":
      return dstack(operands);
    case "column_stack":
      return column_stack(operands);
    case "append":
      return append(operands[0], operands[1], axis);
    case "atleast_1d":
      return atleast_1d(...operands);
    case "atleast_2d":
      return atleast_2d(...operands);
    default:
      return atleast_3d(...operands);
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
      : flat(result);
  assert.deepEqual(values, expected.values, label);
  result?.dispose?.();
}

test("joining functions agree exactly with fixed NumPy fixtures", async () => {
  assert.equal(fixture.numpy, "2.5.4");
  assert.equal(fixture.criterion, "exact");
  const ctx = context();
  try {
    for (const c of fixture.cases) {
      const label = `${c.id}: ${JSON.stringify(c.kwargs)} ${c.operands.map((o) => `${o.dtype}${JSON.stringify(o.shape)}`).join(" ")}`;
      const operands = c.operands.map((o) => unpack(ctx, o));
      const result = call(c, operands);
      if (Array.isArray(c.expected)) {
        assert.ok(Object.isFrozen(result), label);
        assert.equal(result.length, c.expected.length, label);
        for (let i = 0; i < result.length; i++) {
          await check(result[i], c.expected[i], `${label} [${i}]`);
        }
      } else {
        await check(result, c.expected, label);
      }
      for (const o of operands) o.dispose();
    }
  } finally {
    await ctx.close();
  }
});

test("concatenate copies its operands and reads them at execution start", async () => {
  const ctx = context();
  try {
    const a = ctx.array([1, 2], { dtype: "int32" });
    const b = ctx.array([3], { dtype: "int32" });
    const joined = concatenate([a, b]);
    // Changes made before the job starts are visible to it.
    await a.set([0], 10);
    const first = await joined.compute();
    assert.deepEqual(first.toArray(), [10, 2, 3]);
    // The result is an independent copy, as NumPy's concatenate returns.
    await a.set([1], 20);
    assert.deepEqual(first.toArray(), [10, 2, 3]);
    assert.deepEqual((await joined.compute()).toArray(), [10, 20, 3]);
    // One shared operand is read once and copied twice.
    const twice = concatenate([a, a, a.get(0).reshape([1])]);
    assert.deepEqual((await twice.compute()).toArray(), [10, 20, 10, 20, 10]);
    // Computed operands join like arrays.
    const computed = stack([a.binaryOp("mul", 2), a], -1);
    assert.deepEqual((await computed.compute()).toArray(), [
      [20, 10],
      [40, 20],
    ]);
  } finally {
    await ctx.close();
  }
});

test("joining rejects invalid operands like NumPy", async () => {
  const ctx = context();
  const other = context();
  try {
    const m = ctx.zeros([2, 3], "float64");
    const v = ctx.zeros([3], "float64");
    const s = ctx.zeros([], "float64");
    const mismatch = { code: "NMATH_SHAPE_MISMATCH" };
    assert.throws(() => concatenate([]), mismatch);
    assert.throws(() => concatenate([s, s]), mismatch);
    assert.throws(() => concatenate([m, v]), mismatch);
    assert.throws(
      () => concatenate([m, ctx.zeros([2, 2], "float64")]),
      mismatch,
    );
    assert.throws(() => concatenate([m, m], 2), {
      code: "NMATH_INDEX_OUT_OF_BOUNDS",
    });
    assert.throws(() => concatenate([m, m], -3), {
      code: "NMATH_INDEX_OUT_OF_BOUNDS",
    });
    assert.throws(() => concatenate([m, m], 0.5), {
      code: "NMATH_INDEX_INVALID",
    });
    assert.throws(() => concatenate([m, other.zeros([2, 3], "float64")]), {
      code: "NMATH_CONTEXT_MISMATCH",
    });
    assert.throws(() => concatenate([ctx.array(["a"], { dtype: "str" }), v]), {
      code: "NMATH_DTYPE_UNSUPPORTED",
    });
    assert.throws(() => concatenate(m), TypeError);
    assert.throws(() => concatenate([m, [1, 2, 3]]), TypeError);
    assert.throws(() => stack([]), mismatch);
    assert.throws(() => stack([v, ctx.zeros([2], "float64")]), mismatch);
    assert.throws(() => stack([v, m]), mismatch);
    assert.throws(() => stack([v, v], 2), {
      code: "NMATH_INDEX_OUT_OF_BOUNDS",
    });
    assert.throws(() => stack([v, v], -3), {
      code: "NMATH_INDEX_OUT_OF_BOUNDS",
    });
    assert.throws(() => vstack([m, ctx.zeros([2], "float64")]), mismatch);
    assert.throws(() => hstack([v, m]), mismatch);
    assert.throws(() => dstack([]), mismatch);
    assert.throws(() => column_stack([v, ctx.zeros([2], "float64")]), mismatch);
    // NumPy returns an empty tuple when no operand is given.
    assert.deepEqual(atleast_1d(), []);
    assert.ok(Object.isFrozen(atleast_2d()));
    // Datetimes join only within one unit; NumPy would convert to the finer unit.
    const seconds = ctx.array(new BigInt64Array([1n]), {
      dtype: "datetime64",
      unit: "s",
    });
    const days = ctx.array(new BigInt64Array([1n]), {
      dtype: "datetime64",
      unit: "D",
    });
    assert.throws(() => concatenate([seconds, days]), {
      code: "NMATH_DTYPE_UNSUPPORTED",
    });
    await assert.rejects(concatenate([v, v]).compute({ device: "gpu" }), {
      code: "NMATH_GPU_UNSUPPORTED_OPERATION",
    });
  } finally {
    await ctx.close();
    await other.close();
  }
});

test("concatenate charges its operands and result to the context budget", async () => {
  // Two 8 KiB operands, their execution snapshots and the 16 KiB result do
  // not fit in 40 000 bytes.
  const ctx = context({ cpuMemoryBudget: 40_000n });
  try {
    const a = ctx.zeros([1024], "float64");
    const b = ctx.zeros([1024], "float64");
    await assert.rejects(concatenate([a, b]).compute(), {
      code: "NMATH_MEMORY_BUDGET_EXCEEDED",
    });
    // The failed job released its charges; only the two operands remain.
    assert.equal(ctx.metrics().cpuMemory.used, 16_384n);
  } finally {
    await ctx.close();
  }
});

test("cancellation stops a queued concatenation", async () => {
  const ctx = context();
  try {
    const a = ctx.zeros([4], "int8");
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(
      concatenate([a, a]).compute({ signal: controller.signal }),
      { code: "NMATH_CANCELLED" },
    );
  } finally {
    await ctx.close();
  }
});
