// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { Complex, Temporal } from "../dist/index.js";
import {
  empty,
  empty_like,
  eye,
  full_like,
  identity,
  linspace,
  logspace,
  ones_like,
  zeros_like,
} from "../dist/numpy.js";
import { context } from "./helpers.mjs";

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/creation.json", import.meta.url), "utf8"),
);

function view(hex) {
  return new DataView(Uint8Array.from(Buffer.from(hex, "hex")).buffer);
}

function decode(hex, width) {
  const v = view(hex);
  return width === 8
    ? v.getFloat64(0)
    : width === 4
      ? v.getFloat32(0)
      : v.getFloat16(0);
}

function f64(hex) {
  return decode(hex, 8);
}

function bitsOf(value, width) {
  const view = new DataView(new ArrayBuffer(width));
  if (width === 8) view.setFloat64(0, value);
  else if (width === 4) view.setFloat32(0, value);
  else view.setFloat16(0, value);
  return Buffer.from(view.buffer).toString("hex");
}

/** Signed position of a float in the ordered sequence of representable values. */
function ordinal(value, width) {
  const view = new DataView(new ArrayBuffer(8));
  if (width === 8) {
    view.setFloat64(0, value);
    const b = view.getBigInt64(0);
    return b < 0n ? -(b & 0x7fff_ffff_ffff_ffffn) : b;
  }
  view.setFloat32(0, value);
  const b = BigInt(view.getInt32(0));
  return b < 0n ? -(b & 0x7fff_ffffn) : b;
}

const WIDTH = {
  float16: 2,
  float32: 4,
  float64: 8,
  complex64: 4,
  complex128: 8,
};

/** Exact element bits; NaN matches NaN regardless of payload or sign. */
function sameBits(actual, expectedHex, width, label) {
  const expectedNaN = Number.isNaN(decode(expectedHex, width));
  if (expectedNaN) assert.ok(Number.isNaN(actual), label);
  else assert.equal(bitsOf(actual, width), expectedHex, label);
}

function assertExact(array, expected, label) {
  assert.equal(array.dtype, expected.dtype, label);
  assert.deepEqual(array.shape, expected.shape, label);
  const dtype = expected.dtype;
  if (dtype === "complex64" || dtype === "complex128") {
    const parts = array.toTypedArray();
    expected.values.forEach(([re, im], i) => {
      sameBits(parts[2 * i], re, WIDTH[dtype], `${label}[${i}].real`);
      sameBits(parts[2 * i + 1], im, WIDTH[dtype], `${label}[${i}].imag`);
    });
  } else if (dtype in WIDTH) {
    const values = array.toTypedArray();
    expected.values.forEach((hex, i) =>
      sameBits(values[i], hex, WIDTH[dtype], `${label}[${i}]`),
    );
  } else if (dtype === "bool") {
    assert.deepEqual(array.toArray().flat(Infinity), expected.values, label);
  } else {
    const values = Array.from(array.toTypedArray(), String);
    assert.deepEqual(values, expected.values, label);
  }
}

test("linspace matches NumPy 2.5.4 bit for bit, including retstep", async () => {
  assert.equal(fixture.numpy, "2.5.4");
  const ctx = context();
  try {
    let count = 0;
    for (const c of fixture.cases.filter((c) => c.fn === "linspace")) {
      const a = c.args;
      const label = JSON.stringify(a);
      const options = { endpoint: a.endpoint };
      if (a.dtype !== null) options.dtype = a.dtype;
      const samples = linspace(ctx, f64(a.start), f64(a.stop), a.num, options);
      assertExact(samples, c.expected, label);
      const [withStep, step] = linspace(ctx, f64(a.start), f64(a.stop), a.num, {
        ...options,
        retstep: true,
      });
      assertExact(withStep, c.expected, label);
      sameBits(step, c.step, 8, `${label} step`);
      count++;
    }
    assert.ok(count >= 40);
    // Defaults: num 50, endpoint true, float64.
    const defaults = linspace(ctx, 0, 1);
    assert.equal(defaults.dtype, "float64");
    assert.deepEqual(defaults.shape, [50]);
    assert.equal(defaults.item(49), 1);
    assert.ok(Object.isFrozen(linspace(ctx, 0, 1, 3, { retstep: true })));
  } finally {
    await ctx.close();
  }
});

test("logspace matches NumPy 2.5.4 within the stated ULP tolerance", async () => {
  const ctx = context();
  try {
    let maxUlp = 0n;
    for (const c of fixture.cases.filter((c) => c.fn === "logspace")) {
      const a = c.args;
      const label = JSON.stringify(a);
      const options = { endpoint: a.endpoint, base: f64(a.base) };
      if (a.dtype !== null) options.dtype = a.dtype;
      const r = logspace(ctx, f64(a.start), f64(a.stop), a.num, options);
      assert.equal(r.dtype, c.expected.dtype, label);
      assert.deepEqual(r.shape, c.expected.shape, label);
      const dtype = c.expected.dtype;
      if (dtype === "float64" || dtype === "float32") {
        // JS Math.pow and the platform C pow may round differently.
        const width = WIDTH[dtype];
        const tolerance = dtype === "float64" ? 4n : 1n;
        const values = r.toTypedArray();
        c.expected.values.forEach((hex, i) => {
          const expected = decode(hex, width);
          if (Number.isNaN(expected)) {
            assert.ok(Number.isNaN(values[i]), label);
            return;
          }
          if (!Number.isFinite(expected)) {
            assert.equal(values[i], expected, label);
            return;
          }
          let ulp = ordinal(values[i], width) - ordinal(expected, width);
          if (ulp < 0n) ulp = -ulp;
          if (dtype === "float64" && ulp > maxUlp) maxUlp = ulp;
          assert.ok(
            ulp <= tolerance,
            `${label}[${i}]: ${values[i]} vs ${expected}`,
          );
        });
      } else {
        assertExact(r, c.expected, label);
      }
    }
    assert.ok(maxUlp <= 4n);
    assert.deepEqual(logspace(ctx, 0, 1).shape, [50]);
  } finally {
    await ctx.close();
  }
});

test("eye and identity match NumPy 2.5.4 exactly", async () => {
  const ctx = context();
  try {
    for (const c of fixture.cases) {
      const a = c.args;
      const label = JSON.stringify(a);
      if (c.fn === "eye") {
        assertExact(
          eye(ctx, a.N, a.M ?? undefined, { k: a.k, dtype: a.dtype }),
          c.expected,
          label,
        );
      } else if (c.fn === "identity") {
        assertExact(identity(ctx, a.n, a.dtype), c.expected, label);
      }
    }
    const square = eye(ctx, 2);
    assert.equal(square.dtype, "float64");
    assert.deepEqual(square.toArray(), [
      [1, 0],
      [0, 1],
    ]);
    assert.deepEqual(eye(ctx, 2, 2, { dtype: "int64" }).toArray(), [
      [1n, 0n],
      [0n, 1n],
    ]);
    const complex = identity(ctx, 2, "complex128").toArray();
    assert.ok(complex[0][0] instanceof Complex);
    assert.deepEqual(
      [complex[0][0].real, complex[0][0].imag, complex[0][1].real],
      [1, 0, 0],
    );
  } finally {
    await ctx.close();
  }
});

test("like constructors keep shape, dtype, metadata and context", async () => {
  const ctx = context();
  const other = context();
  try {
    const a = ctx.array([1.5, 2, 3, 4, 5, 6], {
      dtype: "float32",
      shape: [2, 3],
    });
    const z = zeros_like(a);
    assert.equal(z.context, ctx);
    assert.equal(z.dtype, "float32");
    assert.deepEqual(z.shape, [2, 3]);
    assert.deepEqual(Array.from(z.toTypedArray()), [0, 0, 0, 0, 0, 0]);
    assert.equal(z.sharesStorage(a), false);
    assert.deepEqual(ones_like(a, "int64").toArray(), [
      [1n, 1n, 1n],
      [1n, 1n, 1n],
    ]);
    assert.deepEqual(
      ones_like(a, "bool").toArray().flat(),
      Array(6).fill(true),
    );
    assert.deepEqual(
      Array.from(full_like(a, 7.25).toTypedArray()),
      Array(6).fill(7.25),
    );
    assert.deepEqual(full_like(a, 3, "int32").toArray(), [
      [3, 3, 3],
      [3, 3, 3],
    ]);
    const c = full_like(a, new Complex(1, -2), "complex64").toArray();
    assert.deepEqual([c[1][2].real, c[1][2].imag], [1, -2]);
    assert.deepEqual(
      Array.from(empty_like(a).toTypedArray()),
      Array(6).fill(0),
    );
    assert.deepEqual(empty_like(a, "uint8").toArray(), [
      [0, 0, 0],
      [0, 0, 0],
    ]);
    // Expressions use the context of their array operands.
    const e = a.add(1);
    assert.equal(zeros_like(e).context, ctx);
    assert.deepEqual(zeros_like(e).shape, [2, 3]);
    assert.equal(zeros_like(e).dtype, "float32");
    // Strings keep their width; ones become "1" as in NumPy.
    const s = ctx.array(["ab", "c"], { dtype: "str" });
    assert.deepEqual(zeros_like(s).toArray(), ["", ""]);
    assert.equal(zeros_like(s).stringWidth, 2);
    assert.deepEqual(ones_like(s).toArray(), ["1", "1"]);
    assert.deepEqual(full_like(s, "xy").toArray(), ["xy", "xy"]);
    assert.throws(() => full_like(s, "xyz"), RangeError);
    // Temporal values keep their unit; ones are one tick.
    const d = ctx.array([5n, 6n], { dtype: "datetime64", unit: "s" });
    assert.equal(zeros_like(d).unit, "s");
    assert.deepEqual(
      ones_like(d)
        .toArray()
        .map((t) => [t.ticks, t.unit]),
      [
        [1n, "s"],
        [1n, "s"],
      ],
    );
    assert.deepEqual(
      full_like(d, new Temporal(9n, "s", "datetime64"))
        .toArray()
        .map((t) => t.ticks),
      [9n, 9n],
    );
    assert.throws(
      () => full_like(d, new Temporal(9n, "ms", "datetime64")),
      TypeError,
    );
    // empty zero-fills.
    const blank = empty(other, [2, 2], "int32");
    assert.equal(blank.context, other);
    assert.deepEqual(blank.toArray(), [
      [0, 0],
      [0, 0],
    ]);
  } finally {
    await ctx.close();
    await other.close();
  }
});

test("creation helpers reject invalid arguments and budget overruns", async () => {
  const ctx = context();
  const tiny = context({ cpuMemoryBudget: 64n });
  try {
    const shape = { code: "NMATH_SHAPE_MISMATCH" };
    assert.throws(() => linspace(ctx, 0, 1, -1), shape);
    assert.throws(() => linspace(ctx, 0, 1, 2.5), shape);
    assert.throws(() => linspace(ctx, 0, 1, 2 ** 32), shape);
    assert.throws(() => logspace(ctx, 0, 1, -3), shape);
    assert.throws(() => eye(ctx, -1), shape);
    assert.throws(() => eye(ctx, 2, 1.5), shape);
    assert.throws(() => identity(ctx, Number.NaN), shape);
    assert.throws(() => empty(ctx, [-1], "float64"), shape);
    assert.throws(() => linspace(ctx, "0", 1), TypeError);
    assert.throws(() => linspace(ctx, 0, 1n), TypeError);
    assert.throws(() => linspace(ctx, 0, 1, 3, { axis: 0 }), TypeError);
    assert.throws(() => linspace(ctx, 0, 1, 3, { endpoint: 1 }), TypeError);
    assert.throws(() => logspace(ctx, 0, 1, 3, { base: 2n }), TypeError);
    assert.throws(() => eye(ctx, 3, 3, { k: 0.5 }), TypeError);
    assert.throws(() => eye(ctx, 3, 3, { order: "C" }), TypeError);
    const dtype = { code: "NMATH_DTYPE_UNSUPPORTED" };
    assert.throws(() => linspace(ctx, 0, 1, 3, { dtype: "str" }), dtype);
    assert.throws(() => logspace(ctx, 0, 1, 3, { dtype: "datetime64" }), dtype);
    assert.throws(() => eye(ctx, 2, 2, { dtype: "float128" }), dtype);
    // Integer casts outside the target range are platform-dependent in NumPy.
    assert.throws(
      () => linspace(ctx, 0, 300, 3, { dtype: "uint8" }),
      RangeError,
    );
    assert.throws(
      () => linspace(ctx, -1, 1, 3, { dtype: "uint32" }),
      RangeError,
    );
    assert.throws(
      () => linspace(ctx, 0, 2 ** 63, 2, { dtype: "int64" }),
      RangeError,
    );
    assert.throws(
      () => linspace(ctx, 0, Infinity, 2, { dtype: "int32" }),
      RangeError,
    );
    assert.throws(() => logspace(ctx, 0, 3, 2, { dtype: "int8" }), RangeError);
    // Budgets are checked before input buffers are built.
    const budget = { code: "NMATH_MEMORY_BUDGET_EXCEEDED" };
    assert.throws(() => linspace(tiny, 0, 1, 9), budget);
    assert.throws(() => logspace(tiny, 0, 1, 9), budget);
    assert.throws(() => eye(tiny, 3), budget);
    assert.throws(() => identity(tiny, 3), budget);
    assert.throws(() => eye(tiny, 100_000, 100_000), budget);
    assert.throws(() => empty(tiny, [9], "float64"), budget);
    const small = tiny.zeros([5], "float64");
    assert.throws(() => zeros_like(small, "complex128"), budget);
    assert.throws(() => ones_like(small, "complex64"), budget);
    assert.throws(() => empty_like(small, "complex128"), budget);
    small.dispose();
    const fits = linspace(tiny, 0, 1, 8);
    assert.deepEqual(fits.shape, [8]);
    fits.dispose();
    assert.deepEqual(eye(tiny, 2, 4, { dtype: "int8" }).shape, [2, 4]);
    // Contextless values and closed contexts fail explicitly.
    const closed = context();
    const source = closed.zeros([2], "float64");
    await closed.close();
    assert.throws(() => zeros_like(source), { code: "NMATH_CONTEXT_CLOSED" });
    assert.throws(() => linspace(closed, 0, 1, 2), {
      code: "NMATH_CONTEXT_CLOSED",
    });
  } finally {
    await ctx.close();
    await tiny.close();
  }
});
