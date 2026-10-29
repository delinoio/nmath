// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { Complex, indices, mask, slice, add } from "../dist/index.js";
import { context } from "./helpers.mjs";

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/complex-dtypes.json", import.meta.url)),
);
const number = (x) =>
  typeof x === "number"
    ? x
    : { nan: NaN, inf: Infinity, "-inf": -Infinity, "-0": -0 }[x];
const scalar = (z) => new Complex(...z.map(number));
const array = (ctx, r) =>
  ctx.array(r.values.map(scalar), { dtype: r.dtype, shape: r.shape });
function check(result, expected, tolerance = 0, signedZero = false) {
  const values =
    result instanceof Complex ? [result] : result.toArray().flat(Infinity);
  if (!(result instanceof Complex)) {
    assert.equal(result.dtype, expected.dtype);
    assert.deepEqual(result.shape, expected.shape);
  }
  assert.equal(values.length, expected.values.length);
  values.forEach((z, i) => {
    assert.ok(z instanceof Complex);
    for (const [j, key] of ["real", "imag"].entries()) {
      const want = number(expected.values[i][j]),
        got = z[key];
      if (Number.isNaN(want))
        assert.ok(Number.isNaN(got), `${i}.${key}: expected NaN, got ${got}`);
      else if (tolerance && Number.isFinite(want))
        assert.ok(
          Math.abs(got - want) < tolerance,
          `${i}.${key}: ${got} != ${want}`,
        );
      else if (want === 0 && !signedZero) assert.equal(got === 0, true);
      else assert.equal(got, want, `${i}.${key}`);
    }
  });
}

for (const c of fixture.cases) {
  test(`${c.dtype}: NumPy ${fixture.numpy} arithmetic, reductions, matmul and scalar transport`, async () => {
    const ctx = context();
    try {
      const a = array(ctx, c.a),
        b = array(ctx, c.b);
      check(a, c.a);
      assert.ok(a.item(0, 0) instanceof Complex);
      assert.ok(
        a.toTypedArray() instanceof
          (c.dtype === "complex64" ? Float32Array : Float64Array),
      );
      check(
        ctx.array(a.toTypedArray(), { dtype: c.dtype, shape: a.shape }),
        c.a,
      );
      for (const [op, expected] of Object.entries(c.operations)) {
        const expr = ["sum", "mean", "min", "max"].includes(op)
          ? a[op](0)
          : ["neg", "pos"].includes(op)
            ? a[op]()
            : op === "pow"
              ? a.pow(2n)
              : a[op](b);
        assert.equal(expr.dtype, expected.dtype, op);
        const result = await expr.compute();
        check(result, expected);
        result.dispose();
      }
      check(await a.get(0, 0).compute(), { values: [c.a.values[0]] });
      assert.equal(a.add(new Complex(1, 2)).dtype, c.dtype);
      assert.equal(add(new Complex(1, 2), a).dtype, c.dtype);
      assert.throws(() => a.mod(b), { code: "NMATH_DTYPE_UNSUPPORTED" });
      assert.throws(() => Number(a.item(0, 0)), TypeError);
      assert.equal(a.item(0, 0).toNumber(), 1);
      assert.deepEqual(ctx.zeros([1], c.dtype).item(0), new Complex(0));
      assert.deepEqual(
        ctx.full([1], new Complex(-0, Infinity), c.dtype).item(0),
        new Complex(-0, Infinity),
      );
    } finally {
      await ctx.close();
    }
  });

  test(`${c.dtype}: casts, reduction order and exceptional means match NumPy`, async () => {
    const ctx = context();
    try {
      const input = array(ctx, c.castsInput);
      for (const [dtype, expected] of Object.entries(c.casts)) {
        const out = await input.astype(dtype).compute();
        assert.deepEqual(
          out.toArray(),
          dtype === "int64" || dtype === "uint64"
            ? expected.map(BigInt)
            : expected,
          dtype,
        );
        out.dispose();
      }
      const p = array(ctx, c.pairwiseInput);
      for (const op of ["sum", "mean"])
        check(await p[op]().compute(), c.pairwise[op]);
      for (const op of ["min", "max"]) {
        const a = ctx.array([new Complex(-0, -0), new Complex(0, 0)], {
          dtype: c.dtype,
        });
        const r = await a[op]().compute();
        assert.ok(Object.is(r.real, -0) && Object.is(r.imag, -0));
      }
      const exceptional = ctx.array([new Complex(Infinity, 0)], {
        dtype: c.dtype,
      });
      const mean = await exceptional.mean().compute();
      assert.equal(mean.real, Infinity);
      assert.ok(Number.isNaN(mean.imag));
      const maskResult = await array(ctx, c.a)
        .get(mask([true, false]))
        .compute();
      check(maskResult, {
        dtype: c.dtype,
        shape: [1, 2],
        values: c.a.values.slice(0, 2),
      });
    } finally {
      await ctx.close();
    }
  });

  test(`${c.dtype}: exceptional division and power components match NumPy`, async () => {
    const ctx = context();
    try {
      for (const item of c.exceptional) {
        const a = array(ctx, item.a),
          b = array(ctx, item.b);
        const result = await a[item.op](b).compute();
        try {
          check(result, item.result, 0, true);
        } catch (error) {
          throw new Error(JSON.stringify(item), { cause: error });
        }
        result.dispose();
        a.dispose();
        b.dispose();
      }
      for (const op of ["sum", "mean"]) {
        const result = await ctx.zeros([0], c.dtype)[op]().compute();
        assert.ok(result instanceof Complex);
        if (op === "mean")
          assert.ok(Number.isNaN(result.real) && Number.isNaN(result.imag));
        else assert.deepEqual(result, new Complex(0));
      }
      for (const op of ["min", "max"]) {
        const a = ctx.array(
          [new Complex(2, 1), new Complex(2, -1), new Complex(1, NaN)],
          { dtype: c.dtype },
        );
        assert.ok(Number.isNaN((await a[op]().compute()).imag));
      }
    } finally {
      await ctx.close();
    }
  });

  test(`${c.dtype}: views, writes, copies, casts and NumPy NPY interchange`, async () => {
    const ctx = context(),
      other = context();
    const dir = mkdtempSync(join(tmpdir(), "nmath-complex-"));
    try {
      const a = array(ctx, c.a),
        view = await a.get(slice(), 1).compute();
      assert.ok(view.sharesStorage(a));
      const gather = await a.get(indices([1, 0]), 0).compute();
      assert.equal(gather.sharesStorage(a), false);
      check(gather, {
        dtype: c.dtype,
        shape: [2],
        values: [c.a.values[2], c.a.values[0]],
      });
      await a.set([0, 1], new Complex(9, -7));
      assert.deepEqual(view.item(0), new Complex(9, -7));
      const cancelled = new AbortController();
      cancelled.abort();
      await assert.rejects(
        a.set([], new Complex(1, 1), { signal: cancelled.signal }),
        { code: "NMATH_CANCELLED" },
      );
      assert.deepEqual(view.item(0), new Complex(9, -7));
      const copy = await other.copy(a);
      assert.deepEqual(copy.toArray(), a.toArray());
      const real = await a.astype("float64").compute();
      assert.equal(real.item(0, 1), 9);
      await assert.rejects(real.set([], new Complex(2, 3)), {
        code: "NMATH_DTYPE_UNSUPPORTED",
      });
      assert.equal(real.item(0, 1), 9);
      const booleans = await ctx
        .array([new Complex(0, 1), new Complex(0, 0)], { dtype: c.dtype })
        .astype("bool")
        .compute();
      assert.deepEqual(booleans.toArray(), [true, false]);
      for (const name of ["npy", "bigEndianNpy", "fortranNpy"]) {
        const path = join(dir, `${name}.npy`);
        writeFileSync(path, Buffer.from(c[name], "base64"));
        const loaded = await ctx.load(path);
        check(loaded, c.a);
        const saved = join(dir, `${name}-saved.npy`);
        await loaded.save(saved);
        assert.deepEqual(readFileSync(saved), Buffer.from(c.npy, "base64"));
      }
      const special = ctx.array(
        [new Complex(-0, -0), new Complex(Infinity, NaN)],
        { dtype: c.dtype },
      );
      check(
        special,
        {
          dtype: c.dtype,
          shape: [2],
          values: [
            ["-0", "-0"],
            ["inf", "nan"],
          ],
        },
        0,
        true,
      );
      const tight = context({
        cpuMemoryBudget: c.dtype === "complex64" ? 8 : 16,
      });
      try {
        const x = tight.zeros([1], c.dtype);
        await assert.rejects(x.neg().compute(), {
          code: "NMATH_MEMORY_BUDGET_EXCEEDED",
        });
        assert.equal(
          tight.metrics().cpuMemory.used,
          BigInt(c.dtype === "complex64" ? 8 : 16),
        );
        x.dispose();
      } finally {
        await tight.close();
      }
    } finally {
      await ctx.close();
      await other.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });
}

test("complex strong and weak promotions match NumPy 2.5.4", async () => {
  const ctx = context();
  try {
    for (const [d, row] of Object.entries(fixture.promotion))
      for (const [e, want] of Object.entries(row)) {
        const a = ctx.zeros([1], d),
          b = ctx.zeros([1], e);
        assert.equal(a.add(b).dtype, want);
        assert.equal(b.add(a).dtype, want);
        const out = await a.add(b).compute();
        assert.equal(out.dtype, want);
        out.dispose();
        a.dispose();
        b.dispose();
      }
    for (const [d, want] of Object.entries(fixture.weak)) {
      const a = ctx.zeros([1], d);
      const expr = a.add(new Complex(1, 2));
      assert.equal(expr.dtype, want);
      const out = await expr.compute();
      assert.deepEqual(out.item(0), new Complex(1, 2));
      out.dispose();
      a.dispose();
    }
  } finally {
    await ctx.close();
  }
});

test("complex128 fractional power uses the matching upstream decimal=7 criterion", async () => {
  const ctx = context();
  try {
    const a = array(ctx, fixture.fractionalPower.a),
      result = await a.pow(0.5).compute();
    check(result, fixture.fractionalPower.result, 1.5e-7);
  } finally {
    await ctx.close();
  }
});

test("complex inputs reject malformed exchange and implicit real conversions", async () => {
  const ctx = context();
  try {
    assert.throws(
      () => ctx.array(new Float32Array([1]), { dtype: "complex64" }),
      TypeError,
    );
    assert.throws(
      () => ctx.array(new Float64Array([1, 2]), { dtype: "complex64" }),
      TypeError,
    );
    assert.throws(
      () => ctx.array([new Complex(1, 2)], { dtype: "float64" }),
      TypeError,
    );
    assert.throws(() => ctx.full([1], new Complex(1, 2), "bool"), TypeError);
    const a = ctx.array([new Complex(1.5, -2.5)], { dtype: "complex128" });
    const narrow = await a.astype("complex64").compute();
    assert.equal(narrow.dtype, "complex64");
    assert.deepEqual(narrow.item(0), new Complex(1.5, -2.5));
    const real = ctx.array([-0, 2], { dtype: "float64" });
    const promoted = await real.astype("complex128").compute();
    assert.deepEqual(promoted.item(0), new Complex(-0, 0));
    await assert.rejects(a.neg().compute({ device: "gpu" }), {
      code: "NMATH_GPU_UNSUPPORTED_OPERATION",
    });
  } finally {
    await ctx.close();
  }
});
