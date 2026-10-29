// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { Complex, indices, slice, add } from "../dist/index.js";
import { context } from "./helpers.mjs";

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/float16-dtypes.json", import.meta.url)),
);
const array = (ctx, r) =>
  ctx.arrayFromFloat16Bits(Uint16Array.from(r.bits), { shape: r.shape });
const nan = (bits) => (bits & 0x7c00) === 0x7c00 && (bits & 0x3ff) !== 0;
function check(result, expected) {
  const bits =
    typeof result === "number"
      ? new Uint16Array(Float16Array.of(result).buffer)
      : result.toFloat16Bits();
  if (typeof result !== "number") {
    assert.equal(result.dtype, "float16");
    assert.deepEqual(result.shape, expected.shape);
  }
  assert.equal(bits.length, expected.bits.length);
  bits.forEach((got, i) => {
    const want = expected.bits[i];
    if (nan(want)) assert.ok(nan(got), `${i}: expected NaN`);
    else assert.equal(got, want, `element ${i}`);
  });
}

test(`float16: NumPy ${fixture.numpy} exact arithmetic, reductions and matmul`, async () => {
  const ctx = context();
  try {
    for (const c of fixture.cases) {
      // Public reductions use variadic axes; no arguments means all axes.
      if (Array.isArray(c.axis) && c.axis.length === 0) continue;
      const a = array(ctx, c.a),
        b = c.b && array(ctx, c.b);
      const expr = ["sum", "mean", "min", "max"].includes(c.op)
        ? a[c.op](
            ...(c.axis === null
              ? []
              : Array.isArray(c.axis)
                ? c.axis
                : [c.axis]),
          )
        : b
          ? a[c.op](b)
          : a[c.op]();
      assert.equal(expr.dtype, "float16");
      const out = await expr.compute();
      try {
        check(out, c.result);
      } catch (e) {
        e.message += ` (${c.op}, shape=${c.a.shape}, axis=${c.axis})`;
        throw e;
      }
      out.dispose?.();
      a.dispose();
      b?.dispose();
    }
  } finally {
    await ctx.close();
  }
});

test("float16: promotion, weak scalars and exact rounding", async () => {
  const ctx = context();
  try {
    const a = ctx.array([1], { dtype: "float16" });
    for (const [dtype, expected] of Object.entries(fixture.promotion)) {
      const b = ctx.zeros([1], dtype);
      assert.equal(a.add(b).dtype, expected);
      assert.equal(b.add(a).dtype, expected);
      b.dispose();
    }
    for (const scalar of [1, 1n, true]) {
      assert.equal(a.add(scalar).dtype, "float16");
      assert.equal(add(scalar, a).dtype, "float16");
      const out = await a.add(scalar).compute();
      assert.equal(out.item(0), 2);
      out.dispose();
    }
    assert.equal(a.add(new Complex(1)).dtype, "complex64");
    const weak = await a.add(1.00048828125).compute();
    assert.equal(weak.item(0), 2);
    weak.dispose();
    const integerSource = array(ctx, fixture.integerCasts.input);
    for (const [dtype, expected] of Object.entries(
      fixture.integerCasts.outputs,
    )) {
      const result = await integerSource.astype(dtype).compute();
      assert.deepEqual(
        result.toArray(),
        dtype === "int64" || dtype === "uint64"
          ? expected.map(BigInt)
          : expected,
      );
      result.dispose();
    }
    integerSource.dispose();
    const source = ctx.array(fixture.casts.values, { dtype: "float64" });
    const cast = await source.astype("float16").compute();
    check(cast, fixture.casts.result);
    check(
      ctx.array(fixture.casts.values, { dtype: "float16" }),
      fixture.casts.result,
    );
    assert.equal(ctx.full([1], 1.00048828125, "float16").item(0), 1);
    const metrics = ctx.metrics();
    await assert.rejects(a.add(1).compute({ device: "gpu" }), {
      code: "NMATH_GPU_UNSUPPORTED_OPERATION",
    });
    assert.deepEqual(ctx.metrics().executions, metrics.executions);
    cast.dispose();
    source.dispose();
    a.dispose();
  } finally {
    await ctx.close();
  }
});

test("float16: exact bits, views, indexing, assignment, copies and lifetime", async () => {
  const ctx = context(),
    other = context();
  try {
    const bits = Uint16Array.of(0x3c00, 0x8000, 1, 0x7e01, 0x7c00);
    const a = ctx.arrayFromFloat16Bits(bits);
    bits[0] = 0;
    assert.equal(a.item(0), 1);
    assert.equal(ctx.metrics().cpuMemory.used, 10n);
    const typed = a.toTypedArray();
    assert.ok(typed instanceof Float16Array);
    assert.deepEqual(
      ctx.array(typed, { dtype: "float16" }).toFloat16Bits(),
      a.toFloat16Bits(),
    );
    const view = await a.get(slice(0, 3)).compute();
    const gathered = await a.get(indices([2, 0])).compute();
    assert.deepEqual([...gathered.toFloat16Bits()], [1, 0x3c00]);
    await a.set([0], 2);
    assert.equal(view.item(0), 2);
    assert.equal(gathered.item(1), 1);
    const copied = await other.copy(a);
    assert.deepEqual(copied.toFloat16Bits(), a.toFloat16Bits());
    const f16 = globalThis.Float16Array;
    try {
      globalThis.Float16Array = undefined;
      assert.deepEqual(copied.toFloat16Bits(), a.toFloat16Bits());
      assert.throws(() => a.toTypedArray(), /no Float16Array/);
      const raw = other.arrayFromFloat16Bits(a.toFloat16Bits());
      assert.equal(raw.item(0), 2);
      raw.dispose();
    } finally {
      globalThis.Float16Array = f16;
    }
    a.dispose();
    assert.throws(() => a.toFloat16Bits(), { code: "NMATH_DISPOSED" });
    assert.equal(view.item(0), 2);
    view.dispose();
    gathered.dispose();
    copied.dispose();
    assert.equal(other.metrics().cpuMemory.used, 0n);
  } finally {
    await ctx.close();
    await other.close();
  }
});

test("float16: NPY endian/layout imports and NPY/NPZ exact bits roundtrip", async () => {
  const ctx = context(),
    dir = mkdtempSync(join(tmpdir(), "nmath-f16-"));
  try {
    for (const [i, data] of fixture.npy.entries()) {
      const path = join(dir, `input-${i}.npy`);
      writeFileSync(path, Buffer.from(data, "base64"));
      const loaded = await ctx.load(path);
      check(loaded, fixture.cases[0].a);
      loaded.dispose();
    }
    const a = ctx.arrayFromFloat16Bits(
      Uint16Array.of(0x8000, 1, 0x7e01, 0x7c00),
    );
    const path = join(dir, "saved.npy");
    await a.save(path);
    const loaded = await ctx.load(path);
    assert.deepEqual(loaded.toFloat16Bits(), a.toFloat16Bits());
    loaded.dispose();
    for (const compressed of [false, true]) {
      const path = join(dir, `saved-${compressed}.npz`);
      await ctx.saveNpz(path, new Map([["half", a]]), { compressed });
      const entries = await ctx.loadNpz(path);
      assert.deepEqual(entries.get("half").toFloat16Bits(), a.toFloat16Bits());
      entries.get("half").dispose();
    }
    a.dispose();
    assert.equal(ctx.metrics().cpuMemory.used, 0n);
  } finally {
    await ctx.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("float16: two-byte budget admission and cancellation", async () => {
  const ctx = context({ cpuMemoryBudget: 4 });
  try {
    const a = ctx.array([1, 2], { dtype: "float16" });
    assert.equal(ctx.metrics().cpuMemory.used, 4n);
    assert.throws(() => ctx.zeros([1], "float16"), {
      code: "NMATH_MEMORY_BUDGET_EXCEEDED",
    });
    const expr = a.add(1);
    await assert.rejects(expr.compute(), {
      code: "NMATH_MEMORY_BUDGET_EXCEEDED",
    });

    assert.deepEqual(a.toArray(), [1, 2]);
    const controller = new AbortController();
    controller.abort();
    const sum = a.sum();
    await assert.rejects(sum.compute({ signal: controller.signal }), {
      code: "NMATH_CANCELLED",
    });

    a.dispose();
  } finally {
    await ctx.close();
  }
});
