// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
  Temporal,
  createContext,
  indices,
  slice,
  add,
  sub,
} from "../dist/index.js";
import { context } from "./helpers.mjs";
const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/text-temporal.json", import.meta.url)),
);
function input(ctx, r) {
  const data =
    r.dtype === "str"
      ? r.values
      : r.dtype === "int64" || r.unit
        ? BigInt64Array.from(r.values, BigInt)
        : r.values;
  return ctx.array(data, {
    dtype: r.dtype,
    shape: r.shape,
    unit: r.unit,
    stringWidth: r.stringWidth,
  });
}
function metadata(target) {
  if (target.startsWith("U"))
    return { dtype: "str", stringWidth: Number(target.slice(1)) };
  const [dtype, unit] = target.slice(0, -1).split("[");
  return { dtype, unit };
}
function check(result, r) {
  if (result?.dtype && typeof result.dispose === "function") {
    assert.equal(result.dtype, r.dtype);
    assert.deepEqual(result.shape, r.shape);
    if (r.unit) assert.equal(result.unit, r.unit);
    if (r.stringWidth) assert.equal(result.stringWidth, r.stringWidth);
  }
  const flat = result?.toArray ? result.toArray().flat(Infinity) : [result];
  assert.deepEqual(
    flat.map((x) =>
      x instanceof Temporal
        ? x.ticks.toString()
        : typeof x === "bigint"
          ? x.toString()
          : Number.isNaN(x)
            ? "NaN"
            : x,
    ),
    r.values,
  );
  for (const x of flat)
    if (r.unit) {
      assert.ok(x instanceof Temporal);
      assert.equal(x.unit, r.unit);
      assert.equal(x.dtype, r.dtype);
    }
}

test(`text and temporal: exact NumPy ${fixture.numpy} oracle cases`, async () => {
  const ctx = context();
  try {
    for (const [i, c] of fixture.cases.entries()) {
      const a = input(ctx, c.a),
        b = c.b && input(ctx, c.b);
      let out;
      const run = async () => {
        const target = c.op === "cast" ? metadata(c.target) : undefined;
        const e = target
          ? a.astype(target.dtype, target)
          : b
            ? a[c.op](b)
            : ["sum", "mean", "min", "max"].includes(c.op)
              ? a[c.op](c.axis ?? undefined)
              : a[c.op]();
        out = await e.compute();
        if (!c.error) check(out, c.result);
      };
      try {
        if (c.error)
          await assert.rejects(
            run,
            /NMATH_DTYPE_UNSUPPORTED|NMATH_SHAPE_MISMATCH/,
          );
        else await run();
      } catch (e) {
        throw new Error(
          `case ${i}: ${c.a.dtype}[${c.a.unit ?? c.a.stringWidth}] ${c.op} ${c.target ?? ""}: ${e.message}`,
          { cause: e },
        );
      } finally {
        out?.dispose?.();
        a.dispose();
        b?.dispose();
      }
    }
  } finally {
    await ctx.close();
  }
});

test("text and temporal: endian/Fortran NPY and compressed NPZ preserve metadata and ticks", async () => {
  const ctx = context(),
    dir = mkdtempSync(join(tmpdir(), "nmath-text-temporal-"));
  try {
    for (const [i, f] of fixture.files.entries()) {
      const path = join(dir, `${i}.npy`);
      writeFileSync(path, Buffer.from(f.npy, "base64"));
      const a = await ctx.load(path);
      check(a, f.result);
      await a.save(join(dir, `${i}-out.npy`));
      const b = await ctx.load(join(dir, `${i}-out.npy`));
      check(b, f.result);
      b.dispose();
      await ctx.saveNpz(join(dir, `${i}.npz`), new Map([["data", a]]), {
        compressed: true,
      });
      const archive = await ctx.loadNpz(join(dir, `${i}.npz`));
      check(archive.get("data"), f.result);
      archive.get("data").dispose();
      a.dispose();
    }
  } finally {
    await ctx.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("text and temporal: views, writes, copies, indexing and disposal", async () => {
  const ctx = context(),
    other = context();
  try {
    for (const data of [
      ctx.array(["한😀", "é", "a\0b"], { dtype: "str" }),
      ctx.array(new BigInt64Array([-1n, 9007199254740993n, -(1n << 63n)]), {
        dtype: "datetime64",
        unit: "ns",
      }),
    ]) {
      const view = await data.get(slice(undefined, undefined, -1)).compute();
      assert.ok(view.sharesStorage(data));
      const taken = await data.get(indices([2, 0])).compute();
      assert.ok(!taken.sharesStorage(data));
      const copied = await other.copy(data);
      assert.equal(copied.dtypeDescriptor, data.dtypeDescriptor);
      assert.deepEqual(copied.toArray(), data.toArray());
      const replacement =
        data.dtype === "str" ? "🦊" : new Temporal(42n, "ns", "datetime64");
      await data.set([0], replacement);
      assert.deepEqual(view.item(-1), replacement);
      assert.notDeepEqual(taken.item(-1), replacement);
      await assert.rejects(
        data.set(
          [0],
          data.dtype === "str" ? new Temporal(1n, "ns", "datetime64") : "bad",
        ),
        /NMATH_DTYPE_UNSUPPORTED/,
      );
      assert.deepEqual(data.item(0), replacement);
      const pending = data.get(indices([0])).compute();
      data.dispose();
      const out = await pending;
      assert.deepEqual(out.item(0), replacement);
      out.dispose();
      assert.throws(() => data.item(0), /NMATH_DISPOSED/);
      assert.deepEqual(view.item(-1), replacement);
      view.dispose();
      taken.dispose();
      copied.dispose();
    }
    // Native expression handles retain their input storage until GC.
    assert.equal(other.metrics().cpuMemory.used, 0n);
  } finally {
    await ctx.close();
    await other.close();
  }
});

test("text and temporal: exact scalars, explicit Date conversion and rejection", async () => {
  const ctx = context();
  try {
    const t = new Temporal(-1n, "ns", "datetime64");
    assert.equal(t.toDate().toISOString(), "1969-12-31T23:59:59.999Z");
    assert.throws(() => Number(t), /explicit/);
    assert.throws(
      () => new Temporal(1n << 63n, "ns", "datetime64"),
      RangeError,
    );
    assert.throws(
      () => new Temporal(0n, "0s", "datetime64"),
      /NMATH_DTYPE_UNSUPPORTED/,
    );
    assert.throws(
      () => new Temporal(9223372036854775807n, "Y", "datetime64").toDate(),
      /overflow/,
    );
    const nat = new Temporal(-(1n << 63n), "ns", "datetime64");
    assert.ok(nat.isNaT);
    assert.ok(Number.isNaN(nat.toDate().getTime()));
    const a = ctx.array(new BigInt64Array([9007199254740993n]), {
      dtype: "datetime64",
      unit: "ns",
    });
    assert.equal((await a.get(0).compute()).ticks, 9007199254740993n);
    const duration = ctx.array(new BigInt64Array([2n]), {
      dtype: "timedelta64",
      unit: "ns",
    });
    const result = await add(a, duration).compute();
    assert.equal(result.item(0).ticks, 9007199254740995n);
    result.dispose();
    const difference = await sub(a, a).compute();
    assert.equal(difference.dtype, "timedelta64");
    difference.dispose();
    await assert.rejects(
      a.compute({ device: "gpu" }),
      /NMATH_GPU_UNSUPPORTED_OPERATION/,
    );
    assert.throws(() => a.add(a), /NMATH_DTYPE_UNSUPPORTED/);
    assert.throws(
      () => ctx.array([new Date()], { dtype: "datetime64" }),
      /exact int64/,
    );
    assert.throws(
      () => ctx.array(["😀"], { dtype: "str", stringWidth: 0 }),
      /width/,
    );
    assert.throws(
      () => ctx.array(["ab"], { dtype: "str", stringWidth: 1 }),
      /width/,
    );
    assert.throws(() => ctx.array(["\ud800"], { dtype: "str" }), /surrogates/);
    const short = ctx.array(["a"], { dtype: "str" });
    assert.throws(() => short.add("\ud800"), /surrogates/);
    await assert.rejects(short.set([0], "long"), /truncate/);
    assert.equal(short.item(0), "a");
    short.dispose();
    const seconds = ctx.array(new BigInt64Array([1n]), {
      dtype: "datetime64",
      unit: "s",
    });
    await assert.rejects(a.set([], seconds), /matching kind and unit/);
    seconds.dispose();
    a.dispose();
    duration.dispose();
  } finally {
    await ctx.close();
  }
});

test("text and temporal: storage and execution budgets reject before allocation", async () => {
  const make = (n) =>
    createContext({
      maxConcurrentJobs: 1,
      maxQueueLength: 2,
      cpuMemoryBudget: n,
      gpuMemoryBudget: 0,
    });
  for (const [dtype, data, metadata, bytes] of [
    ["str", ["😀a", "한글"], { stringWidth: 2 }, 16],
    ["datetime64", new BigInt64Array([1n, 2n]), { unit: "ns" }, 16],
  ]) {
    const low = make(bytes - 1);
    assert.throws(
      () => low.array(data, { dtype, ...metadata }),
      /NMATH_MEMORY_BUDGET_EXCEEDED/,
    );
    assert.equal(low.metrics().cpuMemory.used, 0n);
    await low.close();
    const ctx = make(bytes);
    const a = ctx.array(data, { dtype, ...metadata });
    assert.equal(ctx.metrics().cpuMemory.used, BigInt(bytes));
    await assert.rejects(a.compute(), /NMATH_MEMORY_BUDGET_EXCEEDED/);
    assert.equal(ctx.metrics().cpuMemory.used, BigInt(bytes));
    const view = await a
      .get(slice())
      .compute()
      .catch((e) => e); // A snapshot still needs budget.
    assert.match(String(view), /NMATH_MEMORY_BUDGET_EXCEEDED/);
    a.dispose();
    await ctx.close();
  }
});

test("text and temporal: table selection, filtering and cross-entry conversions retain values", async () => {
  const pl = await import("../dist/polars.js");
  const ctx = context();
  try {
    const strings = ctx.array(["한글", "😀", "x"], { dtype: "str" });
    const dates = ctx.array(
      new BigInt64Array([-1n, 9007199254740993n, -(1n << 63n)]),
      { dtype: "datetime64", unit: "ns" },
    );
    const keep = ctx.array([true, false, true], { dtype: "bool" });
    const frame = new pl.DataFrame([
      new pl.Series("text", strings),
      new pl.Series("time", dates),
      new pl.Series("keep", keep),
    ]);
    const result = await frame
      .lazy()
      .filter("keep")
      .select("text", "time")
      .collect();
    assert.deepEqual(result.getColumn("text").values.toArray(), ["한글", "x"]);
    assert.deepEqual(
      [...result.getColumn("time").values.toTypedArray()],
      [-1n, -(1n << 63n)],
    );
    assert.equal(result.getColumn("time").values.unit, "ns");
    const pandas = result.toPandas();
    assert.deepEqual(pandas.get("text").values.toArray(), ["한글", "x"]);
    assert.equal(pandas.get("time").values.item(1).isNaT, true);
    await assert.rejects(
      frame.lazy().sort("text").collect(),
      /NMATH_DTYPE_UNSUPPORTED/,
    );
    for (const name of result.columns) {
      const column = result.getColumn(name);
      column.values.dispose();
      column.validity?.dispose();
    }
    strings.dispose();
    dates.dispose();
    keep.dispose();
  } finally {
    await ctx.close();
  }
});
