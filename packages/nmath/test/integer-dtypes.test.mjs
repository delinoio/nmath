// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { indices, slice } from "../dist/index.js";
import { context } from "./helpers.mjs";

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/integer-dtypes.json", import.meta.url)),
);
const constructors = {
  int8: Int8Array,
  int16: Int16Array,
  uint8: Uint8Array,
  uint16: Uint16Array,
};

for (const c of fixture.cases) {
  test(`${c.dtype}: NumPy ${fixture.numpy} arithmetic, reductions and native exchange`, async () => {
    const ctx = context();
    try {
      const a = ctx.array(c.a, { dtype: c.dtype });
      const b = ctx.array(new constructors[c.dtype](c.b), { dtype: c.dtype });
      assert.equal(a.item(0, 1), c.max);
      assert.ok(a.toTypedArray() instanceof constructors[c.dtype]);
      assert.deepEqual(a.toArray(), c.a);
      for (const [op, expected] of Object.entries(c.operations)) {
        const expr = ["sum", "mean", "min", "max"].includes(op)
          ? a[op](0)
          : op === "neg"
            ? a.neg()
            : a[op](b);
        assert.equal(expr.dtype, expected.dtype, op);
        const result = await expr.compute();
        const values = expected.dtype.endsWith("int64")
          ? expected.values.map(BigInt)
          : expected.values;
        assert.deepEqual(result.toArray(), values, op);
        result.dispose();
      }
      assert.equal(await a.get(0, 1).compute(), c.max);
      assert.equal(a.add(1n).dtype, c.dtype);
      assert.equal(a.add(1).dtype, "float64");
      for (const scalar of [BigInt(c.min) - 1n, BigInt(c.max) + 1n]) {
        assert.throws(() => a.add(scalar), { code: "NMATH_DTYPE_UNSUPPORTED" });
      }
      assert.deepEqual(ctx.full([2], 3, c.dtype).toArray(), [3, 3]);
      assert.deepEqual(ctx.zeros([2], c.dtype).toArray(), [0, 0]);
    } finally {
      await ctx.close();
    }
  });

  test(`${c.dtype}: views, atomic writes, copies, budgets and NPY`, async () => {
    const ctx = context();
    const other = context();
    const dir = mkdtempSync(join(tmpdir(), "nmath-integer-"));
    try {
      const a = ctx.array(c.a, { dtype: c.dtype });
      const view = await a.get(slice(), 1).compute();
      assert.ok(view.sharesStorage(a));
      const gathered = await a.get(indices([1, 0])).compute();
      assert.deepEqual(gathered.toArray(), [c.a[1], c.a[0]]);
      await a.set([0, 1], 4n);
      assert.equal(view.item(0), 4);
      const controller = new AbortController();
      controller.abort();
      await assert.rejects(a.set([], 5n, { signal: controller.signal }), {
        code: "NMATH_CANCELLED",
      });
      assert.equal(a.item(0, 1), 4);
      const copy = await other.copy(a);
      assert.equal(copy.dtype, c.dtype);
      assert.deepEqual(copy.toArray(), a.toArray());
      const path = join(dir, "reference.npy");
      writeFileSync(path, Buffer.from(c.npy, "base64"));
      const loaded = await ctx.load(path);
      assert.equal(loaded.dtype, c.dtype);
      assert.deepEqual(loaded.toArray(), c.a);
      const output = join(dir, "output.npy");
      await loaded.save(output);
      assert.deepEqual((await ctx.load(output)).toArray(), c.a);
      const width = constructors[c.dtype].BYTES_PER_ELEMENT;
      const tight = context({ cpuMemoryBudget: width * 4 });
      try {
        const exact = tight.array(c.a, { dtype: c.dtype });
        assert.equal(tight.metrics().cpuMemory.used, BigInt(width * 4));
        await assert.rejects(exact.add(1n).compute(), {
          code: "NMATH_MEMORY_BUDGET_EXCEEDED",
        });
        // A rejected snapshot adds no allocation.
        assert.equal(tight.metrics().cpuMemory.used, BigInt(width * 4));
        exact.dispose();
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

test("all numeric array promotions match NumPy 2.5.4", async () => {
  const ctx = context();
  try {
    for (const [a, row] of Object.entries(fixture.promotion)) {
      for (const [b, expected] of Object.entries(row)) {
        const left = ctx.zeros([1], a),
          right = ctx.zeros([1], b);
        const expr = left.add(right);
        assert.equal(expr.dtype, expected, `${a} + ${b}`);
        const result = await expr.compute();
        assert.equal(result.dtype, expected);
        result.dispose();
        left.dispose();
        right.dispose();
      }
    }
  } finally {
    await ctx.close();
  }
});
