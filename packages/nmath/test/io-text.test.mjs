// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { context } from "./helpers.mjs";
import * as pandas from "../dist/pandas.js";
import * as polars from "../dist/polars.js";
const schema = new Map([
  ["signed", "int64"],
  ["unsigned", "uint64"],
  ["small", "int8"],
  ["flag", "bool"],
]);
const fixtures = fileURLToPath(
  new URL(
    "../../../crates/nmath-core/tests/fixtures/io-text/",
    import.meta.url,
  ),
);
function dispose(table) {
  for (const c of table.columns.values()) c.dispose();
  for (const c of table.validity.values()) c.dispose();
}
for (const format of ["csv", "json"]) {
  test(`${format}: pinned pandas and Polars files retain nullable exact numeric schema`, async () => {
    const ctx = context();
    try {
      for (const original of ["pandas", "polars"]) {
        const t = await ctx.readText(
          join(fixtures, `${original}.${format}`),
          format,
          { schema },
        );
        assert.deepEqual(t.columns.get("signed").toArray(), [
          -9223372036854775808n,
          0n,
          9223372036854775807n,
        ]);
        assert.deepEqual(t.columns.get("unsigned").toArray(), [
          18446744073709551615n,
          0n,
          0n,
        ]);
        assert.equal(t.columns.get("small").dtype, "int8");
        assert.deepEqual(t.validity.get("signed").toArray(), [
          true,
          false,
          true,
        ]);
        dispose(t);
      }
    } finally {
      await ctx.close();
    }
  });
  test(`${format}: public frame entrypoints, lazy write, atomic overwrite and cancellation`, async () => {
    const ctx = context();
    const dir = await mkdtemp(join(tmpdir(), "nmath-text-"));
    try {
      const frame = await pandas[`read_${format}`](
        ctx,
        join(fixtures, `pandas.${format}`),
        { schema },
      );
      const dest = join(dir, "out");
      await frame[format === "csv" ? "toCsv" : "toJson"](ctx, dest);
      const before = await readFile(dest);
      await assert.rejects(
        frame[format === "csv" ? "toCsv" : "toJson"](ctx, dest),
        { code: "NMATH_IO_EXISTS" },
      );
      const abort = new AbortController();
      abort.abort();
      await assert.rejects(
        frame[format === "csv" ? "toCsv" : "toJson"](ctx, dest, {
          overwrite: true,
          signal: abort.signal,
        }),
        { code: "NMATH_CANCELLED" },
      );
      assert.deepEqual(await readFile(dest), before);
      const p = await polars[`read_${format}`](ctx, dest, { schema });
      assert.deepEqual(p.getColumn("unsigned").values.toArray(), [
        18446744073709551615n,
        0n,
        0n,
      ]);
      await p[format === "csv" ? "writeCsv" : "writeJson"](ctx, dest, {
        overwrite: true,
      });
      const values = ctx.array([1n, 2n], { dtype: "int64" });
      await ctx.writeText(
        dest,
        format,
        new Map([["lazy", values.add(2n)]]),
        new Map(),
        { overwrite: true },
      );
      const lazy = await ctx.readText(dest, format);
      assert.deepEqual(lazy.columns.get("lazy").toArray(), [3n, 4n]);
      dispose(lazy);
      values.dispose();
      for (const f of [frame, p]) {
        for (const c of f.columnMap.values()) c.dispose();
        for (const c of f.validity.values()) c.dispose();
      }
    } finally {
      await ctx.close();
      await rm(dir, { recursive: true, force: true });
    }
  });
}
test("text IO: malformed files, budgets, context ownership, custom pandas index", async () => {
  const dir = await mkdtemp(join(tmpdir(), "nmath-text-"));
  const ctx = context();
  const tiny = context({ cpuMemoryBudget: 10n });
  const other = context();
  try {
    const path = join(dir, "file");
    await writeFile(path, '[{"a":1,"a":2}]');
    await assert.rejects(ctx.readText(path, "json"), {
      code: "NMATH_IO_FORMAT",
    });
    await assert.rejects(tiny.readText(path, "json"), {
      code: "NMATH_MEMORY_BUDGET_EXCEEDED",
    });
    const a = ctx.array([1], { dtype: "int8" });
    await assert.rejects(other.writeText(path, "csv", new Map([["a", a]])), {
      code: "NMATH_CONTEXT_MISMATCH",
    });
    const frame = new pandas.DataFrame(new Map([["a", a]]), {
      index: ["custom"],
    });
    assert.throws(() => frame.toCsv(ctx, path), /dropIndex/);
    await frame.toCsv(ctx, path, { overwrite: true, dropIndex: true });
    a.dispose();
  } finally {
    await ctx.close();
    await tiny.close();
    await other.close();
    await rm(dir, { recursive: true, force: true });
  }
});
