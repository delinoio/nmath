// SPDX-License-Identifier: Apache-2.0
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, rm, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { context } from "./helpers.mjs";
import * as pandas from "../dist/pandas.js";
import * as polars from "../dist/polars.js";

for (const format of ["arrow", "parquet"]) {
  test(`${format}: scheduled nullable bigint roundtrip, overwrite and preservation`, async () => {
    const dir = await mkdtemp(join(tmpdir(), "nmath-columnar-"));
    const ctx = context();
    try {
      const path = join(dir, "data");
      const signed = ctx.array(
        [-9223372036854775808n, 0n, 9223372036854775807n],
        { dtype: "int64" },
      );
      const unsigned = ctx.array([0n, 0n, 18446744073709551615n], {
        dtype: "uint64",
      });
      const mask = ctx.array([true, false, true], { dtype: "bool" });
      const columns = new Map([
        ["signed", signed],
        ["unsigned", unsigned],
      ]);
      const validity = new Map([
        ["signed", mask],
        ["unsigned", mask],
      ]);
      const promise = ctx.writeColumnar(path, format, columns, validity);
      assert.ok(promise instanceof Promise);
      await promise;
      const bytes = await readFile(path);
      await assert.rejects(ctx.writeColumnar(path, format, columns, validity), {
        code: "NMATH_IO_EXISTS",
      });
      const abort = new AbortController();
      abort.abort();
      await assert.rejects(
        ctx.writeColumnar(path, format, columns, validity, {
          overwrite: true,
          signal: abort.signal,
        }),
        { code: "NMATH_CANCELLED" },
      );
      assert.deepEqual(await readFile(path), bytes);
      assert.deepEqual(await readdir(dir), ["data"]);
      const loaded = await ctx.readColumnar(path, format);
      assert.equal(loaded.rows, 3);
      assert.equal(loaded.columns.get("signed").dtype, "int64");
      assert.equal(
        loaded.columns.get("unsigned").item(2),
        18446744073709551615n,
      );
      assert.equal(loaded.columns.get("signed").item(0), -9223372036854775808n);
      assert.equal(loaded.validity.get("signed").item(1), false);
      for (const value of [
        ...loaded.columns.values(),
        ...loaded.validity.values(),
      ])
        value.dispose();
      const frame = new polars.DataFrame(columns, validity);
      if (format === "arrow")
        await frame.write_ipc(ctx, path, { overwrite: true });
      else await frame.write_parquet(ctx, path, { overwrite: true });
      const adapted = await (format === "arrow"
        ? pandas.read_feather(ctx, path)
        : pandas.read_parquet(ctx, path));
      assert.equal(adapted.columnMap.get("unsigned").dtype, "uint64");
      assert.equal(adapted.validity.get("signed").item(1), false);
      const indexed = new pandas.DataFrame(columns, {
        validity,
        index: ["a", "b", "c"],
      });
      await assert.rejects(indexed.to_parquet(ctx, path, { overwrite: true }), {
        code: "NMATH_DTYPE_UNSUPPORTED",
      });
      const small = context({ cpuMemoryBudget: 1n });
      try {
        await assert.rejects(small.readColumnar(path, format), {
          code: "NMATH_MEMORY_BUDGET_EXCEEDED",
        });
      } finally {
        await small.close();
      }
      await writeFile(path, bytes.subarray(0, bytes.length - 1));
      await assert.rejects(ctx.readColumnar(path, format), {
        code: "NMATH_IO_FORMAT",
      });
      assert.ok(ctx.metrics().executions.some((e) => e.kind === "file_write"));
    } finally {
      await ctx.close();
      await rm(dir, { recursive: true, force: true });
    }
  });
  test(`${format}: typed empty table retains schema`, async () => {
    const dir = await mkdtemp(join(tmpdir(), "nmath-columnar-empty-"));
    const ctx = context();
    try {
      const path = join(dir, "data");
      const values = ctx.array([], { dtype: "uint64" });
      await ctx.writeColumnar(path, format, new Map([["empty", values]]));
      const out = await ctx.readColumnar(path, format);
      assert.equal(out.rows, 0);
      assert.equal(out.columns.get("empty").dtype, "uint64");
      assert.equal(out.validity.size, 0);
    } finally {
      await ctx.close();
      await rm(dir, { recursive: true, force: true });
    }
  });
}
