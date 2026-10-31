// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { DataFrame as PandasFrame } from "../dist/pandas.js";
import { DataFrame as PolarsFrame, Series } from "../dist/polars.js";
import { context } from "./helpers.mjs";

function pandasFrame(ctx, key, sub, value, validity) {
  return new PandasFrame(
    {
      key: ctx.array(key, { dtype: "int64" }),
      sub: ctx.array(sub, { dtype: "int64" }),
      value: ctx.array(value, { dtype: "int64" }),
    },
    validity ? { validity } : {},
  );
}

function polarsFrame(ctx, key, sub, value, validity) {
  return new PolarsFrame([
    new Series(
      "key",
      ctx.array(key, { dtype: "int64" }),
      validity ? ctx.array(validity, { dtype: "bool" }) : undefined,
    ),
    new Series("sub", ctx.array(sub, { dtype: "int64" })),
    new Series("value", ctx.array(value, { dtype: "int64" })),
  ]);
}

test("pandas pivot reshapes unique pairs and keeps missing cells null", async () => {
  const ctx = context();
  try {
    const frame = pandasFrame(ctx, [1n, 1n, 2n], [10n, 20n, 10n], [5n, 6n, 7n]);
    const pivoted = await frame.pivot("key", "sub", "value");
    assert.deepEqual(pivoted.columns, ["key", "10", "20"]);
    assert.deepEqual(pivoted.get("key").values.toArray(), [1n, 2n]);
    assert.deepEqual(pivoted.get("10").values.toArray(), [5n, 7n]);
    assert.deepEqual(pivoted.get("20").values.toArray(), [6n, 0n]);
    assert.deepEqual(pivoted.get("20").validity.toArray(), [true, false]);
    assert.deepEqual(pivoted.get("10").validity.toArray(), [true, true]);

    const duplicate = pandasFrame(ctx, [1n, 1n], [10n, 10n], [1n, 2n]);
    await assert.rejects(duplicate.pivot("key", "sub", "value"), {
      code: "NMATH_INDEX_INVALID",
    });
  } finally {
    await ctx.close();
  }
});

test("pandas pivotTable aggregates duplicate pairs", async () => {
  const ctx = context();
  try {
    const frame = pandasFrame(
      ctx,
      [1n, 1n, 2n, 2n],
      [10n, 10n, 10n, 20n],
      [1n, 2n, 3n, 4n],
    );
    const summed = await frame.pivotTable("key", "sub", "value", {
      aggfunc: "sum",
    });
    assert.deepEqual(pivotedValues(summed, "10"), [3n, 3n]);
    assert.deepEqual(summed.get("20").values.toArray(), [0n, 4n]);
    assert.deepEqual(summed.get("20").validity.toArray(), [false, true]);

    const meaned = await frame.pivotTable("key", "sub", "value", {
      aggfunc: "mean",
    });
    assert.equal(meaned.get("10").values.dtype, "float64");
    assert.deepEqual(pivotedValues(meaned, "10"), [1.5, 3]);

    const counted = await frame.pivotTable("key", "sub", "value", {
      aggfunc: "count",
    });
    assert.deepEqual(pivotedValues(counted, "10"), [2n, 1n]);
  } finally {
    await ctx.close();
  }
});

function pivotedValues(frame, name) {
  return frame.get(name).values.toArray();
}

test("Polars pivot preserves first-appearance order and aggregation options", async () => {
  const ctx = context();
  try {
    const frame = polarsFrame(ctx, [2n, 1n, 2n], [20n, 10n, 10n], [4n, 5n, 6n]);
    const pivoted = await frame.pivot("sub", "key", "value").collect();
    assert.deepEqual(pivoted.columns, ["key", "20", "10"]);
    assert.deepEqual(pivoted.getColumn("key").values.toArray(), [2n, 1n]);
    assert.deepEqual(pivoted.getColumn("20").values.toArray(), [4n, 0n]);
    assert.deepEqual(pivoted.getColumn("20").validity.toArray(), [true, false]);
    assert.deepEqual(pivoted.getColumn("10").values.toArray(), [6n, 5n]);
    assert.deepEqual(pivoted.getColumn("10").validity.toArray(), [true, true]);

    const summed = await frame.pivot("sub", "key", "value", "sum").collect();
    assert.deepEqual(summed.getColumn("10").values.toArray(), [6n, 5n]);
    assert.deepEqual(summed.getColumn("10").validity.toArray(), [true, true]);
    const counted = await frame.pivot("sub", "key", "value", "count").collect();
    assert.equal(counted.getColumn("10").values.dtype, "uint32");
    assert.deepEqual(counted.getColumn("10").values.toArray(), [1, 1]);
  } finally {
    await ctx.close();
  }
});

test("pivot keys exclude nulls and GPU execution is rejected", async () => {
  const ctx = context();
  try {
    const validity = new Map([
      ["key", ctx.array([true, false, true], { dtype: "bool" })],
    ]);
    const frame = pandasFrame(
      ctx,
      [1n, 2n, 3n],
      [10n, 10n, 20n],
      [1n, 2n, 3n],
      validity,
    );
    const pivoted = await frame.pivotTable("key", "sub", "value", {
      aggfunc: "sum",
    });
    assert.deepEqual(pivoted.get("key").values.toArray(), [1n, 3n]);

    await assert.rejects(
      frame.pivot("key", "sub", "value", { device: "gpu" }),
      { code: "NMATH_GPU_UNSUPPORTED_OPERATION" },
    );
  } finally {
    await ctx.close();
  }
});
