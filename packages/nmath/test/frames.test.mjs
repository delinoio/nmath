// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";

import * as np from "../dist/numpy.js";
import * as pd from "../dist/pandas.js";
import * as pl from "../dist/polars.js";
import { context } from "./helpers.mjs";

test("numpy entry point builds the same lazy expressions", async () => {
  const ctx = context();
  const a = np.arange(ctx, 0, 6, 1, "float64");
  const b = np.ones(ctx, [6], "float64");
  assert.deepEqual(
    (await np.multiply(np.add(a, b), 2).compute()).toArray(),
    [2, 4, 6, 8, 10, 12],
  );
  assert.equal(await np.sum(a).compute(), 15);
  const rng = np.random.defaultRng(ctx, 42);
  assert.equal((await rng.random([1]).compute()).item(0), 0.7739560485559633);
  await ctx.close();
});

test("pandas Series and DataFrame keep names and compute lazily", async () => {
  const ctx = context();
  const price = new pd.Series(
    ctx.array([1, 2, 3], { dtype: "float64" }),
    "price",
  );
  const qty = new pd.Series(ctx.array([2n, 2n, 4n], { dtype: "int64" }), "qty");
  const df = new pd.DataFrame({ price, qty }).assign({ total: price.mul(qty) });
  assert.deepEqual(df.columns, ["price", "qty", "total"]);
  assert.deepEqual(df.shape, [3, 3]);
  assert.equal(df.dtypes.total, "float64");
  const computed = await df.compute();
  assert.deepEqual(
    (await computed.get("total").toArray()).toArray(),
    [2, 4, 12],
  );
  assert.equal(await df.sum().total.compute(), 18);
  assert.throws(
    () =>
      new pd.DataFrame({ a: price, b: ctx.array([1], { dtype: "float64" }) }),
    /NMATH_SHAPE_MISMATCH/,
  );
  await ctx.close();
});

test("polars lazy frames collect selected expressions", async () => {
  const ctx = context();
  const df = new pl.DataFrame([
    new pl.Series("a", ctx.array([1, 2, 3], { dtype: "int32" })),
    new pl.Series("b", ctx.array([10, 20, 30], { dtype: "int32" })),
  ]);
  const out = await df
    .lazy()
    .withColumns(pl.col("a").mul(pl.col("b")).alias("ab"))
    .select("a", "ab")
    .collect();
  assert.deepEqual(out.columns, ["a", "ab"]);
  assert.deepEqual(out.getColumn("ab").values.toArray(), [10, 40, 90]);
  assert.equal(out.height, 3);
  await ctx.close();
});
