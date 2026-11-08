// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import * as pd from "../dist/pandas.js";
import * as pl from "../dist/polars.js";
import { context } from "./helpers.mjs";
const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/tables.json", import.meta.url), "utf8"),
);
function series(ctx, name, input) {
  return new pl.Series(
    name,
    ctx.array(
      input.map((v) => BigInt(v ?? 0)),
      { dtype: "int64" },
    ),
    input.includes(null)
      ? ctx.array(
          input.map((v) => v !== null),
          { dtype: "bool" },
        )
      : undefined,
  );
}
function rows(frame, names = frame.columns) {
  const n = frame instanceof pd.DataFrame ? frame.shape[0] : frame.height;
  return Array.from({ length: n }, (_, i) =>
    names.map((name) => {
      const col =
        frame instanceof pd.DataFrame ? frame.get(name) : frame.getColumn(name);
      return col.validity && !col.validity.item(i)
        ? null
        : Number(col.values.item(i));
    }),
  );
}

test("pandas floating reductions skip NaNs while core reductions propagate them", async () => {
  const ctx = context();
  try {
    for (const dtype of ["float32", "float64"]) {
      for (const values of [
        [NaN, 1, 3],
        [1, NaN, 3],
        [1, 3, NaN],
      ]) {
        const input = ctx.array(values, { dtype });
        const series = new pd.Series(input);
        for (const [name, expected] of Object.entries({
          sum: 4,
          mean: 2,
          min: 1,
          max: 3,
        })) {
          assert.equal(series[name]().dtype, dtype);
          assert.equal(await series[name]().compute(), expected);
          assert.ok(Number.isNaN(await input[name]().compute()));
        }
        assert.equal(await new pd.DataFrame({ x: input }).sum().x.compute(), 4);
      }
      for (const values of [[], [NaN, NaN]]) {
        const input = ctx.array(values, { dtype });
        const series = new pd.Series(input);
        assert.equal(await series.sum().compute(), 0);
        assert.ok(Number.isNaN(await series.mean().compute()));
        assert.ok(Number.isNaN(await series.min().compute()));
        assert.ok(Number.isNaN(await series.max().compute()));
        assert.equal(await new pd.DataFrame({ x: input }).sum().x.compute(), 0);
      }
      const plain = new pd.Series(ctx.array([1, 3], { dtype }));
      assert.equal(await plain.mean().compute(), 2);
      const infinite = new pd.Series(ctx.array([Infinity, 1], { dtype }));
      assert.equal(await infinite.sum().compute(), Infinity);
      assert.equal(await infinite.min().compute(), 1);
      const opposed = new pd.Series(
        ctx.array([Infinity, -Infinity], { dtype }),
      );
      assert.ok(Number.isNaN(await opposed.sum().compute()));
    }
    const values = ctx.array([1, NaN, 3], { dtype: "float64" });
    const validity = ctx.array([true, false, true], { dtype: "bool" });
    const nullable = new pd.Series(values, "x", undefined, validity);
    assert.throws(() => nullable.sum(), /Nullable Series reductions/);
    assert.throws(
      () => new pd.DataFrame({ x: nullable }).sum(),
      /Nullable frame reductions/,
    );
  } finally {
    await ctx.close();
  }
});

test("pandas reduction keeps execution-start snapshots, cancellation and budgets", async () => {
  const ctx = context({ maxConcurrentJobs: 1, maxQueueLength: 8 });
  try {
    const blocker = ctx.zeros([512, 512], "float64");
    const running = blocker.matmul(blocker).compute();
    const input = ctx.array([1, NaN, 3], { dtype: "float64" });
    const write = input.set([0], 5);
    const queued = new pd.Series(input).sum().compute();
    await write;
    assert.equal(await queued, 8);
    await running;
    const runningAgain = blocker.matmul(blocker).compute();
    const queuedCancel = new AbortController();
    const pending = new pd.Series(input)
      .sum()
      .compute({ signal: queuedCancel.signal });
    queuedCancel.abort();
    await assert.rejects(pending, { code: "NMATH_CANCELLED" });
    await runningAgain;
    const cancelled = new AbortController();
    cancelled.abort();
    await assert.rejects(
      new pd.Series(input).sum().compute({ signal: cancelled.signal }),
      {
        code: "NMATH_CANCELLED",
      },
    );
    await assert.rejects(
      new pd.Series(input).sum().compute({ device: "gpu" }),
      { code: "NMATH_GPU_UNSUPPORTED_OPERATION" },
    );
  } finally {
    await ctx.close();
  }
  const tiny = context({ cpuMemoryBudget: 24n });
  try {
    const input = tiny.array([1, NaN, 3], { dtype: "float64" });
    await assert.rejects(new pd.Series(input).mean().compute(), {
      code: "NMATH_MEMORY_BUDGET_EXCEEDED",
    });
    assert.equal(tiny.metrics().cpuMemory.used, 24n);
  } finally {
    await tiny.close();
  }
});
test("fixed pandas and Polars duplicate/null joins, groups, stable sort and empty groups", async () => {
  const ctx = context();
  try {
    const { k, v, rk, rv } = fixture.input;
    const a = new pl.DataFrame([series(ctx, "k", k), series(ctx, "v", v)]);
    const b = new pl.DataFrame([series(ctx, "k", rk), series(ctx, "w", rv)]);
    const p = a.toPandas(),
      q = b.toPandas();
    assert.deepEqual(
      rows(await p.merge(q, "k", { how: "left" }), ["k", "v", "w"]),
      fixture.pandasJoin,
    );
    assert.deepEqual(
      rows(await a.lazy().join(b.lazy(), "k", { how: "left" }).collect(), [
        "k",
        "v",
        "w",
      ]),
      fixture.polarsJoin,
    );
    assert.deepEqual(
      rows(await p.groupBy("k", "v", "sum")),
      fixture.pandasGroup,
    );
    assert.deepEqual(
      rows(await a.lazy().groupBy("k", "v", "sum").collect()),
      fixture.polarsGroup,
    );
    assert.deepEqual(rows(await p.sortValues("k")), fixture.pandasSort);
    assert.deepEqual(
      rows(await a.lazy().sort("k").collect()),
      fixture.polarsSort,
    );
    const empty = new pl.DataFrame([
      series(ctx, "k", []),
      series(ctx, "v", []),
    ]);
    assert.deepEqual(
      rows(await empty.toPandas().groupBy("k", "v", "sum")),
      fixture.emptyGroup,
    );
  } finally {
    await ctx.close();
  }
});
test("pandas dropna excludes float NaN keys while Polars retains them", async () => {
  const ctx = context();
  try {
    for (const dtype of ["float32", "float64"]) {
      for (const mask of [undefined, [true, true]]) {
        const frame = new pl.DataFrame([
          new pl.Series(
            "k",
            ctx.array([1, Number.NaN], { dtype }),
            mask && ctx.array(mask, { dtype: "bool" }),
          ),
          new pl.Series("v", ctx.array([10, 20], { dtype: "int32" })),
        ]);
        const pandas = frame.toPandas();
        for (const [op, expected] of [
          ["sum", 10],
          ["mean", 10],
          ["min", 10],
          ["max", 10],
          ["count", 1],
        ]) {
          const omitted = await pandas.groupBy("k", "v", op);
          const explicit = await pandas.groupBy("k", "v", op, {
            dropna: true,
          });
          assert.deepEqual(rows(omitted), [[1, expected]], `${dtype}/${op}`);
          assert.deepEqual(rows(explicit), [[1, expected]], `${dtype}/${op}`);
          assert.equal(omitted.get("k").dtype, dtype);
          assert.deepEqual(
            rows(await pandas.groupBy("k", "v", op, { dropna: false })),
            [
              [1, expected],
              [Number.NaN, op === "count" ? 1 : 20],
            ],
            `${dtype}/${op}/dropna:false`,
          );
        }
        assert.deepEqual(
          rows(await frame.lazy().groupBy("k", "v", "sum").collect()),
          [
            [1, 10],
            [Number.NaN, 20],
          ],
        );
      }

      const frame = new pl.DataFrame([
        new pl.Series(
          "k",
          ctx.array([2, Number.NaN, 1, Number.NaN, 2, 1, 3], { dtype }),
          ctx.array([true, true, true, true, true, true, false], {
            dtype: "bool",
          }),
        ),
        new pl.Series(
          "v",
          ctx.array([4, 20, 3, 30, 5, 7, 100], { dtype: "int32" }),
        ),
      ]);
      const pandas = frame.toPandas();
      assert.deepEqual(rows(await pandas.groupBy("k", "v", "sum")), [
        [1, 10],
        [2, 9],
      ]);
      assert.deepEqual(
        rows(await pandas.groupBy("k", "v", "sum", { sort: false })),
        [
          [2, 9],
          [1, 10],
        ],
      );
      assert.deepEqual(
        rows(
          await pandas.groupBy("k", "v", "sum", { dropna: false, sort: false }),
        ),
        [
          [2, 9],
          [Number.NaN, 50],
          [1, 10],
          [null, 100],
        ],
      );
      assert.deepEqual(
        rows(await frame.lazy().groupBy("k", "v", "sum").collect()),
        [
          [2, 9],
          [Number.NaN, 50],
          [1, 10],
          [null, 100],
        ],
      );

      const allMissing = new pl.DataFrame([
        new pl.Series("k", ctx.array([Number.NaN], { dtype })),
        new pl.Series("v", ctx.array([10], { dtype: "int32" })),
      ]);
      const empty = await allMissing.toPandas().groupBy("k", "v", "sum");
      assert.deepEqual(rows(empty), []);
      assert.equal(empty.get("k").dtype, dtype);
      assert.equal(empty.get("v").dtype, "int32");
    }
  } finally {
    await ctx.close();
  }
});
test("join coalesces only the right key after resolving retained names", async () => {
  const ctx = context();
  try {
    for (const [suffix, leftName] of [
      [undefined, "k_right"],
      ["_rhs", "k_rhs"],
    ]) {
      const left = new pl.DataFrame([
        series(ctx, "k", [1, 2]),
        series(ctx, leftName, [7, null]),
      ]);
      const right = new pl.DataFrame([
        series(ctx, "k", [1]),
        series(ctx, "w", [9]),
      ]);
      const options = { how: "left", suffix };
      for (const result of [
        await left.toPandas().merge(right.toPandas(), "k", options),
        await left.lazy().join(right.lazy(), "k", options).collect(),
      ]) {
        assert.deepEqual(result.columns, ["k", leftName, "w"]);
        assert.deepEqual(rows(result), [
          [1, 7, 9],
          [2, null, null],
        ]);
        const retained =
          result instanceof pd.DataFrame
            ? result.get(leftName)
            : result.getColumn(leftName);
        assert.ok(retained.validity);
        assert.equal(retained.validity.item(1), false);
      }
    }
    const left = new pl.DataFrame([
      series(ctx, "k", [1, 1]),
      series(ctx, "k_right", [7, 8]),
    ]);
    const right = new pl.DataFrame([
      series(ctx, "k", [1, 1]),
      series(ctx, "w", [9, 10]),
    ]);
    for (const result of [
      await left.toPandas().merge(right.toPandas(), "k"),
      await left.lazy().join(right.lazy(), "k").collect(),
    ]) {
      assert.deepEqual(result.columns, ["k", "k_right", "w"]);
      assert.deepEqual(rows(result), [
        [1, 7, 9],
        [1, 7, 10],
        [1, 8, 9],
        [1, 8, 10],
      ]);
    }
  } finally {
    await ctx.close();
  }
});
test("join rejects retained collisions and preserves differently named keys", async () => {
  const ctx = context();
  try {
    for (const suffix of [undefined, "_rhs", ""]) {
      const overlapping = "v" + (suffix ?? "_right");
      const left = new pl.DataFrame([
        series(ctx, "k", [1]),
        series(ctx, "v", [2]),
        ...(suffix === "" ? [] : [series(ctx, overlapping, [3])]),
      ]);
      const right = new pl.DataFrame([
        series(ctx, "k", [1]),
        series(ctx, "v", [4]),
      ]);
      for (const join of [
        () => left.toPandas().merge(right.toPandas(), "k", { suffix }),
        () => left.lazy().join(right.lazy(), "k", { suffix }),
      ]) {
        await assert.rejects(
          async () => join(),
          (error) => error.code === "NMATH_INDEX_INVALID",
        );
      }
    }
    const left = new pl.DataFrame([series(ctx, "k", [1])]);
    const right = new pl.DataFrame([
      series(ctx, "r", [1]),
      series(ctx, "w", [9]),
    ]);
    for (const result of [
      await left.toPandas().merge(right.toPandas(), "k", { rightKey: "r" }),
      await left.lazy().join(right.lazy(), "k", { rightKey: "r" }).collect(),
    ]) {
      assert.deepEqual(result.columns, ["k", "r", "w"]);
      assert.deepEqual(rows(result), [[1, 1, 9]]);
    }
  } finally {
    await ctx.close();
  }
});
test("one native execution per multi-column plan and independent repeated collection", async () => {
  const ctx = context();
  try {
    const x = ctx.array([3, 1, 2], { dtype: "int32" }),
      keep = ctx.array([true, false, true], { dtype: "bool" });
    const frame = new pl.DataFrame([
      new pl.Series("x", x),
      new pl.Series("y", x.mul(2n)),
      new pl.Series("keep", keep),
    ]);
    const query = frame.lazy().filter("keep").sort("x").select("x", "y");
    const before = ctx.metrics().totalSubmitted;
    assert.deepEqual(rows(await query.collect()), [
      [2, 4],
      [3, 6],
    ]);
    assert.equal(ctx.metrics().totalSubmitted - before, 1);
    await x.set([0], 8n);
    assert.deepEqual(rows(await query.collect()), [
      [2, 4],
      [8, 16],
    ]);
  } finally {
    await ctx.close();
  }
});
test("pandas labels align, filter/sort retain labels, conversions preserve masks or reject loss", async () => {
  const ctx = context();
  try {
    const x = ctx.array([2n, 1n], { dtype: "int64" });
    const a = new pd.Series(x, "x", ["b", "a"]);
    const b = new pd.Series(ctx.array([10n, 20n], { dtype: "int64" }), "y", [
      "a",
      "b",
    ]);
    assert.deepEqual((await a.add(b).toArray()).toArray(), [22n, 11n]);
    assert.deepEqual((await a.loc("a").compute()).index, ["a"]);
    assert.throws(() => a.toPolars(), /NMATH_INDEX_INVALID/);
    assert.equal(a.toPolars({ dropIndex: true }).name, "x");
    const p = new pd.DataFrame({ x: a });
    const sorted = await p.sortValues("x");
    assert.deepEqual(sorted.index, ["a", "b"]);
    const nullable = series(ctx, "n", [1, null]);
    const roundtrip = nullable.toPandas().toPolars();
    assert.equal(roundtrip.validity.item(1), false);
    assert.equal(roundtrip.values.dtype, "int64");
    await assert.rejects(nullable.toPandas().toArray(), /NMATH_INDEX_INVALID/);
  } finally {
    await ctx.close();
  }
});
test("pandas Series arithmetic rejects mismatched duplicate labels in either operand", async () => {
  const ctx = context();
  try {
    const make = (values, index, name = "left") =>
      new pd.Series(ctx.array(values, { dtype: "float64" }), name, index);
    const duplicate = make([10, 20], ["a", "a"]);
    const unique = make([1, 2], ["a", "b"], "right");
    const reordered = make([1, 2, 3], ["a", "a", "b"]);
    const differentlyOrdered = make([1, 2, 3], ["a", "b", "a"]);
    const operations = ["add", "sub", "mul", "div", "mod", "pow"];

    for (const operation of operations) {
      assert.throws(
        () => duplicate[operation](unique),
        (error) => error.code === "NMATH_INDEX_INVALID",
        `${operation}: duplicate left index`,
      );
      assert.throws(
        () => unique[operation](duplicate),
        (error) => error.code === "NMATH_INDEX_INVALID",
        `${operation}: duplicate right index`,
      );
      assert.throws(
        () => reordered[operation](differentlyOrdered),
        (error) => error.code === "NMATH_INDEX_INVALID",
        `${operation}: differently ordered duplicate indexes`,
      );
    }

    const sameLabels = make([1, 2], ["a", "a"], "right");
    const leftUnique = make([10, 20], ["a", "b"]);
    const rightReordered = make([1, 2], ["b", "a"], "right");
    const expected = {
      add: [
        [11, 22],
        [12, 21],
      ],
      sub: [
        [9, 18],
        [8, 19],
      ],
      mul: [
        [10, 40],
        [20, 20],
      ],
      div: [
        [10, 10],
        [5, 20],
      ],
      mod: [
        [0, 0],
        [0, 0],
      ],
      pow: [
        [10, 400],
        [100, 20],
      ],
    };
    for (const operation of operations) {
      const positional = duplicate[operation](sameLabels);
      assert.equal(positional.name, "left");
      assert.deepEqual(positional.index, ["a", "a"]);
      assert.deepEqual(
        (await positional.toArray()).toArray(),
        expected[operation][0],
        `${operation}: identical duplicate indexes align positionally`,
      );

      const aligned = leftUnique[operation](rightReordered);
      assert.equal(aligned.name, "left");
      assert.deepEqual(aligned.index, ["a", "b"]);
      assert.deepEqual(
        (await aligned.toArray()).toArray(),
        expected[operation][1],
        `${operation}: unique matching indexes reorder right-hand values`,
      );
    }
    assert.throws(
      () => unique.add(make([1, 2], ["a", "c"])),
      (error) => error.code === "NMATH_INDEX_INVALID",
    );

    const numeric = make([3, 4], [1, 2]).add(make([5, 6], [2, 1]));
    assert.deepEqual(numeric.index, [1, 2]);
    assert.deepEqual((await numeric.toArray()).toArray(), [9, 9]);
    const empty = make([], []).add(make([], []));
    assert.deepEqual((await empty.toArray()).toArray(), []);
    assert.deepEqual((await unique.add(2).toArray()).toArray(), [3, 4]);
    const nullable = new pd.Series(
      ctx.array([1, 2], { dtype: "float64" }),
      "nullable",
      ["a", "b"],
      ctx.array([true, false], { dtype: "bool" }),
    );
    assert.throws(
      () => nullable.add(unique),
      (error) => error.code === "NMATH_INDEX_INVALID",
    );
  } finally {
    await ctx.close();
  }
});
test("table errors reject mixed contexts, malformed masks, GPU and cancellation without leaking names", async () => {
  const a = context(),
    b = context();
  try {
    const x = a.array([1, 2], { dtype: "int32" }),
      y = b.array([1, 2], { dtype: "int32" });
    assert.throws(
      () => new pd.DataFrame({ "private-column": x, y }),
      /NMATH_CONTEXT_MISMATCH/,
    );
    const frame = new pl.DataFrame([new pl.Series("private-column", x)]);
    assert.throws(
      () => frame.getColumn("secret-name"),
      (e) =>
        e.code === "NMATH_INDEX_INVALID" && !e.message.includes("secret-name"),
    );
    await assert.rejects(
      frame.lazy().collect({ device: "gpu" }),
      (e) => e.code === "NMATH_GPU_UNSUPPORTED_OPERATION",
    );
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(
      frame.lazy().collect({ signal: controller.signal }),
      (e) => e.code === "NMATH_CANCELLED",
    );
    assert.throws(
      () => new pl.Series("n", x, a.array([true], { dtype: "bool" })),
      /NMATH_SHAPE_MISMATCH/,
    );
  } finally {
    await a.close();
    await b.close();
  }
});
test("fixed-version group result dtypes and all-null groups", async () => {
  const ctx = context();
  try {
    for (const row of fixture.aggregationDtypes) {
      const values =
        row.input === "bool"
          ? [true, true]
          : row.input.includes("64") && !row.input.includes("float")
            ? [1n, 1n]
            : [1, 1];
      const frame = new pl.DataFrame([
        new pl.Series("k", ctx.array([1, 1], { dtype: "int32" })),
        new pl.Series("v", ctx.array(values, { dtype: row.input })),
      ]);
      const p = await frame.toPandas().groupBy("k", "v", row.op);
      const l = await frame.lazy().groupBy("k", "v", row.op).collect();
      assert.equal(
        p.get("v").dtype,
        row.pandas,
        `${row.input}/${row.op}/pandas`,
      );
      assert.equal(
        l.getColumn("v").dtype,
        row.polars,
        `${row.input}/${row.op}/polars`,
      );
      assert.equal(Number(p.get("v").values.item(0)), row.value);
      assert.equal(Number(l.getColumn("v").values.item(0)), row.value);
    }
    const frame = new pl.DataFrame([
      series(ctx, "k", [1, 1, 2]),
      series(ctx, "v", [null, null, 4]),
    ]);
    for (const op of ["sum", "mean", "min", "max", "count"])
      assert.deepEqual(
        rows(await frame.toPandas().groupBy("k", "v", op)),
        fixture.allNullGroups[op],
      );
    const overflow = new pl.DataFrame([
      new pl.Series("k", ctx.array([1, 1], { dtype: "int32" })),
      new pl.Series("v", ctx.array([127, 127], { dtype: "int8" })),
    ]);
    assert.equal(
      (await overflow.toPandas().groupBy("k", "v", "sum")).get("v").dtype,
      "int64",
    );
  } finally {
    await ctx.close();
  }
});
test("table result ownership and failed intermediate budgets release native allocations", async () => {
  const ctx = context();
  try {
    const x = ctx.array([1, 2], { dtype: "int32" });
    const baseline = ctx.metrics().cpuMemory.used;
    const out = await new pl.DataFrame([new pl.Series("x", x)])
      .lazy()
      .collect();
    assert.notEqual(out.getColumn("x").values.nativeArray, x.nativeArray);
    out.getColumn("x").values.dispose();
    assert.equal(ctx.metrics().cpuMemory.used, baseline);
  } finally {
    await ctx.close();
  }
  const small = context({ cpuMemoryBudget: 16n });
  try {
    const x = small.array([1n, 2n], { dtype: "int64" });
    await assert.rejects(
      new pl.DataFrame([new pl.Series("x", x)]).lazy().collect(),
      (e) => e.code === "NMATH_MEMORY_BUDGET_EXCEEDED",
    );
    assert.equal(small.metrics().cpuMemory.used, 16n);
  } finally {
    await small.close();
  }
});
