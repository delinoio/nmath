// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  expand_dims,
  flip,
  fliplr,
  flipud,
  moveaxis,
  ravel,
  squeeze,
  swapaxes,
} from "../dist/numpy.js";
import { NMathError, slice } from "../dist/index.js";
import { context } from "./helpers.mjs";

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/axes.json", import.meta.url), "utf8"),
);
const functions = {
  squeeze,
  expand_dims,
  swapaxes,
  moveaxis,
  ravel,
  flip,
  fliplr,
  flipud,
};

/**
 * NumPy error classes map to NMath codes: `AxisError` to an out-of-bounds
 * index, a non-integer axis (`TypeError`) and repeated or mismatched axis
 * lists to an invalid index, and other `ValueError`s to a shape mismatch.
 */
function expectedCode({ class: name, message }) {
  if (name === "AxisError") return "NMATH_INDEX_OUT_OF_BOUNDS";
  if (name === "TypeError") return "NMATH_INDEX_INVALID";
  if (/repeated|duplicate|same number of elements/.test(message))
    return "NMATH_INDEX_INVALID";
  return "NMATH_SHAPE_MISMATCH";
}

function flat(result) {
  return typeof result === "number"
    ? [result]
    : result.toArray().flat(Infinity);
}

test("axis helpers match NumPy shapes, values and errors", async () => {
  assert.equal(fixture.numpy, "2.5.4");
  const ctx = context();
  try {
    for (const c of fixture.cases) {
      const label = `${c.function}(${JSON.stringify(c.shape)}, ${JSON.stringify(c.args)})`;
      const size = c.shape.reduce((n, m) => n * m, 1);
      const a = ctx
        .array(
          Array.from({ length: size }, (_, i) => i),
          { dtype: "int32" },
        )
        .reshape(c.shape);
      const call = () => functions[c.function](a, ...c.args);
      if (c.error) {
        assert.throws(
          call,
          (error) =>
            error instanceof NMathError && error.code === expectedCode(c.error),
          label,
        );
        continue;
      }
      const expr = call();
      assert.equal(expr.dtype, "int32", label);
      assert.deepEqual(expr.shape, c.result.shape, label);
      assert.deepEqual(flat(await expr.compute()), c.result.values, label);
    }
  } finally {
    await ctx.close();
  }
});

test("axis helpers preserve dtypes and read actual-start values", async () => {
  const ctx = context();
  try {
    const a = ctx.array(
      [
        [1n, 2n, 3n],
        [4n, 5n, 6n],
      ],
      { dtype: "int64" },
    );
    const flipped = flip(a);
    const moved = moveaxis(a, 0, -1);
    const raveled = ravel(expand_dims(a, 1));
    await a.set([0, 0], 10n);
    assert.equal(flipped.dtype, "int64");
    assert.deepEqual((await flipped.compute()).toArray(), [
      [6n, 5n, 4n],
      [3n, 2n, 10n],
    ]);
    assert.deepEqual((await moved.compute()).toArray(), [
      [10n, 4n],
      [2n, 5n],
      [3n, 6n],
    ]);
    assert.deepEqual((await raveled.compute()).toArray(), [
      10n,
      2n,
      3n,
      4n,
      5n,
      6n,
    ]);

    const words = ctx.array(["a", "b", "c"], { dtype: "str" });
    assert.deepEqual((await flipud(words).compute()).toArray(), [
      "c",
      "b",
      "a",
    ]);
    const half = ctx.zeros([1, 2], "float16");
    assert.equal(squeeze(half).dtype, "float16");
    assert.deepEqual(swapaxes(half, 0, 1).shape, [2, 1]);
    const scalar = await squeeze(ctx.zeros([1, 1], "float64")).compute();
    assert.equal(scalar, 0);
  } finally {
    await ctx.close();
  }
});

test("axis helpers reject unsupported arguments", async () => {
  const ctx = context();
  try {
    const a = ctx.zeros([2, 3], "float32");
    assert.throws(() => ravel(a, "F"), TypeError);
    for (const axis of ["0", 1n, null, Number.NaN]) {
      assert.throws(
        () => flip(a, axis),
        (error) => error.code === "NMATH_INDEX_INVALID",
      );
    }
    assert.throws(
      () => moveaxis(a, [0], [1, 0]),
      (error) => error.code === "NMATH_INDEX_INVALID",
    );
  } finally {
    await ctx.close();
  }
});

test("ravel views contiguous inputs and copies strided ones", async () => {
  const ctx = context();
  try {
    const a = ctx.array([0, 1, 2, 3, 4, 5], { dtype: "int32", shape: [2, 3] });
    const view = await ravel(a).compute();
    assert.ok(view.sharesStorage(a));
    for (const [expr, values] of [
      [ravel(a.get(0, slice(undefined, undefined, 2))), [0, 2]],
      [ravel(flip(a)), [5, 4, 3, 2, 1, 0]],
      [ravel(swapaxes(a, 0, 1)), [0, 3, 1, 4, 2, 5]],
    ]) {
      const flat = await expr.compute();
      assert.deepEqual(flat.toArray(), values);
      assert.ok(!flat.sharesStorage(a));
      await flat.set([0], 9);
      assert.deepEqual(a.toArray(), [
        [0, 1, 2],
        [3, 4, 5],
      ]);
      flat.dispose();
    }
    // A stepped view that is already materialized also copies.
    const stepped = await a.get(0, slice(undefined, undefined, 2)).compute();
    assert.ok(stepped.sharesStorage(a));
    const copied = await ravel(stepped).compute();
    assert.ok(!copied.sharesStorage(a));
    for (const array of [view, stepped, copied]) array.dispose();
  } finally {
    await ctx.close();
  }
});

test("axis helpers keep the layout nodes' GPU admission", async () => {
  const ctx = context();
  try {
    const a = ctx.zeros([2, 3], "float32");
    for (const expr of [
      squeeze(expand_dims(a, 0)),
      swapaxes(a, 0, 1),
      moveaxis(a, 0, 1),
      ravel(a),
      flip(a),
      fliplr(a),
      flipud(a),
    ]) {
      await assert.rejects(
        expr.compute({ device: "gpu" }),
        (error) => error.code === "NMATH_GPU_UNSUPPORTED_OPERATION",
      );
    }
  } finally {
    await ctx.close();
  }
});
