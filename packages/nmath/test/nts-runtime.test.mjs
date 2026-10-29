// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";

import * as __nmath from "../dist/nts-runtime.js";
import { context } from "./helpers.mjs";

test("operator helpers keep ordinary JS semantics for JS values", () => {
  assert.equal(__nmath.add(1, 2), 3);
  assert.equal(__nmath.add("a", 1), "a1");
  assert.equal(__nmath.mod(-7, 3), -1);
  assert.equal(__nmath.pow(2n, 10n), 1024n);
  assert.equal(__nmath.neg(5), -5);
  assert.equal(__nmath.get({ x: 1 }, "x"), 1);
  assert.equal(__nmath.get([4, 5], 1), 5);
});

test("operator helpers build lazy NMath expressions", async () => {
  const ctx = context();
  const a = ctx.array(
    [
      [1, 2],
      [3, 4],
    ],
    { dtype: "float64" },
  );
  // a * 2 + -a[[0, slice(undefined, undefined, -1)]]
  const expr = __nmath.add(
    __nmath.mul(a, 2),
    __nmath.neg(__nmath.get(a, [0, __nmath.slice(undefined, undefined, -1)])),
  );
  assert.deepEqual((await expr.compute()).toArray(), [
    [0, 3],
    [4, 7],
  ]);
  // NumPy mod semantics, not JS remainder semantics.
  const m = await __nmath
    .mod(ctx.array([-7], { dtype: "float64" }), 3)
    .compute();
  assert.deepEqual(m.toArray(), [2]);
  await ctx.close();
});

test("set awaits the write and resolves to the assigned value", async () => {
  const ctx = context();
  const a = ctx.array([0, 0], { dtype: "int32" });
  assert.equal(await __nmath.set(a, 1, 9), 9);
  assert.deepEqual(a.toArray(), [0, 9]);
  const plain = [1, 2];
  assert.equal(await __nmath.set(plain, 0, 5), 5);
  assert.deepEqual(plain, [5, 2]);
  await ctx.close();
});
