// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";

import * as __nmath from "../dist/nts-runtime.js";
import { context } from "./helpers.mjs";

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
  await ctx.close();
});
