// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { ellipsis, indices, mask, newaxis, slice } from "../dist/index.js";
import { get, set } from "../dist/nts-runtime.js";
import { context } from "./helpers.mjs";

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/newaxis.json", import.meta.url), "utf8"),
);

function decode(s) {
  switch (s.kind) {
    case "newaxis":
      return newaxis;
    case "ellipsis":
      return ellipsis;
    case "index":
      return s.value;
    case "slice":
      return slice(
        s.start ?? undefined,
        s.stop ?? undefined,
        s.step ?? undefined,
      );
    case "mask":
      return mask(s.values);
    default:
      return indices(s.values);
  }
}

test("newaxis and ellipsis reads match NumPy", async () => {
  assert.equal(fixture.numpy, "2.5.4");
  const ctx = context();
  try {
    const values = Array.from({ length: 24 }, (_, i) => i);
    const a = ctx.array(values, { dtype: "int32" }).reshape([2, 3, 4]);
    for (const c of fixture.cases) {
      const label = JSON.stringify(c.index);
      const expr = a.get(...c.index.map(decode));
      assert.deepEqual(expr.shape, c.shape, label);
      const result = await expr.compute();
      const flat =
        typeof result === "number" ? [result] : result.toArray().flat(Infinity);
      assert.deepEqual(flat, c.values, label);
    }
    // .nts bracket reads lower to the same runtime helper.
    assert.deepEqual(get(a, [newaxis, ellipsis, 0]).shape, [1, 2, 3]);
  } finally {
    await ctx.close();
  }
});

test("several array selectors broadcast in reads and writes", async () => {
  const ctx = context();
  try {
    const a = ctx.zeros([3, 4], "int32");
    await a.set([indices([0, 2]), indices([1, 3])], 9n);
    assert.deepEqual(a.toArray(), [
      [0, 9, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 9],
    ]);
    await a.set([mask([false, true, false]), indices([0, 2])], 4n);
    assert.deepEqual(a.toArray()[1], [4, 0, 4, 0]);
    assert.deepEqual(a.get(indices([0, 1]), indices([1, 2])).shape, [2]);
    assert.throws(() => a.get(indices([0, 1]), indices([0, 1, 2])), {
      code: "NMATH_INDEX_INVALID",
    });
    assert.throws(() => a.get(indices([0, 3]), indices([0, 1])), {
      code: "NMATH_INDEX_OUT_OF_BOUNDS",
    });
  } finally {
    await ctx.close();
  }
});

test("ellipsis writes and newaxis rejections", async () => {
  const ctx = context();
  try {
    const a = ctx.zeros([2, 3], "int32");
    await a.set([ellipsis, 1], 5n);
    assert.deepEqual(a.toArray(), [
      [0, 5, 0],
      [0, 5, 0],
    ]);
    await set(a, [0, ellipsis], 7n);
    assert.deepEqual(a.toArray()[0], [7, 7, 7]);
    assert.throws(() => a.set([newaxis, 0], 1n), {
      code: "NMATH_INDEX_INVALID",
    });
    assert.throws(() => a.get(ellipsis, ellipsis), {
      code: "NMATH_INDEX_INVALID",
    });
    assert.throws(() => a.get(0, 0, ellipsis, 0), {
      code: "NMATH_INDEX_INVALID",
    });
  } finally {
    await ctx.close();
  }
});
