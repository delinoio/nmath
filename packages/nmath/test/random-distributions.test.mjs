// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { context } from "./helpers.mjs";

const fixtureRoot = new URL("./fixtures/random/", import.meta.url);
const calls = readFileSync(new URL("mixed.txt", fixtureRoot), "utf8")
  .trim()
  .split("\n")
  .map((line) => line.split("|"));
function expression(generator, [op, dtype, a, b, endpoint, size]) {
  const shape = [Number(size)];
  switch (op) {
    case "random":
    case "standard_normal":
      return generator[op](shape, dtype);
    case "integers":
      return generator.integers(BigInt(a), BigInt(b), shape, {
        dtype,
        endpoint: endpoint === "1",
      });
    case "exponential":
      return generator.exponential(Number(a), shape);
    default:
      return generator[op](Number(a), Number(b), shape);
  }
}

test("mixed NumPy calls match exact bytes in shared-generator request order", async () => {
  const ctx = context({ maxQueueLength: 64 });
  try {
    const rng = ctx.generator(12345);
    const results = await Promise.all(
      calls.map((call) => expression(rng, call).compute()),
    );
    for (let i = 0; i < results.length; i++) {
      const result = results[i];
      const array = result.toTypedArray();
      const bytes = Buffer.from(
        array.buffer,
        array.byteOffset,
        array.byteLength,
      );
      assert.deepEqual(
        bytes,
        readFileSync(new URL(calls[i][6], fixtureRoot)),
        `${calls[i][0]} ${calls[i][1]}`,
      );
      result.dispose();
    }
  } finally {
    await ctx.close();
  }
});

test("repeated expressions advance independently and errors preserve cached words", async () => {
  const ctx = context();
  try {
    const rng = ctx.generator(12345);
    const oracle = ctx.generator(12345);
    const expr = rng.random([3], "float32");
    for (let i = 0; i < 3; i++) {
      assert.throws(() => rng.integers(0, 257, [1], { dtype: "uint8" }));
      assert.throws(() => rng.normal(0, -1, [1]));
      assert.throws(() => rng.normal(0, -0, [1]));
      assert.throws(() => rng.exponential(-0, [1]));
      assert.throws(() => rng.uniform(0, -0, [1]));
      const signal = AbortSignal.abort();
      await assert.rejects(expr.compute({ signal }));
      const a = await expr.compute();
      const b = await oracle.random([3], "float32").compute();
      assert.deepEqual(a.toArray(), b.toArray());
      a.dispose();
      b.dispose();
    }
    assert.throws(() =>
      rng.integers(Number.MAX_SAFE_INTEGER + 1, undefined, [1]),
    );
    assert.throws(() => rng.random([-1]));
    assert.equal(typeof (await rng.integers(0, 9, []).compute()), "bigint");
  } finally {
    await ctx.close();
  }
});
