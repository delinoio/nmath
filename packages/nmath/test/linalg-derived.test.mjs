// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { Complex } from "../dist/index.js";
import { linalg } from "../dist/numpy.js";
import { context } from "./helpers.mjs";

const fixture = JSON.parse(
  readFileSync(
    new URL("./fixtures/linalg-derived.json", import.meta.url),
    "utf8",
  ),
);

function decodeValue(v, dtype) {
  if (dtype.startsWith("complex")) return new Complex(v[0], v[1]);
  if (dtype === "int64" || dtype === "uint64") return BigInt(v);
  if (v === null) return Number.NaN;
  if (v === "inf") return Number.POSITIVE_INFINITY;
  if (v === "-inf") return Number.NEGATIVE_INFINITY;
  return v;
}

function flat(result) {
  const list =
    result !== null && typeof result === "object" && "toArray" in result
      ? [result.toArray()].flat(Infinity)
      : [result];
  return list.map((v) =>
    v instanceof Complex
      ? [v.real, v.imag]
      : typeof v === "bigint"
        ? Number(v)
        : v,
  );
}

function order(o) {
  if (o === "inf") return Number.POSITIVE_INFINITY;
  if (o === "-inf") return Number.NEGATIVE_INFINITY;
  return o ?? undefined;
}

function call(c, a) {
  switch (c.fn) {
    case "norm":
      return linalg.norm(
        a,
        order(c.args.ord),
        c.args.axis,
        c.args.keepdims ?? false,
      );
    case "pinv":
      return linalg.pinv(a, "rtol" in c.args ? { rtol: c.args.rtol } : {});
    case "cond":
      return linalg.cond(a, c.args.p);
    case "matrix_rank":
      return linalg.matrix_rank(
        a,
        c.args.tol === undefined ? {} : { tol: c.args.tol },
      );
    default:
      return linalg.matrix_power(a, c.args.n);
  }
}

test("norm, matrix_rank, pinv, cond and matrix_power match NumPy", async () => {
  assert.equal(fixture.numpy, "2.5.4");
  const ctx = context();
  try {
    for (const [i, c] of fixture.cases.entries()) {
      const label = `case ${i}: ${c.fn} ${JSON.stringify(c.args)} ${c.input.dtype} ${JSON.stringify(c.input.shape)}`;
      const values = c.input.values.map((v) => decodeValue(v, c.input.dtype));
      const a = ctx.array(values, {
        dtype: c.input.dtype,
        shape: c.input.shape,
      });
      const expr = call(c, a);
      assert.deepEqual(expr.shape, c.expected.shape, label);
      assert.equal(expr.dtype, c.expected.dtype, label);
      const got = flat(await expr.compute());
      const want = c.expected.values.map((v) => {
        const d = decodeValue(v, c.expected.dtype);
        return d instanceof Complex
          ? [d.real, d.imag]
          : typeof d === "bigint"
            ? Number(d)
            : d;
      });
      if (c.atLeast !== undefined) {
        assert.ok(got[0] > c.atLeast, `${label}: ${got[0]} <= ${c.atLeast}`);
        continue;
      }
      const single = ["float32", "complex64"].includes(c.input.dtype);
      const decimal = c.fn === "pinv" ? (single ? 5 : 11) : single ? 6 : 12;
      const exact = !/^(float|complex)/.test(c.expected.dtype);
      got.flat().forEach((x, j) => {
        const y = want.flat()[j];
        if (exact || !Number.isFinite(y)) {
          assert.ok(
            Object.is(x, y) || x === y,
            `${label} element ${j}: ${x} vs ${y}`,
          );
          return;
        }
        const scale = c.fn === "matrix_power" ? Math.max(1, Math.abs(y)) : 1;
        assert.ok(
          Math.abs(x - y) < 1.5 * 10 ** -decimal * scale,
          `${label} element ${j}: ${x} vs ${y}`,
        );
      });
    }
  } finally {
    await ctx.close();
  }
});

test("derived linalg rejections", async () => {
  const ctx = context();
  try {
    const t = ctx.zeros([2, 3, 4], "float64");
    assert.throws(() => linalg.norm(t, 1), { code: "NMATH_SHAPE_MISMATCH" });
    assert.throws(() => linalg.norm(t, "fro", 0), {
      code: "NMATH_INDEX_INVALID",
    });
    assert.throws(() => linalg.norm(t, 3, [0, 1]), {
      code: "NMATH_INDEX_INVALID",
    });
    assert.throws(() => linalg.norm(t, 1, [1, 1]), {
      code: "NMATH_INDEX_INVALID",
    });
    assert.throws(() => linalg.norm(t, 1, 3), {
      code: "NMATH_INDEX_OUT_OF_BOUNDS",
    });
    assert.throws(() => linalg.norm(ctx.zeros([0], "float64"), Infinity), {
      code: "NMATH_SHAPE_MISMATCH",
    });
    const sq = ctx.zeros([2, 2], "float64");
    assert.throws(() => linalg.cond(sq, 1), { code: "NMATH_INDEX_INVALID" });
    assert.throws(() => linalg.cond(ctx.zeros([0, 0], "float64")), {
      code: "NMATH_SHAPE_MISMATCH",
    });
    assert.throws(() => linalg.matrix_power(ctx.zeros([2, 3], "float64"), 2), {
      code: "NMATH_SHAPE_MISMATCH",
    });
    assert.throws(() => linalg.matrix_power(sq, 1.5), {
      code: "NMATH_INDEX_INVALID",
    });
    assert.throws(() => linalg.matrix_rank(sq, { tol: 1, rtol: 1 }), {
      code: "NMATH_INDEX_INVALID",
    });
    assert.throws(() => linalg.pinv(sq, { rcond: 1, rtol: 1 }), {
      code: "NMATH_INDEX_INVALID",
    });
    await assert.rejects(linalg.norm(sq).compute({ device: "gpu" }), {
      code: "NMATH_GPU_UNSUPPORTED_OPERATION",
    });
  } finally {
    await ctx.close();
  }
});
