// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { linalg, matmul } from "../dist/numpy.js";
import { context } from "./helpers.mjs";

const close = (a, b, tol = 1e-9) =>
  a.flat(Infinity).every((x, i) => Math.abs(x - b.flat(Infinity)[i]) <= tol);

test("linalg factorizations satisfy NumPy invariants", async () => {
  const ctx = context();
  try {
    const a = ctx.array(
      [
        [4, 2, 0.6],
        [2, 5, 1],
        [0.6, 1, 3],
      ],
      { dtype: "float64" },
    );
    const l = linalg.cholesky(a);
    const rebuilt = await matmul(l, l.transpose()).compute();
    assert.ok(close(rebuilt.toArray(), a.toArray()));
    const upper = (
      await linalg.cholesky(a, { upper: true }).compute()
    ).toArray();
    assert.equal(upper[1][0], 0);

    const [q, r] = linalg.qr(a);
    assert.ok(close((await matmul(q, r).compute()).toArray(), a.toArray()));
    const named = linalg.qr(a);
    assert.equal(named.Q, named[0]);
    assert.equal(named.R, named[1]);
    assert.deepEqual(
      linalg.qr(ctx.zeros([4, 2], "float32"), "complete").Q.shape,
      [4, 4],
    );
    assert.deepEqual(
      linalg.qr(ctx.zeros([4, 2], "float32"), "r").shape,
      [2, 2],
    );
    assert.throws(() => linalg.qr(a, "raw"), { code: "NMATH_INDEX_INVALID" });

    const { eigenvalues, eigenvectors } = linalg.eigh(a);
    const w = (await eigenvalues.compute()).toArray();
    assert.ok(w[0] <= w[1] && w[1] <= w[2]);
    const av = (await matmul(a, eigenvectors).compute()).toArray();
    const v = (await eigenvectors.compute()).toArray();
    assert.ok(
      close(
        av,
        v.map((row) => row.map((x, j) => x * w[j])),
      ),
    );
    assert.deepEqual((await linalg.eigvalsh(a, "U").compute()).toArray(), w);

    const tall = ctx.array(
      [
        [1, 2],
        [3, 4],
        [5, 6],
      ],
      { dtype: "float64" },
    );
    const svd = linalg.svd(tall, { full_matrices: false });
    assert.deepEqual(svd.U.shape, [3, 2]);
    const s = (await svd.S.compute()).toArray();
    assert.ok(s[0] >= s[1]);
    assert.deepEqual(linalg.svd(tall).U.shape, [3, 3]);
    assert.deepEqual(
      (await linalg.svd(tall, { compute_uv: false }).compute()).toArray(),
      s,
    );
    assert.deepEqual((await linalg.svdvals(tall).compute()).toArray(), s);

    const { sign, logabsdet } = linalg.slogdet(a);
    const det = await linalg.det(a).compute();
    assert.ok(
      Math.abs(
        (await sign.compute()) * Math.exp(await logabsdet.compute()) - det,
      ) < 1e-9,
    );
  } finally {
    await ctx.close();
  }
});

test("linalg factorization errors are stable", async () => {
  const ctx = context();
  try {
    const indefinite = ctx.array(
      [
        [1, 2],
        [2, 1],
      ],
      { dtype: "float64" },
    );
    await assert.rejects(linalg.cholesky(indefinite).compute(), {
      code: "NMATH_LINALG_NOT_POSITIVE_DEFINITE",
    });
    await assert.rejects(
      linalg.svdvals(ctx.array([[Number.NaN]], { dtype: "float64" })).compute(),
      { code: "NMATH_LINALG_NO_CONVERGENCE" },
    );
    assert.throws(() => linalg.eigh(ctx.zeros([2, 3], "float64")), {
      code: "NMATH_SHAPE_MISMATCH",
    });
    assert.throws(() => linalg.eigh(indefinite, "X"), {
      code: "NMATH_INDEX_INVALID",
    });
    assert.throws(() => linalg.svd(indefinite, { out: 1 }), {
      code: "NMATH_INDEX_INVALID",
    });
    const singular = linalg.slogdet(
      ctx.array(
        [
          [1, 2],
          [2, 4],
        ],
        { dtype: "float64" },
      ),
    );
    assert.equal(await singular.sign.compute(), 0);
    assert.equal(await singular.logabsdet.compute(), Number.NEGATIVE_INFINITY);
    await assert.rejects(
      linalg.svdvals(indefinite).compute({ device: "gpu" }),
      {
        code: "NMATH_GPU_UNSUPPORTED_OPERATION",
      },
    );
  } finally {
    await ctx.close();
  }
});
