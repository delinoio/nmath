// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { linalg, matmul } from "../dist/numpy.js";
import { context } from "./helpers.mjs";

test("eig returns complex eigenpairs that satisfy A v = v w", async () => {
  const ctx = context();
  try {
    const rotation = ctx.array(
      [
        [0, -1],
        [1, 0],
      ],
      { dtype: "float64" },
    );
    const [w, v] = linalg.eig(rotation);
    assert.equal(w.dtype, "complex128");
    const values = (await w.compute()).toArray();
    const imag = values.map((z) => z.imag).sort((a, b) => a - b);
    assert.ok(Math.abs(imag[0] + 1) < 1e-12 && Math.abs(imag[1] - 1) < 1e-12);
    const av = (await matmul(rotation, v).compute()).toArray();
    const vv = (await v.compute()).toArray();
    for (let i = 0; i < 2; i += 1) {
      for (let j = 0; j < 2; j += 1) {
        const re =
          vv[i][j].real * values[j].real - vv[i][j].imag * values[j].imag;
        const im =
          vv[i][j].real * values[j].imag + vv[i][j].imag * values[j].real;
        assert.ok(
          Math.abs(av[i][j].real - re) < 1e-12 &&
            Math.abs(av[i][j].imag - im) < 1e-12,
        );
      }
    }
    assert.equal(linalg.eig(rotation).eigenvalues.shape.length, 1);
    assert.equal(
      linalg.eigvals(ctx.zeros([3, 3], "float32")).dtype,
      "complex64",
    );
    assert.throws(() => linalg.eig(ctx.zeros([2, 3], "float64")), {
      code: "NMATH_SHAPE_MISMATCH",
    });
    await assert.rejects(linalg.eigvals(rotation).compute({ device: "gpu" }), {
      code: "NMATH_GPU_UNSUPPORTED_OPERATION",
    });
  } finally {
    await ctx.close();
  }
});
