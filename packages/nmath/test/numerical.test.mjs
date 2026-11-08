// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { createContext, Complex } from "../dist/index.js";
import { linalg, fft } from "../dist/numpy.js";
const context = (budget = 1n << 24n) =>
  createContext({
    maxConcurrentJobs: 1,
    maxQueueLength: 4,
    cpuMemoryBudget: budget,
    gpuMemoryBudget: 0n,
  });

test("linalg is lazy, preserves complex transport and broadcasts solve batches", async () => {
  const c = context();
  try {
    const a = c.array(
      [
        [new Complex(1, 2), new Complex(2, 3)],
        [new Complex(3, 4), new Complex(4, 5)],
      ],
      { dtype: "complex128" },
    );
    const d = linalg.det(a);
    assert.equal(d.dtype, "complex128");
    assert.deepEqual(d.shape, []);
    const result = await d.compute();
    assert.ok(result instanceof Complex);
    assert.ok(Math.abs(result.real) < 1e-12);
    assert.ok(Math.abs(result.imag + 4) < 1e-12);
    const b = c.array(
      [
        [
          [2, 0],
          [0, 4],
        ],
        [
          [1, 0],
          [0, 2],
        ],
      ],
      { dtype: "float64" },
    );
    const rhs = c.array([2, 4], { dtype: "float64" });
    const x = await linalg.solve(b, rhs).compute();
    assert.deepEqual(x.toArray(), [
      [1, 1],
      [2, 2],
    ]);
    x.dispose();
    const inverse = await linalg.inv(b).compute();
    assert.deepEqual(inverse.toArray(), [
      [
        [0.5, 0],
        [0, 0.25],
      ],
      [
        [1, 0],
        [0, 0.5],
      ],
    ]);
    inverse.dispose();
  } finally {
    await c.close();
  }
});
test("linalg keeps finite extreme inverse and solve results", async () => {
  const c = context();
  try {
    for (const scale of [2 ** 600, 2 ** -600]) {
      const reciprocal = 1 / scale;
      const one = c.array([[scale]], { dtype: "float64" });
      const rhs = c.array([scale], { dtype: "float64" });
      const inverse = await linalg.inv(one).compute();
      const solution = await linalg.solve(one, rhs).compute();
      assert.equal(inverse.dtype, "float64");
      assert.deepEqual(inverse.shape, [1, 1]);
      assert.deepEqual(inverse.toArray(), [[reciprocal]]);
      assert.deepEqual(solution.toArray(), [1]);
      inverse.dispose();
      solution.dispose();

      const diagonal = c.array(
        [
          [scale, 0],
          [0, scale],
        ],
        { dtype: "float64" },
      );
      const diagonalRhs = c.array([scale, scale], { dtype: "float64" });
      const diagonalInverse = await linalg.inv(diagonal).compute();
      const diagonalSolution = await linalg
        .solve(diagonal, diagonalRhs)
        .compute();
      assert.deepEqual(diagonalInverse.toArray(), [
        [reciprocal, 0],
        [0, reciprocal],
      ]);
      assert.deepEqual(diagonalSolution.toArray(), [1, 1]);
      diagonalInverse.dispose();
      diagonalSolution.dispose();

      const z = new Complex(scale, scale);
      const complexOne = c.array([[z]], { dtype: "complex128" });
      const complexRhs = c.array([z], { dtype: "complex128" });
      const complexSolution = await linalg
        .solve(complexOne, complexRhs)
        .compute();
      const [complexValue] = complexSolution.toArray();
      assert.ok(Math.abs(complexValue.real - 1) < 1e-12);
      assert.ok(Math.abs(complexValue.imag) < 1e-12);
      complexSolution.dispose();

      const lower = c.array(
        [
          [z, new Complex(0, 0)],
          [z, z],
        ],
        { dtype: "complex128" },
      );
      const lowerRhs = c.array([z, new Complex(2 * scale, 2 * scale)], {
        dtype: "complex128",
      });
      const lowerSolution = await linalg.solve(lower, lowerRhs).compute();
      for (const value of lowerSolution.toArray()) {
        assert.ok(Math.abs(value.real - 1) < 1e-12);
        assert.ok(Math.abs(value.imag) < 1e-12);
      }
      lowerSolution.dispose();
    }
  } finally {
    await c.close();
  }
});
test("FFT axes, normalization, odd real reconstruction and typed transport", async () => {
  const c = context();
  try {
    const a = c.array([1, 2, 3], { dtype: "float32" });
    for (const norm of ["backward", "forward", "ortho"]) {
      const result = await fft
        .irfft(fft.rfft(a, { norm }), { n: 3, norm })
        .compute();
      assert.equal(result.dtype, "float32");
      assert.deepEqual(result.toArray(), [1, 2, 3]);
      result.dispose();
    }
    const m = c.array(
      [
        [1, 2],
        [3, 4],
      ],
      { dtype: "float64" },
    );
    const out = await fft.fft(m, { axis: 0 }).compute();
    assert.deepEqual([...out.toTypedArray()], [4, 0, 6, 0, -2, 0, -2, 0]);
    out.dispose();
    const padded = await fft.fft(a, { n: 4 }).compute();
    assert.deepEqual(padded.shape, [4]);
    padded.dispose();
    assert.throws(() => fft.fft(a, { out: a }), /Unsupported FFT option/);
    assert.throws(() => fft.fft(a, { n: 0 }), RangeError);
    assert.throws(() => fft.fft(a, { axis: 3 }), {
      code: "NMATH_SHAPE_MISMATCH",
    });
  } finally {
    await c.close();
  }
});
test("failures preserve context accounting and reject GPU before admission", async () => {
  const c = context();
  try {
    const singular = c.array(
      [
        [1, 2],
        [2, 4],
      ],
      { dtype: "float64" },
    );
    await assert.rejects(linalg.inv(singular).compute(), {
      code: "NMATH_LINALG_SINGULAR",
    });
    assert.equal(await linalg.det(singular).compute(), 0);
    const a = c.array([1, 2, 3], { dtype: "float32" });
    await assert.rejects(fft.fft(a).compute({ device: "gpu" }), {
      code: "NMATH_GPU_UNSUPPORTED_OPERATION",
    });
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(fft.fft(a).compute({ signal: controller.signal }), {
      code: "NMATH_CANCELLED",
    });
    const foreign = context();
    try {
      const rhs = foreign.array([1, 2], { dtype: "float64" });
      assert.throws(() => linalg.solve(singular, rhs), {
        code: "NMATH_CONTEXT_MISMATCH",
      });
    } finally {
      await foreign.close();
    }
  } finally {
    await c.close();
  }
  const tiny = context(128n);
  try {
    const a = tiny.array(
      [
        [1, 0],
        [0, 1],
      ],
      { dtype: "float64" },
    );
    await assert.rejects(linalg.inv(a).compute(), {
      code: "NMATH_MEMORY_BUDGET_EXCEEDED",
    });
  } finally {
    await tiny.close();
  }
});
