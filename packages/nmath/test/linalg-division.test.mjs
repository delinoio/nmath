// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { createContext, Complex } from "../dist/index.js";
import { linalg } from "../dist/numpy.js";

const context = () =>
  createContext({
    maxConcurrentJobs: 1,
    maxQueueLength: 4,
    cpuMemoryBudget: 1n << 24n,
    gpuMemoryBudget: 0n,
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
