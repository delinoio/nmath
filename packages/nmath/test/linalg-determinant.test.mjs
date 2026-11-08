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

test("determinants retain compensated magnitudes and phases", async () => {
  const c = context();
  try {
    for (const diagonal of [
      [1e100, 1e100, 1e100, 1e100, 1e-100, 1e-100, 1e-100, 1e-100],
      [1e-100, 1e-100, 1e-100, 1e-100, 1e100, 1e100, 1e100, 1e100],
    ]) {
      const values = Array(64).fill(0);
      diagonal.forEach((value, i) => {
        values[i * 8 + i] = value;
      });
      const a = c.array(values, { dtype: "float64", shape: [8, 8] });
      const result = await linalg.det(a).compute();
      assert.ok(Number.isFinite(result) && result !== 0);
      assert.ok(Math.abs(result - 1) < 1e-12, `${result}`);
    }

    const swapped = c.array(
      [
        [0, 1],
        [1, 0],
      ],
      { dtype: "float64" },
    );
    assert.equal(await linalg.det(swapped).compute(), -1);
    const complex = c.array(
      [
        [new Complex(1e100, 1e100), new Complex(0, 0)],
        [new Complex(0, 0), new Complex(1e-100, 0)],
      ],
      { dtype: "complex128" },
    );
    const phase = await linalg.det(complex).compute();
    assert.ok(Math.abs(phase.real - 1) < 1e-12);
    assert.ok(Math.abs(phase.imag - 1) < 1e-12);

    const batch = c.array(
      [
        [
          [2, 0],
          [0, 3],
        ],
        [
          [0, 1],
          [1, 0],
        ],
      ],
      { dtype: "float32" },
    );
    const expression = linalg.det(batch);
    assert.equal(expression.dtype, "float32");
    assert.deepEqual(expression.shape, [2]);
    const results = await expression.compute();
    assert.deepEqual(results.toArray(), [6, -1]);
    results.dispose();
  } finally {
    await c.close();
  }
});
