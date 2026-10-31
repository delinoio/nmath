// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { createContext } from "../dist/index.js";
import { fft } from "../dist/numpy.js";

const context = () =>
  createContext({
    maxConcurrentJobs: 1,
    maxQueueLength: 4,
    cpuMemoryBudget: 1n << 24n,
    gpuMemoryBudget: 0n,
  });

function close(actual, expected, atol = 1e-9) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < expected.length; i++) {
    const a = actual[i];
    const b = expected[i];
    if (typeof b === "number") {
      assert.ok(Math.abs(a - b) <= atol, `element ${i}: ${a} != ${b}`);
    } else {
      assert.ok(
        Math.abs(a.real - b[0]) <= atol && Math.abs(a.imag - b[1]) <= atol,
        `element ${i}: ${a.real}+${a.imag}j != ${b[0]}+${b[1]}j`,
      );
    }
  }
}

test("multi-axis transforms match the fixed NumPy results", async () => {
  const ctx = context();
  try {
    const m = ctx.array(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { dtype: "float64" },
    );
    const transformed = fft.fft2(m);
    assert.equal(transformed.dtype, "complex128");
    assert.deepEqual(transformed.shape, [2, 3]);
    const result = await transformed.compute();
    close(
      [...result.toTypedArray()],
      [
        21, 0, -3, 1.7320508075688772, -3, -1.7320508075688772, -9, 0, 0, 0, 0,
        0,
      ],
    );
    result.dispose();

    const restored = await fft.ifft2(fft.fft2(m)).compute();
    close([...restored.toTypedArray()], [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]);
    restored.dispose();

    const padded = await fft.fft2(m, { s: [4, 2] }).compute();
    assert.deepEqual(padded.shape, [4, 2]);
    close(
      [...padded.toTypedArray()],
      [12, 0, -2, 0, 3, -9, -1, 1, -6, 0, 0, 0, 3, 9, -1, -1],
    );
    padded.dispose();

    const cube = ctx.array([0, 1, 2, 3, 4, 5, 6, 7], {
      dtype: "float64",
      shape: [2, 2, 2],
    });
    const full = await fft.fftn(cube).compute();
    close(
      [...full.toTypedArray()],
      [28, 0, -4, 0, -8, 0, 0, 0, -16, 0, 0, 0, 0, 0, 0, 0],
    );
    full.dispose();

    const subset = await fft
      .fftn(cube, { axes: [0, 2], norm: "ortho" })
      .compute();
    close(
      [...subset.toTypedArray()],
      [5, 0, -1, 0, 9, 0, -1, 0, -4, 0, 0, 0, -4, 0, 0, 0],
    );
    subset.dispose();

    const inverse = await fft.ifftn(fft.fftn(cube)).compute();
    close(
      [...inverse.toTypedArray()],
      [0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0],
    );
    inverse.dispose();
  } finally {
    await ctx.close();
  }
});

test("multi-axis options and ranks are validated", async () => {
  const ctx = context();
  try {
    const vector = ctx.zeros([4], "float64");
    assert.throws(() => fft.fft2(vector), RangeError);
    const m = ctx.zeros([2, 3], "float64");
    assert.throws(() => fft.fftn(m, { axes: [0, 0] }), RangeError);
    assert.throws(() => fft.fftn(m, { axes: [2] }), RangeError);
    assert.throws(() => fft.fft2(m, { s: [2] }), RangeError);
    assert.throws(() => fft.fft2(m, { s: [0, 2] }), RangeError);
    assert.throws(() => fft.fftn(m, { out: m }), /Unsupported FFT option/);
  } finally {
    await ctx.close();
  }
});

test("fftshift and ifftshift match NumPy axis rolls", async () => {
  const ctx = context();
  try {
    const even = ctx.array([0, 1, 2, 3], { dtype: "int32" });
    assert.deepEqual(
      (await fft.fftshift(even).compute()).toArray(),
      [2, 3, 0, 1],
    );
    assert.deepEqual(
      (await fft.ifftshift(even).compute()).toArray(),
      [2, 3, 0, 1],
    );
    const odd = ctx.array([0, 1, 2, 3, 4], { dtype: "int32" });
    assert.deepEqual(
      (await fft.fftshift(odd).compute()).toArray(),
      [3, 4, 0, 1, 2],
    );
    assert.deepEqual(
      (await fft.ifftshift(odd).compute()).toArray(),
      [2, 3, 4, 0, 1],
    );

    const matrix = ctx.array([0, 1, 2, 3, 4, 5], {
      dtype: "int32",
      shape: [2, 3],
    });
    assert.deepEqual((await fft.fftshift(matrix).compute()).toArray(), [
      [5, 3, 4],
      [2, 0, 1],
    ]);
    assert.deepEqual((await fft.fftshift(matrix, [0]).compute()).toArray(), [
      [3, 4, 5],
      [0, 1, 2],
    ]);

    const strings = ctx.array(["a", "b", "c"], { dtype: "str" });
    assert.deepEqual((await fft.ifftshift(strings).compute()).toArray(), [
      "b",
      "c",
      "a",
    ]);

    assert.throws(() => fft.fftshift(matrix, [0, 0]), RangeError);
    assert.throws(() => fft.fftshift(matrix, [2]), {
      code: "NMATH_INDEX_OUT_OF_BOUNDS",
    });
    await assert.rejects(fft.fftshift(matrix).compute({ device: "gpu" }), {
      code: "NMATH_GPU_UNSUPPORTED_OPERATION",
    });
  } finally {
    await ctx.close();
  }
});

test("frequency helpers match NumPy values", async () => {
  const ctx = context();
  try {
    const four = fft.fftfreq(ctx, 4);
    assert.equal(four.dtype, "float64");
    close(four.toArray(), [0, 0.25, -0.5, -0.25]);
    close(fft.fftfreq(ctx, 5).toArray(), [0, 0.2, 0.4, -0.4, -0.2]);
    close(fft.fftfreq(ctx, 4, 2).toArray(), [0, 0.125, -0.25, -0.125]);
    close(fft.rfftfreq(ctx, 4).toArray(), [0, 0.25, 0.5]);
    close(fft.rfftfreq(ctx, 5).toArray(), [0, 0.2, 0.4]);
    close(fft.rfftfreq(ctx, 5, 4).toArray(), [0, 0.05, 0.1]);
    assert.throws(() => fft.fftfreq(ctx, 0), RangeError);
    assert.throws(() => fft.rfftfreq(ctx, -1), RangeError);
    assert.throws(() => fft.fftfreq(ctx, 4, Number.NaN), RangeError);
  } finally {
    await ctx.close();
  }
});
