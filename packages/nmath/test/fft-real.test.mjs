// SPDX-License-Identifier: Apache-2.0
// Relations from NumPy 2.5.4 numpy/fft/tests/test_pocketfft.py, atol 1e-6.
import assert from "node:assert/strict";
import test from "node:test";
import { Complex, slice } from "../dist/index.js";
import { fft } from "../dist/numpy.js";
import { context } from "./helpers.mjs";

let seed = 20261106;
function random() {
  seed = (seed * 1103515245 + 12345) % 2 ** 31;
  return seed / 2 ** 31;
}
const values = (n) => Array.from({ length: n }, random);

function flat(result) {
  return [result.toArray()]
    .flat(Infinity)
    .map((v) => (v instanceof Complex ? [v.real, v.imag] : [v, 0]));
}
async function close(a, b, label, scale = 1) {
  const x = flat(await a.compute());
  const y = flat(await b.compute());
  assert.equal(x.length, y.length, label);
  x.forEach(([re, im], i) => {
    const d = Math.hypot(re - y[i][0] * scale, im - y[i][1] * scale);
    assert.ok(d <= 1e-6, `${label} element ${i}: ${d}`);
  });
}

test("rfft2 and rfftn match the complex transforms and invert", async () => {
  const ctx = context();
  try {
    const x2 = ctx.array(values(30 * 20), {
      dtype: "float64",
      shape: [30, 20],
    });
    const r2 = fft.rfft2(x2);
    assert.deepEqual(r2.shape, [30, 11]);
    assert.equal(r2.dtype, "complex128");
    await close(fft.fft2(x2).get(slice(), slice(0, 11)), r2, "rfft2");
    await close(
      fft.rfft2(x2, { norm: "ortho" }),
      r2,
      "rfft2 ortho",
      1 / Math.sqrt(600),
    );
    await close(
      fft.rfft2(x2, { norm: "forward" }),
      r2,
      "rfft2 forward",
      1 / 600,
    );
    for (const norm of ["backward", "ortho", "forward"]) {
      await close(
        fft.irfft2(fft.rfft2(x2, { norm }), { norm }),
        x2,
        `irfft2 ${norm}`,
      );
    }
    const x3 = ctx.array(values(30 * 20 * 10), {
      dtype: "float64",
      shape: [30, 20, 10],
    });
    const r3 = fft.rfftn(x3);
    assert.deepEqual(r3.shape, [30, 20, 6]);
    await close(fft.fftn(x3).get(slice(), slice(), slice(0, 6)), r3, "rfftn");
    await close(
      fft.rfftn(x3, { norm: "forward" }),
      r3,
      "rfftn forward",
      1 / 6000,
    );
    for (const norm of ["backward", "ortho", "forward"]) {
      await close(
        fft.irfftn(fft.rfftn(x3, { norm }), { norm }),
        x3,
        `irfftn ${norm}`,
      );
    }
    assert.deepEqual(fft.rfftn(x3, { s: [10, 21] }).shape, [30, 10, 11]);
    assert.deepEqual(fft.irfftn(r3, { axes: [0, 2] }).shape, [30, 20, 10]);
    assert.equal(
      fft.irfft2(fft.rfft2(ctx.zeros([4, 4], "float32"))).dtype,
      "float32",
    );
  } finally {
    await ctx.close();
  }
});

test("hfft and ihfft follow NumPy's Hermitian relations", async () => {
  const ctx = context();
  try {
    const re = values(14);
    const im = values(14);
    const head = random();
    const tail = random();
    const herm = [
      new Complex(head, 0),
      ...re.map((r, i) => new Complex(r, im[i])),
      new Complex(tail, 0),
    ];
    const full = [
      ...herm,
      ...re.map((r, i) => new Complex(r, -im[i])).reverse(),
    ];
    const xHerm = ctx.array(herm, { dtype: "complex128" });
    const x = ctx.array(full, { dtype: "complex128" });
    const h = fft.hfft(xHerm);
    assert.deepEqual(h.shape, [30]);
    assert.equal(h.dtype, "float64");
    await close(fft.fft(x), h, "hfft vs fft");
    await close(
      fft.hfft(xHerm, { norm: "ortho" }),
      h,
      "hfft ortho",
      1 / Math.sqrt(30),
    );
    await close(
      fft.hfft(xHerm, { norm: "forward" }),
      h,
      "hfft forward",
      1 / 30,
    );
    for (const norm of ["backward", "ortho", "forward"]) {
      await close(
        fft.ihfft(fft.hfft(xHerm, { norm }), { norm }),
        xHerm,
        `ihfft ${norm}`,
      );
    }
    assert.equal(fft.ihfft(ctx.zeros([8], "float32")).dtype, "complex64");
    assert.throws(() => fft.hfft(xHerm, { norm: "nope" }), TypeError);
    assert.throws(
      () => fft.rfftn(ctx.zeros([2, 2], "float64"), { axes: [0, 0] }),
      RangeError,
    );
    await assert.rejects(
      fft.rfft2(ctx.zeros([2, 2], "float64")).compute({ device: "gpu" }),
      {
        code: "NMATH_GPU_UNSUPPORTED_OPERATION",
      },
    );
  } finally {
    await ctx.close();
  }
});
