// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import { context } from "./helpers.mjs";

function float32Bits(values) {
  const view = new DataView(
    values.buffer,
    values.byteOffset,
    values.byteLength,
  );
  return Array.from({ length: values.length }, (_, i) =>
    view.getUint32(i * Float32Array.BYTES_PER_ELEMENT, true),
  );
}

test("wide integer array casts and assignments round directly to float32 components", async () => {
  const ctx = context();
  const midpoint = (1n << 62n) + (1n << 38n);
  const signed = ctx.array(
    new BigInt64Array([
      midpoint + 1n,
      -(midpoint + 1n),
      midpoint,
      midpoint - 1n,
      0n,
    ]),
    { dtype: "int64" },
  );
  const unsigned = ctx.array(
    new BigUint64Array([midpoint + 1n, (1n << 63n) + (1n << 39n) + 1n]),
    { dtype: "uint64" },
  );
  try {
    for (const [source, expected] of [
      [signed, [0x5e800001, 0xde800001, 0x5e800000, 0x5e800000, 0]],
      [unsigned, [0x5e800001, 0x5f000001]],
    ]) {
      const real = await source.astype("float32").compute();
      assert.deepEqual(float32Bits(real.toTypedArray()), expected);
      const complex = await source.astype("complex64").compute();
      const components = float32Bits(complex.toTypedArray());
      assert.deepEqual(
        components,
        expected.flatMap((bits) => [bits, 0]),
      );
      const realTarget = ctx.zeros([expected.length], "float32");
      await realTarget.set([], source);
      assert.deepEqual(float32Bits(realTarget.toTypedArray()), expected);
      const complexTarget = ctx.zeros([expected.length], "complex64");
      await complexTarget.set([], source);
      assert.deepEqual(
        float32Bits(complexTarget.toTypedArray()),
        expected.flatMap((bits) => [bits, 0]),
      );
      real.dispose();
      complex.dispose();
      realTarget.dispose();
      complexTarget.dispose();
    }
    const scalarTarget = ctx.zeros([1], "float32");
    await scalarTarget.set([0], midpoint + 1n);
    assert.deepEqual(float32Bits(scalarTarget.toTypedArray()), [0x5e800000]);
    const complexScalarTarget = ctx.zeros([1], "complex64");
    await complexScalarTarget.set([0], midpoint + 1n);
    assert.deepEqual(
      float32Bits(complexScalarTarget.toTypedArray()),
      [0x5e800000, 0],
    );
    const weak = await ctx
      .zeros([1], "float32")
      .add(midpoint + 1n)
      .compute();
    assert.deepEqual(float32Bits(weak.toTypedArray()), [0x5e800000]);
    const double = await signed.astype("float64").compute();
    assert.equal(double.item(0), Number(midpoint + 1n));
    const doubleComplex = await signed.astype("complex128").compute();
    assert.equal(doubleComplex.item(0).real, Number(midpoint + 1n));
    const small = ctx.array(new Int32Array([16_777_217]), { dtype: "int32" });
    const smallCast = await small.astype("float32").compute();
    assert.deepEqual(float32Bits(smallCast.toTypedArray()), [0x4b800000]);
    scalarTarget.dispose();
    complexScalarTarget.dispose();
    weak.dispose();
    double.dispose();
    doubleComplex.dispose();
    small.dispose();
    smallCast.dispose();
  } finally {
    await ctx.close();
  }
});
