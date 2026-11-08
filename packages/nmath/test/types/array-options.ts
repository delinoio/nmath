// SPDX-License-Identifier: Apache-2.0
import {
  createContext,
  NMathArray,
  type ArrayOptions,
} from "../../src/index.js";
import { array as numpyArray } from "../../src/numpy.js";

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;
declare function exact<T extends true>(): void;

const ctx = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 1,
  cpuMemoryBudget: 1024,
  gpuMemoryBudget: 0,
});
const data = new Float32Array([1, 2]);
const bits = new Uint16Array([0x3c00, 0x4000]);

const optional: ArrayOptions<"float32", readonly [1, 2]> = {
  dtype: "float32",
};
const optionalCore = ctx.array(data, optional);
const optionalNumpy = numpyArray(ctx, data, optional);
const optionalBits: { shape?: readonly [1, 2] } = {};
const optionalFloat16 = ctx.arrayFromFloat16Bits(bits, optionalBits);
exact<Equal<typeof optionalCore, NMathArray<"float32", number[]>>>();
exact<Equal<typeof optionalNumpy, NMathArray<"float32", number[]>>>();
exact<Equal<typeof optionalFloat16, NMathArray<"float16", number[]>>>();
// @ts-expect-error an optional shape can fall back to [2]
const wrongCore: readonly [1, 2] = optionalCore.shape;
// @ts-expect-error an optional shape can fall back to [2]
const wrongNumpy: readonly [1, 2] = optionalNumpy.shape;
// @ts-expect-error an optional shape can fall back to [2]
const wrongFloat16: readonly [1, 2] = optionalFloat16.shape;

const optionalPresent: ArrayOptions<"float32", readonly [1, 2]> = {
  dtype: "float32",
  shape: [1, 2],
};
const optionalPresentCore = ctx.array(data, optionalPresent);
const optionalPresentNumpy = numpyArray(ctx, data, optionalPresent);
exact<Equal<typeof optionalPresentCore.shape, number[]>>();
exact<Equal<typeof optionalPresentNumpy.shape, number[]>>();
const possiblyUndefined = {
  dtype: "float32" as const,
  shape: undefined as readonly [1, 2] | undefined,
};
const undefinedShape = ctx.array(data, possiblyUndefined);
exact<Equal<typeof undefinedShape.shape, number[]>>();

const optionalScalar: ArrayOptions<"float32", readonly []> = {
  dtype: "float32",
};
const vector = ctx.array(data, optionalScalar);
const oneElementVector = ctx.array(new Float32Array([1]), optionalScalar);
const optionalScalarBits: { shape?: readonly [] } = {};
const bitsVector = ctx.arrayFromFloat16Bits(bits, optionalScalarBits);
const oneBitVector = ctx.arrayFromFloat16Bits(
  new Uint16Array([0x3c00]),
  optionalScalarBits,
);
exact<Equal<typeof vector.shape, number[]>>();
exact<Equal<typeof oneElementVector.shape, number[]>>();
exact<Equal<typeof bitsVector.shape, number[]>>();
exact<Equal<typeof oneBitVector.shape, number[]>>();
async function scalarResultNeedsNarrowing() {
  // @ts-expect-error an absent scalar shape creates a vector
  const claimedNumber: number = await vector.compute();
  // @ts-expect-error an absent scalar shape creates a vector
  const claimedBitNumber: number = await bitsVector.compute();
  void claimedNumber;
  void claimedBitNumber;
}
void scalarResultNeedsNarrowing;

const required = ctx.array(data, { dtype: "float32", shape: [1, 2] });
const requiredNumpy = numpyArray(ctx, data, {
  dtype: "float32",
  shape: [1, 2],
});
const requiredBits = ctx.arrayFromFloat16Bits(bits, { shape: [1, 2] });
exact<Equal<typeof required.shape, readonly [1, 2]>>();
exact<Equal<typeof requiredNumpy.shape, readonly [1, 2]>>();
exact<Equal<typeof requiredBits.shape, readonly [1, 2]>>();

const rankZero = ctx.array(new Float32Array([1]), {
  dtype: "float32",
  shape: [],
});
const rankZeroNumpy = numpyArray(ctx, new Float32Array([1]), {
  dtype: "float32",
  shape: [],
});
const rankZeroBits = ctx.arrayFromFloat16Bits(new Uint16Array([0x3c00]), {
  shape: [],
});
exact<Equal<typeof rankZero.shape, readonly []>>();
exact<Equal<typeof rankZeroNumpy.shape, readonly []>>();
exact<Equal<typeof rankZeroBits.shape, readonly []>>();
exact<Equal<ReturnType<typeof rankZero.compute>, Promise<number>>>();

const empty = ctx.array(new Float32Array(), { dtype: "float32", shape: [0] });
const emptyNumpy = numpyArray(ctx, new Float32Array(), {
  dtype: "float32",
  shape: [0],
});
const emptyBits = ctx.arrayFromFloat16Bits(new Uint16Array(), { shape: [0] });
exact<Equal<typeof empty.shape, readonly [0]>>();
exact<Equal<typeof emptyNumpy.shape, readonly [0]>>();
exact<Equal<typeof emptyBits.shape, readonly [0]>>();

const omitted = ctx.array(data, { dtype: "float32" });
const omittedNumpy = numpyArray(ctx, data, { dtype: "float32" });
const omittedBits = ctx.arrayFromFloat16Bits(bits);
const nested = ctx.array([[1, 2]], { dtype: "float32" });
exact<Equal<typeof omitted.shape, number[]>>();
exact<Equal<typeof omittedNumpy.shape, number[]>>();
exact<Equal<typeof omittedBits.shape, number[]>>();
exact<Equal<typeof nested.shape, number[]>>();
