// SPDX-License-Identifier: Apache-2.0
import { createContext, type Expr } from "../../src/index.js";
import { linalg, fft } from "../../src/numpy.js";
const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 1,
  cpuMemoryBudget: 1024n,
  gpuMemoryBudget: 0n,
});
const a = c.zeros([2, 3, 3] as const, "float32");
const inverse: Expr<"float32", readonly [2, 3, 3]> = linalg.inv(a);
const determinant: Expr<"float32", [2]> = linalg.det(a);
const solution: Expr<"float32", [2, 3]> = linalg.solve(
  a,
  c.zeros([3] as const, "float32"),
);
const openShape: readonly [2, ...number[]] = [2, 2];
const openRhsShape: readonly [2, ...number[]] = [2, 1];
const openMatrix = c.array(new Float32Array([1, 0, 0, 1]), {
  dtype: "float32",
  shape: openShape,
});
const openRhs = c.array(new Float32Array([1, 2]), {
  dtype: "float32",
  shape: openRhsShape,
});
const fixedMatrix = c.zeros([2, 2] as const, "float32");
const openInverse: Expr<"float32", readonly [2, ...number[]]> =
  linalg.inv(openMatrix);
const openDeterminant: Expr<"float32", number[]> = linalg.det(openMatrix);
const openSolution: Expr<"float32", number[]> = linalg.solve(
  openMatrix,
  fixedMatrix,
);
const openRhsSolution: Expr<"float32", number[]> = linalg.solve(
  fixedMatrix,
  openRhs,
);
const mutableShape: [2, ...number[]] = [2, 2];
const mutableMatrix = c.zeros(mutableShape, "float32");
const mutableInverse: Expr<"float32", [2, ...number[]]> =
  linalg.inv(mutableMatrix);
const mutableDeterminant: Expr<"float32", number[]> = linalg.det(mutableMatrix);
const mutableSolution: Expr<"float32", number[]> = linalg.solve(
  mutableMatrix,
  fixedMatrix,
);
declare const suffixShape: readonly [...number[], 2, 2];
declare const mixedShape: readonly [2, ...number[], 2, 2];
const suffixMatrix = c.zeros(suffixShape, "float32");
const mixedMatrix = c.zeros(mixedShape, "float32");
const suffixInverse: Expr<"float32", typeof suffixShape> =
  linalg.inv(suffixMatrix);
const mixedDeterminant: Expr<"float32", number[]> = linalg.det(mixedMatrix);
const suffixSolution: Expr<"float32", number[]> = linalg.solve(
  suffixMatrix,
  fixedMatrix,
);
const dynamicShape: number[] = [2, 2];
const dynamicMatrix = c.zeros(dynamicShape, "float32");
const dynamicInverse: Expr<"float32", number[]> = linalg.inv(dynamicMatrix);
const dynamicSolution: Expr<"float32", number[]> = linalg.solve(
  dynamicMatrix,
  fixedMatrix,
);
declare const nonsquareSuffixShape: readonly [...number[], 2, 3];
declare const nonsquareMixedShape: readonly [2, ...number[], 2, 3];
declare const incompatibleRhsShape: readonly [...number[], 3, 1];
// @ts-expect-error known nonsquare trailing dimensions remain invalid
linalg.inv(c.zeros(nonsquareSuffixShape, "float32"));
// @ts-expect-error mixed open shapes retain known nonsquare trailing dimensions
linalg.det(c.zeros(nonsquareMixedShape, "float32"));
// @ts-expect-error a fixed rank-one input cannot be a matrix
linalg.inv(c.zeros([2] as const, "float32"));
// @ts-expect-error known solve core dimensions remain incompatible with open batches
linalg.solve(suffixMatrix, c.zeros([3] as const, "float32"));
// @ts-expect-error known RHS core dimensions remain incompatible with an open batch
linalg.solve(fixedMatrix, c.zeros(incompatibleRhsShape, "float32"));
const transform: Expr<"complex64", readonly [2, 3, 3]> = fft.fft(a);
// @ts-expect-error known nonsquare input
linalg.inv(c.zeros([2, 3] as const, "float64"));
// @ts-expect-error solve core dimensions differ
linalg.solve(a, c.zeros([4] as const, "float32"));
// @ts-expect-error float16 linalg is unsupported
linalg.inv(c.zeros([2, 2] as const, "float16"));
// @ts-expect-error rfft requires real input
fft.rfft(c.zeros([3] as const, "complex64"));
void [
  inverse,
  determinant,
  solution,
  openInverse,
  openDeterminant,
  openSolution,
  openRhsSolution,
  mutableInverse,
  mutableDeterminant,
  mutableSolution,
  suffixInverse,
  mixedDeterminant,
  suffixSolution,
  dynamicInverse,
  dynamicSolution,
  transform,
];
const unchanged: Expr<"complex64", readonly [2, 3, 3]> = fft.fft(a);
const resized: Expr<"complex64", [2, 3, 8]> = fft.fft(a, { n: 8 });
const resizedFirst: Expr<"complex64", [8, 3, 3]> = fft.fft(a, {
  n: 8,
  axis: 0,
});
const realLast: Expr<"complex64", [2, 3, number]> = fft.rfft(a);
void [unchanged, resized, resizedFirst, realLast];
declare const dynamicOptions: import("../../src/fft.js").FftOptions;
const dynamicTransform: Expr<"complex64", readonly [number, number, number]> =
  fft.fft(a, dynamicOptions);
void dynamicTransform;

const optionalAxis: { n: 8; axis?: number } = { n: 8, axis: 0 };
const optionalLiteralAxis: { n: 8; axis?: 0 } = { n: 8 };
for (const options of [optionalAxis, optionalLiteralAxis]) {
  const forward = fft.fft(a, options);
  const inverse = fft.ifft(a, options);
  const real = fft.rfft(a, options);
  const inverseReal = fft.irfft(a, options);
  const dimensions: readonly [number, number, number] = forward.shape;
  // @ts-expect-error an optional axis may change a non-last dimension
  const wrongForward: readonly [2, 3, 8] = forward.shape;
  // @ts-expect-error inverse transforms have the same axis uncertainty
  const wrongInverse: readonly [2, 3, 8] = inverse.shape;
  // @ts-expect-error real transforms cannot retain the leading dimensions
  const wrongReal: readonly [2, 3, number] = real.shape;
  // @ts-expect-error inverse real transforms cannot retain the leading dimensions
  const wrongInverseReal: readonly [2, 3, number] = inverseReal.shape;
  void [dimensions, wrongForward, wrongInverse, wrongReal, wrongInverseReal];
}
