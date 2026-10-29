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
const transform: Expr<"complex64", readonly [2, 3, 3]> = fft.fft(a);
// @ts-expect-error known nonsquare input
linalg.inv(c.zeros([2, 3] as const, "float64"));
// @ts-expect-error solve core dimensions differ
linalg.solve(a, c.zeros([4] as const, "float32"));
// @ts-expect-error float16 linalg is unsupported
linalg.inv(c.zeros([2, 2] as const, "float16"));
// @ts-expect-error rfft requires real input
fft.rfft(c.zeros([3] as const, "complex64"));
void [inverse, determinant, solution, transform];
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
