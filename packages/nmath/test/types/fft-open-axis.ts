// SPDX-License-Identifier: Apache-2.0
import { createContext, type Expr } from "../../src/index.js";
import { fft, ifft, rfft, irfft } from "../../src/fft.js";

const context = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 1,
  cpuMemoryBudget: 1024n,
  gpuMemoryBudget: 0n,
});

const suffixShape: readonly [...number[], 2] = [2];
const input = context.zeros(suffixShape, "float32");
const forward = fft(input, { n: 3, axis: 0 });
const inverse = ifft(input, { n: 3, axis: 0 });
const real = rfft(input, { n: 6, axis: 0 });
const inverseReal = irfft(input, { n: 3, axis: 0 });

const forwardShape: typeof forward.shape = [3];
const inverseShape: typeof inverse.shape = [3];
const realShape: typeof real.shape = [4];
const inverseRealShape: typeof inverseReal.shape = [3];
// @ts-expect-error The transformed axis may be the required suffix.
const wrongSuffix: readonly [...number[], 2] = forward.shape;
const forwardDtype: Expr<"complex64", readonly [...number[], number]> = forward;
const inverseDtype: Expr<"complex64", readonly [...number[], number]> = inverse;
const realDtype: Expr<"complex64", readonly [...number[], number]> = real;
const inverseRealDtype: Expr<"float32", readonly [...number[], number]> =
  inverseReal;

const mixedShape: readonly [5, ...number[], 2] = [5, 2];
const mixedInput = context.zeros(mixedShape, "float32");
const mixed = fft(mixedInput, { n: 3, axis: 1 });
const mixedResult: typeof mixed.shape = [5, 3];
const knownPrefix: 5 = mixed.shape[0];
// @ts-expect-error The unresolved tail cannot guarantee its final literal.
const wrongMixedSuffix: readonly [5, ...number[], 2] = mixed.shape;

const fixed = context.zeros([2, 4] as const, "float32");
const fixedAxis: Expr<"complex64", [3, 4]> = fft(fixed, { n: 3, axis: 0 });
const lastAxis: Expr<"complex64", [...number[], 3]> = fft(input, {
  n: 3,
  axis: -1,
});
const unchanged: Expr<"complex64", typeof suffixShape> = fft(input);
declare const dynamicAxis: number;
const dynamic: Expr<"complex64", readonly [...number[], number]> = fft(input, {
  n: 3,
  axis: dynamicAxis,
});

void [
  forwardShape,
  inverseShape,
  realShape,
  inverseRealShape,
  wrongSuffix,
  forwardDtype,
  inverseDtype,
  realDtype,
  inverseRealDtype,
  mixedResult,
  knownPrefix,
  wrongMixedSuffix,
  fixedAxis,
  lastAxis,
  unchanged,
  dynamic,
];
