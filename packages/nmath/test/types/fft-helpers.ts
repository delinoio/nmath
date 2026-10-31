// SPDX-License-Identifier: Apache-2.0
import { createContext, type Expr, type NMathArray } from "../../src/index.js";
import { fft } from "../../src/numpy.js";

const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});

const matrix = c.zeros([2, 3], "float64");
const two: Expr<"complex128", [2, 3]> = fft.fft2(matrix);
const inverseTwo: Expr<"complex128", [2, 3]> = fft.ifft2(matrix);
const full: Expr<"complex128", [2, 3]> = fft.fftn(matrix);
const inverseFull: Expr<"complex128", [2, 3]> = fft.ifftn(matrix);
const resized: Expr<"complex128", readonly [number, number]> = fft.fft2(
  matrix,
  {
    s: [4, 2],
  },
);
const subset: Expr<"complex128", [2, 3]> = fft.fftn(matrix, {
  axes: [0, 1],
  norm: "ortho",
});

const single = c.zeros([4], "float32");
const complexSingle: Expr<"complex64", readonly [4]> = fft.fft(single);

const shifted: Expr<"float64", [2, 3]> = fft.fftshift(matrix);
const inverseShifted: Expr<"float64", [2, 3]> = fft.ifftshift(matrix, [-1]);
const strings = c.array(["a", "b", "c"], { dtype: "str" });
const shiftedStrings: Expr<"str", number[]> = fft.fftshift(strings);

const frequencies: NMathArray<"float64", [number]> = fft.fftfreq(c, 4);
const realFrequencies: NMathArray<"float64", [number]> = fft.rfftfreq(
  c,
  4,
  0.5,
);

// The runtime rejects fft2 on a rank-one array; the static signature keeps it
// valid because the rank check depends on the runtime shape.
void fft.fft2(single);
// @ts-expect-error resized transforms use dynamic dimensions
const wrongShape: Expr<"complex128", [2, 3]> = fft.fft2(matrix, { s: [4, 2] });

void two;
void inverseTwo;
void full;
void inverseFull;
void resized;
void subset;
void complexSingle;
void shifted;
void inverseShifted;
void shiftedStrings;
void frequencies;
void realFrequencies;
void wrongShape;
