// SPDX-License-Identifier: Apache-2.0
import { createContext, type Expr } from "../../src/index.js";
import { fft } from "../../src/numpy.js";

const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});
const real = c.zeros([4, 6], "float32");
const spectrum = c.zeros([4, 4], "complex128");
const r2: Expr<"complex64", readonly [number, number]> = fft.rfft2(real);
const i2: Expr<"float64", readonly [number, number]> = fft.irfft2(spectrum);
const rn: Expr<"complex64", readonly [number, number]> = fft.rfftn(real);
const h: Expr<"float64", readonly [number, number]> = fft.hfft(spectrum);
const ih: Expr<"complex64", readonly [number, number]> = fft.ihfft(real);
// @ts-expect-error rfftn rejects complex input
fft.rfftn(spectrum);
void [r2, i2, rn, h, ih];
