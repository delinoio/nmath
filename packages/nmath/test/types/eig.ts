// SPDX-License-Identifier: Apache-2.0
import { createContext, type Expr } from "../../src/index.js";
import { linalg } from "../../src/numpy.js";

const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});

const single = c.zeros([2, 3, 3], "float32");
const ints = c.zeros([4, 4], "int32");
const result = linalg.eig(single);
const w: Expr<"complex64", [2, 3]> = result.eigenvalues;
const v: Expr<"complex64", readonly [2, 3, 3]> = result[1];
const wide: Expr<"complex128", [4]> = linalg.eigvals(ints);
// @ts-expect-error eig requires square matrices
linalg.eig(c.zeros([2, 3], "float64"));
void [w, v, wide];
