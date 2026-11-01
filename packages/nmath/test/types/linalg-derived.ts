// SPDX-License-Identifier: Apache-2.0
import { createContext, type Expr } from "../../src/index.js";
import { linalg } from "../../src/numpy.js";

const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});

const batch = c.zeros([2, 4, 3], "complex64");
const ints = c.zeros([3, 3], "int32");
const n: Expr<"float32", number[]> = linalg.norm(batch, "fro", [1, 2]);
const rank: Expr<"int64", [2]> = linalg.matrix_rank(batch);
const inverse: Expr<"complex64", [2, 3, 4]> = linalg.pinv(batch);
const condition: Expr<"float32", [2]> = linalg.cond(batch);
const power: Expr<"int32", readonly [3, 3]> = linalg.matrix_power(ints, 3);
const negative: Expr<"float64", readonly [3, 3]> = linalg.matrix_power(
  ints,
  -2,
);

// @ts-expect-error matrix_power requires square matrices
linalg.matrix_power(batch, 2);
// @ts-expect-error cond supports only p = 2 and -2
linalg.cond(batch, 1);

void [n, rank, inverse, condition, power, negative];
