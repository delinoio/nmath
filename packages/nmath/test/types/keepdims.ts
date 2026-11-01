// SPDX-License-Identifier: Apache-2.0
import { createContext, type Expr } from "../../src/index.js";
import { argmax, max, sum, var as variance } from "../../src/numpy.js";

const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});

const a = c.zeros([2, 3, 4], "int32");
const listed: Expr<"int64", [3]> = sum(a, [0, 2]);
const kept: Expr<"int64", [1, 3, 1]> = sum(a, [0, 2], { keepdims: true });
const keptAll: Expr<"int64", [1, 1, 1]> = sum(a, undefined, { keepdims: true });
const scalar: Expr<"int64", []> = sum(a);
const none: Expr<"int64", [2, 3, 4]> = sum(a, []);
const method: Expr<"int32", [2, 1, 4]> = a.max(1, { keepdims: true });
const index: Expr<"int64", [2, 1, 4]> = argmax(a, 1, { keepdims: true });
const spread: Expr<"float64", [2, 3, 1]> = variance(a, 2, 0, {
  keepdims: true,
});
const negative: Expr<"int32", number[]> = max(a, [-1]);
const dynamic: Expr<"int64", number[]> = sum(a, [0], {
  keepdims: Math.random() > 0.5,
});

// @ts-expect-error keepdims keeps reduced axes
const wrong: Expr<"int64", [3]> = sum(a, [0, 2], { keepdims: true });
// @ts-expect-error argmax selects one axis
argmax(a, [0, 1]);

void [listed, kept, keptAll, scalar, none, method, index, spread];
void [negative, dynamic, wrong];
