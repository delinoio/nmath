// SPDX-License-Identifier: Apache-2.0
import { createContext, type Expr } from "../../src/index.js";
import { matmul } from "../../src/numpy.js";
const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 1,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});
const a = c.zeros([2, 1, 3, 4] as const, "float32");
const b = c.zeros([5, 4, 6] as const, "float32");
const batched: Expr<"float32", [2, 5, 3, 6]> = matmul(a, b);
const method: Expr<"float32", [2, 5, 3, 6]> = a.matmul(b);
const vector = c.zeros([4] as const, "float32");
const leftVector: Expr<"float32", [5, 6]> = matmul(vector, b);
const rightVector: Expr<"float32", [2, 1, 3]> = matmul(a, vector);
const dot: Expr<"float32", []> = matmul(vector, vector);
const promoted: Expr<"float64", [2, 1, 3, 6]> = matmul(
  a,
  c.zeros([4, 6] as const, "int32"),
);
declare const unknown: number;
const dynamic = c.zeros([unknown, 4, 6] as const, "float32");
const partlyKnown: Expr<"float32", [2, number, 3, 6]> = matmul(a, dynamic);
const zero = c.zeros([0, 3, 4] as const, "float32");
const empty: Expr<"float32", [0, 3, 6]> = matmul(
  zero,
  c.zeros([1, 4, 6] as const, "float32"),
);
// @ts-expect-error incompatible static contraction dimensions
matmul(a, c.zeros([5, 7, 6] as const, "float32"));
// @ts-expect-error incompatible static leading batch dimensions
matmul(a, c.zeros([3, 5, 4, 6] as const, "float32"));
// @ts-expect-error method rejects incompatible static batch dimensions too
a.matmul(c.zeros([3, 5, 4, 6] as const, "float32"));
// @ts-expect-error scalars cannot be matrix operands
matmul(c.zeros([] as const, "float32"), vector);
// @ts-expect-error strings cannot be matrix operands
matmul(c.zeros([4] as const, "str"), vector);
// @ts-expect-error datetime values cannot be matrix operands
matmul(vector, c.zeros([4] as const, "datetime64"));
void [
  batched,
  method,
  leftVector,
  rightVector,
  dot,
  promoted,
  partlyKnown,
  empty,
];
