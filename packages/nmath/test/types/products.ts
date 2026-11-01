// SPDX-License-Identifier: Apache-2.0
import { createContext, type Expr } from "../../src/index.js";
import {
  clip,
  diagonal,
  diff,
  dot,
  inner,
  outer,
  trace,
  vdot,
} from "../../src/numpy.js";

const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});
const m = c.zeros([2, 3], "float32");
const n = c.zeros([3, 4], "int32");
const v = c.zeros([3], "float64");
const t = c.zeros([5, 2, 3], "int8");
const s = c.zeros([], "complex128");
const flags = c.zeros([4, 4], "bool");
declare const unknown: number;
const dynamic = c.zeros([unknown, 3], "float32");

const matrix: Expr<"float64", [2, 4]> = dot(m, n);
const matrixVector: Expr<"float64", [2]> = dot(m, v);
const vectorDot: Expr<"float64", []> = dot(v, v);
const batched: Expr<"float32", [2, 5, 2]> = dot(m, c.zeros([5, 3, 2], "int8"));
const scaled: Expr<"complex128", [2, 3]> = dot(s, m);
const partly: Expr<"float32", [number, 2]> = dot(
  dynamic,
  c.zeros([3, 2], "float32"),
);
const conjugated: Expr<"float64", []> = vdot(m, n);
const contracted: Expr<"float32", [2, 5, 2]> = inner(m, t);
const products: Expr<"float64", [3, 2]> = outer(v, c.zeros([2], "int64"));
const flattened: Expr<"float32", [number, number]> = outer(m, m);
const diag: Expr<"bool", [number]> = diagonal(flags, 1);
const batchDiag: Expr<"int8", [3, number]> = diagonal(t);
const otherAxes: Expr<"int8", number[]> = diagonal(t, 0, 1, 2);
const total: Expr<"int64", []> = trace(flags);
const batchTotal: Expr<"int64", [3]> = trace(t);
const floatTotal: Expr<"float32", []> = trace(m);
const clipped: Expr<"float32", readonly [2, 3]> = clip(m, 0, 1);
const intClip: Expr<"int32", readonly [3, 4]> = clip(n, 0n, null);
const floatClip: Expr<"float64", readonly [3, 4]> = clip(n, undefined, 2.5);
const arrayClip: Expr<"float64", [2, 3]> = clip(m, v, null);
// Both bounds resolve one dtype with the array before either comparison.
const mixedClip: Expr<"float64", readonly [3, 4]> = clip(n, 1000n, 2000.5);
const difference: Expr<"float32", [number, number]> = diff(m);
const unchanged: Expr<"float32", readonly [2, 3]> = diff(m, 0);
const boolDiff: Expr<"bool", [number, number]> = diff(flags, 2, 0);
const durations: Expr<"timedelta64", [number]> = diff(
  c.zeros([4], "datetime64"),
);

// @ts-expect-error dot contraction dimensions must match
dot(m, c.zeros([4, 2], "float32"));
// @ts-expect-error inner contraction dimensions must match
inner(m, c.zeros([4], "float32"));
// @ts-expect-error dot rejects string operands
dot(c.zeros([3], "str"), v);
// @ts-expect-error vdot rejects temporal operands
vdot(c.zeros([3], "datetime64"), v);
// @ts-expect-error diagonal requires at least two dimensions
diagonal(v);
// @ts-expect-error trace rejects strings
trace(c.zeros([2, 2], "str"));
// @ts-expect-error clip rejects temporal input
clip(c.zeros([2], "timedelta64"), 0n, 1n);
// @ts-expect-error clip rejects string bounds
clip(m, "a", null);
// @ts-expect-error diff rejects strings
diff(c.zeros([2], "str"));

void [
  matrix,
  matrixVector,
  vectorDot,
  batched,
  scaled,
  partly,
  conjugated,
  contracted,
  products,
  flattened,
  diag,
  batchDiag,
  otherAxes,
  total,
  batchTotal,
  floatTotal,
  clipped,
  mixedClip,
  intClip,
  floatClip,
  arrayClip,
  difference,
  unchanged,
  boolDiff,
  durations,
];
