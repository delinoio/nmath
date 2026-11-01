// SPDX-License-Identifier: Apache-2.0
import { createContext, type Expr, type Lazy } from "../../src/index.js";
import { std, var as variance } from "../../src/numpy.js";

const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});

const ints = c.zeros([2, 3], "int16");
const flat: Expr<"float64", []> = ints.var();
const along: Expr<"float64", [3]> = ints.var(0, 1);
const rows: Expr<"float64", [2]> = ints.std(1);
const negative: Expr<"float64", number[]> = ints.var(-1, 0);
const helperVar: Expr<"float64", []> = variance(ints);
const helperStd: Expr<"float64", [2]> = std(ints, 1, 1);

const half = c.zeros([4], "float16");
const halfVar: Expr<"float16", []> = half.var();
const single = c.zeros([4], "float32");
const singleStd: Expr<"float32", []> = single.std();
const complex = c.zeros([4], "complex64");
const complexVar: Expr<"float32", []> = complex.var();
const doubleComplex = c.zeros([4], "complex128");
const doubleComplexStd: Expr<"float64", []> = doubleComplex.std();

declare const n: number;
const dynamic = c.zeros([n, 3], "float32");
const dynamicVar: Expr<"float32", [number]> = dynamic.var(1);

const strings = c.array(["a", "b"], { dtype: "str" });
// @ts-expect-error string arrays do not support variance
strings.var();

declare const times: Lazy<"datetime64", [3]>;
// @ts-expect-error temporal arrays do not support variance
times.std();

// @ts-expect-error integer variance is float64, not float32
const wrongDtype: Expr<"float32", []> = ints.var();
// @ts-expect-error an axis removes its dimension
const wrongShape: Expr<"float64", [2, 3]> = ints.var(0);

void flat;
void along;
void rows;
void negative;
void helperVar;
void helperStd;
void halfVar;
void singleStd;
void complexVar;
void doubleComplexStd;
void dynamicVar;
void wrongDtype;
void wrongShape;
