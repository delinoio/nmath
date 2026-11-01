// SPDX-License-Identifier: Apache-2.0
import { createContext, type Expr, type Lazy } from "../../src/index.js";
import { cumsum, cumprod } from "../../src/numpy.js";

const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});

const ints = c.zeros([2, 3], "int16");
const flat: Expr<"int64", [number]> = ints.cumsum();
const along: Expr<"int64", [2, 3]> = ints.cumprod(1);
const negative: Expr<"int64", [2, 3]> = ints.cumsum(-1);
const arrayFlat: Expr<"int64", [number]> = cumsum(ints);
const arrayAlong: Expr<"int64", [2, 3]> = cumprod(ints, 0);

const floats = c.zeros([2, 3], "float32");
const half = c.zeros([4], "float16");
const floatScan: Expr<"float32", [2, 3]> = floats.cumsum(1);
const halfScan: Expr<"float16", [number]> = half.cumprod();
const complex = c.zeros([4], "complex128");
const complexScan: Expr<"complex128", [number]> = complex.cumsum();

declare const n: number;
const dynamic = c.zeros([n, 3], "float64");
const dynamicScan: Expr<"float64", [number, 3]> = dynamic.cumsum(0);

const strings = c.array(["a", "b"], { dtype: "str" });
// @ts-expect-error string arrays do not support cumulative scans
strings.cumsum();

declare const times: Lazy<"datetime64", [3]>;
// @ts-expect-error temporal arrays do not support cumulative scans
times.cumprod();

// @ts-expect-error an axis-less scan flattens to one dimension
const wrongFlat: Expr<"int64", [2, 3]> = ints.cumsum();
// @ts-expect-error the promoted scan dtype is int64, not int16
const wrongDtype: Expr<"int16", [number]> = ints.cumsum();

void flat;
void along;
void negative;
void arrayFlat;
void arrayAlong;
void floatScan;
void halfScan;
void complexScan;
void dynamicScan;
void wrongFlat;
void wrongDtype;
