// SPDX-License-Identifier: Apache-2.0
import { createContext, type Expr, type Lazy } from "../../src/index.js";
import { argmax, argmin, prod } from "../../src/numpy.js";

const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});

const ints = c.zeros([2, 3], "int16");
const product: Expr<"int64", [3]> = ints.prod(0);
const total: Expr<"int64", []> = ints.prod();
const rows: Expr<"int64", [2]> = ints.argmax(1);
const first: Expr<"int64", []> = ints.argmin();
const arrayProduct: Expr<"int64", [3]> = prod(ints, 0);
const arrayIndex: Expr<"int64", [2]> = argmin(ints, 1);
const arrayMaximum: Expr<"int64", []> = argmax(ints);

const floats = c.zeros([2, 3], "float32");
const floatProduct: Expr<"float32", [2]> = floats.prod(1);
const complex = c.zeros([4], "complex128");
const complexProduct: Expr<"complex128", []> = complex.prod();

declare const n: number;
const dynamic = c.zeros([n, 3], "float32");
const dynamicProduct: Expr<"float32", [number]> = dynamic.prod(1);
const dynamicIndex: Expr<"int64", [number]> = dynamic.argmin(1);
const flatDynamic: Expr<"int64", []> = dynamic.argmax();

const strings = c.array(["a", "b"], { dtype: "str" });
// @ts-expect-error string arrays do not support products
strings.prod();
// @ts-expect-error string arrays do not support index reductions
strings.argmin();

declare const times: Lazy<"datetime64", [3]>;
// @ts-expect-error temporal arrays do not support products
times.prod();
// @ts-expect-error temporal arrays do not support index reductions
times.argmax();

declare const durations: Lazy<"timedelta64", [3]>;
// @ts-expect-error duration arrays do not support products
durations.prod();
// @ts-expect-error duration arrays do not support index reductions
durations.argmin();

// @ts-expect-error the promoted product dtype is int64, not int16
const wrongProduct: Expr<"int16", [3]> = ints.prod(0);
// @ts-expect-error index reductions always use int64
const wrongIndex: Expr<"float64", []> = ints.argmin();
// @ts-expect-error reductions produce rank-0 results without an axis
const wrongShape: Expr<"int64", [2]> = ints.argmax();

void product;
void total;
void rows;
void first;
void arrayProduct;
void arrayIndex;
void arrayMaximum;
void floatProduct;
void complexProduct;
void dynamicProduct;
void dynamicIndex;
void flatDynamic;
void wrongProduct;
void wrongIndex;
void wrongShape;
