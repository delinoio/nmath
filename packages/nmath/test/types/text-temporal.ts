// SPDX-License-Identifier: Apache-2.0
import {
  createContext,
  Temporal,
  type Expr,
  type NMathArray,
  add,
  sub,
  mul,
} from "../../src/index.js";
const ctx = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 1,
  cpuMemoryBudget: 100000,
  gpuMemoryBudget: 0,
});
const strings = ctx.array(["😀", "한글"], {
  dtype: "str",
  shape: [2] as const,
});
const dates = ctx.array(new BigInt64Array([1n, 2n]), {
  dtype: "datetime64",
  unit: "ns",
  shape: [2] as const,
});
const deltas = ctx.array(new BigInt64Array([1n, 2n]), {
  dtype: "timedelta64",
  unit: "ms",
  shape: [2] as const,
});
const stringResult: Promise<string> = strings.get(0).compute();
const temporalResult: Promise<Temporal<"datetime64">> = dates.get(0).compute();
const difference: Expr<"timedelta64", readonly [2]> = sub(dates, dates);
const shifted: Expr<"datetime64", readonly [2]> = add(dates, deltas);
const scaled: Expr<"timedelta64", readonly [2]> = mul(deltas, 2n);
const mean: Promise<Temporal<"timedelta64">> = deltas.mean().compute();
const copy: Promise<NMathArray<"str", readonly [2]>> = ctx.copy(strings);
// @ts-expect-error datetime addition is not defined
add(dates, dates);
// @ts-expect-error string subtraction is not defined
strings.sub(strings);
// @ts-expect-error string reduction is not defined
strings.sum();
// @ts-expect-error datetime mean is not defined
dates.mean();
// @ts-expect-error temporal values are not numbers
const numeric: number = dates.item(0);
// @ts-expect-error durations cannot convert to Date
new Temporal(1n, "s", "timedelta64").toDate();
// @ts-expect-error strings require JS strings
ctx.array([1], { dtype: "str" });
void [
  stringResult,
  temporalResult,
  difference,
  shifted,
  scaled,
  mean,
  copy,
  numeric,
];

import * as np from "../../src/numpy.js";
const namespaceMean: Promise<Temporal<"timedelta64">> = np
  .mean(deltas)
  .compute();
const namespaceMinimum: Promise<Temporal<"datetime64">> = np
  .min(dates)
  .compute();
// @ts-expect-error datetime mean is not defined through the namespace
np.mean(dates);
// @ts-expect-error string minima are not supported
np.min(strings);
void [namespaceMean, namespaceMinimum];
