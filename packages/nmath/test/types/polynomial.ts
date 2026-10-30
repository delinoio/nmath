// SPDX-License-Identifier: Apache-2.0
import { createContext, type Expr } from "../../src/index.js";
import { polynomial, polyval } from "../../src/numpy.js";
const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 1,
  cpuMemoryBudget: 1024n,
  gpuMemoryBudget: 0n,
});
const coefficients = c.zeros([3, 2] as const, "float32");
const points = c.zeros([2] as const, "float32");
const tensor: Expr<"float32", [2, 2]> = polynomial.polynomial.polyval(
  points,
  coefficients,
);
const broadcast: Expr<"float32", [2]> = polynomial.polynomial.polyval(
  points,
  coefficients,
  { tensor: false },
);
const legacy: Expr<"float32", [2]> = polyval(coefficients, points);
const integerCoefficients: Expr<"float64", [2]> = polynomial.polynomial.polyval(
  points,
  c.zeros([3] as const, "int8"),
);
const booleanPoints: Expr<"float64", [2, 2]> = polynomial.polynomial.polyval(
  c.zeros([2] as const, "bool"),
  coefficients,
);
const empty: Expr<"int64", readonly [2]> = polyval(
  c.zeros([0] as const, "complex128"),
  c.zeros([2] as const, "int64"),
);
const constant: Expr<"float64", [2]> = polynomial.polynomial.polyval(
  points,
  c.zeros([] as const, "int32"),
);
const prefix: Expr<"complex64", [2, number, 4]> = polynomial.polynomial.polyval(
  c.zeros([4] as const, "float32"),
  c.zeros([3, 2, 7 as number] as const, "complex64"),
);
declare const optionalTensor: { tensor?: false };
const optional: Expr<"float32", [2, 2] | [2]> = polynomial.polynomial.polyval(
  points,
  coefficients,
  optionalTensor,
);
// @ts-expect-error omitted optional tensor defaults to true, so the output may have rank two
const wrongOptional: Expr<"float32", [2]> = polynomial.polynomial.polyval(
  points,
  coefficients,
  optionalTensor,
);
// @ts-expect-error no coefficient axis for legacy evaluation
polyval(c.zeros([] as const, "float32"), points);
// @ts-expect-error empty ascending coefficient axis
polynomial.polynomial.polyval(points, c.zeros([0] as const, "float32"));
// @ts-expect-error incompatible static broadcast shapes
polynomial.polynomial.polyval(c.zeros([4] as const, "float32"), coefficients, {
  tensor: false,
});
// @ts-expect-error descending evaluation always broadcasts
polyval(coefficients, c.zeros([4] as const, "float32"));
// @ts-expect-error string coefficients are outside polynomial dtypes
polyval(c.zeros([3] as const, "str"), points);
// @ts-expect-error boolean tensor option required
polynomial.polynomial.polyval(points, coefficients, { tensor: "yes" });
void [
  tensor,
  broadcast,
  legacy,
  integerCoefficients,
  booleanPoints,
  empty,
  constant,
  prefix,
  optional,
  wrongOptional,
];

declare const count: 0 | 3;
const maybeEmpty = polyval(c.zeros([count, 2] as const, "complex64"), points);
const dtypeAlternative: "float32" | "complex64" = maybeEmpty.dtype;
// @ts-expect-error an empty descending coefficient array keeps the points' dtype
const wrongDtype: "complex64" = maybeEmpty.dtype;
void [dtypeAlternative, wrongDtype];
