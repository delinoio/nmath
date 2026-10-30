// SPDX-License-Identifier: Apache-2.0
import { createContext, type Expr, Complex } from "../../src/index.js";
import { polynomial, polyder, type DerivativeShape } from "../../src/numpy.js";
const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 4,
  cpuMemoryBudget: 4096n,
  gpuMemoryBudget: 0n,
});
const p = polynomial.polynomial;
const a = c.zeros([4, 3, 2] as const, "float32");
const first: Expr<"float32", [3, 3, 2]> = p.polyder(a);
const axis: Expr<"float32", [4, 1, 2]> = p.polyder(a, { m: 2, axis: -2 });
const high: Expr<"float32", [4, 3, 1]> = p.polyder(a, {
  m: 1000000000000,
  axis: -1,
});
const integer: Expr<"float64", [2]> = p.polyder(c.zeros([3] as const, "int8"));
const complex: Expr<"complex64", [2]> = p.polyder(
  c.zeros([3] as const, "complex64"),
  { scl: new Complex(1, 2) },
);
const scalar: Expr<"float64", [1]> = p.polyder(c.zeros([] as const, "bool"));
const empty: Expr<"float32", [2, 0, 3]> = p.polyder(
  c.zeros([2, 0, 3] as const, "float32"),
  { axis: 1 },
);
const dynamicAxis: Expr<"float32", [number, number, number]> = p.polyder(a, {
  axis: 1 as number,
});
const dynamicOrder: Expr<"float32", [number, 3, 2]> = p.polyder(a, {
  m: 1 as number,
});
const partial: Expr<"float32", [4, number, 2]> = p.polyder(
  c.zeros([4, 9 as number, 2] as const, "float32"),
  { m: 2, axis: 1 },
);
const legacy: Expr<"float64", [2]> = polyder(c.zeros([3] as const, "float32"));
const noOp: Expr<"int8", [3]> = polyder(c.zeros([3] as const, "int8"), 0);
const legacyComplex: Expr<"complex128", [1]> = polyder(
  c.zeros([3] as const, "complex64"),
  2,
);
const legacyEmpty: Expr<"int64", [0]> = polyder(
  c.zeros([1] as const, "int32"),
  4,
);
const maybeNoOp: Expr<"float32" | "float64", [number]> = polyder(
  c.zeros([3] as const, "float32"),
  1 as number,
);
declare const optional: { m?: 0; axis?: -1 };
const optionalResult: Expr<"float32", [4, 3, 2] | [3, 3, 2] | [4, 3, 1]> =
  p.polyder(a, optional);
const large: DerivativeShape<[1000000, 2], 999998> = [2, 2];
// @ts-expect-error statically negative order
p.polyder(a, { m: -1 });
// @ts-expect-error fractional order
polyder(c.zeros([3] as const, "float32"), 0.5);
// @ts-expect-error out-of-range axis
p.polyder(a, { axis: 3 });
// @ts-expect-error negative axis beyond rank
p.polyder(a, { axis: -4 });
// @ts-expect-error fractional axis
p.polyder(a, { axis: 0.5 });
// @ts-expect-error legacy requires rank one
polyder(a);
// @ts-expect-error legacy scalar has no coefficient axis
polyder(c.zeros([] as const, "float64"));
// @ts-expect-error nonnumeric coefficients
p.polyder(c.zeros([3] as const, "str"));
// @ts-expect-error unsupported derivative option
p.polyder(a, { out: a });
// @ts-expect-error rank and unaffected dimensions remain known
const wrongShape: Expr<"float32", [number, 4, 2]> = p.polyder(a, {
  m: 2 as number,
});
void [
  first,
  axis,
  high,
  integer,
  complex,
  scalar,
  empty,
  dynamicAxis,
  dynamicOrder,
  partial,
  legacy,
  noOp,
  legacyComplex,
  legacyEmpty,
  maybeNoOp,
  optionalResult,
  large,
  wrongShape,
];

// @ts-expect-error complex scale cannot be applied to known active real coefficients
p.polyder(c.zeros([3] as const, "float32"), { scl: new Complex(1) });
p.polyder(c.zeros([3] as const, "float32"), { m: 0, scl: new Complex(1) });
p.polyder(c.zeros([3] as const, "float32"), { m: 5, scl: new Complex(1) });
