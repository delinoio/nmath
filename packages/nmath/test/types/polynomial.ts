// SPDX-License-Identifier: Apache-2.0
import { createContext, type Expr } from "../../src/index.js";
import { polynomial, polyval } from "../../src/numpy.js";
type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;
declare function exact<T extends true>(): void;
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

type CoefficientShapes = readonly [0, 3] | readonly [2, 3];
declare const coefficientShape: CoefficientShapes;
const unionPoints = c.zeros([1] as const, "float32");
const unionResult = polyval(c.zeros(coefficientShape, "int32"), unionPoints);
exact<Equal<typeof unionResult.dtype, "float32" | "float64">>();
exact<Equal<typeof unionResult.shape, readonly [1] | [3]>>();
// @ts-expect-error the empty coefficients retain the points' dtype
const onlyNonemptyDtype: "float64" = unionResult.dtype;
// @ts-expect-error the empty coefficients retain the points' shape
const onlyNonemptyShape: readonly [3] = unionResult.shape;
async function unionStorage() {
  const computed = await unionResult.compute();
  exact<Equal<typeof computed.dtype, "float32" | "float64">>();
  exact<Equal<typeof computed.shape, readonly [1] | [3]>>();
  exact<
    Equal<ReturnType<typeof computed.toTypedArray>, Float32Array | Float64Array>
  >();
  // @ts-expect-error the empty result uses Float32Array storage
  const onlyNonemptyStorage: Float64Array = computed.toTypedArray();
  void onlyNonemptyStorage;
}
void [onlyNonemptyDtype, onlyNonemptyShape, unionStorage];

declare const reversedShape: readonly [2, 3] | readonly [0, 3];
const reversed = polyval(c.zeros(reversedShape, "int32"), unionPoints);
exact<Equal<typeof reversed.dtype, "float32" | "float64">>();
exact<Equal<typeof reversed.shape, readonly [1] | [3]>>();

declare const differentTail: readonly [0, 3] | readonly [2, 4];
const differentTailResult = polyval(
  c.zeros(differentTail, "int32"),
  unionPoints,
);
exact<Equal<typeof differentTailResult.shape, readonly [1] | [4]>>();

const fixedEmpty = polyval(c.zeros([0, 3] as const, "int32"), unionPoints);
exact<Equal<typeof fixedEmpty.dtype, "float32">>();
exact<Equal<typeof fixedEmpty.shape, readonly [1]>>();
const fixedNonempty = polyval(c.zeros([2, 3] as const, "int32"), unionPoints);
exact<Equal<typeof fixedNonempty.dtype, "float64">>();
exact<Equal<typeof fixedNonempty.shape, [3]>>();
declare const dynamicLength: number;
const dynamic = polyval(
  c.zeros([dynamicLength, 3] as const, "int32"),
  unionPoints,
);
exact<Equal<typeof dynamic.dtype, "float32" | "float64">>();
exact<Equal<typeof dynamic.shape, readonly [1] | [3]>>();

const equalDtype = polyval(c.zeros(coefficientShape, "float32"), unionPoints);
exact<Equal<typeof equalDtype.dtype, "float32">>();
exact<Equal<typeof equalDtype.shape, readonly [1] | [3]>>();
