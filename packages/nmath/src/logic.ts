// SPDX-License-Identifier: Apache-2.0
// NumPy truth reductions and closeness tests. Each function composes existing lazy
// nodes in the order of NumPy 2.5.4 `numpy/_core/fromnumeric.py` and
// `numpy/_core/numeric.py`; no dedicated native node exists.

import { Complex, Expr, Lazy, type BinaryOperator } from "./core.js";
import { NMathError, native } from "./errors.js";
import { binding } from "./native.js";
import type {
  Broadcast,
  BroadcastCheck,
  ComplexDType,
  MutableShape,
  NumericDType,
  ReduceOptions,
  ReduceShape,
  Shape,
} from "./types.js";

type Arg<D extends NumericDType, S extends Shape> = Lazy<NoInfer<D>, S> & {
  readonly dtype: D;
};

/** Weak JS scalars accepted by the closeness tests. */
export type CloseScalar = number | bigint | boolean | Complex;

/** `isclose` and `allclose` keyword arguments. Tolerances are JS numbers. */
export interface CloseOptions {
  readonly rtol?: number;
  readonly atol?: number;
  readonly equal_nan?: boolean;
}

/** `array_equal` keyword arguments. */
export interface ArrayEqualOptions {
  readonly equal_nan?: boolean;
}

type Value = Lazy<any, any> | CloseScalar;

function invalid(message: string): never {
  throw new NMathError(
    "NMATH_INDEX_INVALID",
    `NMATH_INDEX_INVALID: ${message}`,
  );
}

function checkKeys(
  options: object | undefined,
  allowed: readonly string[],
  name: string,
): void {
  if (options === undefined) return;
  if (options === null || typeof options !== "object") {
    invalid(`${name} options must be an object.`);
  }
  for (const key of Object.keys(options)) {
    if (!allowed.includes(key)) invalid(`Unknown ${name} option '${key}'.`);
  }
}

function flag(value: unknown, name: string): boolean {
  if (value === undefined) return false;
  if (typeof value !== "boolean") invalid(`${name} must be a boolean.`);
  return value;
}

/** Rejects text and temporal arrays, which have no truth or closeness semantics here. */
function numeric(x: Value, name: string): void {
  if (x instanceof Lazy) {
    const dtype = x.dtype as string;
    if (dtype === "str" || dtype === "datetime64" || dtype === "timedelta64") {
      throw new NMathError(
        "NMATH_DTYPE_UNSUPPORTED",
        `NMATH_DTYPE_UNSUPPORTED: ${name} does not support ${dtype} arrays. Convert the operand to a numeric dtype explicitly.`,
      );
    }
    return;
  }
  if (
    typeof x !== "number" &&
    typeof x !== "bigint" &&
    typeof x !== "boolean" &&
    !(x instanceof Complex)
  ) {
    throw new TypeError(
      `${name} operands must be NMath arrays, expressions, numbers, bigints, booleans or Complex scalars.`,
    );
  }
}

function lazy(x: unknown, name: string): Lazy<any, any> {
  if (!(x instanceof Lazy)) {
    throw new TypeError(`${name} requires an NMath array or expression.`);
  }
  numeric(x, name);
  return x;
}

function binary(op: BinaryOperator, a: Value, b: Value): Expr<any, any> {
  if (a instanceof Lazy) return a.binaryOp(op, b);
  if (b instanceof Lazy) return b.reversedBinaryOp(op, a);
  throw new TypeError(
    "At least one operand must be an NMath array or expression.",
  );
}

function math(op: string, x: Lazy<any, any>): Expr<any, any> {
  return new Expr(
    native(() => binding.NativeExpr.math(op, x.nativeExpr())),
    x.context,
  );
}

/** NumPy truthiness: nonzero, NaN, and complex values with any nonzero component. */
function truth(x: Lazy<any, any>): Expr<any, any> {
  return x.binaryOp("ne", false);
}

function isComplexDType(dtype: string): dtype is ComplexDType {
  return dtype === "complex64" || dtype === "complex128";
}

/**
 * Elementwise `abs`. The `absolute` math node has no complex loop yet, so complex
 * values use the native vector infinity norm over an appended length-1 axis,
 * which is `hypot(re, im)` computed in double precision and stored in the real
 * dtype. Remove this workaround when `absolute` accepts complex input.
 */
function abs(x: Lazy<any, any>): Expr<any, any> {
  if (!isComplexDType(x.dtype)) return math("absolute", x);
  const column = new Expr(
    native(() =>
      binding.NativeExpr.reshape(x.nativeExpr(), [...x.shape, 1], undefined),
    ),
    x.context,
  );
  return new Expr(
    native(() =>
      binding.NativeExpr.norm(column.nativeExpr(), "vector_inf", undefined),
    ),
    x.context,
  );
}

function scalarAbs(x: CloseScalar): number {
  if (x instanceof Complex) return Math.hypot(x.real, x.imag);
  return Math.abs(Number(x));
}

function scalarIsNaN(x: CloseScalar): boolean {
  if (x instanceof Complex) return Number.isNaN(x.real) || Number.isNaN(x.imag);
  return typeof x === "number" && Number.isNaN(x);
}

function scalarIsFinite(x: CloseScalar): boolean {
  if (x instanceof Complex)
    return Number.isFinite(x.real) && Number.isFinite(x.imag);
  return typeof x !== "number" || Number.isFinite(x);
}

/** A strong rank-0 bool constant bound to a context, for statically known results. */
function constant(value: boolean, source: Lazy<any, any>): Expr<"bool", []> {
  return new Expr<any, any>(
    native(() =>
      binding.NativeExpr.cast(binding.NativeExpr.boolScalar(value), "bool"),
    ),
    source.context,
  );
}

function reduceTruth(
  name: string,
  x: unknown,
  axis: number | readonly number[] | undefined,
  options: ReduceOptions | undefined,
): Expr<any, any> {
  const input = lazy(x, name);
  checkKeys(options, ["keepdims"], name);
  return truth(input).reduceOp("sum", axis, options ?? {});
}

/**
 * NumPy `any`: whether any element along the axes is truthy. Empty reductions
 * give `false`. The result is the int64 count of truthy elements compared with
 * zero, so the dtype stays bool.
 */
function any_<
  D extends NumericDType,
  S extends Shape,
  const A extends number | readonly number[] | undefined = undefined,
  K extends boolean = false,
>(
  a: Arg<D, S>,
  axis?: A,
  options?: ReduceOptions<K>,
): Expr<"bool", ReduceShape<S, A, K>>;
function any_(
  a: Lazy<any, any>,
  axis?: number | readonly number[],
  options?: ReduceOptions,
): Expr<any, any> {
  return reduceTruth("any", a, axis, options).binaryOp("gt", 0n);
}

/**
 * NumPy `all`: whether every element along the axes is truthy. Empty
 * reductions give `true`. The result is the int64 count of falsy elements
 * compared with zero.
 */
function all_<
  D extends NumericDType,
  S extends Shape,
  const A extends number | readonly number[] | undefined = undefined,
  K extends boolean = false,
>(
  a: Arg<D, S>,
  axis?: A,
  options?: ReduceOptions<K>,
): Expr<"bool", ReduceShape<S, A, K>>;
function all_(
  a: Lazy<any, any>,
  axis?: number | readonly number[],
  options?: ReduceOptions,
): Expr<any, any> {
  const input = lazy(a, "all");
  checkKeys(options, ["keepdims"], "all");
  return input
    .binaryOp("eq", false)
    .reduceOp("sum", axis, options ?? {})
    .binaryOp("eq", 0n);
}

export { any_ as any, all_ as all };

/**
 * NumPy `count_nonzero`: the number of truthy elements as int64 (NumPy `intp`).
 * A rank-0 result computes to a `bigint`.
 */
export function count_nonzero<
  D extends NumericDType,
  S extends Shape,
  const A extends number | readonly number[] | undefined = undefined,
  K extends boolean = false,
>(
  a: Arg<D, S>,
  axis?: A,
  options?: ReduceOptions<K>,
): Expr<"int64", ReduceShape<S, A, K>>;
export function count_nonzero(
  a: Lazy<any, any>,
  axis?: number | readonly number[],
  options?: ReduceOptions,
): Expr<any, any> {
  return reduceTruth("count_nonzero", a, axis, options);
}

/** NumPy `result_type(y, 1.)`: integer and bool references become float64. */
function inexact(y: Value): Value {
  if (y instanceof Lazy) {
    const dtype = y.dtype as string;
    return dtype.startsWith("float") || isComplexDType(dtype)
      ? y
      : y.astype("float64");
  }
  return typeof y === "bigint" || typeof y === "boolean" ? Number(y) : y;
}

function and(a: Expr<any, any>, b: Expr<any, any> | boolean): Expr<any, any> {
  return a.binaryOp("logical_and", b);
}

function or(a: Expr<any, any>, b: Expr<any, any> | boolean): Expr<any, any> {
  return a.binaryOp("logical_or", b);
}

function closeness(
  name: string,
  a: Value,
  b: Value,
  options: CloseOptions | undefined,
): Expr<any, any> {
  numeric(a, name);
  numeric(b, name);
  if (!(a instanceof Lazy) && !(b instanceof Lazy)) {
    throw new TypeError(
      `${name} requires at least one NMath array or expression.`,
    );
  }
  checkKeys(options, ["rtol", "atol", "equal_nan"], name);
  const rtol = options?.rtol ?? 1e-5;
  const atol = options?.atol ?? 1e-8;
  if (typeof rtol !== "number" || typeof atol !== "number") {
    invalid(`${name} rtol and atol must be numbers.`);
  }
  const equalNan = flag(options?.equal_nan, "equal_nan");
  const x = a;
  const y = inexact(b);
  // result = less_equal(abs(x - y), atol + rtol * abs(y)) & isfinite(y) | (x == y)
  const distance = abs(binary("sub", x, y));
  const tolerance =
    y instanceof Lazy
      ? abs(y).binaryOp("mul", rtol).binaryOp("add", atol)
      : atol + rtol * scalarAbs(y);
  const within = distance.binaryOp("le", tolerance);
  const finite = y instanceof Lazy ? math("isfinite", y) : scalarIsFinite(y);
  const same = binary("eq", x, y);
  let result =
    finite === false
      ? same
      : or(finite === true ? within : and(within, finite), same);
  if (equalNan) {
    const xNan = x instanceof Lazy ? math("isnan", x) : scalarIsNaN(x);
    const yNan = y instanceof Lazy ? math("isnan", y) : scalarIsNaN(y);
    if (xNan instanceof Expr && yNan instanceof Expr) {
      result = or(result, and(xNan, yNan));
    } else if (xNan instanceof Expr || yNan instanceof Expr) {
      const lazyNan = (xNan instanceof Expr ? xNan : yNan) as Expr<any, any>;
      const known = (xNan instanceof Expr ? yNan : xNan) as boolean;
      if (known) result = or(result, lazyNan);
    }
  }
  return result;
}

/**
 * NumPy `isclose`: `abs(a - b) <= atol + rtol * abs(b)` where `b` is finite, or
 * `a == b`, so infinities are close only to equal infinities. NaNs are close
 * only with `equal_nan`. `b` is promoted to an inexact dtype first, as in NumPy,
 * and the comparison is asymmetric in `a` and `b`. Operands broadcast.
 */
export function isclose<
  A extends NumericDType,
  AS extends Shape,
  B extends NumericDType,
  BS extends Shape,
>(
  a: Arg<A, AS>,
  b: Arg<B, BS> & BroadcastCheck<AS, BS>,
  options?: CloseOptions,
): Expr<"bool", Broadcast<AS, BS>>;
export function isclose<A extends NumericDType, AS extends Shape>(
  a: Arg<A, AS>,
  b: CloseScalar,
  options?: CloseOptions,
): Expr<"bool", MutableShape<AS>>;
export function isclose<B extends NumericDType, BS extends Shape>(
  a: CloseScalar,
  b: Arg<B, BS>,
  options?: CloseOptions,
): Expr<"bool", MutableShape<BS>>;
export function isclose(
  a: Value,
  b: Value,
  options?: CloseOptions,
): Expr<any, any> {
  return closeness("isclose", a, b, options);
}

/**
 * NumPy `allclose`: `all(isclose(a, b, ...))`. The rank-0 bool result computes
 * to a JS boolean.
 */
export function allclose<
  A extends NumericDType,
  AS extends Shape,
  B extends NumericDType,
  BS extends Shape,
>(
  a: Arg<A, AS>,
  b: Arg<B, BS> & BroadcastCheck<AS, BS>,
  options?: CloseOptions,
): Expr<"bool", []>;
export function allclose<A extends NumericDType, AS extends Shape>(
  a: Arg<A, AS>,
  b: CloseScalar,
  options?: CloseOptions,
): Expr<"bool", []>;
export function allclose<B extends NumericDType, BS extends Shape>(
  a: CloseScalar,
  b: Arg<B, BS>,
  options?: CloseOptions,
): Expr<"bool", []>;
export function allclose(
  a: Value,
  b: Value,
  options?: CloseOptions,
): Expr<any, any> {
  return all_(closeness("allclose", a, b, options));
}

function sameShape(a: Shape, b: Shape): boolean {
  return a.length === b.length && a.every((n, i) => n === b[i]);
}

function broadcastable(a: Shape, b: Shape): boolean {
  for (let i = 1; i <= Math.min(a.length, b.length); i++) {
    const m = a[a.length - i];
    const n = b[b.length - i];
    if (m !== n && m !== 1 && n !== 1) return false;
  }
  return true;
}

/**
 * NumPy `array_equal`: true when both inputs have the same shape and equal
 * elements. Shapes are known before execution, so different shapes give a
 * rank-0 bool expression that computes to `false` without reading the inputs.
 * With `equal_nan`, NaNs in the same positions compare equal; a complex value
 * counts as NaN when either component is NaN.
 */
export function array_equal<
  A extends NumericDType,
  AS extends Shape,
  B extends NumericDType,
  BS extends Shape,
>(
  a1: Arg<A, AS>,
  a2: Arg<B, BS>,
  options?: ArrayEqualOptions,
): Expr<"bool", []>;
export function array_equal(
  a1: Lazy<any, any>,
  a2: Lazy<any, any>,
  options?: ArrayEqualOptions,
): Expr<any, any> {
  const x = lazy(a1, "array_equal");
  const y = lazy(a2, "array_equal");
  checkKeys(options, ["equal_nan"], "array_equal");
  const equalNan = flag(options?.equal_nan, "equal_nan");
  if (!sameShape(x.shape, y.shape)) return constant(false, x);
  const same = x.binaryOp("eq", y);
  return all_(
    equalNan ? or(same, and(math("isnan", x), math("isnan", y))) : same,
  );
}

/**
 * NumPy `array_equiv`: true when the inputs broadcast and all broadcast
 * elements are equal. Shapes that do not broadcast give a rank-0 bool
 * expression that computes to `false`.
 */
export function array_equiv<
  A extends NumericDType,
  AS extends Shape,
  B extends NumericDType,
  BS extends Shape,
>(a1: Arg<A, AS>, a2: Arg<B, BS>): Expr<"bool", []>;
export function array_equiv<A extends NumericDType, AS extends Shape>(
  a1: Arg<A, AS>,
  a2: CloseScalar,
): Expr<"bool", []>;
export function array_equiv(
  a1: Lazy<any, any>,
  a2: Lazy<any, any> | CloseScalar,
): Expr<any, any> {
  const x = lazy(a1, "array_equiv");
  numeric(a2, "array_equiv");
  if (a2 instanceof Lazy && !broadcastable(x.shape, a2.shape)) {
    return constant(false, x);
  }
  return all_(x.binaryOp("eq", a2));
}
