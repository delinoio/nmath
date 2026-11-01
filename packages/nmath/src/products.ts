// SPDX-License-Identifier: Apache-2.0
// NumPy products, diagonals, clip and diff. Each function composes existing lazy
// nodes (matmul, transpose, reshape, indexing, elementwise ufuncs and the sum
// reduction) in the order NumPy 2.5.4 defines them.

import { type Complex, Expr, Lazy, ellipsis, indices, slice } from "./core.js";
import { NMathError } from "./errors.js";
import { conjugate, maximum, minimum } from "./ufunc.js";
import type {
  Broadcast,
  ComputableDType,
  NumericDType,
  Promote,
  PromoteWeak,
  Shape,
  SumDType,
} from "./types.js";

type Arg<D extends ComputableDType, S extends Shape> = Lazy<NoInfer<D>, S> & {
  readonly dtype: D;
};
type Dims<S extends Shape> = S extends readonly [...infer M extends number[]]
  ? M
  : number[];
type IsLiteral<N extends number> = number extends N ? false : true;
/** Unknown dimensions are compatible with anything; literal ones must be equal. */
type SameDim<A extends number, B extends number> =
  IsLiteral<A> extends true
    ? IsLiteral<B> extends true
      ? A extends B
        ? true
        : false
      : true
    : true;
type NonEmpty<S extends Shape> = S extends readonly [] ? never : S;

/**
 * `np.dot` result shape: a 0-d operand multiplies elementwise; otherwise the
 * last axis of `a` contracts with the second-to-last axis of `b` (its only
 * axis when `b` is one-dimensional).
 */
export type DotShape<A extends Shape, B extends Shape> = A extends readonly []
  ? Dims<B>
  : B extends readonly []
    ? Dims<A>
    : number[] extends A | B
      ? number[]
      : A extends readonly [
            ...infer AH extends number[],
            infer K extends number,
          ]
        ? B extends readonly [infer K2 extends number]
          ? SameDim<K, K2> extends true
            ? AH
            : never
          : B extends readonly [
                ...infer BH extends number[],
                infer K2 extends number,
                infer M extends number,
              ]
            ? SameDim<K, K2> extends true
              ? [...AH, ...BH, M]
              : never
            : number[]
        : number[];
/** `np.inner` result shape: the last axes contract; a 0-d operand multiplies. */
export type InnerShape<A extends Shape, B extends Shape> = A extends readonly []
  ? Dims<B>
  : B extends readonly []
    ? Dims<A>
    : number[] extends A | B
      ? number[]
      : A extends readonly [
            ...infer AH extends number[],
            infer K extends number,
          ]
        ? B extends readonly [
            ...infer BH extends number[],
            infer K2 extends number,
          ]
          ? SameDim<K, K2> extends true
            ? [...AH, ...BH]
            : never
          : number[]
        : number[];
/** `np.outer` flattens both operands; one-dimensional literal lengths are kept. */
export type OuterShape<A extends Shape, B extends Shape> = [
  A extends readonly [infer M extends number] ? M : number,
  B extends readonly [infer N extends number] ? N : number,
];
type ContractionCheck<S extends Shape> = [S] extends [never]
  ? { readonly "contraction dimensions do not match": true }
  : unknown;

/**
 * `np.diagonal` result shape. With the default axes `0` and `1` the remaining
 * axes stay in order and the diagonal is appended as a dynamic last axis.
 * Other axis choices give a dynamic shape.
 */
export type DiagonalShape<
  S extends Shape,
  A1 extends number,
  A2 extends number,
> = number[] extends S
  ? number[]
  : [A1, A2] extends [0, 1]
    ? S extends readonly [number, number, ...infer R extends number[]]
      ? [...R, number]
      : never
    : S extends readonly [number, number, ...number[]]
      ? number[]
      : never;
/** `np.trace` result shape: the two diagonal axes are removed. */
export type TraceShape<
  S extends Shape,
  A1 extends number,
  A2 extends number,
> = number[] extends S
  ? number[]
  : [A1, A2] extends [0, 1]
    ? S extends readonly [number, number, ...infer R extends number[]]
      ? R
      : never
    : S extends readonly [number, number, ...number[]]
      ? number[]
      : never;
type MatrixCheck<S extends Shape> = number[] extends S
  ? unknown
  : S extends readonly [number, number, ...number[]]
    ? unknown
    : { readonly "diagonal requires at least two dimensions": S };

/** A clip bound: an array, a weak JS scalar, or `null`/`undefined` for no bound. */
export type ClipBound =
  | Lazy<NumericDType, Shape>
  | number
  | bigint
  | boolean
  | Complex
  | null
  | undefined;
type StrongBound<D extends NumericDType, B> = B extends {
  readonly dtype: infer D2 extends NumericDType;
}
  ? Promote<D, D2>
  : D;
type WeakBound<D extends NumericDType, B> = B extends
  number | bigint | boolean | Complex
  ? PromoteWeak<D, B>
  : D;
type BoundShape<S extends Shape, B> = B extends {
  readonly shape: infer S2 extends Shape;
}
  ? Broadcast<S, S2>
  : Dims<S>;
/**
 * `np.clip` dtype: like the three-input `um.clip`, array bounds promote with
 * the array first and weak scalar bounds then apply together.
 */
export type ClipDType<D extends NumericDType, Lo, Hi> =
  StrongBound<StrongBound<D, Lo>, Hi> extends infer M extends NumericDType
    ? WeakBound<WeakBound<M, Lo>, Hi>
    : never;
/** `np.clip` shape: array bounds broadcast with the input. */
export type ClipShape<S extends Shape, Lo, Hi> =
  BoundShape<S, Lo> extends infer M extends Shape ? BoundShape<M, Hi> : never;

/** `np.diff` dtype: bool uses `not_equal`, datetimes subtract to timedeltas. */
export type DiffDType<D extends ComputableDType> = D extends "datetime64"
  ? "timedelta64"
  : D;
/** `np.diff` keeps the rank; dimensions along the axis shrink, so they become dynamic. */
export type DiffShape<S extends Shape, N extends number> = N extends 0
  ? Dims<S>
  : number[] extends S
    ? number[]
    : { -readonly [K in keyof S]: number };

function requireLazy(...values: unknown[]): void {
  for (const value of values) {
    if (!(value instanceof Lazy)) {
      throw new TypeError("The operand must be an NMath array or expression.");
    }
  }
}

function product(dims: readonly number[]): number {
  return dims.reduce((total, n) => total * n, 1);
}

function reshape(x: Lazy<any, any>, shape: number[]): Expr<any, any> {
  return x.reshape(shape as never) as Expr<any, any>;
}

function mismatch(message: string): never {
  throw new NMathError(
    "NMATH_SHAPE_MISMATCH",
    `NMATH_SHAPE_MISMATCH: ${message} Check the operand shapes against the operation requirements.`,
  );
}

function integer(value: unknown, what: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value)) {
    throw new NMathError(
      "NMATH_INDEX_INVALID",
      `NMATH_INDEX_INVALID: ${what} must be a safe integer.`,
    );
  }
  return value;
}

/** Normalizes a possibly negative axis like NumPy's `normalize_axis_index`. */
function normalizeAxis(axis: unknown, ndim: number, what: string): number {
  const value = integer(axis, what);
  if (value < -ndim || value >= ndim) {
    throw new NMathError(
      "NMATH_INDEX_OUT_OF_BOUNDS",
      `NMATH_INDEX_OUT_OF_BOUNDS: ${what} ${value} is out of bounds for an array of dimension ${ndim}. Use an axis inside the array rank.`,
    );
  }
  return value < 0 ? value + ndim : value;
}

/**
 * `np.dot(a, b)`. A 0-d operand multiplies elementwise. Otherwise the last
 * axis of `a` contracts with the second-to-last axis of `b`, or with its only
 * axis when `b` is one-dimensional; the result shape is
 * `a.shape[:-1] + b.shape[:-2] + b.shape[-1:]`. Mismatched contraction
 * lengths fail with `NMATH_SHAPE_MISMATCH`, like `matmul`.
 */
export function dot<
  A extends NumericDType,
  AS extends Shape,
  B extends NumericDType,
  BS extends Shape,
>(
  a: Arg<A, AS>,
  b: Arg<B, BS> & ContractionCheck<DotShape<AS, BS>>,
): Expr<Promote<A, B>, DotShape<AS, BS>>;
export function dot(a: Lazy<any, any>, b: Lazy<any, any>): Expr<any, any> {
  requireLazy(a, b);
  if (a.ndim === 0 || b.ndim === 0) {
    return a.binaryOp("mul", b);
  }
  // For at most two dimensions, matmul and dot agree, including vector promotion.
  if (a.ndim <= 2 && b.ndim <= 2) {
    return a.matmul(b as never);
  }
  const n = a.shape[a.ndim - 1];
  const k = b.ndim === 1 ? 0 : b.ndim - 2;
  if (n !== b.shape[k]) {
    mismatch(
      `dot contraction lengths do not match: ${n} (last axis of a) != ${b.shape[k]} (axis ${k} of b).`,
    );
  }
  const lead = a.shape.slice(0, -1);
  const left = reshape(a, [product(lead), n]);
  if (b.ndim === 1) {
    return reshape(left.matmul(b as never), lead);
  }
  // Move the contraction axis of b first and flatten the rest:
  // (..., n, m) -> (n, ..., m) -> (n, prod(...) * m).
  const rest = [...b.shape.slice(0, -2), b.shape[b.ndim - 1]];
  const order = [k, ...b.shape.slice(0, -2).map((_: number, i: number) => i)];
  order.push(b.ndim - 1);
  const right = reshape(b.transpose(order as never), [n, product(rest)]);
  return reshape(left.matmul(right as never), [...lead, ...rest]);
}

/**
 * `np.vdot(a, b)`: flattens both operands, conjugates the first when it is
 * complex and returns the 0-d sum of products. The sizes must match.
 */
export function vdot<
  A extends NumericDType,
  AS extends Shape,
  B extends NumericDType,
  BS extends Shape,
>(a: Arg<A, AS>, b: Arg<B, BS>): Expr<Promote<A, B>, []>;
export function vdot(a: Lazy<any, any>, b: Lazy<any, any>): Expr<any, any> {
  requireLazy(a, b);
  const m = product(a.shape);
  const n = product(b.shape);
  if (m !== n) {
    mismatch(`vdot operands have different sizes: ${m} != ${n}.`);
  }
  const left = reshape(a, [m]);
  const right = reshape(b, [n]);
  const first =
    a.dtype === "complex64" || a.dtype === "complex128"
      ? conjugate(left)
      : left;
  return first.matmul(right as never);
}

/**
 * `np.inner(a, b)`: contracts the last axes without conjugation. The result
 * shape is `a.shape[:-1] + b.shape[:-1]`; a 0-d operand multiplies.
 */
export function inner<
  A extends NumericDType,
  AS extends Shape,
  B extends NumericDType,
  BS extends Shape,
>(
  a: Arg<A, AS>,
  b: Arg<B, BS> & ContractionCheck<InnerShape<AS, BS>>,
): Expr<Promote<A, B>, InnerShape<AS, BS>>;
export function inner(a: Lazy<any, any>, b: Lazy<any, any>): Expr<any, any> {
  requireLazy(a, b);
  if (a.ndim === 0 || b.ndim === 0) {
    return a.binaryOp("mul", b);
  }
  const n = a.shape[a.ndim - 1];
  const m = b.shape[b.ndim - 1];
  if (n !== m) {
    mismatch(`inner contraction lengths do not match: ${n} != ${m}.`);
  }
  const leadA = a.shape.slice(0, -1);
  const leadB = b.shape.slice(0, -1);
  const left = reshape(a, [product(leadA), n]);
  const right = reshape(b, [product(leadB), m]).transpose([1, 0] as never);
  return reshape(left.matmul(right as never), [...leadA, ...leadB]);
}

/** `np.outer(a, b)`: flattens both operands and multiplies to shape `(M, N)`. */
export function outer<
  A extends NumericDType,
  AS extends Shape,
  B extends NumericDType,
  BS extends Shape,
>(a: Arg<A, AS>, b: Arg<B, BS>): Expr<Promote<A, B>, OuterShape<AS, BS>>;
export function outer(a: Lazy<any, any>, b: Lazy<any, any>): Expr<any, any> {
  requireLazy(a, b);
  return reshape(a, [product(a.shape), 1]).binaryOp(
    "mul",
    reshape(b, [1, product(b.shape)]),
  );
}

/**
 * `np.diagonal(a, offset, axis1, axis2)`. The two axes are removed and the
 * diagonal is appended as the last axis. Offsets beyond the matrix give an
 * empty last axis. Requires at least two dimensions and distinct axes.
 */
export function diagonal<
  D extends ComputableDType,
  S extends Shape,
  const A1 extends number = 0,
  const A2 extends number = 1,
>(
  a: Arg<D, S> & MatrixCheck<S>,
  offset?: number,
  axis1?: A1,
  axis2?: A2,
): Expr<D, DiagonalShape<S, A1, A2>>;
export function diagonal(
  a: Lazy<any, any>,
  offset = 0,
  axis1 = 0,
  axis2 = 1,
): Expr<any, any> {
  requireLazy(a);
  const nd = a.ndim;
  if (nd < 2) {
    mismatch("diagonal requires an array of at least two dimensions.");
  }
  const k = integer(offset, "The diagonal offset");
  const first = normalizeAxis(axis1, nd, "axis1");
  const second = normalizeAxis(axis2, nd, "axis2");
  if (first === second) {
    mismatch("diagonal axis1 and axis2 cannot be the same.");
  }
  const rest = a.shape
    .map((_: number, i: number) => i)
    .filter((i: number) => i !== first && i !== second);
  const order = [...rest, first, second];
  const moved = order.every((axis, i) => axis === i)
    ? a
    : a.transpose(order as never);
  const rows: number = a.shape[first];
  const cols: number = a.shape[second];
  const length = Math.max(
    0,
    k >= 0 ? Math.min(rows, cols - k) : Math.min(rows + k, cols),
  );
  const rowStart = Math.max(0, -k);
  const colStart = Math.max(0, k);
  // Two adjacent array selectors form one advanced block that stays last.
  return moved.get(
    ellipsis,
    indices(Array.from({ length }, (_, i) => rowStart + i)),
    indices(Array.from({ length }, (_, i) => colStart + i)),
  );
}

/**
 * `np.trace(a, offset, axis1, axis2)`: the sum along the diagonal with
 * NumPy's sum dtype promotion (bool and small integers become 64-bit).
 */
export function trace<
  D extends Exclude<ComputableDType, "str" | "datetime64">,
  S extends Shape,
  const A1 extends number = 0,
  const A2 extends number = 1,
>(
  a: Arg<D, S> & MatrixCheck<S>,
  offset?: number,
  axis1?: A1,
  axis2?: A2,
): Expr<SumDType<D>, TraceShape<S, A1, A2>>;
export function trace(
  a: Lazy<any, any>,
  offset = 0,
  axis1 = 0,
  axis2 = 1,
): Expr<any, any> {
  return diagonal(a as never, offset, axis1, axis2).reduceOp("sum", -1);
}

const INTEGER_RANGE: Record<string, readonly [bigint, bigint]> = {
  int8: [-(1n << 7n), (1n << 7n) - 1n],
  int16: [-(1n << 15n), (1n << 15n) - 1n],
  int32: [-(1n << 31n), (1n << 31n) - 1n],
  int64: [-(1n << 63n), (1n << 63n) - 1n],
  uint8: [0n, (1n << 8n) - 1n],
  uint16: [0n, (1n << 16n) - 1n],
  uint32: [0n, (1n << 32n) - 1n],
  uint64: [0n, (1n << 64n) - 1n],
};

/**
 * `np.clip(a, a_min, a_max)`. `null` or `undefined` means no bound. With both
 * bounds, the array and bounds first resolve one dtype as NumPy's three-input
 * `um.clip` does, then apply as `minimum(maximum(a, a_min), a_max)`, so NaN
 * propagates and `a_min > a_max` yields `a_max`. Weak JS scalars keep the
 * array dtype where NEP 50 allows; a `number` is a Python float and makes
 * integer input float64. As in NumPy, a `bigint` bound at or beyond the integer dtype's
 * limit is dropped; without bounds the result is `positive(a)`, which fails
 * for bool input.
 */
export function clip<
  D extends NumericDType,
  S extends Shape,
  const Lo extends ClipBound,
  const Hi extends ClipBound,
>(
  a: Arg<D, S>,
  a_min: Lo,
  a_max: Hi,
): Expr<ClipDType<D, Lo, Hi>, ClipShape<S, Lo, Hi>>;
export function clip(
  a: Lazy<any, any>,
  a_min: ClipBound,
  a_max: ClipBound,
): Expr<any, any> {
  requireLazy(a);
  let lo = a_min ?? undefined;
  let hi = a_max ?? undefined;
  const range = INTEGER_RANGE[a.dtype];
  if (range !== undefined) {
    // NumPy's _clip drops Python-integer bounds outside the dtype instead of
    // promoting them (NEP 50); bigint is the JS counterpart of a Python int.
    if (typeof lo === "bigint" && lo <= range[0]) lo = undefined;
    if (typeof hi === "bigint" && hi >= range[1]) hi = undefined;
  }
  if (lo === undefined && hi === undefined) {
    // NumPy returns positive(a). The cast only satisfies the conditional
    // receiver of pos(); the native node still rejects bool at runtime.
    return (a as Lazy<"float64", Shape>).pos();
  }
  if (lo === undefined) {
    return minimum(a, hi as never) as Expr<any, any>;
  }
  if (hi === undefined) {
    return maximum(a, lo as never) as Expr<any, any>;
  }
  const dtype = clipDType(a, lo, hi);
  const x = a.dtype === dtype ? a : a.astype(dtype);
  const lower = maximum(x, lo as never) as Expr<any, any>;
  return minimum(lower, hi as never) as never;
}

const WEAK_RANK: Record<string, number> = {
  boolean: 0,
  bigint: 1,
  number: 2,
};

/**
 * The common dtype of the three-input `um.clip`: array bounds promote with
 * the array, then weak scalar bounds apply from the highest kind down, so a
 * float bound widens an integer array before an integer bound is checked
 * against it. The probe expressions only carry dtypes and are never computed.
 */
function clipDType(
  a: Lazy<any, any>,
  lo: Exclude<ClipBound, null | undefined>,
  hi: Exclude<ClipBound, null | undefined>,
): ComputableDType {
  let probe: Lazy<any, any> = a;
  const scalars: Exclude<ClipBound, Lazy<any, any> | null | undefined>[] = [];
  for (const bound of [lo, hi]) {
    if (bound instanceof Lazy) probe = maximum(probe, bound as never);
    else scalars.push(bound);
  }
  const rank = (s: (typeof scalars)[number]) => WEAK_RANK[typeof s] ?? 3;
  scalars.sort((x, y) => rank(y) - rank(x));
  for (const scalar of scalars) probe = maximum(probe, scalar as never);
  return probe.dtype;
}

/**
 * `np.diff(a, n, axis)`: the `n`-th discrete difference along `axis` by
 * repeated slicing. Bool input uses `not_equal`, datetimes subtract to
 * timedeltas, and `n = 0` returns the input unchanged as a view. A negative
 * or non-integer `n` fails with `NMATH_INDEX_INVALID`; 0-d input fails with
 * `NMATH_SHAPE_MISMATCH` unless `n` is 0. `prepend` and `append` remain
 * targets.
 */
export function diff<
  D extends Exclude<ComputableDType, "str">,
  S extends Shape,
  const N extends number = 1,
>(
  a: Arg<D, S>,
  n?: N,
  axis?: number,
): Expr<N extends 0 ? D : DiffDType<D>, DiffShape<S, N>>;
export function diff(a: Lazy<any, any>, n = 1, axis = -1): Expr<any, any> {
  requireLazy(a);
  const order = integer(n, "The difference order n");
  if (order < 0) {
    throw new NMathError(
      "NMATH_INDEX_INVALID",
      `NMATH_INDEX_INVALID: The difference order n must be non-negative but got ${order}.`,
    );
  }
  if (order === 0) {
    return a.get(ellipsis);
  }
  const nd = a.ndim;
  if (nd === 0) {
    mismatch("diff requires input that is at least one dimensional.");
  }
  const ax = normalizeAxis(axis, nd, "axis");
  const later = Array.from({ length: nd }, (_, i) =>
    i === ax ? slice(1) : slice(),
  );
  const earlier = Array.from({ length: nd }, (_, i) =>
    i === ax ? slice(undefined, -1) : slice(),
  );
  const op = a.dtype === "bool" ? "ne" : "sub";
  let result: Lazy<any, any> = a;
  for (let i = 0; i < order; i += 1) {
    result = result.get(...later).binaryOp(op, result.get(...earlier));
  }
  return result as Expr<any, any>;
}
