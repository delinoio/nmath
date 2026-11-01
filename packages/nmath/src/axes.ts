// SPDX-License-Identifier: Apache-2.0
// NumPy axis-manipulation helpers. Each function composes the existing lazy
// layout nodes (`reshape`, `transpose` and slice reads) in the order NumPy
// 2.5.4 uses, so results keep the dtype, context and snapshot behavior of
// those nodes. Axis validation runs when the expression is built.

import { Expr, type Lazy, type ReadSelector, slice } from "./core.js";
import { NMathError, native } from "./errors.js";
import { binding } from "./native.js";
import type { ComputableDType, MutableShape, Shape } from "./types.js";

// ---------------------------------------------------------------------------
// Static shape helpers. Literal integer axes against tuple shapes give exact
// result shapes; non-literal axes or shapes fall back to dynamic dimensions.

type Ranks = readonly unknown[];
type RankOf<T extends Ranks> = { -readonly [K in keyof T]: 0 };
type DynamicDims<S extends Shape> = { -readonly [K in keyof S]: number };
type IsUnion<T, U = T> = T extends T ? ([U] extends [T] ? false : true) : never;
type NegativeAxis<
  T extends Ranks,
  P extends number,
  C extends 0[] = [],
> = C["length"] extends P
  ? T["length"]
  : T extends readonly [...infer H, unknown]
    ? NegativeAxis<H, P, [...C, 0]>
    : never;
type PositiveAxis<
  T extends Ranks,
  A extends number,
  I extends 0[] = [],
> = T extends readonly [unknown, ...infer R]
  ? I["length"] extends A
    ? A
    : PositiveAxis<R, A, [...I, 0]>
  : never;
/**
 * The normalized index of axis `A` for rank `T["length"]`: `never` when the
 * literal is out of range or not an integer, `number` when it is not literal.
 */
type AxisIndex<T extends Ranks, A extends number> = number extends A
  ? number
  : true extends IsUnion<A>
    ? number
    : `${A}` extends `-${infer P extends number}`
      ? `${P}` extends `${bigint}`
        ? NegativeAxis<T, P>
        : never
      : `${A}` extends `${bigint}`
        ? PositiveAxis<T, A>
        : never;
/** Normalizes every axis; `never` if one is invalid, `number[]` if the list is dynamic. */
type AxisList<
  T extends Ranks,
  A extends readonly number[],
  Out extends number[] = [],
> = A extends readonly [infer H extends number, ...infer R extends number[]]
  ? AxisIndex<T, H> extends infer N extends number
    ? [N] extends [never]
      ? never
      : AxisList<T, R, [...Out, N]>
    : never
  : number extends A["length"]
    ? number[]
    : Out;
type HasRepeat<A extends readonly number[]> = A extends readonly [
  infer H,
  ...infer R extends number[],
]
  ? H extends R[number]
    ? true
    : HasRepeat<R>
  : false;
type IsDynamic<L extends readonly number[]> = number extends L["length"]
  ? true
  : number extends L[number]
    ? true
    : false;
type ListOf<A> = A extends readonly number[] ? A : A extends number ? [A] : [];
type Valid<R> = [R] extends [never] ? never : unknown;

type DropOnes<S extends Shape> = S extends readonly [
  infer H extends number,
  ...infer R extends Shape,
]
  ? H extends 1
    ? DropOnes<R>
    : [H, ...DropOnes<R>]
  : [];
type DropAxes<
  S extends Shape,
  Axes extends number,
  I extends 0[] = [],
> = S extends readonly [infer H extends number, ...infer R extends Shape]
  ? I["length"] extends Axes
    ? number extends H
      ? DropAxes<R, Axes, [...I, 0]>
      : H extends 1
        ? DropAxes<R, Axes, [...I, 0]>
        : never
    : DropAxes<R, Axes, [...I, 0]> extends infer Rest extends number[]
      ? [Rest] extends [never]
        ? never
        : [H, ...Rest]
      : never
  : [];

/**
 * `np.squeeze` result shape; selecting a literal non-1 dimension is rejected.
 * An axis type that admits `undefined` yields both the omitted-axis and the
 * explicit-axis alternatives.
 */
export type SqueezeShape<
  S extends Shape,
  A extends number | readonly number[] | undefined = undefined,
> = number extends S["length"]
  ? number[]
  : undefined extends A
    ? [A] extends [undefined]
      ? SqueezeAllShape<S>
      : SqueezeAllShape<S> | SqueezeAxesShape<S, Exclude<A, undefined>>
    : SqueezeAxesShape<S, A>;
type SqueezeAllShape<S extends Shape> = number extends S[number]
  ? number[]
  : DropOnes<S>;
// Each alternative of a union axis type is a separate selection.
type SqueezeAxesShape<
  S extends Shape,
  A extends number | readonly number[] | undefined,
> = A extends unknown ? SqueezeAxesOne<S, A> : never;
type SqueezeAxesOne<
  S extends Shape,
  A extends number | readonly number[] | undefined,
> = S extends readonly []
  ? // NumPy accepts the integer axes 0 and -1 for zero-dimensional input.
    A extends number
    ? number extends A
      ? []
      : A extends 0 | -1
        ? []
        : never
    : AxisList<[], ListOf<A>> extends infer L extends number[]
      ? [L] extends [never]
        ? never
        : []
      : never
  : AxisList<RankOf<S>, ListOf<A>> extends infer L extends number[]
    ? [L] extends [never]
      ? never
      : IsDynamic<L> extends true
        ? number[]
        : HasRepeat<L> extends true
          ? never
          : DropAxes<S, L[number]>
    : never;

type InsertOnes<
  S extends Shape,
  Axes extends number,
  Remaining extends Ranks,
  I extends 0[] = [],
> = Remaining extends readonly [unknown, ...infer Rest]
  ? I["length"] extends Axes
    ? InsertOnes<S, Axes, Rest, [...I, 0]> extends infer R extends number[]
      ? [R] extends [never]
        ? never
        : [1, ...R]
      : never
    : S extends readonly [infer H extends number, ...infer T extends Shape]
      ? InsertOnes<T, Axes, Rest, [...I, 0]> extends infer R extends number[]
        ? [R] extends [never]
          ? never
          : [H, ...R]
        : never
      : never
  : [];
type ExpandRanks<S extends Shape, L extends readonly number[]> = [
  ...RankOf<S>,
  ...RankOf<L>,
];

/**
 * `np.expand_dims` result shape; axes normalize against the output rank. Each
 * alternative of a union axis type is a separate selection.
 */
export type ExpandDimsShape<
  S extends Shape,
  A extends number | readonly number[],
> = A extends unknown ? ExpandDimsOne<S, A> : never;
type ExpandDimsOne<
  S extends Shape,
  A extends number | readonly number[],
> = number extends S["length"]
  ? number[]
  : number extends ListOf<A>["length"]
    ? number[]
    : AxisList<ExpandRanks<S, ListOf<A>>, ListOf<A>> extends infer L extends
          number[]
      ? [L] extends [never]
        ? never
        : IsDynamic<L> extends true
          ? number[]
          : HasRepeat<L> extends true
            ? never
            : InsertOnes<S, L[number], ExpandRanks<S, ListOf<A>>>
      : never;

type Swap<S extends Shape, N1 extends number, N2 extends number> = {
  -readonly [K in keyof S]: K extends `${N1}`
    ? S[N2]
    : K extends `${N2}`
      ? S[N1]
      : S[K];
};

/** `np.swapaxes` result shape. */
export type SwapAxesShape<
  S extends Shape,
  A1 extends number,
  A2 extends number,
> = number extends S["length"]
  ? number[]
  : SwapKnown<S, AxisIndex<RankOf<S>, A1>, AxisIndex<RankOf<S>, A2>>;
type SwapKnown<S extends Shape, N1 extends number, N2 extends number> = [
  N1,
] extends [never]
  ? never
  : [N2] extends [never]
    ? never
    : number extends N1 | N2
      ? DynamicDims<S>
      : Swap<S, N1, N2>;

type KeepAxes<
  S extends Shape,
  Axes extends number,
  I extends 0[] = [],
> = S extends readonly [infer H extends number, ...infer R extends Shape]
  ? I["length"] extends Axes
    ? KeepAxes<R, Axes, [...I, 0]>
    : [H, ...KeepAxes<R, Axes, [...I, 0]>]
  : [];
type Lookup<
  Dst extends readonly number[],
  Src extends readonly number[],
  I extends number,
> = Dst extends readonly [infer H, ...infer DR extends number[]]
  ? Src extends readonly [infer SH extends number, ...infer SR extends number[]]
    ? H extends I
      ? SH
      : Lookup<DR, SR, I>
    : never
  : never;
type Place<
  S extends Shape,
  Src extends readonly number[],
  Dst extends readonly number[],
  Rest extends Shape,
  Remaining extends Ranks,
  I extends 0[] = [],
> = Remaining extends readonly [unknown, ...infer More]
  ? I["length"] extends Dst[number]
    ? [
        S[Lookup<Dst, Src, I["length"]>],
        ...Place<S, Src, Dst, Rest, More, [...I, 0]>,
      ]
    : Rest extends readonly [infer H extends number, ...infer T extends Shape]
      ? [H, ...Place<S, Src, Dst, T, More, [...I, 0]>]
      : []
  : [];
type MoveKnown<S extends Shape, Src extends number[], Dst extends number[]> = [
  Src,
] extends [never]
  ? never
  : [Dst] extends [never]
    ? never
    : IsDynamic<Src> extends true
      ? DynamicDims<S>
      : IsDynamic<Dst> extends true
        ? DynamicDims<S>
        : HasRepeat<Src> extends true
          ? never
          : HasRepeat<Dst> extends true
            ? never
            : Src["length"] extends Dst["length"]
              ? Place<S, Src, Dst, KeepAxes<S, Src[number]>, RankOf<S>>
              : never;

/** `np.moveaxis` result shape; union axis types distribute like the others. */
export type MoveAxisShape<
  S extends Shape,
  Src extends number | readonly number[],
  Dst extends number | readonly number[],
> = Src extends unknown
  ? Dst extends unknown
    ? MoveAxisOne<S, Src, Dst>
    : never
  : never;
type MoveAxisOne<
  S extends Shape,
  Src extends number | readonly number[],
  Dst extends number | readonly number[],
> = number extends S["length"]
  ? number[]
  : MoveKnown<
      S,
      AxisList<RankOf<S>, ListOf<Src>>,
      AxisList<RankOf<S>, ListOf<Dst>>
    >;

/** Static axis validation for `np.flip`; the shape itself never changes. */
export type FlipCheck<
  S extends Shape,
  A extends number | readonly number[] | undefined,
> = A extends unknown ? FlipCheckOne<S, A> : never;
type FlipCheckOne<
  S extends Shape,
  A extends number | readonly number[] | undefined,
> = [A] extends [undefined]
  ? unknown
  : number extends S["length"]
    ? unknown
    : AxisList<RankOf<S>, ListOf<A>> extends infer L extends number[]
      ? [L] extends [never]
        ? never
        : IsDynamic<L> extends true
          ? unknown
          : HasRepeat<L> extends true
            ? never
            : unknown
      : never;

/** Rejects tuple shapes with fewer than `N` dimensions. */
export type MinRankCheck<
  S extends Shape,
  N extends 1 | 2,
  Name extends string,
> = number extends S["length"]
  ? unknown
  : S extends readonly [number, ...(N extends 2 ? [number] : []), ...number[]]
    ? unknown
    : { readonly [K in `${Name} requires at least ${N} dimensions`]: S };

// ---------------------------------------------------------------------------
// Runtime validation, following `normalize_axis_index` and
// `normalize_axis_tuple` with the error codes the native reductions use.

type Axes = number | readonly number[];

/** The layout methods each helper composes, without their static checks. */
interface Layout {
  readonly shape: readonly number[];
  reshape(shape: readonly number[]): Expr<any, any>;
  transpose(axes?: readonly number[]): Expr<any, any>;
  get(...selectors: ReadSelector[]): Expr<any, any>;
}

function layout(a: Lazy<any, any>): Layout {
  return a as unknown as Layout;
}

const HINT = "Use an index inside the axis length or a valid selector.";
const SHAPE_HINT =
  "Check the operand shapes against the operation requirements.";

function normalizeAxis(axis: unknown, rank: number, name?: string): number {
  if (typeof axis !== "number" || !Number.isSafeInteger(axis)) {
    throw new NMathError(
      "NMATH_INDEX_INVALID",
      `NMATH_INDEX_INVALID: ${name === undefined ? "An axis" : `Axis ${name}`} must be an integer. ${HINT}`,
    );
  }
  if (axis < -rank || axis >= rank) {
    const where = name === undefined ? "" : ` in ${name}`;
    throw new NMathError(
      "NMATH_INDEX_OUT_OF_BOUNDS",
      `NMATH_INDEX_OUT_OF_BOUNDS: Axis ${axis}${where} is out of bounds for an array of dimension ${rank}. ${HINT}`,
    );
  }
  return axis < 0 ? axis + rank : axis;
}

function axisValues(axis: unknown, name?: string): readonly unknown[] {
  if (Array.isArray(axis)) return axis;
  if (typeof axis === "number") return [axis];
  throw new NMathError(
    "NMATH_INDEX_INVALID",
    `NMATH_INDEX_INVALID: ${name === undefined ? "An axis" : `Axis ${name}`} must be an integer or an array of integers. ${HINT}`,
  );
}

function repeated(axis: unknown, name?: string): NMathError {
  const where = name === undefined ? "" : ` in ${name}`;
  return new NMathError(
    "NMATH_INDEX_INVALID",
    `NMATH_INDEX_INVALID: Axis ${String(axis)}${where} is repeated. ${HINT}`,
  );
}

/** NumPy `normalize_axis_tuple`: every axis is range-checked before repeats. */
function normalizeAxes(axis: unknown, rank: number, name?: string): number[] {
  const values = axisValues(axis, name);
  const axes = values.map((value) => normalizeAxis(value, rank, name));
  axes.forEach((value, i) => {
    if (axes.indexOf(value) !== i) throw repeated(values[i], name);
  });
  return axes;
}

function shapeError(message: string): NMathError {
  return new NMathError(
    "NMATH_SHAPE_MISMATCH",
    `NMATH_SHAPE_MISMATCH: ${message} ${SHAPE_HINT}`,
  );
}

// ---------------------------------------------------------------------------
// Public functions

/**
 * `np.squeeze(a, axis)`: removes length-1 axes, or only the selected ones.
 * Selecting an axis whose length is not 1 fails with `NMATH_SHAPE_MISMATCH`.
 */
export function squeeze<D extends ComputableDType, S extends Shape>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
): Expr<D, SqueezeShape<S>>;
// A required `axis` keeps `undefined` in `A` when the argument may be
// undefined; an optional parameter would strip it during inference.
export function squeeze<
  D extends ComputableDType,
  S extends Shape,
  const A extends Axes | undefined,
>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis: A & Valid<SqueezeShape<S, A>>,
): Expr<D, SqueezeShape<S, A>>;
export function squeeze(a: Lazy<any, any>, axis?: Axes): Expr<any, any> {
  const source = layout(a);
  const shape = source.shape;
  const rank = shape.length;
  if (axis === undefined) {
    return source.reshape(shape.filter((n) => n !== 1));
  }
  // NumPy's `PyArray_ConvertMultiAxis` lets a scalar integer axis of 0 or -1
  // through for zero-dimensional input for backwards compatibility. Remove
  // this branch if NumPy drops that legacy case.
  if (rank === 0 && (axis === 0 || axis === -1)) {
    return source.reshape([]);
  }
  // `PyArray_ConvertMultiAxis` checks range and repeats one axis at a time.
  const selected = new Set<number>();
  for (const value of axisValues(axis)) {
    const normalized = normalizeAxis(value, rank);
    if (selected.has(normalized)) throw repeated(value);
    selected.add(normalized);
  }
  for (const index of selected) {
    if (shape[index] !== 1) {
      throw shapeError(
        `Cannot squeeze axis ${index} with length ${shape[index]}; only length-1 axes can be removed.`,
      );
    }
  }
  return source.reshape(shape.filter((_, i) => !selected.has(i)));
}

/**
 * `np.expand_dims(a, axis)`: inserts length-1 axes. Axes refer to positions
 * in the result, whose rank is the input rank plus the number of axes.
 */
export function expand_dims<
  D extends ComputableDType,
  S extends Shape,
  const A extends Axes,
>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis: A & Valid<ExpandDimsShape<S, A>>,
): Expr<D, ExpandDimsShape<S, A>> {
  const source = layout(a);
  const values = axisValues(axis);
  const rank = source.shape.length + values.length;
  const inserted = new Set(normalizeAxes(values, rank));
  const dims = source.shape[Symbol.iterator]();
  return source.reshape(
    Array.from({ length: rank }, (_, i) =>
      inserted.has(i) ? 1 : (dims.next().value as number),
    ),
  );
}

/** `np.swapaxes(a, axis1, axis2)`: exchanges two axes. */
export function swapaxes<
  D extends ComputableDType,
  S extends Shape,
  const A1 extends number,
  const A2 extends number,
>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis1: A1 & Valid<SwapAxesShape<S, A1, A2>>,
  axis2: A2 & Valid<SwapAxesShape<S, A1, A2>>,
): Expr<D, SwapAxesShape<S, A1, A2>> {
  const source = layout(a);
  const rank = source.shape.length;
  const first = normalizeAxis(axis1, rank, "axis1");
  const second = normalizeAxis(axis2, rank, "axis2");
  const order = Array.from({ length: rank }, (_, i) =>
    i === first ? second : i === second ? first : i,
  );
  return source.transpose(order);
}

/**
 * `np.moveaxis(a, source, destination)`: moves axes to new positions; the
 * other axes keep their relative order.
 */
export function moveaxis<
  D extends ComputableDType,
  S extends Shape,
  const Src extends Axes,
  const Dst extends Axes,
>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  source: Src & Valid<MoveAxisShape<S, Src, Dst>>,
  destination: Dst & Valid<MoveAxisShape<S, Src, Dst>>,
): Expr<D, MoveAxisShape<S, Src, Dst>> {
  const input = layout(a);
  const rank = input.shape.length;
  const from = normalizeAxes(source, rank, "source");
  const to = normalizeAxes(destination, rank, "destination");
  if (from.length !== to.length) {
    throw new NMathError(
      "NMATH_INDEX_INVALID",
      `NMATH_INDEX_INVALID: source and destination must have the same number of axes. ${HINT}`,
    );
  }
  // NumPy inserts each source axis at its destination in ascending order.
  const order = Array.from({ length: rank }, (_, i) => i).filter(
    (i) => !from.includes(i),
  );
  to.map((dest, i) => [dest, from[i]] as const)
    .sort(([x], [y]) => x - y)
    .forEach(([dest, src]) => order.splice(dest, 0, src));
  return input.transpose(order);
}

/** `np.ravel(a)`: the C-order one-dimensional view or copy of `a`. */
export function ravel<D extends ComputableDType, S extends Shape>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  order: "C" = "C",
): Expr<D, [number]> {
  if (order !== "C")
    throw new TypeError("Ravel currently supports order 'C' only.");
  // NumPy returns a contiguous result: a view of a C-contiguous input and a
  // copy otherwise, which a plain reshape would not guarantee.
  return new Expr(
    native(() => binding.NativeExpr.ravel(a.nativeExpr())),
    a.context,
  );
}

/**
 * `np.flip(a, axis)`: reverses the selected axes, or every axis when `axis`
 * is omitted, with negative-step slice reads.
 */
export function flip<
  D extends ComputableDType,
  S extends Shape,
  const A extends Axes | undefined = undefined,
>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis?: A & FlipCheck<S, A>,
): Expr<D, MutableShape<S>> {
  const source = layout(a);
  const rank = source.shape.length;
  const reversed =
    axis === undefined
      ? new Set(source.shape.map((_, i) => i))
      : new Set(normalizeAxes(axis, rank));
  return source.get(
    ...source.shape.map((_, i) =>
      reversed.has(i) ? slice(undefined, undefined, -1) : slice(),
    ),
  );
}

/** `np.fliplr(m)`: reverses axis 1. The input needs at least two dimensions. */
export function fliplr<D extends ComputableDType, S extends Shape>(
  m: Lazy<NoInfer<D>, S> & { readonly dtype: D } & MinRankCheck<S, 2, "fliplr">,
): Expr<D, MutableShape<S>> {
  const source = layout(m);
  if (source.shape.length < 2)
    throw shapeError("fliplr requires an input with at least 2 dimensions.");
  return source.get(slice(), slice(undefined, undefined, -1));
}

/** `np.flipud(m)`: reverses axis 0. The input needs at least one dimension. */
export function flipud<D extends ComputableDType, S extends Shape>(
  m: Lazy<NoInfer<D>, S> & { readonly dtype: D } & MinRankCheck<S, 1, "flipud">,
): Expr<D, MutableShape<S>> {
  const source = layout(m);
  if (source.shape.length < 1)
    throw shapeError("flipud requires an input with at least 1 dimension.");
  return source.get(slice(undefined, undefined, -1));
}
