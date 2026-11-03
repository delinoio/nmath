// SPDX-License-Identifier: Apache-2.0
// NumPy joining helpers. `concatenate` is one native lazy node; the other
// functions reshape their operands with the existing layout node and then
// concatenate them in the order NumPy 2.5.4's `shape_base` defines.

import { Expr, Lazy, type NMathValue } from "./core.js";
import { NMathError, native } from "./errors.js";
import { binding } from "./native.js";
import type { AxisOf, DynamicDims, ReplaceDim } from "./layout-types.js";
import type {
  ComputableDType,
  MutableShape,
  Promote,
  Shape,
  TemporalDType,
} from "./types.js";

// ---------------------------------------------------------------------------
// Static types. Literal axes against tuple shapes keep every dimension except
// the joined one, which becomes dynamic; other cases fall back to dynamic
// dimensions.

// `Lazy<any, any>` rejects some rank-0 values through method `this` types, so
// operands use the dtype- and shape-independent brand.
type Part = NMathValue & {
  readonly dtype: ComputableDType;
  readonly shape: Shape;
};
/** A non-empty list of operands to join. */
export type JoinOperands = readonly [Part, ...Part[]] | readonly Part[];
type DTypeOf<P> = P extends { readonly dtype: infer D extends ComputableDType }
  ? D
  : never;
type ShapeOf<P> = P extends { readonly shape: infer S extends Shape }
  ? S
  : never;
type Insert<
  S extends Shape,
  I extends number,
  N extends number,
  Out extends number[] = [],
> = Out["length"] extends I
  ? [...Out, N, ...MutableShape<S>]
  : S extends readonly [infer H extends number, ...infer R extends Shape]
    ? Insert<R, I, N, [...Out, H]>
    : never;
type Count<P extends readonly unknown[]> = number extends P["length"]
  ? number
  : P["length"];

/** Temporal dtypes join only with the same dtype; others use `Promote`. */
type Join2<
  A extends ComputableDType,
  B extends ComputableDType,
> = A extends TemporalDType ? (B extends A ? A : never) : Promote<A, B>;
/**
 * NumPy's legacy type-number order, highest first. Pairwise promotion is not
 * associative, so NumPy promotes every operand with the highest-ordered
 * ("main") dtype and combines those results.
 */
type MainOrder = [
  "float16",
  "timedelta64",
  "datetime64",
  "str",
  "complex128",
  "complex64",
  "float64",
  "float32",
  "uint64",
  "int64",
  "uint32",
  "int32",
  "uint16",
  "int16",
  "uint8",
  "int8",
  "bool",
];
type MainDType<D, O = MainOrder> = O extends readonly [infer H, ...infer R]
  ? H extends D
    ? H
    : MainDType<D, R>
  : never;
/** One tuple of single dtypes per combination of union operand dtypes. */
type DTypeAlternatives<P extends readonly Part[]> = P extends readonly [
  infer H,
  ...infer R extends Part[],
]
  ? DTypeOf<H> extends infer D
    ? D extends ComputableDType
      ? [D, ...DTypeAlternatives<R>]
      : never
    : never
  : [];
type JoinFold<
  T extends readonly ComputableDType[],
  M extends ComputableDType,
  Acc extends ComputableDType = M,
> = T extends readonly [
  infer H extends ComputableDType,
  ...infer R extends ComputableDType[],
]
  ? JoinFold<R, M, Join2<Acc, Join2<M, H>>>
  : Acc;
type JoinAlternative<T> = T extends readonly ComputableDType[]
  ? JoinFold<T, MainDType<T[number]>>
  : never;
/** The NumPy `result_type` of every operand, `never` when they cannot join. */
export type JoinDType<P extends readonly Part[]> = number extends P["length"]
  ? DTypeOf<P[number]>
  : JoinAlternative<DTypeAlternatives<P>>;
type JoinCheck<P extends readonly Part[]> = [JoinDType<P>] extends [never]
  ? { readonly "operand dtypes cannot be joined": P }
  : unknown;
/** Rejects a statically invalid axis or a zero-dimensional operand. */
type ShapeCheck<S> = [S] extends [never]
  ? { readonly "the axis or operand rank is invalid": true }
  : unknown;

/** `np.concatenate` shape: the joined axis is dynamic; `null` flattens. */
export type ConcatenateShape<
  P extends readonly Part[],
  A extends number | null,
> = A extends null
  ? [number]
  : ShapeOf<P[0]> extends infer S extends Shape
    ? number extends S["length"]
      ? number[]
      : S extends readonly []
        ? never
        : AxisOf<S, A & number> extends infer I extends number
          ? [I] extends [never]
            ? never
            : number extends I
              ? DynamicDims<S>
              : ReplaceDim<S, I>
          : never
    : never;

/**
 * `np.stack` shape: inserts the operand count at the normalized axis. A union
 * axis gives the union of each alternative's shape.
 */
export type StackShape<
  P extends readonly Part[],
  A extends number,
> = A extends unknown
  ? ShapeOf<P[0]> extends infer S extends Shape
    ? number extends S["length"]
      ? number[]
      : AxisOf<[...S, number], A> extends infer I extends number
        ? [I] extends [never]
          ? never
          : number extends I
            ? [number, ...DynamicDims<S>]
            : Insert<S, I, Count<P>>
        : never
    : never
  : never;

/** `np.atleast_1d` shape. */
export type Atleast1DShape<S extends Shape> = number extends S["length"]
  ? number[]
  : S extends readonly []
    ? [1]
    : MutableShape<S>;
/** `np.atleast_2d` shape: a vector becomes one row. */
export type Atleast2DShape<S extends Shape> = number extends S["length"]
  ? number[]
  : S extends readonly []
    ? [1, 1]
    : S extends readonly [infer N extends number]
      ? [1, N]
      : MutableShape<S>;
/** `np.atleast_3d` shape: a vector `(N,)` becomes `(1, N, 1)` and a matrix `(M, N)` becomes `(M, N, 1)`. */
export type Atleast3DShape<S extends Shape> = number extends S["length"]
  ? number[]
  : S extends readonly []
    ? [1, 1, 1]
    : S extends readonly [infer N extends number]
      ? [1, N, 1]
      : S extends readonly [infer M extends number, infer N extends number]
        ? [M, N, 1]
        : MutableShape<S>;
type ColumnShape<S extends Shape> = number extends S["length"]
  ? number[]
  : S extends readonly []
    ? [1, 1]
    : S extends readonly [infer N extends number]
      ? [N, 1]
      : MutableShape<S>;

type JoinedAt<S extends Shape, I extends number> = number extends S["length"]
  ? number[]
  : ReplaceDim<S, I> extends infer R extends Shape
    ? R
    : never;
/** `np.vstack` shape: operands become at least 2-D and join along axis 0. */
export type VStackShape<P extends readonly Part[]> = JoinedAt<
  Atleast2DShape<ShapeOf<P[0]>>,
  0
>;
/** `np.hstack` shape: 1-D operands join along axis 0, others along axis 1. */
export type HStackShape<P extends readonly Part[]> =
  Atleast1DShape<ShapeOf<P[0]>> extends infer S extends Shape
    ? S extends readonly [number]
      ? [number]
      : JoinedAt<S, 1>
    : never;
/** `np.dstack` shape: operands become at least 3-D and join along axis 2. */
export type DStackShape<P extends readonly Part[]> = JoinedAt<
  Atleast3DShape<ShapeOf<P[0]>>,
  2
>;
/** `np.column_stack` shape: 0-d and 1-D operands become columns joined along axis 1. */
export type ColumnStackShape<P extends readonly Part[]> = JoinedAt<
  ColumnShape<ShapeOf<P[0]>>,
  1
>;

type Atleast<
  P extends readonly Part[],
  F extends "1d" | "2d" | "3d",
> = P extends readonly [infer One]
  ? AtleastOne<One, F>
  : { readonly [K in keyof P]: AtleastOne<P[K], F> };
type AtleastOne<P, F extends "1d" | "2d" | "3d"> = Expr<
  DTypeOf<P>,
  F extends "1d"
    ? Atleast1DShape<ShapeOf<P>>
    : F extends "2d"
      ? Atleast2DShape<ShapeOf<P>>
      : Atleast3DShape<ShapeOf<P>>
>;

// ---------------------------------------------------------------------------
// Runtime helpers

/** The layout method each helper composes, without its static checks. */
interface Layout {
  readonly shape: readonly number[];
  reshape(shape: readonly number[]): Expr<any, any>;
}

function operands(arrays: unknown, name: string): Lazy<any, any>[] {
  if (!Array.isArray(arrays)) {
    throw new TypeError(`${name} requires an array of NMath values.`);
  }
  for (const value of arrays) {
    if (!(value instanceof Lazy)) {
      throw new TypeError(
        `${name} requires NMath arrays or expressions as operands.`,
      );
    }
  }
  return arrays;
}

function reshape(a: Lazy<any, any>, shape: readonly number[]): Expr<any, any> {
  return (a as unknown as Layout).reshape(shape);
}

function axisInteger(axis: unknown): number {
  if (typeof axis !== "number" || !Number.isSafeInteger(axis)) {
    throw new NMathError(
      "NMATH_INDEX_INVALID",
      "NMATH_INDEX_INVALID: An axis must be an integer. Use an index inside the axis length or a valid selector.",
    );
  }
  return axis;
}

function shapeError(message: string): NMathError {
  return new NMathError(
    "NMATH_SHAPE_MISMATCH",
    `NMATH_SHAPE_MISMATCH: ${message} Check the operand shapes against the operation requirements.`,
  );
}

/** Joins along a validated integer axis with the native concatenate node. */
function join(parts: readonly Lazy<any, any>[], axis: number): Expr<any, any> {
  const context = parts.find((part) => part.context !== undefined)?.context;
  return new Expr(
    native(() =>
      binding.NativeExpr.concat(
        parts.map((part) => part.nativeExpr()),
        axis,
      ),
    ),
    context,
  );
}

function atleast1d(a: Lazy<any, any>): Expr<any, any> {
  return reshape(a, a.ndim === 0 ? [1] : a.shape);
}

function atleast2d(a: Lazy<any, any>): Expr<any, any> {
  if (a.ndim === 0) return reshape(a, [1, 1]);
  if (a.ndim === 1) return reshape(a, [1, a.shape[0]]);
  return reshape(a, a.shape);
}

function atleast3d(a: Lazy<any, any>): Expr<any, any> {
  if (a.ndim === 0) return reshape(a, [1, 1, 1]);
  if (a.ndim === 1) return reshape(a, [1, a.shape[0], 1]);
  if (a.ndim === 2) return reshape(a, [...a.shape, 1]);
  return reshape(a, a.shape);
}

function each(
  arys: unknown[],
  name: string,
  convert: (a: Lazy<any, any>) => Expr<any, any>,
): Expr<any, any> | readonly Expr<any, any>[] {
  const results = operands(arys, name).map(convert);
  return results.length === 1 ? results[0] : Object.freeze(results);
}

// ---------------------------------------------------------------------------
// Public functions

/**
 * `np.concatenate(arrays, axis=0)`: joins the operands along an existing
 * axis. The operands promote to one dtype and must match every dimension
 * except `axis`. `axis: null` flattens every operand first. The result is a
 * new array, never a view.
 */
export function concatenate<
  const P extends JoinOperands,
  const A extends number | null = 0,
>(
  arrays: P & JoinCheck<P> & ShapeCheck<ConcatenateShape<P, A>>,
  axis?: A,
): Expr<JoinDType<P>, ConcatenateShape<P, A>>;
export function concatenate(
  arrays: readonly Lazy<any, any>[],
  axis: number | null = 0,
): Expr<any, any> {
  const parts = operands(arrays, "concatenate");
  if (axis === null) {
    return join(
      parts.map((part) => reshape(part, [-1])),
      0,
    );
  }
  return join(parts, axisInteger(axis));
}

/**
 * `np.stack(arrays, axis=0)`: joins operands of one shape along a new axis.
 * `axis` refers to a position in the result, whose rank is one higher.
 */
export function stack<const P extends JoinOperands, const A extends number = 0>(
  arrays: P & JoinCheck<P> & ShapeCheck<StackShape<P, A>>,
  axis?: A,
): Expr<JoinDType<P>, StackShape<P, A>>;
export function stack(
  arrays: readonly Lazy<any, any>[],
  axis: number = 0,
): Expr<any, any> {
  const parts = operands(arrays, "stack");
  if (parts.length === 0) {
    throw shapeError("stack needs at least one array.");
  }
  const shape = parts[0].shape;
  if (
    parts.some(
      (part) =>
        part.ndim !== shape.length ||
        part.shape.some((n: number, i: number) => n !== shape[i]),
    )
  ) {
    throw shapeError("All input arrays to stack must have the same shape.");
  }
  const rank = shape.length + 1;
  const value = axisInteger(axis);
  if (value < -rank || value >= rank) {
    throw new NMathError(
      "NMATH_INDEX_OUT_OF_BOUNDS",
      `NMATH_INDEX_OUT_OF_BOUNDS: Axis ${value} is out of bounds for an array of dimension ${rank}. Use an index inside the axis length or a valid selector.`,
    );
  }
  const normalized = value < 0 ? value + rank : value;
  const expanded = [
    ...shape.slice(0, normalized),
    1,
    ...shape.slice(normalized),
  ];
  return join(
    parts.map((part) => reshape(part, expanded)),
    normalized,
  );
}

/** `np.vstack(tup)`: makes every operand at least 2-D and joins along axis 0. */
export function vstack<const P extends JoinOperands>(
  tup: P & JoinCheck<P>,
): Expr<JoinDType<P>, VStackShape<P>>;
export function vstack(tup: readonly Lazy<any, any>[]): Expr<any, any> {
  return join(operands(tup, "vstack").map(atleast2d), 0);
}

/**
 * `np.hstack(tup)`: makes every operand at least 1-D and joins along axis 1,
 * or along axis 0 when the first operand is 1-D.
 */
export function hstack<const P extends JoinOperands>(
  tup: P & JoinCheck<P>,
): Expr<JoinDType<P>, HStackShape<P>>;
export function hstack(tup: readonly Lazy<any, any>[]): Expr<any, any> {
  const parts = operands(tup, "hstack").map(atleast1d);
  return join(parts, parts.length > 0 && parts[0].ndim === 1 ? 0 : 1);
}

/** `np.dstack(tup)`: makes every operand at least 3-D and joins along axis 2. */
export function dstack<const P extends JoinOperands>(
  tup: P & JoinCheck<P>,
): Expr<JoinDType<P>, DStackShape<P>>;
export function dstack(tup: readonly Lazy<any, any>[]): Expr<any, any> {
  return join(operands(tup, "dstack").map(atleast3d), 2);
}

/**
 * `np.column_stack(tup)`: 0-d and 1-D operands become columns, and every
 * operand joins along axis 1.
 */
export function column_stack<const P extends JoinOperands>(
  tup: P & JoinCheck<P>,
): Expr<JoinDType<P>, ColumnStackShape<P>>;
export function column_stack(tup: readonly Lazy<any, any>[]): Expr<any, any> {
  return join(
    operands(tup, "column_stack").map((part) =>
      part.ndim === 0
        ? reshape(part, [1, 1])
        : part.ndim === 1
          ? reshape(part, [part.shape[0], 1])
          : part,
    ),
    1,
  );
}

/**
 * `np.append(arr, values, axis=None)`: without an axis both operands are
 * flattened and joined; with an axis this is `concatenate([arr, values], axis)`.
 */
export function append<
  A extends ComputableDType,
  AS extends Shape,
  V extends ComputableDType,
  VS extends Shape,
  const X extends number | null = null,
>(
  arr: Lazy<NoInfer<A>, AS> & { readonly dtype: A },
  values: Lazy<NoInfer<V>, VS> & { readonly dtype: V } & JoinCheck<
      [Part & { readonly dtype: A }, Part & { readonly dtype: V }]
    >,
  axis?: X &
    ShapeCheck<
      ConcatenateShape<
        [Part & { readonly shape: AS }, Part & { readonly shape: VS }],
        X
      >
    >,
): Expr<
  Join2<A, V>,
  ConcatenateShape<
    [Part & { readonly shape: AS }, Part & { readonly shape: VS }],
    X
  >
>;
export function append(
  arr: Lazy<any, any>,
  values: Lazy<any, any>,
  axis: number | null = null,
): Expr<any, any> {
  return concatenate([arr, values] as never, axis as never);
}

/**
 * `np.atleast_1d(*arys)`: 0-d operands become shape `(1,)`. One operand
 * returns one expression; several return a frozen tuple.
 */
export function atleast_1d<const P extends readonly [Part, ...Part[]]>(
  ...arys: P
): Atleast<P, "1d">;
export function atleast_1d(...arys: Lazy<any, any>[]): unknown {
  return each(arys, "atleast_1d", atleast1d);
}

/** `np.atleast_2d(*arys)`: 0-d becomes `(1, 1)` and 1-D `(N,)` becomes `(1, N)`. */
export function atleast_2d<const P extends readonly [Part, ...Part[]]>(
  ...arys: P
): Atleast<P, "2d">;
export function atleast_2d(...arys: Lazy<any, any>[]): unknown {
  return each(arys, "atleast_2d", atleast2d);
}

/**
 * `np.atleast_3d(*arys)`: 0-d becomes `(1, 1, 1)`, 1-D `(N,)` becomes
 * `(1, N, 1)` and 2-D `(M, N)` becomes `(M, N, 1)`.
 */
export function atleast_3d<const P extends readonly [Part, ...Part[]]>(
  ...arys: P
): Atleast<P, "3d">;
export function atleast_3d(...arys: Lazy<any, any>[]): unknown {
  return each(arys, "atleast_3d", atleast3d);
}
