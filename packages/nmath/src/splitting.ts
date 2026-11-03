// SPDX-License-Identifier: Apache-2.0
// NumPy splitting, tiling, repetition and roll helpers. Splits return lazy
// slice views; `repeat` and `tile` select with integer-array indexes, and
// `roll` joins two slices per axis with the native concatenate node, so their
// results are new arrays as in NumPy 2.5.4.

import { Expr, Lazy, indices, slice } from "./core.js";
import { NMathError } from "./errors.js";
import { concatenate } from "./joining.js";
import type { AxisOf, DynamicDims, ReplaceDim } from "./layout-types.js";
import type { ComputableDType, MutableShape, Shape } from "./types.js";

// ---------------------------------------------------------------------------
// Static types. The split or repeated axis becomes dynamic; the other
// dimensions of a literal tuple shape are kept.

type Arg<D extends ComputableDType, S extends Shape> = Lazy<NoInfer<D>, S> & {
  readonly dtype: D;
};
type Valid<R> = [R] extends [never] ? never : unknown;
type Zeros<T extends readonly unknown[]> = { -readonly [K in keyof T]: 0 };
type ListOf<A> = A extends readonly number[] ? A : A extends number ? [A] : [];

/** The shape of each part along a normalized axis; `never` when the axis is invalid. */
export type SplitShape<
  S extends Shape,
  A extends number,
> = number extends S["length"]
  ? number[]
  : AxisOf<S, A> extends infer I extends number
    ? [I] extends [never]
      ? never
      : number extends I
        ? DynamicDims<S>
        : ReplaceDim<S, I> extends infer R extends Shape
          ? R
          : never
    : never;
/** `np.hsplit` part shape: axis 1, or axis 0 for 1-D input; 0-d input is rejected. */
export type HSplitShape<S extends Shape> = number extends S["length"]
  ? number[]
  : S extends readonly []
    ? never
    : S extends readonly [number]
      ? [number]
      : SplitShape<S, 1>;
type MinRank<S extends Shape, N extends 2 | 3> = number extends S["length"]
  ? unknown
  : S extends readonly [number, number, ...(N extends 3 ? [number] : [])]
    ? unknown
    : S extends readonly [
          number,
          number,
          ...(N extends 3 ? [number] : []),
          ...number[],
        ]
      ? unknown
      : never;

/**
 * `np.repeat` shape: flattened without an axis, otherwise the axis is
 * dynamic. A 0-d input repeats as a 1-D array, so it accepts axis 0 or -1.
 */
export type RepeatShape<
  S extends Shape,
  A extends number | null,
> = A extends null
  ? [number]
  : SplitShape<S extends readonly [] ? [number] : S, A & number>;
/** `np.tile` shape: the rank is the larger of the input rank and the repetition count. */
export type TileShape<
  S extends Shape,
  R extends readonly number[],
> = number extends S["length"]
  ? number[]
  : number extends R["length"]
    ? number[]
    : Zeros<R> extends readonly [...Zeros<S>, 0, ...0[]]
      ? DynamicDims<R>
      : DynamicDims<S>;
type RollCheck<
  S extends Shape,
  A extends number | readonly number[] | null,
> = A extends null
  ? unknown
  : number extends S["length"]
    ? unknown
    : AxisCheck<S, ListOf<A>>;
type AxisCheck<
  S extends Shape,
  L extends readonly number[],
> = L extends readonly [infer H extends number, ...infer R extends number[]]
  ? [AxisOf<S, H>] extends [never]
    ? never
    : AxisCheck<S, R>
  : unknown;

// ---------------------------------------------------------------------------
// Runtime helpers

const HINT = "Use an index inside the axis length or a valid selector.";

function requireLazy(value: unknown, name: string): Lazy<any, any> {
  if (!(value instanceof Lazy)) {
    throw new TypeError(`${name} requires an NMath array or expression.`);
  }
  return value;
}

function invalid(message: string): NMathError {
  return new NMathError(
    "NMATH_INDEX_INVALID",
    `NMATH_INDEX_INVALID: ${message} ${HINT}`,
  );
}

function shapeError(message: string): NMathError {
  return new NMathError(
    "NMATH_SHAPE_MISMATCH",
    `NMATH_SHAPE_MISMATCH: ${message} Check the operand shapes against the operation requirements.`,
  );
}

function integer(value: unknown, what: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value)) {
    throw invalid(`${what} must be a safe integer.`);
  }
  return value;
}

/** NumPy `normalize_axis_index` with the native error codes. */
function normalizeAxis(axis: unknown, rank: number): number {
  const value = integer(axis, "An axis");
  if (value < -rank || value >= rank) {
    throw new NMathError(
      "NMATH_INDEX_OUT_OF_BOUNDS",
      `NMATH_INDEX_OUT_OF_BOUNDS: Axis ${value} is out of bounds for an array of dimension ${rank}. ${HINT}`,
    );
  }
  return value < 0 ? value + rank : value;
}

/** Integers from a scalar or a flat list, as NumPy reads `int` and 1-D sequences. */
function integers(value: unknown, what: string): number[] {
  const values = Array.isArray(value) ? value : [value];
  return values.map((item) => integer(item, what));
}

function select(
  x: Lazy<any, any>,
  axis: number,
  selector: ReturnType<typeof slice> | ReturnType<typeof indices>,
): Expr<any, any> {
  return x.get(
    ...Array.from({ length: x.ndim }, (_, i) =>
      i === axis ? selector : slice(),
    ),
  ) as Expr<any, any>;
}

function flatten(x: Lazy<any, any>): Expr<any, any> {
  return x.reshape([-1] as never) as Expr<any, any>;
}

/** NumPy `array_split` division points along `axis`, then one slice view per part. */
function parts(
  ary: Lazy<any, any>,
  sections: unknown,
  axis: number,
  equal: boolean,
): readonly Expr<any, any>[] {
  const total = ary.shape[axis] as number;
  let points: number[];
  if (Array.isArray(sections)) {
    points = [0, ...integers(sections, "A split index"), total];
  } else {
    const count = integer(sections, "The number of sections");
    if (count <= 0) {
      throw invalid("The number of sections must be larger than 0.");
    }
    if (equal && total % count !== 0) {
      throw shapeError("The array split does not result in an equal division.");
    }
    const each = Math.floor(total / count);
    const extras = total % count;
    points = [0];
    for (let i = 0; i < count; i += 1) {
      points.push(points[i] + each + (i < extras ? 1 : 0));
    }
  }
  const result: Expr<any, any>[] = [];
  for (let i = 0; i + 1 < points.length; i += 1) {
    result.push(select(ary, axis, slice(points[i], points[i + 1])));
  }
  return Object.freeze(result);
}

/** Repeats every position along `axis` by its count with one index-array read. */
function repeatAxis(
  x: Lazy<any, any>,
  counts: readonly number[],
  axis: number,
): Expr<any, any> {
  const length = x.shape[axis] as number;
  const positions: number[] = [];
  for (let i = 0; i < length; i += 1) {
    const count = counts.length === 1 ? counts[0] : counts[i];
    for (let k = 0; k < count; k += 1) positions.push(i);
  }
  return select(x, axis, indices(positions));
}

// ---------------------------------------------------------------------------
// Public functions

/**
 * `np.array_split(ary, indices_or_sections, axis=0)`: a section count splits
 * the axis into parts whose lengths differ by at most one, the longer parts
 * first; a list of indexes splits before each index with Python slice rules.
 * Parts are lazy views and the result is a frozen array.
 */
export function array_split<
  D extends ComputableDType,
  S extends Shape,
  const A extends number = 0,
>(
  ary: Arg<D, S>,
  indices_or_sections: number | readonly number[],
  axis?: A & Valid<SplitShape<S, A>>,
): readonly Expr<D, SplitShape<S, A>>[];
export function array_split(
  ary: Lazy<any, any>,
  indices_or_sections: number | readonly number[],
  axis = 0,
): readonly Expr<any, any>[] {
  const source = requireLazy(ary, "array_split");
  return parts(
    source,
    indices_or_sections,
    normalizeAxis(axis, source.ndim),
    false,
  );
}

/**
 * `np.split(ary, indices_or_sections, axis=0)`: like `array_split`, but a
 * section count must divide the axis length (`NMATH_SHAPE_MISMATCH`).
 */
export function split<
  D extends ComputableDType,
  S extends Shape,
  const A extends number = 0,
>(
  ary: Arg<D, S>,
  indices_or_sections: number | readonly number[],
  axis?: A & Valid<SplitShape<S, A>>,
): readonly Expr<D, SplitShape<S, A>>[];
export function split(
  ary: Lazy<any, any>,
  indices_or_sections: number | readonly number[],
  axis = 0,
): readonly Expr<any, any>[] {
  const source = requireLazy(ary, "split");
  return parts(
    source,
    indices_or_sections,
    normalizeAxis(axis, source.ndim),
    true,
  );
}

/** `np.hsplit`: splits along axis 1, or along axis 0 for 1-D input. */
export function hsplit<D extends ComputableDType, S extends Shape>(
  ary: Arg<D, S> & Valid<HSplitShape<S>>,
  indices_or_sections: number | readonly number[],
): readonly Expr<D, HSplitShape<S>>[];
export function hsplit(
  ary: Lazy<any, any>,
  indices_or_sections: number | readonly number[],
): readonly Expr<any, any>[] {
  const source = requireLazy(ary, "hsplit");
  if (source.ndim === 0) {
    throw shapeError("hsplit only works on arrays of 1 or more dimensions.");
  }
  return parts(source, indices_or_sections, source.ndim > 1 ? 1 : 0, true);
}

/** `np.vsplit`: splits along axis 0; the input needs at least 2 dimensions. */
export function vsplit<D extends ComputableDType, S extends Shape>(
  ary: Arg<D, S> & MinRank<S, 2>,
  indices_or_sections: number | readonly number[],
): readonly Expr<D, SplitShape<S, 0>>[];
export function vsplit(
  ary: Lazy<any, any>,
  indices_or_sections: number | readonly number[],
): readonly Expr<any, any>[] {
  const source = requireLazy(ary, "vsplit");
  if (source.ndim < 2) {
    throw shapeError("vsplit only works on arrays of 2 or more dimensions.");
  }
  return parts(source, indices_or_sections, 0, true);
}

/** `np.dsplit`: splits along axis 2; the input needs at least 3 dimensions. */
export function dsplit<D extends ComputableDType, S extends Shape>(
  ary: Arg<D, S> & MinRank<S, 3>,
  indices_or_sections: number | readonly number[],
): readonly Expr<D, SplitShape<S, 2>>[];
export function dsplit(
  ary: Lazy<any, any>,
  indices_or_sections: number | readonly number[],
): readonly Expr<any, any>[] {
  const source = requireLazy(ary, "dsplit");
  if (source.ndim < 3) {
    throw shapeError("dsplit only works on arrays of 3 or more dimensions.");
  }
  return parts(source, indices_or_sections, 2, true);
}

/**
 * `np.repeat(a, repeats, axis=None)`: repeats each element, or each position
 * along `axis`. `repeats` is one count or one count per position; without an
 * axis the input is flattened first. Negative counts fail with
 * `NMATH_INDEX_INVALID` and a count list of the wrong length with
 * `NMATH_SHAPE_BROADCAST`.
 */
export function repeat<
  D extends ComputableDType,
  S extends Shape,
  const A extends number | null = null,
>(
  a: Arg<D, S>,
  repeats: number | readonly number[],
  axis?: A & Valid<RepeatShape<S, A>>,
): Expr<D, RepeatShape<S, A>>;
export function repeat(
  a: Lazy<any, any>,
  repeats: number | readonly number[],
  axis: number | null = null,
): Expr<any, any> {
  const source = requireLazy(a, "repeat");
  const counts = integers(repeats, "A repeat count");
  if (counts.some((count) => count < 0)) {
    throw invalid("Repeat counts must be non-negative.");
  }
  // NumPy checks the axis of a 0-d input against its 1-D form.
  const x = axis === null || source.ndim === 0 ? flatten(source) : source;
  const ax = axis === null ? 0 : normalizeAxis(axis, x.ndim);
  if (counts.length !== 1 && counts.length !== x.shape[ax]) {
    throw new NMathError(
      "NMATH_SHAPE_BROADCAST",
      `NMATH_SHAPE_BROADCAST: ${counts.length} repeat counts cannot broadcast to axis length ${x.shape[ax]}. Pass one count or one count per position.`,
    );
  }
  return repeatAxis(x, counts, ax);
}

/**
 * `np.tile(A, reps)`: repeats the whole array along each axis. The input
 * gains leading length-1 axes when `reps` is longer, and `reps` gains leading
 * ones when it is shorter. The result is always a new array.
 */
export function tile<
  D extends ComputableDType,
  S extends Shape,
  const R extends number | readonly number[],
>(A: Arg<D, S>, reps: R): Expr<D, TileShape<S, ListOf<R>>>;
export function tile(
  A: Lazy<any, any>,
  reps: number | readonly number[],
): Expr<any, any> {
  const source = requireLazy(A, "tile");
  const counts = integers(reps, "A tile repetition");
  if (counts.some((count) => count < 0)) {
    throw invalid("Tile repetitions must be non-negative.");
  }
  const rank = Math.max(source.ndim, counts.length);
  const shape = [...Array<number>(rank - source.ndim).fill(1), ...source.shape];
  const full = [...Array<number>(rank - counts.length).fill(1), ...counts];
  // NumPy copies even when every repetition is 1.
  let result = source.reshape(shape as never, { copy: true }) as Expr<any, any>;
  full.forEach((count, axis) => {
    if (count === 1) return;
    const positions: number[] = [];
    for (let k = 0; k < count; k += 1) {
      for (let i = 0; i < shape[axis]; i += 1) positions.push(i);
    }
    result = select(result, axis, indices(positions));
  });
  return result;
}

/**
 * `np.roll(a, shift, axis=None)`: rolls elements forward along each axis.
 * Without an axis the input rolls as a flat array and keeps its shape. Shifts
 * and axes broadcast, and shifts for a repeated axis add up. The result is
 * always a new array.
 */
export function roll<
  D extends ComputableDType,
  S extends Shape,
  const A extends number | readonly number[] | null = null,
>(
  a: Arg<D, S>,
  shift: number | readonly number[],
  axis?: A & RollCheck<S, A>,
): Expr<D, MutableShape<S>>;
export function roll(
  a: Lazy<any, any>,
  shift: number | readonly number[],
  axis: number | readonly number[] | null = null,
): Expr<any, any> {
  const source = requireLazy(a, "roll");
  if (axis === null) {
    return roll(flatten(source) as never, shift, 0).reshape(
      source.shape as never,
    ) as Expr<any, any>;
  }
  const shifts = integers(shift, "A shift");
  const axes = integers(axis, "An axis").map((value) =>
    normalizeAxis(value, source.ndim),
  );
  if (
    shifts.length !== axes.length &&
    shifts.length !== 1 &&
    axes.length !== 1
  ) {
    throw new NMathError(
      "NMATH_SHAPE_BROADCAST",
      `NMATH_SHAPE_BROADCAST: ${shifts.length} shifts cannot broadcast with ${axes.length} axes. Pass matching lengths or one value.`,
    );
  }
  const offsets = Array<number>(source.ndim).fill(0);
  // Broadcasting a length-1 list takes the other length, which can be 0.
  const count = shifts.length === 1 ? axes.length : shifts.length;
  for (let i = 0; i < count; i += 1) {
    offsets[axes[axes.length === 1 ? 0 : i]] +=
      shifts[shifts.length === 1 ? 0 : i];
  }
  let result: Lazy<any, any> = source;
  offsets.forEach((offset, ax) => {
    const length = source.shape[ax] as number;
    // Python's modulo is non-negative; an empty axis never rolls.
    const k = length === 0 ? 0 : ((offset % length) + length) % length;
    if (k === 0) return;
    result = concatenate(
      [select(result, ax, slice(-k)), select(result, ax, slice(undefined, -k))],
      ax,
    );
  });
  // NumPy fills a new array even when nothing moves.
  return result === source
    ? (source.reshape(source.shape as never, { copy: true }) as Expr<any, any>)
    : (result as Expr<any, any>);
}
