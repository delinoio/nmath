// SPDX-License-Identifier: Apache-2.0
// Compiler-only helpers. The checker proves NMath operands before lowering;
// ordinary JavaScript never calls these helpers.
import {
  type Expr,
  type Lazy,
  type NMathArray,
  type NMathValue,
  type Operand,
  type ReadSelector,
  type WriteSelector,
} from "./core.js";
import type {
  ComputableDType,
  IndexShape,
  Shape,
  WeakScalar,
} from "./types.js";

// Reuse the public signatures, including dtype promotion and shape checks.
export { add, sub, mul, div, mod, pow } from "./index.js";
export { ellipsis, indices, mask, newaxis, slice } from "./core.js";

export function neg<
  D extends Exclude<ComputableDType, "bool">,
  S extends Shape,
>(a: Lazy<NoInfer<D>, S> & { readonly dtype: D }): Expr<D, S> {
  // TypeScript cannot resolve the conditional receiver for a generic dtype.
  // D excludes bool above; widen only the internal call until it can.
  return (a as unknown as Lazy<ComputableDType, S>).neg() as unknown as Expr<
    D,
    S
  >;
}
export function pos<
  D extends Exclude<ComputableDType, "bool">,
  S extends Shape,
>(a: Lazy<NoInfer<D>, S> & { readonly dtype: D }): Expr<D, S> {
  // Match neg: the public constraint excludes bool before this internal cast.
  return (a as unknown as Lazy<ComputableDType, S>).pos() as unknown as Expr<
    D,
    S
  >;
}

type Key = ReadSelector | readonly ReadSelector[];
type Axes<K extends Key> = K extends readonly ReadSelector[]
  ? K
  : K extends ReadSelector
    ? readonly [K]
    : never;
function axes<K extends Key>(key: K): Axes<K> {
  return (Array.isArray(key) ? key : [key]) as unknown as Axes<K>;
}

export function get<
  D extends ComputableDType,
  S extends Shape,
  const K extends Key,
>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  key: K,
): Expr<D, IndexShape<S, Axes<K>>> {
  // TypeScript cannot relate the spread tuple [...Axes<K>] to Axes<K> inside
  // the deferred IndexShape conditional; both denote the same selectors.
  return a.get(...axes(key)) as unknown as Expr<D, IndexShape<S, Axes<K>>>;
}

/** The assignment expression resolves to its original right-hand value. */
export async function set<
  D extends ComputableDType,
  S extends Shape,
  V extends NMathValue | WeakScalar,
>(
  a: NMathArray<D, S>,
  key: WriteSelector | readonly WriteSelector[],
  value: V,
): Promise<V> {
  // The public Operand alias currently uses Lazy<any, any>, which rejects
  // some scalar Expr instantiations through method variance. The nominal
  // NMathValue constraint accepts those values without erasing V. Remove the
  // assertion when Operand accepts every branded array/expression shape.
  await a.set(axes(key) as readonly WriteSelector[], value as Operand);
  return value;
}
