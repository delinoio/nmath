// SPDX-License-Identifier: Apache-2.0
// Compiler-only helpers. The checker proves NMath operands before lowering;
// ordinary JavaScript never calls these helpers.
import {
  type Expr,
  type Lazy,
  type NMathArray,
  type NMathValue,
  type Operand,
  type Selector,
} from "./core.js";
import type {
  ComputableDType,
  IndexShape,
  Shape,
  WeakScalar,
} from "./types.js";

// Reuse the public signatures, including dtype promotion and shape checks.
export { add, sub, mul, div, mod, pow } from "./index.js";
export { indices, mask, slice } from "./core.js";

export function neg<D extends ComputableDType, S extends Shape>(
  a: Lazy<D, S>,
): Expr<D, S> {
  return a.neg();
}
export function pos<D extends ComputableDType, S extends Shape>(
  a: Lazy<D, S>,
): Expr<D, S> {
  return a.pos();
}

type Key = Selector | readonly Selector[];
type Axes<K extends Key> = K extends readonly Selector[]
  ? K
  : K extends Selector
    ? readonly [K]
    : never;
function axes<K extends Key>(key: K): Axes<K> {
  return (Array.isArray(key) ? key : [key]) as unknown as Axes<K>;
}

export function get<
  D extends ComputableDType,
  S extends Shape,
  const K extends Key,
>(a: Lazy<D, S>, key: K): Expr<D, IndexShape<S, Axes<K>>> {
  return a.get(...axes(key));
}

/** The assignment expression resolves to its original right-hand value. */
export async function set<
  D extends ComputableDType,
  S extends Shape,
  V extends NMathValue | WeakScalar,
>(a: NMathArray<D, S>, key: Key, value: V): Promise<V> {
  // The public Operand alias currently uses Lazy<any, any>, which rejects
  // some scalar Expr instantiations through method variance. The nominal
  // NMathValue constraint accepts those values without erasing V. Remove the
  // assertion when Operand accepts every branded array/expression shape.
  await a.set(axes(key), value as Operand);
  return value;
}
