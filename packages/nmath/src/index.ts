// SPDX-License-Identifier: Apache-2.0
// Core entry point: contexts, arrays, lazy expressions, selectors and errors.

import { type BinaryOperator, type Expr, Lazy, type Operand } from "./core.js";
import type {
  Broadcast,
  BroadcastCheck,
  ComputableDType,
  Promote,
  PromoteWeak,
  Shape,
  TrueDivide,
  WeakScalar,
} from "./types.js";

export {
  Expr,
  Lazy,
  NMathArray,
  NMathContext,
  NMathGenerator,
  createContext,
  indices,
  mask,
  slice,
} from "./core.js";
export type {
  NMathValue,
  ArrayOptions,
  BinaryOperator,
  ComputeOptions,
  ComputeResult,
  ContextMetrics,
  ContextOptions,
  Device,
  ExecutionMeasurement,
  Indices,
  Mask,
  MemoryMeasurement,
  Operand,
  ReduceOperator,
  Selector,
  SignalOptions,
  Slice,
} from "./core.js";
export { ERROR_CODES, NMathError } from "./errors.js";
export type { ErrorCode } from "./errors.js";
export { platformTriple } from "./native.js";
export type * from "./types.js";

/** True for NMath arrays and expressions. */
export function isNMath(value: unknown): value is Lazy<ComputableDType, Shape> {
  return value instanceof Lazy;
}

/**
 * Functional arithmetic with the same dtype promotion and static broadcasting checks as
 * the methods. A scalar may appear on either side.
 */
export interface BinaryFunction {
  <
    D1 extends ComputableDType,
    S1 extends Shape,
    D2 extends ComputableDType,
    S2 extends Shape,
  >(
    a: Lazy<D1, S1>,
    b: Lazy<D2, S2> & BroadcastCheck<S1, S2>,
  ): Expr<Promote<D1, D2>, Broadcast<S1, S2>>;
  <D extends ComputableDType, S extends Shape, W extends WeakScalar>(
    a: Lazy<D, S>,
    b: W,
  ): Expr<PromoteWeak<D, W>, S>;
  <W extends WeakScalar, D extends ComputableDType, S extends Shape>(
    a: W,
    b: Lazy<D, S>,
  ): Expr<PromoteWeak<D, W>, S>;
}

/** {@link BinaryFunction} with true-division result dtypes. */
export interface DivideFunction {
  <
    D1 extends ComputableDType,
    S1 extends Shape,
    D2 extends ComputableDType,
    S2 extends Shape,
  >(
    a: Lazy<D1, S1>,
    b: Lazy<D2, S2> & BroadcastCheck<S1, S2>,
  ): Expr<TrueDivide<Promote<D1, D2>>, Broadcast<S1, S2>>;
  <D extends ComputableDType, S extends Shape, W extends WeakScalar>(
    a: Lazy<D, S>,
    b: W,
  ): Expr<TrueDivide<PromoteWeak<D, W>>, S>;
  <W extends WeakScalar, D extends ComputableDType, S extends Shape>(
    a: W,
    b: Lazy<D, S>,
  ): Expr<TrueDivide<PromoteWeak<D, W>>, S>;
}

export const add: BinaryFunction = (a: Operand, b: Operand) =>
  binary("add", a, b);
export const sub: BinaryFunction = (a: Operand, b: Operand) =>
  binary("sub", a, b);
export const mul: BinaryFunction = (a: Operand, b: Operand) =>
  binary("mul", a, b);
export const div: DivideFunction = (a: Operand, b: Operand) =>
  binary("div", a, b);
export const mod: BinaryFunction = (a: Operand, b: Operand) =>
  binary("mod", a, b);
export const pow: BinaryFunction = (a: Operand, b: Operand) =>
  binary("pow", a, b);

function binary(op: BinaryOperator, a: Operand, b: Operand): Expr<any, any> {
  if (a instanceof Lazy) {
    return a.binaryOp(op, b);
  }
  if (b instanceof Lazy) {
    return b.reversedBinaryOp(op, a);
  }
  throw new TypeError(
    "At least one operand must be an NMath array or expression.",
  );
}
