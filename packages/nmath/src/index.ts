// SPDX-License-Identifier: Apache-2.0
// Core entry point: contexts, arrays, lazy expressions, selectors and errors.

import {
  type BinaryOperator,
  type Complex,
  type Expr,
  Lazy,
  type Operand,
} from "./core.js";
import type {
  BinaryDType,
  BinaryCheck,
  WeakBinary,
  WeakDType,
  BoolArithmetic,
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
  Complex,
  Temporal,
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
  TimeUnit,
  TimeBase,
  DTypeMetadata,
  NMathValue,
  ArrayOptions,
  ReshapeOptions,
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
 * the methods. A scalar may appear on either side. Dtypes are inferred from
 * operand properties so recursive method signatures cannot widen promotion.
 */
export interface BinaryFunction<O extends BinaryOperator = "add"> {
  <
    D1 extends ComputableDType,
    S1 extends Shape,
    D2 extends ComputableDType,
    S2 extends Shape,
  >(
    a: Lazy<NoInfer<D1>, S1> & { readonly dtype: D1 },
    b: Lazy<NoInfer<D2>, S2> & { readonly dtype: D2 } & BroadcastCheck<S1, S2> &
      BinaryCheck<O, D1, D2>,
  ): Expr<BinaryDType<O, D1, D2>, Broadcast<S1, S2>>;
  <D extends ComputableDType, S extends Shape, W extends WeakScalar>(
    a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
    b: W & ([WeakBinary<O, D, W>] extends [never] ? never : unknown),
  ): Expr<WeakBinary<O, D, W>, S>;
  <W extends WeakScalar, D extends ComputableDType, S extends Shape>(
    a: W & BinaryCheck<O, WeakDType<W>, D>,
    b: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  ): Expr<
    D extends "str" | "datetime64" | "timedelta64"
      ? BinaryDType<O, WeakDType<W>, D>
      : WeakBinary<O, D, W>,
    S
  >;
}
export type RemainderFunction = BinaryFunction<"mod">;
export type DivideFunction = BinaryFunction<"div">;
export type PowerFunction = BinaryFunction<"pow">;
export const add: BinaryFunction<"add"> = (a: Operand, b: Operand) =>
  binary("add", a, b);
export const sub: BinaryFunction<"sub"> = (a: Operand, b: Operand) =>
  binary("sub", a, b);
export const mul: BinaryFunction<"mul"> = (a: Operand, b: Operand) =>
  binary("mul", a, b);
export const div: DivideFunction = (a: Operand, b: Operand) =>
  binary("div", a, b);
export const mod: RemainderFunction = (a: Operand, b: Operand) =>
  binary("mod", a, b);
export const pow: PowerFunction = (a: Operand, b: Operand) =>
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

export type {
  ReshapeShape,
  ReshapeCheck,
  TransposeShape,
  TransposeCheck,
} from "./layout-types.js";
