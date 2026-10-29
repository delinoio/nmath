// SPDX-License-Identifier: Apache-2.0
// Core entry point: contexts, arrays, lazy expressions, selectors and errors.

import { type BinaryOperator, Lazy, type Operand } from "./core.js";
import type { ComputableDType, Shape } from "./types.js";

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

/** Functional forms. A scalar may appear on either side. */
export function add(a: Operand, b: Operand) {
  return binary("add", a, b);
}
export function sub(a: Operand, b: Operand) {
  return binary("sub", a, b);
}
export function mul(a: Operand, b: Operand) {
  return binary("mul", a, b);
}
export function div(a: Operand, b: Operand) {
  return binary("div", a, b);
}
export function mod(a: Operand, b: Operand) {
  return binary("mod", a, b);
}
export function pow(a: Operand, b: Operand) {
  return binary("pow", a, b);
}

function binary(op: BinaryOperator, a: Operand, b: Operand) {
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
