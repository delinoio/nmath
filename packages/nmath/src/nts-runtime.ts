// SPDX-License-Identifier: Apache-2.0
// Helpers that compiled `.nts` output imports as `__nmath`. Not a documented API.
//
// Each operator helper has overloads: NMath operands produce a lazy expression; ordinary
// JS operands keep ordinary JS semantics. A union that may be either matches no overload,
// so the type checker requires narrowing. The runtime check below only executes the
// overload the checker already selected; it never guesses between the two meanings.

import {
  type Expr,
  Lazy,
  type NMathArray,
  type NMathValue,
  type Selector,
} from "./core.js";
import {
  add as nAdd,
  div as nDiv,
  mod as nMod,
  mul as nMul,
  pow as nPow,
  sub as nSub,
} from "./index.js";
import type { ComputableDType, Shape, WeakScalar } from "./types.js";

type NValue = NMathValue;
type NOperand = NValue | WeakScalar;
type AnyExpr = Expr<ComputableDType, Shape>;

const isN = (v: unknown): v is Lazy<any, any> => v instanceof Lazy;

export function add(a: NValue, b: NOperand): AnyExpr;
export function add(a: WeakScalar, b: NValue): AnyExpr;
export function add(a: number, b: number): number;
export function add(a: bigint, b: bigint): bigint;
export function add(
  a: string,
  b: string | number | bigint | boolean | null | undefined,
): string;
export function add(
  a: number | bigint | boolean | null | undefined,
  b: string,
): string;
export function add(a: any, b: any): any {
  return isN(a) || isN(b) ? nAdd(a, b) : a + b;
}

export function sub(a: NValue, b: NOperand): AnyExpr;
export function sub(a: WeakScalar, b: NValue): AnyExpr;
export function sub(a: number, b: number): number;
export function sub(a: bigint, b: bigint): bigint;
export function sub(a: any, b: any): any {
  return isN(a) || isN(b) ? nSub(a, b) : a - b;
}

export function mul(a: NValue, b: NOperand): AnyExpr;
export function mul(a: WeakScalar, b: NValue): AnyExpr;
export function mul(a: number, b: number): number;
export function mul(a: bigint, b: bigint): bigint;
export function mul(a: any, b: any): any {
  return isN(a) || isN(b) ? nMul(a, b) : a * b;
}

export function div(a: NValue, b: NOperand): AnyExpr;
export function div(a: WeakScalar, b: NValue): AnyExpr;
export function div(a: number, b: number): number;
export function div(a: bigint, b: bigint): bigint;
export function div(a: any, b: any): any {
  return isN(a) || isN(b) ? nDiv(a, b) : a / b;
}

export function mod(a: NValue, b: NOperand): AnyExpr;
export function mod(a: WeakScalar, b: NValue): AnyExpr;
export function mod(a: number, b: number): number;
export function mod(a: bigint, b: bigint): bigint;
export function mod(a: any, b: any): any {
  return isN(a) || isN(b) ? nMod(a, b) : a % b;
}

export function pow(a: NValue, b: NOperand): AnyExpr;
export function pow(a: WeakScalar, b: NValue): AnyExpr;
export function pow(a: number, b: number): number;
export function pow(a: bigint, b: bigint): bigint;
export function pow(a: any, b: any): any {
  return isN(a) || isN(b) ? nPow(a, b) : a ** b;
}

export function neg(a: NValue): AnyExpr;
export function neg(a: number): number;
export function neg(a: bigint): bigint;
export function neg(a: any): any {
  return isN(a) ? (a as Lazy<ComputableDType, Shape>).neg() : -a;
}

export function pos(a: NValue): AnyExpr;
export function pos(a: number | string | boolean | null | undefined): number;
export function pos(a: any): any {
  return isN(a) ? (a as Lazy<ComputableDType, Shape>).pos() : +a;
}

type Key = Selector | readonly Selector[];

function axes(key: Key): Selector[] {
  return Array.isArray(key) ? [...(key as Selector[])] : [key as Selector];
}

/** `a[key]` read. NMath reads are lazy; other values use ordinary property access. */
export function get(a: NValue, key: Key): AnyExpr;
export function get<T, K extends keyof T>(a: T, key: K): T[K];
export function get(a: any, key: any): any {
  return isN(a) ? a.get(...axes(key)) : a[key];
}

/** `a[key] = value`, awaited by the compiled code. Resolves to `value`. */
export function set<V extends NOperand>(
  a: NMathValue & { set: NMathArray<ComputableDType, Shape>["set"] },
  key: Key,
  value: V,
): Promise<V>;
export function set<T, K extends keyof T, V extends T[K]>(
  a: T,
  key: K,
  value: V,
): Promise<V>;
export async function set(a: any, key: any, value: any): Promise<any> {
  if (isN(a)) {
    await (a as NMathArray<ComputableDType, Shape>).set(axes(key), value);
  } else {
    a[key] = value;
  }
  return value;
}

export { indices, mask, slice } from "./core.js";
