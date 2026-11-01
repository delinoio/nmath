// SPDX-License-Identifier: Apache-2.0
// NumPy-style entry point. Functions take an explicit context where NumPy would use global
// state. Names follow NumPy in camelCase. The compatibility inventory in README.md lists
// which NumPy 2.5.4 APIs exist.

import {
  type ArrayOptions,
  type ReshapeOptions,
  Expr,
  Lazy,
  type NMathArray,
  type NMathContext,
  type NMathGenerator,
  type Operand,
} from "./core.js";
import { add, div, mod, mul, pow, sub } from "./index.js";
import { native } from "./errors.js";
import { binding } from "./native.js";
import type {
  ComputableDType,
  NumericDType,
  Promote,
  MatMulShape,
  MatMulCheck,
  MutableShape,
  SumDType,
  TrueDivide,
  ReduceShape,
  ScalarOf,
  Shape,
  TemporalDType,
  TypedArrayOf,
} from "./types.js";

export { indices, mask, slice } from "./core.js";
import type {
  ReshapeShape,
  ReshapeCheck,
  TransposeShape,
  TransposeCheck,
} from "./layout-types.js";

export function reshape<
  D extends ComputableDType,
  S extends Shape,
  const R extends Shape,
>(
  input: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  shape: R & ReshapeCheck<S, R>,
  options: ReshapeOptions = {},
): Expr<D, ReshapeShape<S, R>> {
  return input.reshape<R>(shape, options);
}
export function transpose<
  D extends ComputableDType,
  S extends Shape,
  const A extends Shape | undefined = undefined,
>(
  input: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axes?: A & TransposeCheck<S, A>,
): Expr<D, TransposeShape<S, A>> {
  return input.transpose<A>(axes);
}

type Nested<T> = T | readonly Nested<T>[];

export function array<
  D extends ComputableDType,
  const S extends Shape = number[],
>(
  context: NMathContext,
  data: TypedArrayOf<D> | Nested<ScalarOf<D>>,
  options: ArrayOptions<D, S>,
): NMathArray<D, S> {
  return context.array(data, options);
}

export function zeros<D extends ComputableDType, const S extends Shape>(
  context: NMathContext,
  shape: S,
  dtype: D,
): NMathArray<D, S> {
  return context.zeros(shape, dtype);
}

export function ones<D extends ComputableDType, const S extends Shape>(
  context: NMathContext,
  shape: S,
  dtype: D,
): NMathArray<D, S> {
  const one = (
    dtype === "int64" || dtype === "uint64" ? 1n : dtype === "bool" ? true : 1
  ) as ScalarOf<D>;
  return context.full(shape, one, dtype);
}

export function full<D extends ComputableDType, const S extends Shape>(
  context: NMathContext,
  shape: S,
  value: ScalarOf<D>,
  dtype: D,
): NMathArray<D, S> {
  return context.full(shape, value, dtype);
}

/** `np.arange(start, stop, step)` for numeric dtypes other than 64-bit integers. */
export function arange<D extends "int32" | "uint32" | "float32" | "float64">(
  context: NMathContext,
  start: number,
  stop: number,
  step = 1,
  dtype: D = "float64" as D,
): NMathArray<D, [number]> {
  if (step === 0) {
    throw new RangeError("arange step must not be zero.");
  }
  const length = Math.max(0, Math.ceil((stop - start) / step));
  context.ensureCpuBudget(length, dtype);
  const values = Array.from({ length }, (_, i) => start + i * step);
  return context.array(values as never, { dtype, shape: [length] as [number] });
}

export const subtract = sub;
export const multiply = mul;
export const divide = div;
export const trueDivide = div;
export const remainder = mod;
export const power = pow;
export { add, mod };

export function negative<T extends Lazy<ComputableDType, Shape>>(x: T) {
  return x.neg();
}
export function positive<T extends Lazy<ComputableDType, Shape>>(x: T) {
  return x.pos();
}
// Public overloads retain dtype/axis inference and reject unsupported reductions.
// The broad internal signatures bridge conditional method receivers; no runtime
// coercion occurs. Remove this bridge if TypeScript resolves those generic receivers.
export function sum<
  D extends Exclude<ComputableDType, "str" | "datetime64">,
  S extends Shape,
  A extends number | undefined = undefined,
>(
  x: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis?: A,
): Expr<SumDType<D>, ReduceShape<S, A>>;
export function sum(x: Lazy<any, any>, axis?: number): Expr<any, any> {
  return x.sum(axis);
}
export function mean<
  D extends Exclude<ComputableDType, "str" | "datetime64">,
  S extends Shape,
  A extends number | undefined = undefined,
>(
  x: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis?: A,
): Expr<D extends "timedelta64" ? D : TrueDivide<D>, ReduceShape<S, A>>;
export function mean(x: Lazy<any, any>, axis?: number): Expr<any, any> {
  return x.mean(axis);
}
export function min<
  D extends Exclude<ComputableDType, "str">,
  S extends Shape,
  A extends number | undefined = undefined,
>(
  x: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis?: A,
): Expr<D, ReduceShape<S, A>>;
export function min(x: Lazy<any, any>, axis?: number): Expr<any, any> {
  return x.min(axis);
}
export function max<
  D extends Exclude<ComputableDType, "str">,
  S extends Shape,
  A extends number | undefined = undefined,
>(
  x: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis?: A,
): Expr<D, ReduceShape<S, A>>;
export function max(x: Lazy<any, any>, axis?: number): Expr<any, any> {
  return x.max(axis);
}
export function prod<
  D extends Exclude<ComputableDType, "str" | TemporalDType>,
  S extends Shape,
  A extends number | undefined = undefined,
>(
  x: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis?: A,
): Expr<SumDType<D>, ReduceShape<S, A>>;
export function prod(x: Lazy<any, any>, axis?: number): Expr<any, any> {
  return x.prod(axis);
}
export function argmin<
  D extends Exclude<ComputableDType, "str" | TemporalDType>,
  S extends Shape,
  A extends number | undefined = undefined,
>(
  x: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis?: A,
): Expr<"int64", ReduceShape<S, A>>;
export function argmin(x: Lazy<any, any>, axis?: number): Expr<any, any> {
  return x.argmin(axis);
}
export function argmax<
  D extends Exclude<ComputableDType, "str" | TemporalDType>,
  S extends Shape,
  A extends number | undefined = undefined,
>(
  x: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis?: A,
): Expr<"int64", ReduceShape<S, A>>;
export function argmax(x: Lazy<any, any>, axis?: number): Expr<any, any> {
  return x.argmax(axis);
}
export function cumsum<
  D extends Exclude<ComputableDType, "str" | TemporalDType>,
  S extends Shape,
  A extends number | undefined = undefined,
>(
  x: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis?: A,
): Expr<SumDType<D>, [A] extends [undefined] ? [number] : MutableShape<S>>;
export function cumsum(x: Lazy<any, any>, axis?: number): Expr<any, any> {
  return x.cumsum(axis);
}
export function cumprod<
  D extends Exclude<ComputableDType, "str" | TemporalDType>,
  S extends Shape,
  A extends number | undefined = undefined,
>(
  x: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis?: A,
): Expr<SumDType<D>, [A] extends [undefined] ? [number] : MutableShape<S>>;
export function cumprod(x: Lazy<any, any>, axis?: number): Expr<any, any> {
  return x.cumprod(axis);
}
export function matmul<
  A extends NumericDType,
  AS extends Shape,
  B extends NumericDType,
  BS extends Shape,
>(
  a: Lazy<NoInfer<A>, AS> & { readonly dtype: A },
  b: Lazy<NoInfer<B>, BS> & { readonly dtype: B } & MatMulCheck<AS, BS>,
): Expr<Promote<A, B>, MatMulShape<AS, BS>> {
  return new Expr(
    native(() => binding.NativeExpr.matmul(a.nativeExpr(), b.nativeExpr())),
    a.context ?? b.context,
  );
}

/** `np.save`: writes NPY without overwriting unless requested. */
export function save(
  path: string,
  x: NMathArray<ComputableDType, Shape>,
  options: { overwrite?: boolean; signal?: AbortSignal } = {},
): Promise<void> {
  return x.save(path, options);
}

/** `np.load` for NPY files. */
export function load(
  context: NMathContext,
  path: string,
  options: { signal?: AbortSignal } = {},
) {
  return context.load(path, options);
}

export const random = {
  /** `np.random.default_rng(seed)`. */
  defaultRng(context: NMathContext, seed: number | bigint): NMathGenerator {
    return context.generator(seed);
  },
};

export type { Operand };

/** Explicit JS adaptation of NumPy positional and keyword archive members. */
export interface NpzInputs {
  positional?: readonly NMathArray<any, any>[];
  named?: ReadonlyMap<string, NMathArray<any, any>>;
}

function saveArchive(
  context: NMathContext,
  path: string,
  inputs: NpzInputs,
  compressed: boolean,
  options: { overwrite?: boolean; signal?: AbortSignal },
): Promise<void> {
  const arrays = new Map(inputs.named);
  for (const [i, array] of (inputs.positional ?? []).entries()) {
    const name = `arr_${i}`;
    if (arrays.has(name)) {
      return Promise.reject(
        new TypeError(`Duplicate NPZ member name: ${name}.`),
      );
    }
    arrays.set(name, array);
  }
  return context.saveNpz(path.endsWith(".npz") ? path : `${path}.npz`, arrays, {
    ...options,
    compressed,
  });
}

export function savez(
  context: NMathContext,
  path: string,
  inputs: NpzInputs = {},
  options: { overwrite?: boolean; signal?: AbortSignal } = {},
): Promise<void> {
  return saveArchive(context, path, inputs, false, options);
}

export function savez_compressed(
  context: NMathContext,
  path: string,
  inputs: NpzInputs = {},
  options: { overwrite?: boolean; signal?: AbortSignal } = {},
): Promise<void> {
  return saveArchive(context, path, inputs, true, options);
}

/** Explicit NPZ load; preserves names safely in a Map and scalars as arrays. */
export function loadNpz(
  context: NMathContext,
  path: string,
  options: { signal?: AbortSignal } = {},
) {
  return context.loadNpz(path, options);
}

export * as linalg from "./linalg.js";
export * as fft from "./fft.js";

export { polynomial, polyval, polyder } from "./polynomial.js";
export type {
  PolynomialDType,
  PolynomialShape,
  PolyvalDType,
  PolyvalShape,
  PolyderDType,
  DerivativeShape,
  PolynomialDerivativeOptions,
} from "./polynomial.js";
