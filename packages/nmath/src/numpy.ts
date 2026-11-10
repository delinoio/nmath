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
import {
  add,
  div,
  equal,
  greater,
  greater_equal,
  less,
  less_equal,
  mod,
  mul,
  not_equal,
  pow,
  sub,
  where,
} from "./index.js";
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
  VarianceDType,
  TrueDivide,
  ReduceOptions,
  ReduceShape,
  ScalarOf,
  Shape,
  TemporalDType,
  TypedArrayOf,
} from "./types.js";

export { equal, greater, greater_equal, less, less_equal, not_equal, where };
export { ellipsis, indices, mask, newaxis, slice } from "./core.js";
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
export function transpose<D extends ComputableDType, S extends Shape>(
  input: Lazy<NoInfer<D>, S> & { readonly dtype: D },
): Expr<D, TransposeShape<S, undefined>>;
export function transpose<
  D extends ComputableDType,
  S extends Shape,
  const A extends Shape | undefined,
>(
  input: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axes: A & TransposeCheck<S, A>,
): Expr<D, TransposeShape<S, A>>;
export function transpose<D extends ComputableDType, S extends Shape>(
  input: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axes?: Shape,
): Expr<D, any> {
  // The public overloads check the argument and result; forwarding keeps undefined intact.
  return input.transpose<Shape | undefined>(
    axes as (Shape | undefined) & TransposeCheck<S, Shape | undefined>,
  ) as Expr<D, any>;
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
    dtype === "str"
      ? "1"
      : dtype === "datetime64" ||
          dtype === "timedelta64" ||
          dtype === "int64" ||
          dtype === "uint64"
        ? 1n
        : dtype === "bool"
          ? true
          : 1
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
type ArangeDType = "int32" | "uint32" | "float32" | "float64";

export function arange(
  context: NMathContext,
  start: number,
  stop: number,
  step?: number,
): NMathArray<"float64", [number]>;
export function arange<const D extends ArangeDType | undefined>(
  context: NMathContext,
  start: number,
  stop: number,
  step: number | undefined,
  dtype: D,
): NMathArray<D extends ArangeDType ? D : "float64", [number]>;
export function arange(
  context: NMathContext,
  start: number,
  stop: number,
  step = 1,
  dtype: ArangeDType = "float64",
): NMathArray<ArangeDType, [number]> {
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
  const A extends number | readonly number[] | undefined = undefined,
  K extends boolean = false,
>(
  x: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis?: A,
  options?: ReduceOptions<K>,
): Expr<SumDType<D>, ReduceShape<S, A, K>>;
export function sum(
  x: Lazy<any, any>,
  axis?: number | readonly number[],
  options?: ReduceOptions,
): Expr<any, any> {
  return x.sum(axis, options);
}
export function mean<
  D extends Exclude<ComputableDType, "str" | "datetime64">,
  S extends Shape,
  const A extends number | readonly number[] | undefined = undefined,
  K extends boolean = false,
>(
  x: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis?: A,
  options?: ReduceOptions<K>,
): Expr<D extends "timedelta64" ? D : TrueDivide<D>, ReduceShape<S, A, K>>;
export function mean(
  x: Lazy<any, any>,
  axis?: number | readonly number[],
  options?: ReduceOptions,
): Expr<any, any> {
  return x.mean(axis, options);
}
export function min<
  D extends Exclude<ComputableDType, "str">,
  S extends Shape,
  const A extends number | readonly number[] | undefined = undefined,
  K extends boolean = false,
>(
  x: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis?: A,
  options?: ReduceOptions<K>,
): Expr<D, ReduceShape<S, A, K>>;
export function min(
  x: Lazy<any, any>,
  axis?: number | readonly number[],
  options?: ReduceOptions,
): Expr<any, any> {
  return x.min(axis, options);
}
export function max<
  D extends Exclude<ComputableDType, "str">,
  S extends Shape,
  const A extends number | readonly number[] | undefined = undefined,
  K extends boolean = false,
>(
  x: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis?: A,
  options?: ReduceOptions<K>,
): Expr<D, ReduceShape<S, A, K>>;
export function max(
  x: Lazy<any, any>,
  axis?: number | readonly number[],
  options?: ReduceOptions,
): Expr<any, any> {
  return x.max(axis, options);
}
export function prod<
  D extends Exclude<ComputableDType, "str" | TemporalDType>,
  S extends Shape,
  const A extends number | readonly number[] | undefined = undefined,
  K extends boolean = false,
>(
  x: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis?: A,
  options?: ReduceOptions<K>,
): Expr<SumDType<D>, ReduceShape<S, A, K>>;
export function prod(
  x: Lazy<any, any>,
  axis?: number | readonly number[],
  options?: ReduceOptions,
): Expr<any, any> {
  return x.prod(axis, options);
}
export function argmin<
  D extends Exclude<ComputableDType, "str" | TemporalDType>,
  S extends Shape,
  A extends number | undefined = undefined,
  K extends boolean = false,
>(
  x: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis?: A,
  options?: ReduceOptions<K>,
): Expr<"int64", ReduceShape<S, A, K>>;
export function argmin(
  x: Lazy<any, any>,
  axis?: number,
  options?: ReduceOptions,
): Expr<any, any> {
  return x.argmin(axis, options);
}
export function argmax<
  D extends Exclude<ComputableDType, "str" | TemporalDType>,
  S extends Shape,
  A extends number | undefined = undefined,
  K extends boolean = false,
>(
  x: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis?: A,
  options?: ReduceOptions<K>,
): Expr<"int64", ReduceShape<S, A, K>>;
export function argmax(
  x: Lazy<any, any>,
  axis?: number,
  options?: ReduceOptions,
): Expr<any, any> {
  return x.argmax(axis, options);
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
export function var_<
  D extends Exclude<ComputableDType, "str" | TemporalDType>,
  S extends Shape,
  A extends number | undefined = undefined,
  K extends boolean = false,
>(
  x: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis?: A,
  ddof?: number,
  options?: ReduceOptions<K>,
): Expr<VarianceDType<D>, ReduceShape<S, A, K>>;
export function var_(
  x: Lazy<any, any>,
  axis?: number,
  ddof?: number,
  options?: ReduceOptions,
): Expr<any, any> {
  return x.var(axis, ddof, options);
}
export function std<
  D extends Exclude<ComputableDType, "str" | TemporalDType>,
  S extends Shape,
  A extends number | undefined = undefined,
  K extends boolean = false,
>(
  x: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axis?: A,
  ddof?: number,
  options?: ReduceOptions<K>,
): Expr<VarianceDType<D>, ReduceShape<S, A, K>>;
export function std(
  x: Lazy<any, any>,
  axis?: number,
  ddof?: number,
  options?: ReduceOptions,
): Expr<any, any> {
  return x.std(axis, ddof, options);
}
export { var_ as var };
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

export * from "./ufunc.js";
export * from "./axes.js";
export * from "./creation.js";
export * from "./products.js";
export * from "./logic.js";
export * from "./joining.js";
export * from "./splitting.js";
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
