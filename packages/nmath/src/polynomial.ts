// SPDX-License-Identifier: Apache-2.0
import { Complex, Expr, type Lazy } from "./core.js";
import { native } from "./errors.js";
import { binding } from "./native.js";
import type { NumericalDType } from "./fft.js";
import type { Broadcast, Promote, Shape } from "./types.js";
import type {
  DerivativeActive,
  DerivativeOrder,
  DerivativeShape,
} from "./layout-types.js";
export type { DerivativeShape } from "./layout-types.js";

type Tail<S extends Shape> = S extends readonly [
  number,
  ...infer R extends Shape,
]
  ? R
  : S extends readonly []
    ? []
    : number[];
type MaybeEmpty<S extends Shape> = S extends readonly [
  infer N extends number,
  ...Shape,
]
  ? 0 extends N
    ? true
    : false
  : number extends S["length"]
    ? true
    : false;
type Empty<S extends Shape> = S extends readonly [0, ...Shape] ? true : false;
type NumericFloat<D extends NumericalDType> = D extends
  "float16" | "float32" | "float64" | "complex64" | "complex128"
  ? D
  : "float64";
export type PolynomialDType<
  C extends NumericalDType,
  X extends NumericalDType,
> = Promote<NumericFloat<C>, X extends "bool" ? "int64" : X>;
export type PolynomialShape<
  C extends Shape,
  X extends Shape,
  T extends boolean = true,
> = T extends true ? [...Tail<C>, ...X] : Broadcast<Tail<C>, X>;
export type PolyvalDType<
  C extends NumericalDType,
  CS extends Shape,
  X extends NumericalDType,
> =
  Empty<CS> extends true
    ? X
    : MaybeEmpty<CS> extends true
      ? X | Promote<C, X>
      : Promote<C, X>;
export type PolyvalShape<C extends Shape, X extends Shape> =
  Empty<C> extends true
    ? X
    : MaybeEmpty<C> extends true
      ? X | Broadcast<Tail<C>, X>
      : Broadcast<Tail<C>, X>;
type PolynomialCheck<C extends Shape, X extends Shape, T extends boolean> =
  Empty<C> extends true
    ? never
    : [PolynomialShape<C, X, T>] extends [never]
      ? never
      : unknown;
type PolyvalCheck<C extends Shape, X extends Shape> = C extends readonly []
  ? never
  : [PolyvalShape<C, X>] extends [never]
    ? never
    : unknown;

function evaluate(
  c: Lazy<any, any>,
  x: Lazy<any, any>,
  ascending: boolean,
  tensor: boolean,
): Expr<any, any> {
  return new Expr(
    native(() =>
      binding.NativeExpr.polynomial(
        c.nativeExpr(),
        x.nativeExpr(),
        ascending,
        tensor,
      ),
    ),
    c.context ?? x.context,
  );
}

/** Descending coefficients, as in NumPy's legacy `polyval(p, x)`. */
export function polyval<
  C extends NumericalDType,
  CS extends Shape,
  X extends NumericalDType,
  XS extends Shape,
>(
  coefficients: Lazy<NoInfer<C>, CS> & { readonly dtype: C } & PolyvalCheck<
      CS,
      XS
    >,
  points: Lazy<NoInfer<X>, XS> & { readonly dtype: X },
): Expr<PolyvalDType<C, CS, X>, PolyvalShape<CS, XS>> {
  return evaluate(coefficients, points, false, false);
}

/** Ascending coefficients, with a polynomial batch before the points' axes. */
type Tensor<O extends { tensor?: boolean }> = O extends {
  tensor: infer T extends boolean;
}
  ? T
  : "tensor" extends keyof O
    ? true | Exclude<O["tensor"], undefined>
    : true;
function seriesPolyval<
  C extends NumericalDType,
  CS extends Shape,
  X extends NumericalDType,
  XS extends Shape,
  const O extends { tensor?: boolean } = {},
>(
  points: Lazy<NoInfer<X>, XS> & { readonly dtype: X },
  coefficients: Lazy<NoInfer<C>, CS> & { readonly dtype: C } & PolynomialCheck<
      CS,
      XS,
      Tensor<O>
    >,
  options: O = {} as O,
): Expr<PolynomialDType<C, X>, PolynomialShape<CS, XS, Tensor<O>>> {
  if (
    options === null ||
    typeof options !== "object" ||
    Object.keys(options).some((key) => key !== "tensor")
  ) {
    throw new TypeError(
      "Unsupported polynomial option. Use { tensor: boolean }.",
    );
  }
  if (options.tensor !== undefined && typeof options.tensor !== "boolean") {
    throw new TypeError("Polynomial tensor must be a boolean.");
  }
  return evaluate(coefficients, points, true, options.tensor ?? true);
}

export interface PolynomialDerivativeOptions {
  m?: number;
  axis?: number;
  scl?: number | Complex;
}
type OptionValue<O, K extends PropertyKey, Default extends number> =
  O extends Record<K, infer V extends number>
    ? V
    : K extends keyof O
      ? Default | Extract<O[K], number>
      : Default;
type DerivativeCheck<
  S extends Shape,
  M extends number,
  A extends number,
  Asc extends boolean,
> = [DerivativeShape<S, M, A, Asc>] extends [never] ? never : unknown;
type ScaleCheck<D extends NumericalDType, S extends Shape, O> = O extends {
  scl: Complex;
}
  ? D extends "complex64" | "complex128"
    ? unknown
    : [
          DerivativeActive<
            S,
            OptionValue<O, "m", 1>,
            OptionValue<O, "axis", 0>
          >,
        ] extends [true]
      ? never
      : unknown
  : unknown;
export type PolyderDType<
  D extends NumericalDType,
  M extends number = 1,
> = M extends 0
  ? D
  : number extends M
    ? D | Promote<D, "int64">
    : Promote<D, "int64">;

function derivative(
  c: Lazy<any, any>,
  ascending: boolean,
  m: number,
  axis: number,
  scl: number | Complex,
): Expr<any, any> {
  if (!Number.isSafeInteger(m) || m < 0)
    throw new TypeError("Derivative order must be a nonnegative safe integer.");
  if (!Number.isSafeInteger(axis))
    throw new TypeError("Derivative axis must be a safe integer.");
  if (typeof scl !== "number" && !(scl instanceof Complex))
    throw new TypeError("Derivative scale must be a number or Complex scalar.");
  return new Expr(
    native(() =>
      binding.NativeExpr.polynomialDerivative(
        c.nativeExpr(),
        ascending,
        m,
        axis,
        scl,
      ),
    ),
    c.context,
  );
}

/** Differentiate a descending rank-one coefficient array. */
export function polyder<
  D extends NumericalDType,
  S extends Shape,
  const M extends number = 1,
>(
  coefficients: Lazy<NoInfer<D>, S> & { readonly dtype: D } & DerivativeCheck<
      S,
      M,
      0,
      false
    >,
  m: M & DerivativeOrder<M> = 1 as M & DerivativeOrder<M>,
): Expr<PolyderDType<D, NoInfer<M>>, DerivativeShape<S, NoInfer<M>, 0, false>> {
  return derivative(coefficients, false, m, 0, 1);
}

/** Differentiate ascending coefficients along a selected axis. */
function seriesPolyder<
  D extends NumericalDType,
  S extends Shape,
  const O extends PolynomialDerivativeOptions = {},
>(
  coefficients: Lazy<NoInfer<D>, S> & { readonly dtype: D } & DerivativeCheck<
      S,
      OptionValue<O, "m", 1>,
      OptionValue<O, "axis", 0>,
      true
    >,
  options: O &
    Record<Exclude<keyof O, keyof PolynomialDerivativeOptions>, never> &
    ScaleCheck<D, S, O> = {} as O &
    Record<Exclude<keyof O, keyof PolynomialDerivativeOptions>, never> &
    ScaleCheck<D, S, O>,
): Expr<
  NumericFloat<D>,
  DerivativeShape<S, OptionValue<O, "m", 1>, OptionValue<O, "axis", 0>>
> {
  if (
    options === null ||
    typeof options !== "object" ||
    Array.isArray(options) ||
    Object.keys(options).some((key) => !["m", "axis", "scl"].includes(key))
  )
    throw new TypeError("Unsupported derivative option. Use { m, axis, scl }.");
  return derivative(
    coefficients,
    true,
    options.m === undefined ? 1 : options.m,
    options.axis === undefined ? 0 : options.axis,
    options.scl === undefined ? 1 : options.scl,
  );
}

/** NumPy's polynomial package and power-series namespace. */
export const polynomial = {
  polynomial: { polyval: seriesPolyval, polyder: seriesPolyder },
};
