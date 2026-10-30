// SPDX-License-Identifier: Apache-2.0
import { Expr, type Lazy } from "./core.js";
import { native } from "./errors.js";
import { binding } from "./native.js";
import type { NumericalDType } from "./fft.js";
import type { Broadcast, Promote, Shape } from "./types.js";

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

/** NumPy's polynomial package and power-series namespace. */
export const polynomial = { polynomial: { polyval: seriesPolyval } };
