// SPDX-License-Identifier: Apache-2.0
// NumPy elementwise math ufuncs. Result dtypes follow the NumPy 2.5.4 loop tables.

import { Expr, Lazy } from "./core.js";
import { native } from "./errors.js";
import { binding } from "./native.js";
import type { FloatDType, NumericDType, OrderedDType, Shape } from "./types.js";

/** NumPy's smallest floating-point loop that holds a real input exactly. */
export type MathFloatDType<D extends OrderedDType> = D extends
  "bool" | "int8" | "uint8" | "float16"
  ? "float16"
  : D extends "int16" | "uint16" | "float32"
    ? "float32"
    : "float64";

type Arg<D extends NumericDType, S extends Shape> = Lazy<NoInfer<D>, S> & {
  readonly dtype: D;
};

/** Real-valued floating-point functions of real inputs. */
export interface FloatUfunc {
  <D extends OrderedDType, S extends Shape>(
    x: Arg<D, S>,
  ): Expr<MathFloatDType<D>, S>;
}
/** Bool-valued tests of any numeric input. */
export interface PredicateUfunc {
  <D extends NumericDType, S extends Shape>(x: Arg<D, S>): Expr<"bool", S>;
}
/** Bool-valued tests of real inputs. */
export interface RealPredicateUfunc {
  <D extends OrderedDType, S extends Shape>(x: Arg<D, S>): Expr<"bool", S>;
}
/** Same-dtype functions of real inputs. */
export interface SameDTypeUfunc {
  <D extends OrderedDType, S extends Shape>(x: Arg<D, S>): Expr<D, S>;
}
/** `square`: bools use the int8 loop. */
export interface SquareUfunc {
  <D extends OrderedDType, S extends Shape>(
    x: Arg<D, S>,
  ): Expr<D extends "bool" ? "int8" : D, S>;
}
/** `sign`: NumPy has no bool loop. */
export interface SignUfunc {
  <D extends Exclude<OrderedDType, "bool">, S extends Shape>(
    x: Arg<D, S>,
  ): Expr<D, S>;
}
/** `reciprocal`: floating-point inputs only. */
export interface ReciprocalUfunc {
  <D extends FloatDType, S extends Shape>(x: Arg<D, S>): Expr<D, S>;
}
/** `conjugate`: bools use the int8 loop; complex values negate the imaginary part. */
export interface ConjugateUfunc {
  <D extends NumericDType, S extends Shape>(
    x: Arg<D, S>,
  ): Expr<D extends "bool" ? "int8" : D, S>;
}

function math(op: string, x: Lazy<any, any>): Expr<any, any> {
  if (!(x instanceof Lazy)) {
    throw new TypeError("The operand must be an NMath array or expression.");
  }
  return new Expr(
    native(() => binding.NativeExpr.math(op, x.nativeExpr())),
    x.context,
  );
}
function ufunc<F>(op: string): F {
  return ((x: Lazy<any, any>) => math(op, x)) as F;
}

export const sqrt = ufunc<FloatUfunc>("sqrt");
export const cbrt = ufunc<FloatUfunc>("cbrt");
export const exp = ufunc<FloatUfunc>("exp");
export const exp2 = ufunc<FloatUfunc>("exp2");
export const expm1 = ufunc<FloatUfunc>("expm1");
export const log = ufunc<FloatUfunc>("log");
export const log2 = ufunc<FloatUfunc>("log2");
export const log10 = ufunc<FloatUfunc>("log10");
export const log1p = ufunc<FloatUfunc>("log1p");
export const sin = ufunc<FloatUfunc>("sin");
export const cos = ufunc<FloatUfunc>("cos");
export const tan = ufunc<FloatUfunc>("tan");
export const arcsin = ufunc<FloatUfunc>("arcsin");
export const arccos = ufunc<FloatUfunc>("arccos");
export const arctan = ufunc<FloatUfunc>("arctan");
export const sinh = ufunc<FloatUfunc>("sinh");
export const cosh = ufunc<FloatUfunc>("cosh");
export const tanh = ufunc<FloatUfunc>("tanh");
export const arcsinh = ufunc<FloatUfunc>("arcsinh");
export const arccosh = ufunc<FloatUfunc>("arccosh");
export const arctanh = ufunc<FloatUfunc>("arctanh");
export const deg2rad = ufunc<FloatUfunc>("deg2rad");
export const rad2deg = ufunc<FloatUfunc>("rad2deg");
export const fabs = ufunc<FloatUfunc>("fabs");
export const rint = ufunc<FloatUfunc>("rint");
export const spacing = ufunc<FloatUfunc>("spacing");
export const absolute = ufunc<SameDTypeUfunc>("absolute");
export const square = ufunc<SquareUfunc>("square");
export const reciprocal = ufunc<ReciprocalUfunc>("reciprocal");
export const sign = ufunc<SignUfunc>("sign");
export const floor = ufunc<SameDTypeUfunc>("floor");
export const ceil = ufunc<SameDTypeUfunc>("ceil");
export const trunc = ufunc<SameDTypeUfunc>("trunc");
export const conjugate = ufunc<ConjugateUfunc>("conjugate");
export const isnan = ufunc<PredicateUfunc>("isnan");
export const isinf = ufunc<PredicateUfunc>("isinf");
export const isfinite = ufunc<PredicateUfunc>("isfinite");
export const signbit = ufunc<RealPredicateUfunc>("signbit");
export const logical_not = ufunc<PredicateUfunc>("logical_not");

// NumPy 2 aliases.
export const abs = absolute;
export const conj = conjugate;
export const asin = arcsin;
export const acos = arccos;
export const atan = arctan;
export const asinh = arcsinh;
export const acosh = arccosh;
export const atanh = arctanh;
