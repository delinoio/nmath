// SPDX-License-Identifier: Apache-2.0
// NumPy elementwise math ufuncs. Result dtypes follow the NumPy 2.5.4 loop tables.

import {
  type BinaryOperator,
  type Complex,
  Expr,
  Lazy,
  type Operand,
} from "./core.js";
import { native } from "./errors.js";
import { binding } from "./native.js";
import type {
  Broadcast,
  BroadcastCheck,
  FloatDType,
  IntDType,
  NumericDType,
  OrderedDType,
  Promote,
  PromoteWeak,
  Shape,
} from "./types.js";

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
/** `invert`: bool and integer inputs keep their dtype. */
export interface InvertUfunc {
  <D extends BitwiseDType, S extends Shape>(x: Arg<D, S>): Expr<D, S>;
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
export const invert = ufunc<InvertUfunc>("invert");

/** Dtypes with bitwise loops. */
export type BitwiseDType = "bool" | IntDType;
type Bitwise<D extends NumericDType> = [D] extends [BitwiseDType] ? D : never;
/** Makes the second operand unassignable when no common integer loop exists. */
type BitwiseCheck<A extends NumericDType, B extends NumericDType> = [
  Bitwise<Promote<A, B>>,
] extends [never]
  ? { readonly "operands have no common integer dtype": [A, B] }
  : unknown;
type Shifted<D extends NumericDType> =
  Bitwise<D> extends "bool" ? "int8" : Bitwise<D>;

/**
 * `bitwise_and`, `bitwise_or` and `bitwise_xor`. Scalars must be `bigint` or
 * `boolean`: a JS `number` is a weak float and has no bitwise loop.
 */
export interface BitwiseFunction {
  <
    A extends BitwiseDType,
    AS extends Shape,
    B extends BitwiseDType,
    BS extends Shape,
  >(
    a: Arg<A, AS>,
    b: Arg<B, BS> & BroadcastCheck<AS, BS> & BitwiseCheck<A, B>,
  ): Expr<Bitwise<Promote<A, B>>, Broadcast<AS, BS>>;
  <A extends BitwiseDType, AS extends Shape, W extends bigint | boolean>(
    a: Arg<A, AS>,
    b: W,
  ): Expr<Bitwise<PromoteWeak<A, W>>, AS>;
}
/** `left_shift` and `right_shift`; bools use the int8 loop. */
export interface ShiftFunction {
  <
    A extends BitwiseDType,
    AS extends Shape,
    B extends BitwiseDType,
    BS extends Shape,
  >(
    a: Arg<A, AS>,
    b: Arg<B, BS> & BroadcastCheck<AS, BS> & BitwiseCheck<A, B>,
  ): Expr<Shifted<Promote<A, B>>, Broadcast<AS, BS>>;
  <A extends BitwiseDType, AS extends Shape, W extends bigint | boolean>(
    a: Arg<A, AS>,
    b: W,
  ): Expr<Shifted<PromoteWeak<A, W>>, AS>;
}
/** `logical_and`, `logical_or` and `logical_xor` by truthiness. */
export interface LogicalFunction {
  <
    A extends NumericDType,
    AS extends Shape,
    B extends NumericDType,
    BS extends Shape,
  >(
    a: Arg<A, AS>,
    b: Arg<B, BS> & BroadcastCheck<AS, BS>,
  ): Expr<"bool", Broadcast<AS, BS>>;
  <A extends NumericDType, AS extends Shape>(
    a: Arg<A, AS>,
    b: number | bigint | boolean | Complex,
  ): Expr<"bool", AS>;
}

function binaryUfunc<F>(op: BinaryOperator): F {
  return ((a: Operand, b: Operand) => {
    if (a instanceof Lazy) {
      return a.binaryOp(op, b);
    }
    if (b instanceof Lazy) {
      return b.reversedBinaryOp(op, a);
    }
    throw new TypeError(
      "At least one operand must be an NMath array or expression.",
    );
  }) as F;
}

export const bitwise_and = binaryUfunc<BitwiseFunction>("bitwise_and");
export const bitwise_or = binaryUfunc<BitwiseFunction>("bitwise_or");
export const bitwise_xor = binaryUfunc<BitwiseFunction>("bitwise_xor");
export const left_shift = binaryUfunc<ShiftFunction>("left_shift");
export const right_shift = binaryUfunc<ShiftFunction>("right_shift");
export const logical_and = binaryUfunc<LogicalFunction>("logical_and");
export const logical_or = binaryUfunc<LogicalFunction>("logical_or");
export const logical_xor = binaryUfunc<LogicalFunction>("logical_xor");

/** `maximum`, `minimum`, `fmax` and `fmin`: same promotion as arithmetic. */
export interface ExtremumFunction {
  <
    A extends NumericDType,
    AS extends Shape,
    B extends NumericDType,
    BS extends Shape,
  >(
    a: Arg<A, AS>,
    b: Arg<B, BS> & BroadcastCheck<AS, BS>,
  ): Expr<Promote<A, B>, Broadcast<AS, BS>>;
  <
    A extends NumericDType,
    AS extends Shape,
    W extends number | bigint | boolean | Complex,
  >(
    a: Arg<A, AS>,
    b: W,
  ): Expr<PromoteWeak<A, W>, AS>;
}
type IntLoop<D extends NumericDType> = D extends "bool" ? "int8" : D;
/** `fmod` and `floor_divide` on real operands; bools use the int8 loop. */
export interface DivisionFunction {
  <
    A extends OrderedDType,
    AS extends Shape,
    B extends OrderedDType,
    BS extends Shape,
  >(
    a: Arg<A, AS>,
    b: Arg<B, BS> & BroadcastCheck<AS, BS>,
  ): Expr<IntLoop<Promote<A, B>>, Broadcast<AS, BS>>;
  <
    A extends OrderedDType,
    AS extends Shape,
    W extends number | bigint | boolean,
  >(
    a: Arg<A, AS>,
    b: W,
  ): Expr<IntLoop<PromoteWeak<A, W>>, AS>;
}
type FloatLoop<D extends NumericDType> = D extends OrderedDType
  ? MathFloatDType<D>
  : never;
/** Float-valued binary functions of real operands. Each operand selects its float loop. */
export interface FloatBinaryFunction {
  <
    A extends OrderedDType,
    AS extends Shape,
    B extends OrderedDType,
    BS extends Shape,
  >(
    a: Arg<A, AS>,
    b: Arg<B, BS> & BroadcastCheck<AS, BS>,
  ): Expr<Promote<MathFloatDType<A>, MathFloatDType<B>>, Broadcast<AS, BS>>;
  <
    A extends OrderedDType,
    AS extends Shape,
    W extends number | bigint | boolean,
  >(
    a: Arg<A, AS>,
    b: W,
  ): Expr<FloatLoop<PromoteWeak<A, W>>, AS>;
}

export const maximum = binaryUfunc<ExtremumFunction>("maximum");
export const minimum = binaryUfunc<ExtremumFunction>("minimum");
export const fmax = binaryUfunc<ExtremumFunction>("fmax");
export const fmin = binaryUfunc<ExtremumFunction>("fmin");
export const fmod = binaryUfunc<DivisionFunction>("fmod");
export const floor_divide = binaryUfunc<DivisionFunction>("floor_divide");
export const copysign = binaryUfunc<FloatBinaryFunction>("copysign");
export const nextafter = binaryUfunc<FloatBinaryFunction>("nextafter");
export const arctan2 = binaryUfunc<FloatBinaryFunction>("arctan2");
export const hypot = binaryUfunc<FloatBinaryFunction>("hypot");
export const heaviside = binaryUfunc<FloatBinaryFunction>("heaviside");
export const logaddexp = binaryUfunc<FloatBinaryFunction>("logaddexp");
export const logaddexp2 = binaryUfunc<FloatBinaryFunction>("logaddexp2");

// NumPy 2 aliases.
export const atan2 = arctan2;
export const bitwise_not = invert;
export const bitwise_invert = invert;
export const bitwise_left_shift = left_shift;
export const bitwise_right_shift = right_shift;
export const abs = absolute;
export const conj = conjugate;
export const asin = arcsin;
export const acos = arccos;
export const atan = arctan;
export const asinh = arcsinh;
export const acosh = arccosh;
export const atanh = arctanh;
