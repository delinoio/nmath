// SPDX-License-Identifier: Apache-2.0
import { Expr, type Lazy } from "./core.js";
import { native } from "./errors.js";
import { binding } from "./native.js";
import type { ComputableDType, Shape } from "./types.js";

export type NumericalDType = Extract<
  ComputableDType,
  | "bool"
  | "int8"
  | "int16"
  | "int32"
  | "int64"
  | "uint8"
  | "uint16"
  | "uint32"
  | "uint64"
  | "float16"
  | "float32"
  | "float64"
  | "complex64"
  | "complex128"
>;

export interface FftOptions {
  n?: number;
  axis?: number;
  norm?: "backward" | "forward" | "ortho";
}
export type FftDType<D extends NumericalDType> = D extends
  "float16" | "float32" | "complex64"
  ? "complex64"
  : "complex128";
export type IrfftDType<D extends NumericalDType> = D extends "float16"
  ? "float16"
  : D extends "float32" | "complex64"
    ? "float32"
    : "float64";
type DynamicShape<S extends Shape> = { [K in keyof S]: number };
type LastAxis<S extends Shape, N extends number> = S extends readonly [
  ...infer B extends Shape,
  number,
]
  ? [...B, N]
  : number[];
type PositiveAxis<
  S extends Shape,
  A extends number,
  N extends number,
  I extends unknown[] = [],
> = S extends readonly [infer H extends number, ...infer T extends Shape]
  ? I["length"] extends A
    ? [N, ...T]
    : [H, ...PositiveAxis<T, A, N, [...I, unknown]>]
  : S;
type LiteralFftShape<
  S extends Shape,
  O extends FftOptions = {},
  Real extends boolean = false,
> = O extends { n: infer N extends number }
  ? O extends { axis: infer A extends number }
    ? A extends -1
      ? LastAxis<S, Real extends true ? number : N>
      : number extends A
        ? DynamicShape<S>
        : `${A}` extends `-${string}`
          ? DynamicShape<S>
          : PositiveAxis<S, A, Real extends true ? number : N>
    : LastAxis<S, Real extends true ? number : N>
  : Real extends false
    ? S
    : O extends { axis: infer A extends number }
      ? A extends -1
        ? LastAxis<S, number>
        : number extends A
          ? DynamicShape<S>
          : `${A}` extends `-${string}`
            ? DynamicShape<S>
            : PositiveAxis<S, A, number>
      : LastAxis<S, number>;
export type FftShape<
  S extends Shape,
  O extends FftOptions = {},
  Real extends boolean = false,
> = O extends { n: number }
  ? "axis" extends keyof O
    ? O extends { axis: number }
      ? LiteralFftShape<S, O, Real>
      : DynamicShape<S>
    : LiteralFftShape<S, O, Real>
  : "n" extends keyof O
    ? DynamicShape<S>
    : Real extends true
      ? "axis" extends keyof O
        ? O extends { axis: number }
          ? LiteralFftShape<S, O, Real>
          : DynamicShape<S>
        : LiteralFftShape<S, O, Real>
      : LiteralFftShape<S, O, Real>;
function expression<
  D extends NumericalDType,
  S extends Shape,
  R extends ComputableDType,
  RS extends Shape,
>(op: string, a: Lazy<D, S>, options: FftOptions): Expr<R, RS> {
  for (const key of Object.keys(options))
    if (!["n", "axis", "norm"].includes(key))
      throw new TypeError(`Unsupported FFT option: ${key}`);
  if (
    options.n !== undefined &&
    (!Number.isSafeInteger(options.n) ||
      options.n <= 0 ||
      options.n > 0xffffffff)
  )
    throw new RangeError("FFT n must be a positive unsigned 32-bit integer.");
  if (options.axis !== undefined && !Number.isSafeInteger(options.axis))
    throw new RangeError("FFT axis must be a safe integer.");
  return new Expr(
    native(() =>
      binding.NativeExpr.fft(
        op,
        a.nativeExpr(),
        options.n,
        options.axis ?? -1,
        options.norm ?? "backward",
      ),
    ),
    a.context,
  );
}
export function fft<
  D extends NumericalDType,
  S extends Shape,
  const O extends FftOptions = {},
>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  options: O = {} as O,
): Expr<FftDType<D>, FftShape<S, O>> {
  return expression("fft", a, options);
}
export function ifft<
  D extends NumericalDType,
  S extends Shape,
  const O extends FftOptions = {},
>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  options: O = {} as O,
): Expr<FftDType<D>, FftShape<S, O>> {
  return expression("ifft", a, options);
}
export function rfft<
  D extends Exclude<NumericalDType, "complex64" | "complex128">,
  S extends Shape,
  const O extends FftOptions = {},
>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  options: O = {} as O,
): Expr<FftDType<D>, FftShape<S, O, true>> {
  return expression("rfft", a, options);
}
export function irfft<
  D extends NumericalDType,
  S extends Shape,
  const O extends FftOptions = {},
>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  options: O = {} as O,
): Expr<IrfftDType<D>, FftShape<S, O, true>> {
  return expression("irfft", a, options);
}
