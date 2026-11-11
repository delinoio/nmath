// SPDX-License-Identifier: Apache-2.0
import { Expr, type Lazy, type NMathArray, type NMathContext } from "./core.js";
import { native } from "./errors.js";
import { binding } from "./native.js";
import type { ComputableDType, MutableShape, Shape } from "./types.js";

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
  : DynamicShape<S>;
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

export interface MultiFftOptions {
  s?: readonly number[];
  axes?: readonly number[];
  norm?: "backward" | "forward" | "ortho";
}
type MultiFftShape<S extends Shape, O extends MultiFftOptions> = O extends {
  s: readonly number[];
}
  ? DynamicShape<S>
  : MutableShape<S>;

function multiExpression(
  op: "fft" | "ifft",
  a: Lazy<any, any>,
  options: MultiFftOptions,
  lastTwo: boolean,
): Expr<any, any> {
  for (const key of Object.keys(options))
    if (!["s", "axes", "norm"].includes(key))
      throw new TypeError(`Unsupported FFT option: ${key}`);
  const rank = a.shape.length;
  let axes: readonly number[];
  if (options.axes) axes = [...options.axes];
  else if (lastTwo) {
    if (rank < 2)
      throw new RangeError("fft2 and ifft2 require at least two dimensions.");
    axes = [-2, -1];
  } else axes = Array.from({ length: rank }, (_, axis) => axis);
  const normalized = axes.map((axis) => (axis < 0 ? axis + rank : axis));
  for (const axis of normalized)
    if (!Number.isSafeInteger(axis) || axis < 0 || axis >= rank)
      throw new RangeError(
        "FFT axes must be safe integers inside the input rank.",
      );
  if (new Set(normalized).size !== normalized.length)
    throw new RangeError("FFT axes must not repeat.");
  if (options.s) {
    if (options.s.length !== axes.length)
      throw new RangeError(
        "FFT s must provide one length per transformed axis.",
      );
    for (const length of options.s)
      if (!Number.isSafeInteger(length) || length <= 0 || length > 0xffffffff)
        throw new RangeError(
          "FFT s lengths must be positive unsigned 32-bit integers.",
        );
  }
  let out: any = a;
  axes.forEach((axis, index) => {
    const fftOptions = { n: options.s?.[index], axis, norm: options.norm };
    out = op === "fft" ? fft(out, fftOptions) : ifft(out, fftOptions);
  });
  return out;
}

export function fft2<
  D extends NumericalDType,
  S extends Shape,
  const O extends MultiFftOptions = {},
>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  options: O = {} as O,
): Expr<FftDType<D>, MultiFftShape<S, O>> {
  return multiExpression("fft", a, options, true);
}

export function ifft2<
  D extends NumericalDType,
  S extends Shape,
  const O extends MultiFftOptions = {},
>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  options: O = {} as O,
): Expr<FftDType<D>, MultiFftShape<S, O>> {
  return multiExpression("ifft", a, options, true);
}

export function fftn<
  D extends NumericalDType,
  S extends Shape,
  const O extends MultiFftOptions = {},
>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  options: O = {} as O,
): Expr<FftDType<D>, MultiFftShape<S, O>> {
  return multiExpression("fft", a, options, false);
}

export function ifftn<
  D extends NumericalDType,
  S extends Shape,
  const O extends MultiFftOptions = {},
>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  options: O = {} as O,
): Expr<FftDType<D>, MultiFftShape<S, O>> {
  return multiExpression("ifft", a, options, false);
}

type Norm = "backward" | "forward" | "ortho";
/** Real input dtypes accepted by `rfft` and the transforms built on it. */
export type RealFftDType = Exclude<NumericalDType, "complex64" | "complex128">;
/** NumPy `_swap_direction`: Hermitian transforms reverse the scaling direction. */
function swapDirection(norm: Norm | undefined): Norm {
  if (norm === undefined || norm === "backward") return "forward";
  if (norm === "forward") return "backward";
  if (norm === "ortho") return "ortho";
  throw new TypeError(
    `Invalid norm value ${String(norm)}; should be "backward", "ortho" or "forward".`,
  );
}

function conjugate(a: Lazy<any, any>): Expr<any, any> {
  return new Expr(
    native(() => binding.NativeExpr.math("conjugate", a.nativeExpr())),
    a.context,
  );
}

/** NumPy `_cook_nd_args`: lengths and axes for real multi-axis transforms. */
function realAxes(
  a: Lazy<any, any>,
  options: MultiFftOptions,
  inverse: boolean,
  lastTwo: boolean,
): { s: number[]; axes: number[] } {
  for (const key of Object.keys(options))
    if (!["s", "axes", "norm"].includes(key))
      throw new TypeError(`Unsupported FFT option: ${key}`);
  const rank = a.shape.length;
  let axes: number[];
  if (options.axes) axes = [...options.axes];
  else if (options.s) axes = options.s.map((_, i) => i - options.s!.length);
  else if (lastTwo) axes = [-2, -1];
  else axes = Array.from({ length: rank }, (_, axis) => axis);
  if (lastTwo && rank < 2 && !options.axes)
    throw new RangeError("rfft2 and irfft2 require at least two dimensions.");
  if (axes.length === 0) throw new RangeError("FFT axes must not be empty.");
  const normalized = axes.map((axis) => (axis < 0 ? axis + rank : axis));
  for (const axis of normalized)
    if (!Number.isSafeInteger(axis) || axis < 0 || axis >= rank)
      throw new RangeError(
        "FFT axes must be safe integers inside the input rank.",
      );
  if (new Set(normalized).size !== normalized.length)
    throw new RangeError("FFT axes must not repeat.");
  let s: number[];
  if (options.s) {
    if (options.s.length !== axes.length)
      throw new RangeError(
        "FFT s must provide one length per transformed axis.",
      );
    s = [...options.s];
  } else {
    s = normalized.map((axis) => a.shape[axis]);
    if (inverse)
      s[s.length - 1] = (a.shape[normalized[normalized.length - 1]] - 1) * 2;
  }
  return { s, axes };
}

function rfftnExpression(
  a: Lazy<any, any>,
  options: MultiFftOptions,
  lastTwo: boolean,
) {
  const { s, axes } = realAxes(a, options, false, lastTwo);
  const last = axes.length - 1;
  let out: any = rfft(a, { n: s[last], axis: axes[last], norm: options.norm });
  for (let i = last - 1; i >= 0; i -= 1) {
    out = fft(out, { n: s[i], axis: axes[i], norm: options.norm });
  }
  return out as Expr<any, any>;
}

function irfftnExpression(
  a: Lazy<any, any>,
  options: MultiFftOptions,
  lastTwo: boolean,
) {
  const { s, axes } = realAxes(a, options, true, lastTwo);
  const last = axes.length - 1;
  let out: any = a;
  for (let i = 0; i < last; i += 1) {
    out = ifft(out, { n: s[i], axis: axes[i], norm: options.norm });
  }
  return irfft(out, {
    n: s[last],
    axis: axes[last],
    norm: options.norm,
  }) as Expr<any, any>;
}

/** Real input transform over the last two axes. */
export function rfft2<D extends RealFftDType, S extends Shape>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  options: MultiFftOptions = {},
): Expr<FftDType<D>, DynamicShape<S>> {
  return rfftnExpression(a, options, true);
}

/** Inverse of `rfft2`. */
export function irfft2<D extends NumericalDType, S extends Shape>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  options: MultiFftOptions = {},
): Expr<IrfftDType<D>, DynamicShape<S>> {
  return irfftnExpression(a, options, true);
}

/** Real input transform over all axes, or the given axes, in NumPy's order. */
export function rfftn<D extends RealFftDType, S extends Shape>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  options: MultiFftOptions = {},
): Expr<FftDType<D>, DynamicShape<S>> {
  return rfftnExpression(a, options, false);
}

/** Inverse of `rfftn`; the last axis defaults to `2 * (m - 1)` points. */
export function irfftn<D extends NumericalDType, S extends Shape>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  options: MultiFftOptions = {},
): Expr<IrfftDType<D>, DynamicShape<S>> {
  return irfftnExpression(a, options, false);
}

/** Transform of a Hermitian-symmetric signal given by its first half. */
export function hfft<D extends NumericalDType, S extends Shape>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  options: FftOptions = {},
): Expr<IrfftDType<D>, DynamicShape<S>> {
  const axis = options.axis ?? -1;
  const length = a.shape[axis < 0 ? axis + a.shape.length : axis];
  return irfft(conjugate(a), {
    ...options,
    n: options.n ?? (length - 1) * 2,
    norm: swapDirection(options.norm),
  }) as never;
}

/** Inverse of `hfft`: the conjugated real transform with swapped scaling. */
export function ihfft<D extends RealFftDType, S extends Shape>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  options: FftOptions = {},
): Expr<FftDType<D>, DynamicShape<S>> {
  return conjugate(
    rfft(a as never, {
      ...options,
      norm: swapDirection(options.norm),
    }) as never,
  ) as never;
}

function shiftExpression(
  op: "fftshift" | "ifftshift",
  x: Lazy<any, any>,
  axes?: readonly number[],
): Expr<any, any> {
  if (axes) {
    for (const axis of axes)
      if (!Number.isSafeInteger(axis))
        throw new RangeError("FFT shift axes must be safe integers.");
    if (new Set(axes).size !== axes.length)
      throw new RangeError("FFT shift axes must not repeat.");
  }
  return new Expr(
    native(() =>
      binding.NativeExpr.shift(
        op,
        x.nativeExpr(),
        axes === undefined ? undefined : [...axes],
      ),
    ),
    x.context,
  );
}

export function fftshift<D extends ComputableDType, S extends Shape>(
  x: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axes?: readonly number[],
): Expr<D, MutableShape<S>> {
  return shiftExpression("fftshift", x, axes);
}

export function ifftshift<D extends ComputableDType, S extends Shape>(
  x: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  axes?: readonly number[],
): Expr<D, MutableShape<S>> {
  return shiftExpression("ifftshift", x, axes);
}

function frequency(n: number, d: number): number {
  if (!Number.isSafeInteger(n) || n <= 0 || n > 0xffffffff)
    throw new RangeError(
      "FFT frequency length must be a positive unsigned 32-bit integer.",
    );
  if (!Number.isFinite(d))
    throw new RangeError("FFT sample spacing must be finite.");
  return 1 / (n * d);
}

export function fftfreq(
  ctx: NMathContext,
  n: number,
  d = 1,
): NMathArray<"float64", [number]> {
  const scale = frequency(n, d);
  const values = new Float64Array(n);
  const lastPositive = Math.floor((n - 1) / 2);
  for (let i = 0; i < n; i++)
    values[i] = (i <= lastPositive ? i : i - n) * scale;
  return ctx.array(values, { dtype: "float64" }) as NMathArray<
    "float64",
    [number]
  >;
}

export function rfftfreq(
  ctx: NMathContext,
  n: number,
  d = 1,
): NMathArray<"float64", [number]> {
  const scale = frequency(n, d);
  const values = new Float64Array(Math.floor(n / 2) + 1);
  for (let i = 0; i < values.length; i++) values[i] = i * scale;
  return ctx.array(values, { dtype: "float64" }) as NMathArray<
    "float64",
    [number]
  >;
}
