// SPDX-License-Identifier: Apache-2.0
// NumPy range, identity and `_like` constructors. Values are computed in JS float64 with
// NumPy 2.5.4's operation order (`numpy/_core/function_base.py`,
// `numpy/lib/_twodim_base_impl.py`) and stored as new arrays charged to the context.

import {
  Complex,
  type DTypeMetadata,
  type Lazy,
  type NMathArray,
  type NMathContext,
  Temporal,
  validatedShape,
} from "./core.js";
import { NMathError } from "./errors.js";
import type {
  ComputableDType,
  NumericDType,
  ScalarOf,
  Shape,
} from "./types.js";

/** Dtypes that `linspace`, `logspace`, `eye` and `identity` can produce. */
export type RangeDType = NumericDType;

export interface LinspaceOptions<D extends RangeDType = RangeDType> {
  /** Include `stop` as the last sample. Defaults to `true`. */
  endpoint?: boolean;
  /** Result dtype. Defaults to `float64`. */
  dtype?: D;
  /** Also return the float64 sample spacing, as NumPy's `retstep=True` does. */
  retstep?: boolean;
}

export interface LogspaceOptions<D extends RangeDType = RangeDType> {
  /** Include `base ** stop` as the last sample. Defaults to `true`. */
  endpoint?: boolean;
  /** Scalar base of the log space. Defaults to `10`. */
  base?: number;
  /** Result dtype. Defaults to `float64`. */
  dtype?: D;
}

export interface EyeOptions<D extends RangeDType = RangeDType> {
  /** Diagonal offset: positive above, negative below the main diagonal. */
  k?: number;
  /** Result dtype. Defaults to `float64`. */
  dtype?: D;
}

const RANGE_DTYPES: readonly string[] = [
  "bool",
  "int8",
  "int16",
  "uint8",
  "uint16",
  "int32",
  "int64",
  "uint32",
  "uint64",
  "float16",
  "float32",
  "float64",
  "complex64",
  "complex128",
];

/** Inclusive integer ranges used to reject casts NumPy leaves platform-dependent. */
const INTEGER_RANGES: Readonly<Record<string, readonly [number, number]>> = {
  int8: [-128, 127],
  int16: [-32768, 32767],
  int32: [-2147483648, 2147483647],
  uint8: [0, 255],
  uint16: [0, 65535],
  uint32: [0, 4294967295],
  // 2 ** 63 and 2 ** 64 are exact doubles; the upper bounds are exclusive below.
  int64: [-9223372036854775808, 9223372036854775808],
  uint64: [0, 18446744073709551616],
};

function checkOptions(
  name: string,
  options: object,
  allowed: readonly string[],
): void {
  if (typeof options !== "object" || options === null)
    throw new TypeError(`${name} options must be an object.`);
  for (const key of Object.keys(options))
    if (!allowed.includes(key))
      throw new TypeError(`Unsupported ${name} option: ${key}`);
}

function checkNumber(name: string, value: unknown): number {
  if (typeof value !== "number")
    throw new TypeError(`${name} must be a JS number.`);
  return value;
}

function checkBoolean(name: string, value: unknown): boolean {
  if (typeof value !== "boolean")
    throw new TypeError(`${name} must be a boolean.`);
  return value;
}

function checkRangeDType(name: string, dtype: unknown): RangeDType {
  if (typeof dtype !== "string" || !RANGE_DTYPES.includes(dtype))
    throw new NMathError(
      "NMATH_DTYPE_UNSUPPORTED",
      `NMATH_DTYPE_UNSUPPORTED: ${name} supports bool, integer, floating and complex dtypes, not ${String(dtype)}. Create the values in a numeric dtype and cast explicitly.`,
    );
  return dtype as RangeDType;
}

/**
 * Casts float64 values to `dtype` and creates the array. Integer dtypes apply `round`
 * first (`floor` for linspace, `trunc` for C casts) and reject values outside the target
 * range, where NumPy's cast is undefined behaviour and differs between platforms.
 */
function fromFloat64<D extends RangeDType, S extends Shape>(
  context: NMathContext,
  dtype: D,
  shape: S,
  y: Float64Array,
  round: (x: number) => number,
): NMathArray<D, S> {
  const n = y.length;
  let data: ArrayLike<unknown>;
  switch (dtype) {
    case "float64":
      data = y;
      break;
    case "float32":
      data = Float32Array.from(y);
      break;
    case "float16":
      // The core converts JS numbers to binary16 with one round-to-nearest-even step.
      data = Array.from(y);
      break;
    case "complex64":
    case "complex128": {
      const parts =
        dtype === "complex64"
          ? new Float32Array(2 * n)
          : new Float64Array(2 * n);
      for (let i = 0; i < n; i++) parts[2 * i] = y[i];
      data = parts;
      break;
    }
    case "bool":
      // NumPy casts every nonzero value, including NaN, to true.
      data = Uint8Array.from(y, (v) => (v !== 0 ? 1 : 0));
      break;
    default: {
      const [low, high] = INTEGER_RANGES[dtype];
      const wide = dtype === "int64" || dtype === "uint64";
      const out: (number | bigint)[] = new Array(n);
      for (let i = 0; i < n; i++) {
        const v = round(y[i]);
        if (!(v >= low && (wide ? v < high : v <= high)))
          throw new RangeError(
            `The value ${y[i]} does not fit ${dtype}. Use a wider or floating dtype.`,
          );
        out[i] = wide ? BigInt(v) : v;
      }
      data = out;
    }
  }
  return context.array(data as never, { dtype, shape }) as NMathArray<D, S>;
}

/** `np.linspace` in float64 with NumPy 2.5.4's step, `div` and endpoint rules. */
function linspaceValues(
  start: number,
  stop: number,
  num: number,
  endpoint: boolean,
): { values: Float64Array; step: number } {
  const div = endpoint ? num - 1 : num;
  const delta = stop - start;
  const y = new Float64Array(num);
  let step: number;
  if (div > 0) {
    step = delta / div;
    if (step === 0) {
      // NumPy divides first so that subnormal deltas are not flushed to zero.
      for (let i = 0; i < num; i++) y[i] = (i / div) * delta;
    } else {
      for (let i = 0; i < num; i++) y[i] = i * step;
    }
  } else {
    // Zero samples, or one sample with an endpoint, have an undefined step.
    step = NaN;
    for (let i = 0; i < num; i++) y[i] = i * delta;
  }
  for (let i = 0; i < num; i++) y[i] += start;
  if (endpoint && num > 1) y[num - 1] = stop;
  return { values: y, step };
}

function sampleCount(num: unknown): number {
  return validatedShape([num as number])[0];
}

/**
 * `np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)` for scalar
 * endpoints. Samples are evaluated in float64 and cast once; integer dtypes floor first.
 */
export function linspace<
  const N extends number = 50,
  D extends RangeDType = "float64",
>(
  context: NMathContext,
  start: number,
  stop: number,
  num?: N,
  options?: LinspaceOptions<D> & { retstep?: false },
): NMathArray<D, [N]>;
export function linspace<
  const N extends number = 50,
  D extends RangeDType = "float64",
>(
  context: NMathContext,
  start: number,
  stop: number,
  num: N | undefined,
  options: LinspaceOptions<D> & { retstep: true },
): readonly [samples: NMathArray<D, [N]>, step: number];
export function linspace(
  context: NMathContext,
  start: number,
  stop: number,
  num: number = 50,
  options: LinspaceOptions = {},
): NMathArray<RangeDType, [number]> | readonly [NMathArray<any, any>, number] {
  checkOptions("linspace", options, ["endpoint", "dtype", "retstep"]);
  checkNumber("linspace start", start);
  checkNumber("linspace stop", stop);
  const count = sampleCount(num);
  const endpoint = checkBoolean("endpoint", options.endpoint ?? true);
  const retstep = checkBoolean("retstep", options.retstep ?? false);
  const dtype = checkRangeDType("linspace", options.dtype ?? "float64");
  context.ensureCpuBudget(count, dtype);
  const { values, step } = linspaceValues(start, stop, count, endpoint);
  const samples = fromFloat64(
    context,
    dtype,
    [count] as [number],
    values,
    Math.floor,
  );
  return retstep ? Object.freeze([samples, step] as const) : samples;
}

/**
 * `np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)` for scalar
 * arguments: `base ** linspace(start, stop)` in float64, then one C-style cast. JS
 * `Math.pow` can differ from the platform C `pow` by a few ULP.
 */
export function logspace<
  const N extends number = 50,
  D extends RangeDType = "float64",
>(
  context: NMathContext,
  start: number,
  stop: number,
  num?: N,
  options: LogspaceOptions<D> = {},
): NMathArray<D, [N]> {
  checkOptions("logspace", options, ["endpoint", "base", "dtype"]);
  checkNumber("logspace start", start);
  checkNumber("logspace stop", stop);
  const count = sampleCount(num ?? 50);
  const endpoint = checkBoolean("endpoint", options.endpoint ?? true);
  const base = checkNumber("logspace base", options.base ?? 10);
  const dtype = checkRangeDType("logspace", options.dtype ?? "float64") as D;
  context.ensureCpuBudget(count, dtype);
  const { values } = linspaceValues(start, stop, count, endpoint);
  for (let i = 0; i < count; i++) values[i] = cPow(base, values[i]);
  return fromFloat64(context, dtype, [count] as [N], values, Math.trunc);
}

/**
 * NumPy's float64 power is C99 `pow`, which returns 1 for a base of 1 with any
 * exponent (including NaN) and for a base of -1 with an infinite exponent.
 * ECMAScript `Math.pow` returns NaN in those cases and agrees elsewhere.
 */
function cPow(base: number, exponent: number): number {
  if (base === 1) return 1;
  if (base === -1 && (exponent === Infinity || exponent === -Infinity))
    return 1;
  return Math.pow(base, exponent);
}

/** `np.eye(N, M=None, k=0, dtype=float)`: ones on the `k`-th diagonal, zeros elsewhere. */
export function eye<
  const N extends number,
  const M extends number = N,
  D extends RangeDType = "float64",
>(
  context: NMathContext,
  N: N,
  M?: M,
  options: EyeOptions<D> = {},
): NMathArray<D, [N, M]> {
  checkOptions("eye", options, ["k", "dtype"]);
  const [rows, columns] = validatedShape([N, M ?? N]);
  const k = options.k ?? 0;
  if (!Number.isSafeInteger(k))
    throw new TypeError("The eye diagonal offset k must be a safe integer.");
  const dtype = checkRangeDType("eye", options.dtype ?? "float64") as D;
  context.ensureCpuBudget(rows * columns, dtype);
  const values = new Float64Array(rows * columns);
  // Row r holds its one at column r + k when that column exists.
  const first = Math.max(0, -k);
  const end = Math.min(rows, columns - k);
  for (let r = first; r < end; r++) values[r * columns + r + k] = 1;
  return fromFloat64(
    context,
    dtype,
    [rows, columns] as [N, M],
    values,
    Math.trunc,
  );
}

/** `np.identity(n, dtype=float)`: the square `eye(n)`. */
export function identity<
  const N extends number,
  D extends RangeDType = "float64",
>(
  context: NMathContext,
  n: N,
  dtype: D = "float64" as D,
): NMathArray<D, [N, N]> {
  return eye(context, n, n, { dtype });
}

/** `np.empty(shape, dtype)`. NMath zero-fills; NumPy leaves the contents unspecified. */
export function empty<D extends ComputableDType, const S extends Shape>(
  context: NMathContext,
  shape: S,
  dtype: D,
): NMathArray<D, S> {
  return context.zeros(shape, dtype);
}

function sourceContext(a: Lazy<any, any>): NMathContext {
  const context = a.context;
  if (context === undefined)
    throw new NMathError(
      "NMATH_CONTEXT_MISMATCH",
      "NMATH_CONTEXT_MISMATCH: The value has no array operand, so it has no context. Pass an array or an expression that uses one.",
    );
  return context;
}

/** Keeps string width and time unit when the result dtype equals the source dtype. */
function sourceMetadata(
  a: Lazy<any, any>,
  dtype: ComputableDType,
): DTypeMetadata {
  if (dtype !== a.dtype) return {};
  if (dtype === "str") return { stringWidth: a.stringWidth };
  if (dtype === "datetime64" || dtype === "timedelta64")
    return { unit: a.unit };
  return {};
}

function scalarFor(
  dtype: ComputableDType,
  metadata: DTypeMetadata,
  one: boolean,
): unknown {
  switch (dtype) {
    case "str":
      return one ? "1" : "";
    case "datetime64":
    case "timedelta64":
      return new Temporal(one ? 1n : 0n, metadata.unit ?? "ns", dtype);
    case "complex64":
    case "complex128":
      return new Complex(one ? 1 : 0);
    case "int64":
    case "uint64":
      return one ? 1n : 0n;
    case "bool":
      return one;
    default:
      return one ? 1 : 0;
  }
}

function filledLike(
  a: Lazy<any, any>,
  dtype: ComputableDType | undefined,
  value: (dtype: ComputableDType, metadata: DTypeMetadata) => unknown,
): NMathArray<any, any> {
  const context = sourceContext(a);
  const resultDType = dtype ?? a.dtype;
  const metadata = sourceMetadata(a, resultDType);
  return context.full(
    a.shape,
    value(resultDType, metadata) as never,
    resultDType,
    metadata,
  );
}

/** `np.zeros_like(a, dtype=None)` in `a`'s context, keeping its shape. */
export function zeros_like<
  D extends ComputableDType,
  S extends Shape,
  R extends ComputableDType = D,
>(a: Lazy<NoInfer<D>, S> & { readonly dtype: D }, dtype?: R): NMathArray<R, S> {
  return filledLike(a, dtype, (d, m) => scalarFor(d, m, false));
}

/** `np.ones_like(a, dtype=None)`. Strings become `"1"` and temporal values one tick. */
export function ones_like<
  D extends ComputableDType,
  S extends Shape,
  R extends ComputableDType = D,
>(a: Lazy<NoInfer<D>, S> & { readonly dtype: D }, dtype?: R): NMathArray<R, S> {
  return filledLike(a, dtype, (d, m) => scalarFor(d, m, true));
}

/**
 * `np.full_like(a, fill_value, dtype=None)`. A string source keeps its width and rejects
 * longer values instead of truncating them; a temporal value must use the source unit.
 */
export function full_like<
  D extends ComputableDType,
  S extends Shape,
  R extends ComputableDType = D,
>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D },
  value: ScalarOf<NoInfer<R>>,
  dtype?: R,
): NMathArray<R, S> {
  return filledLike(a, dtype, () => value);
}

/** `np.empty_like(a, dtype=None)`. NMath zero-fills like `zeros_like`. */
export function empty_like<
  D extends ComputableDType,
  S extends Shape,
  R extends ComputableDType = D,
>(a: Lazy<NoInfer<D>, S> & { readonly dtype: D }, dtype?: R): NMathArray<R, S> {
  return zeros_like(a, dtype);
}
