import { NMathGenerator } from "./random.js";
// SPDX-License-Identifier: Apache-2.0
import {
  readText,
  writeText,
  type TextReadOptions,
  type TextWriteOptions,
  type TextFormat,
} from "./io-text.js";
import {
  readColumnar,
  writeColumnar,
  type ColumnarFormat,
  type ColumnarReadOptions,
  type ColumnarWriteOptions,
} from "./columnar.js";
import type { Column, Validity } from "./frame.js";
import { NMathError, native, nativeAsync } from "./errors.js";
import { binding } from "./native.js";
import type {
  NativeArray,
  NativeExpr,
  NativeSelector,
} from "./native-binding.generated.js";
import type {
  BinaryDType,
  BinaryCheck,
  WeakBinary,
  WeakDType,
  TemporalDType,
  NumericDType,
  BoolArithmetic,
  Broadcast,
  BroadcastCheck,
  ComputableDType,
  DType,
  IndexShape,
  MatMulCheck,
  MatMulShape,
  Promote,
  PromoteWeak,
  ReduceShape,
  ScalarOf,
  Shape,
  SumDType,
  TrueDivide,
  TypedArrayOf,
  WeakScalar,
} from "./types.js";

/** A complex scalar with separate IEEE-754 real and imaginary components. */
export class Complex {
  readonly #brand = true;
  constructor(
    readonly real: number,
    readonly imag: number = 0,
  ) {
    if (typeof real !== "number" || typeof imag !== "number") {
      throw new TypeError("Complex components must be numbers.");
    }
    Object.freeze(this);
  }

  /** Explicit lossy conversion: returns the real component. */
  toNumber(): number {
    return this.real;
  }

  valueOf(): never {
    throw new TypeError(
      "Complex values cannot convert implicitly to numbers. Use toNumber() for an explicit lossy conversion.",
    );
  }
}

export type TimeBase =
  | "Y"
  | "M"
  | "W"
  | "D"
  | "h"
  | "m"
  | "s"
  | "ms"
  | "us"
  | "ns"
  | "ps"
  | "fs"
  | "as";
export type TimeUnit = TimeBase | `${number}${TimeBase}`;

/** Exact signed 64-bit ticks and NumPy unit metadata. */
export class Temporal<K extends TemporalDType = TemporalDType> {
  constructor(
    readonly ticks: bigint,
    readonly unit: TimeUnit,
    readonly dtype: K,
  ) {
    if (typeof ticks !== "bigint" || ticks < -(1n << 63n) || ticks >= 1n << 63n)
      throw new RangeError("Temporal ticks must fit signed int64.");
    if (dtype !== "datetime64" && dtype !== "timedelta64")
      throw new TypeError("A temporal dtype is required.");
    native(() => binding.NativeExpr.temporalScalar(ticks, `${dtype}[${unit}]`));
    Object.freeze(this);
  }
  get isNaT(): boolean {
    return this.ticks === -(1n << 63n);
  }
  /** Explicit conversion to the millisecond precision and range of JS Date. */
  toDate(this: Temporal<"datetime64">): Date {
    if (this.dtype !== "datetime64")
      throw new TypeError("Only datetime64 converts to Date.");
    if (this.isNaT) return new Date(NaN);
    const ms = native(() =>
      binding.NativeExpr.temporalMilliseconds(this.ticks, this.unit),
    );
    if (ms < -8640000000000000n || ms > 8640000000000000n)
      throw new RangeError("The datetime is outside the Date range.");
    return new Date(Number(ms));
  }
  valueOf(): never {
    throw new TypeError(
      "Temporal values require explicit conversion. Use ticks or toDate().",
    );
  }
}

function scalarResult(
  value:
    | number
    | bigint
    | boolean
    | string
    | { real: number; imag: number }
    | { ticks: bigint; dtype: string; unit: string },
): WeakScalar {
  if (typeof value !== "object") return value;
  return "ticks" in value
    ? new Temporal(
        value.ticks,
        value.unit as TimeUnit,
        value.dtype as TemporalDType,
      )
    : new Complex(value.real, value.imag);
}

export interface DTypeMetadata {
  unit?: TimeUnit;
  stringWidth?: number;
}
function descriptor(
  dtype: ComputableDType,
  metadata: DTypeMetadata = {},
): string {
  if (dtype === "str") {
    const n = metadata.stringWidth ?? 1;
    if (!Number.isSafeInteger(n) || n <= 0)
      throw new RangeError("String width must be a positive safe integer.");
    return `str[${n}]`;
  }
  return dtype === "datetime64" || dtype === "timedelta64"
    ? `${dtype}[${metadata.unit ?? "ns"}]`
    : dtype;
}

export type Device = "cpu" | "gpu";

export interface ComputeOptions {
  /** Defaults to `cpu`. GPU execution never falls back to the CPU. */
  device?: Device;
  signal?: AbortSignal;
}

export interface SignalOptions {
  signal?: AbortSignal;
}

/** The value `compute()` resolves to: a JS scalar for rank 0, otherwise an array. */
export type ComputeResult<
  D extends ComputableDType,
  S extends Shape,
> = S extends readonly [] ? ScalarOf<D> : NMathArray<D, S>;

// ---------------------------------------------------------------------------
// Selectors

export interface Slice {
  readonly kind: "slice";
  readonly start?: number;
  readonly stop?: number;
  readonly step?: number;
}

export interface Indices {
  readonly kind: "indices";
  readonly values: readonly number[];
}

export interface Mask {
  readonly kind: "mask";
  readonly values: readonly boolean[];
}

export type Selector = number | Slice | Indices | Mask;

/** `slice(start, stop, step)`: `stop` is excluded; `undefined` omits a bound. */
export function slice(start?: number, stop?: number, step?: number): Slice {
  return { kind: "slice", start, stop, step };
}

/** Integer-array selection along one axis. */
export function indices(values: readonly number[]): Indices {
  return { kind: "indices", values: [...values] };
}

/** Boolean-mask selection along one axis. */
export function mask(values: readonly boolean[]): Mask {
  return { kind: "mask", values: [...values] };
}

function checkInt(value: number | undefined, what: string): number | undefined {
  if (value !== undefined && !Number.isSafeInteger(value)) {
    throw new NMathError(
      "NMATH_INDEX_INVALID",
      `NMATH_INDEX_INVALID: ${what} must be an integer. Use an index inside the axis length or a valid selector.`,
    );
  }
  return value;
}

function toNativeSelector(selector: Selector): NativeSelector {
  if (typeof selector === "number") {
    return { kind: "index", index: checkInt(selector, "An index") };
  }
  switch (selector.kind) {
    case "slice":
      return {
        kind: "slice",
        start: checkInt(selector.start, "A slice start"),
        stop: checkInt(selector.stop, "A slice stop"),
        step: checkInt(selector.step, "A slice step"),
      };
    case "indices":
      return {
        kind: "indices",
        indices: selector.values.map((v) => checkInt(v, "An index") as number),
      };
    case "mask":
      return { kind: "mask", mask: [...selector.values] };
  }
}

// ---------------------------------------------------------------------------
// Signals

function withSignal<T>(
  signal: AbortSignal | undefined,
  run: (cancel: InstanceType<typeof binding.NativeCancellation>) => Promise<T>,
): Promise<T> {
  const cancel = new binding.NativeCancellation();
  if (signal?.aborted) {
    cancel.cancel();
  }
  const onAbort = () => cancel.cancel();
  signal?.addEventListener("abort", onAbort, { once: true });
  return nativeAsync(() => run(cancel)).finally(() =>
    signal?.removeEventListener("abort", onAbort),
  );
}

// ---------------------------------------------------------------------------
// Lazy values

/** Anything that can appear as an operand. */
export type Operand = Lazy<any, any> | WeakScalar;

function scalarExpr(value: WeakScalar): NativeExpr {
  if (typeof value === "string") {
    for (const c of value) {
      const cp = c.codePointAt(0)!;
      if (cp >= 0xd800 && cp <= 0xdfff)
        throw new TypeError(
          "Unpaired UTF-16 surrogates cannot be stored as Unicode scalars.",
        );
    }
    return native(() => binding.NativeExpr.textScalar(value));
  }
  if (value instanceof Temporal)
    return native(() =>
      binding.NativeExpr.temporalScalar(
        value.ticks,
        `${value.dtype}[${value.unit}]`,
      ),
    );
  if (value instanceof Complex) {
    return binding.NativeExpr.complexScalar(value.real, value.imag);
  }
  if (typeof value === "number") {
    return binding.NativeExpr.floatScalar(value);
  }
  if (typeof value === "bigint") {
    return native(() => binding.NativeExpr.intScalar(value));
  }
  if (typeof value === "boolean") {
    return binding.NativeExpr.boolScalar(value);
  }
  throw new NMathError(
    "NMATH_DTYPE_UNSUPPORTED",
    `NMATH_DTYPE_UNSUPPORTED: Operands must be NMath values, numbers, bigints, booleans or Complex scalars. Convert the operand to a supported dtype explicitly.`,
  );
}

function operandExpr(value: Operand): NativeExpr {
  return value instanceof Lazy ? value.nativeExpr() : scalarExpr(value);
}

function operandContext(value: Operand): NMathContext | undefined {
  return value instanceof Lazy ? value.context : undefined;
}

function pickContext(
  a: NMathContext | undefined,
  b: NMathContext | undefined,
): NMathContext | undefined {
  return a ?? b;
}

export type BinaryOperator = "add" | "sub" | "mul" | "div" | "mod" | "pow";
export type ReduceOperator = "sum" | "mean" | "min" | "max";

/** Type-only brand that identifies NMath values regardless of dtype and shape. */
export declare const lazyBrand: unique symbol;

/** Any NMath array or expression, for APIs that accept every dtype and shape. */
export interface NMathValue {
  readonly [lazyBrand]: true;
}

/** Shared lazy operations for arrays and expressions. */
export abstract class Lazy<
  D extends ComputableDType,
  S extends Shape,
> implements NMathValue {
  declare readonly [lazyBrand]: true;
  /** @internal */
  abstract nativeExpr(): NativeExpr;
  abstract readonly context: NMathContext | undefined;
  abstract get dtype(): D;
  abstract get shape(): S;
  abstract get dtypeDescriptor(): string;
  get unit(): TimeUnit | undefined {
    return this.dtype === "datetime64" || this.dtype === "timedelta64"
      ? (this.dtypeDescriptor.slice(this.dtype.length + 1, -1) as TimeUnit)
      : undefined;
  }
  get stringWidth(): number | undefined {
    return this.dtype === "str"
      ? Number(this.dtypeDescriptor.slice(4, -1))
      : undefined;
  }

  get ndim(): number {
    return this.shape.length;
  }

  /** @internal */
  binaryOp(op: BinaryOperator, other: Operand): Expr<any, any> {
    const expr = native(() =>
      binding.NativeExpr.binary(op, this.nativeExpr(), operandExpr(other)),
    );
    return new Expr(expr, pickContext(this.context, operandContext(other)));
  }

  /** @internal `scalar op this`. */
  reversedBinaryOp(op: BinaryOperator, scalar: WeakScalar): Expr<any, any> {
    const expr = native(() =>
      binding.NativeExpr.binary(op, scalarExpr(scalar), this.nativeExpr()),
    );
    return new Expr(expr, this.context);
  }

  // Infer dtype from the operand property, not recursively from method results:
  // mixed small integer operations otherwise infer an extra float64 alternative.
  add<D2 extends ComputableDType, S2 extends Shape>(
    other: BinaryCheck<"add", D, D2> &
      Lazy<NoInfer<D2>, S2> & { readonly dtype: D2 } & BroadcastCheck<S, S2>,
  ): Expr<BinaryDType<"add", D, D2>, Broadcast<S, S2>>;
  add<W extends WeakScalar>(
    other: W & ([WeakBinary<"add", D, W>] extends [never] ? never : unknown),
  ): Expr<WeakBinary<"add", D, W>, S>;
  add(other: Operand): Expr<any, any> {
    return this.binaryOp("add", other);
  }

  sub<D2 extends ComputableDType, S2 extends Shape>(
    other: BinaryCheck<"sub", D, D2> &
      Lazy<NoInfer<D2>, S2> & { readonly dtype: D2 } & BroadcastCheck<S, S2>,
  ): Expr<BinaryDType<"sub", D, D2>, Broadcast<S, S2>>;
  sub<W extends WeakScalar>(
    other: W & ([WeakBinary<"sub", D, W>] extends [never] ? never : unknown),
  ): Expr<WeakBinary<"sub", D, W>, S>;
  sub(other: Operand): Expr<any, any> {
    return this.binaryOp("sub", other);
  }

  mul<D2 extends ComputableDType, S2 extends Shape>(
    other: BinaryCheck<"mul", D, D2> &
      Lazy<NoInfer<D2>, S2> & { readonly dtype: D2 } & BroadcastCheck<S, S2>,
  ): Expr<BinaryDType<"mul", D, D2>, Broadcast<S, S2>>;
  mul<W extends WeakScalar>(
    other: W & ([WeakBinary<"mul", D, W>] extends [never] ? never : unknown),
  ): Expr<WeakBinary<"mul", D, W>, S>;
  mul(other: Operand): Expr<any, any> {
    return this.binaryOp("mul", other);
  }

  div<D2 extends ComputableDType, S2 extends Shape>(
    other: BinaryCheck<"div", D, D2> &
      Lazy<NoInfer<D2>, S2> & { readonly dtype: D2 } & BroadcastCheck<S, S2>,
  ): Expr<BinaryDType<"div", D, D2>, Broadcast<S, S2>>;
  div<W extends WeakScalar>(
    other: W & ([WeakBinary<"div", D, W>] extends [never] ? never : unknown),
  ): Expr<WeakBinary<"div", D, W>, S>;
  div(other: Operand): Expr<any, any> {
    return this.binaryOp("div", other);
  }

  mod<
    D2 extends Exclude<ComputableDType, "complex64" | "complex128">,
    S2 extends Shape,
  >(
    this: D extends "complex64" | "complex128" ? never : Lazy<D, S>,
    other: BinaryCheck<"mod", D, D2> &
      Lazy<NoInfer<D2>, S2> & { readonly dtype: D2 } & BroadcastCheck<S, S2>,
  ): Expr<BinaryDType<"mod", D, D2>, Broadcast<S, S2>>;
  mod<W extends Exclude<WeakScalar, Complex>>(
    this: D extends "complex64" | "complex128" ? never : Lazy<D, S>,
    other: W & ([WeakBinary<"mod", D, W>] extends [never] ? never : unknown),
  ): Expr<WeakBinary<"mod", D, W>, S>;
  mod(other: Operand): Expr<any, any> {
    return this.binaryOp("mod", other);
  }

  pow<D2 extends ComputableDType, S2 extends Shape>(
    other: BinaryCheck<"pow", D, D2> &
      Lazy<NoInfer<D2>, S2> & { readonly dtype: D2 } & BroadcastCheck<S, S2>,
  ): Expr<BinaryDType<"pow", D, D2>, Broadcast<S, S2>>;
  pow<W extends WeakScalar>(
    other: W & ([WeakBinary<"pow", D, W>] extends [never] ? never : unknown),
  ): Expr<WeakBinary<"pow", D, W>, S>;
  pow(other: Operand): Expr<any, any> {
    return this.binaryOp("pow", other);
  }

  neg(
    this: [D] extends ["bool" | "str" | "datetime64"] ? never : Lazy<D, S>,
  ): Expr<D, S> {
    return new Expr(
      native(() => binding.NativeExpr.unary("neg", this.nativeExpr())),
      this.context,
    );
  }

  pos(
    this: [D] extends ["bool" | "str" | "datetime64"] ? never : Lazy<D, S>,
  ): Expr<D, S> {
    return new Expr(
      native(() => binding.NativeExpr.unary("pos", this.nativeExpr())),
      this.context,
    );
  }

  /** @internal */
  reduceOp(op: ReduceOperator, axis: number | undefined): Expr<any, any> {
    return new Expr(
      native(() =>
        binding.NativeExpr.reduce(
          op,
          this.nativeExpr(),
          axis === undefined
            ? undefined
            : [checkInt(axis, "An axis") as number],
        ),
      ),
      this.context,
    );
  }

  sum<A extends number | undefined = undefined>(
    this: [D] extends ["str" | "datetime64"]
      ? 0 extends 1 & D
        ? Lazy<D, S>
        : never
      : Lazy<D, S>,
    axis?: A,
  ): Expr<SumDType<D>, ReduceShape<S, A>> {
    return this.reduceOp("sum", axis);
  }

  mean<A extends number | undefined = undefined>(
    this: [D] extends ["str" | "datetime64"]
      ? 0 extends 1 & D
        ? Lazy<D, S>
        : never
      : Lazy<D, S>,
    axis?: A,
  ): Expr<D extends "timedelta64" ? D : TrueDivide<D>, ReduceShape<S, A>> {
    return this.reduceOp("mean", axis);
  }

  min<A extends number | undefined = undefined>(
    this: [D] extends ["str"]
      ? 0 extends 1 & D
        ? Lazy<D, S>
        : never
      : Lazy<D, S>,
    axis?: A,
  ): Expr<D, ReduceShape<S, A>> {
    return this.reduceOp("min", axis);
  }

  max<A extends number | undefined = undefined>(
    this: [D] extends ["str"]
      ? 0 extends 1 & D
        ? Lazy<D, S>
        : never
      : Lazy<D, S>,
    axis?: A,
  ): Expr<D, ReduceShape<S, A>> {
    return this.reduceOp("max", axis);
  }

  matmul<D2 extends NumericDType, S2 extends Shape>(
    this: [D] extends ["str" | TemporalDType]
      ? 0 extends 1 & D
        ? Lazy<D, S>
        : never
      : Lazy<D, S>,
    other: Lazy<NoInfer<D2>, S2> & { readonly dtype: D2 } & MatMulCheck<S, S2>,
  ): Expr<Promote<D, D2>, MatMulShape<S, S2>> {
    return new Expr(
      native(() =>
        binding.NativeExpr.matmul(this.nativeExpr(), other.nativeExpr()),
      ),
      pickContext(this.context, other.context),
    );
  }

  /** Lazy read. Integers drop an axis; slices, `indices` and `mask` keep one. */
  get<const Sel extends readonly Selector[]>(
    ...selectors: Sel
  ): Expr<D, IndexShape<S, Sel>> {
    return new Expr(
      native(() =>
        binding.NativeExpr.index(
          this.nativeExpr(),
          selectors.map(toNativeSelector),
        ),
      ),
      this.context,
    );
  }

  /** Explicit dtype conversion (NumPy `astype`). */
  astype<D2 extends ComputableDType>(
    dtype: D2,
    metadata: DTypeMetadata = {},
  ): Expr<D2, S> {
    return new Expr(
      native(() =>
        binding.NativeExpr.cast(this.nativeExpr(), descriptor(dtype, metadata)),
      ),
      this.context,
    );
  }

  /**
   * Runs one independent job. Inputs are frozen when the job starts, not when this is
   * called. Each call computes again; results are never cached.
   */
  compute(options: ComputeOptions = {}): Promise<ComputeResult<D, S>> {
    const context = this.context;
    if (!context) {
      return Promise.reject(
        new NMathError(
          "NMATH_CONTEXT_MISMATCH",
          "NMATH_CONTEXT_MISMATCH: The expression has no array operand, so it has no context. Combine it with an array from a context.",
        ),
      );
    }
    const expr = this.nativeExpr();
    return withSignal(options.signal, (cancel) =>
      context.nativeContext.compute(expr, options.device ?? "cpu", cancel),
    ).then((array) => fromResult(array, context) as ComputeResult<D, S>);
  }
}

function fromResult(
  array: NativeArray,
  context: NMathContext,
): NMathArray<ComputableDType, Shape> | WeakScalar {
  if (array.shape.length === 0) {
    // Callers only receive the primitive, so release the reservation now.
    try {
      return scalarResult(native(() => array.item([])));
    } finally {
      array.dispose();
    }
  }
  return new NMathArray(array, context);
}

/** An immutable lazy expression. */
export class Expr<D extends ComputableDType, S extends Shape> extends Lazy<
  D,
  S
> {
  readonly #expr: NativeExpr;
  readonly context: NMathContext | undefined;

  /** @internal */
  constructor(expr: NativeExpr, context: NMathContext | undefined) {
    super();
    this.#expr = expr;
    this.context = context;
  }

  /** @internal */
  nativeExpr(): NativeExpr {
    return this.#expr;
  }

  get dtypeDescriptor(): string {
    return this.#expr.dtypeDescriptor;
  }

  get dtype(): D {
    return this.#expr.dtype as D;
  }

  get shape(): S {
    return this.#expr.shape as unknown as S;
  }
}

type Nested<T> = T | readonly Nested<T>[];

/** An array whose storage belongs to one context. Basic indexing results share storage. */
export class NMathArray<
  D extends ComputableDType,
  S extends Shape,
> extends Lazy<D, S> {
  readonly #array: NativeArray;
  readonly context: NMathContext;

  /** @internal */
  constructor(array: NativeArray, context: NMathContext) {
    super();
    this.#array = array;
    this.context = context;
  }

  /** @internal */
  get nativeArray(): NativeArray {
    return this.#array;
  }

  /** @internal */
  nativeExpr(): NativeExpr {
    return native(() => binding.NativeExpr.input(this.#array));
  }

  get dtypeDescriptor(): string {
    return native(() => this.#array.dtypeDescriptor);
  }

  get dtype(): D {
    return native(() => this.#array.dtype) as D;
  }

  get shape(): S {
    return native(() => this.#array.shape) as unknown as S;
  }

  get size(): number {
    return this.shape.reduce((a, b) => a * b, 1);
  }

  get disposed(): boolean {
    return this.#array.disposed;
  }

  /** Reads one element synchronously. */
  item(...index: number[]): ScalarOf<D> {
    const checked = index.map((i) => checkInt(i, "An index") as number);
    return scalarResult(native(() => this.#array.item(checked))) as ScalarOf<D>;
  }

  /** Copies the elements into a typed array in C order. */
  toTypedArray(): TypedArrayOf<D> {
    const bytes = native(() => this.#array.toBytes());
    const copy = new Uint8Array(bytes).buffer;
    return typedArrayFor(this.dtype, copy) as TypedArrayOf<D>;
  }

  /** Copies exact IEEE-754 binary16 encodings, including NaN payloads. */
  toFloat16Bits(this: NMathArray<"float16", S>): Uint16Array {
    if (this.dtype !== "float16")
      throw new TypeError("toFloat16Bits requires float16 storage.");
    return new Uint16Array(
      new Uint8Array(native(() => this.#array.toBytes())).buffer,
    );
  }

  /** Copies the elements into nested JS arrays. */
  toArray(): Nested<ScalarOf<D>> {
    if (
      this.dtype === "str" ||
      this.dtype === "datetime64" ||
      this.dtype === "timedelta64"
    ) {
      const build = (axis: number, indices: number[]): unknown =>
        axis === this.ndim
          ? this.item(...indices)
          : Array.from({ length: this.shape[axis] }, (_, i) =>
              build(axis + 1, [...indices, i]),
            );
      return build(0, []) as Nested<ScalarOf<D>>;
    }
    const typed = this.toTypedArray();
    const flat = (
      this.dtype === "complex64" || this.dtype === "complex128"
        ? Array.from(
            { length: this.size },
            (_, i) =>
              new Complex(Number(typed[2 * i]), Number(typed[2 * i + 1])),
          )
        : Array.from(typed as ArrayLike<number | bigint>, (v) =>
            this.dtype === "bool" ? v !== 0 : v,
          )
    ) as ScalarOf<D>[];
    const build = (axis: number, offset: number): Nested<ScalarOf<D>> => {
      if (axis === this.shape.length) {
        return flat[offset];
      }
      const stride = this.shape
        .slice(axis + 1)
        .reduce((a: number, b: number) => a * b, 1);
      return Array.from({ length: this.shape[axis] }, (_, i) =>
        build(axis + 1, offset + i * stride),
      );
    };
    return build(0, 0);
  }

  /** True when both arrays view the same storage. */
  sharesStorage(other: NMathArray<ComputableDType, Shape>): boolean {
    return native(() => this.#array.sharesStorage(other.nativeArray));
  }

  /**
   * Asynchronous atomic write. Writes to one storage run in request order; cancellation or
   * failure leaves the previous values.
   */
  set(
    selectors: readonly Selector[],
    value: Operand,
    options: SignalOptions = {},
  ): Promise<void> {
    const target = this.#array;
    const nativeSelectors = selectors.map(toNativeSelector);
    const valueExpr = operandExpr(value);
    return withSignal(options.signal, (cancel) =>
      this.context.nativeContext.assign(
        target,
        nativeSelectors,
        valueExpr,
        cancel,
      ),
    );
  }

  /** Writes this array as an NPY file. Existing files are kept unless `overwrite` is set. */
  save(
    path: string,
    options: SignalOptions & { overwrite?: boolean } = {},
  ): Promise<void> {
    const array = this.#array;
    return withSignal(options.signal, (cancel) =>
      this.context.nativeContext.saveNpy(
        array,
        path,
        options.overwrite ?? false,
        cancel,
      ),
    );
  }

  /** Releases this handle. Other views and running jobs keep the storage alive. */
  dispose(): void {
    this.#array.dispose();
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}

function float16Constructor(): Float16ArrayConstructor {
  if (typeof Float16Array === "undefined") {
    throw new TypeError(
      "This Node.js runtime has no Float16Array. Use arrayFromFloat16Bits/toFloat16Bits for exact binary16 transport, or upgrade Node.js 24.",
    );
  }
  return Float16Array;
}

function typedArrayFor(dtype: ComputableDType, buffer: ArrayBuffer) {
  switch (dtype) {
    case "str":
      throw new TypeError("String arrays use toArray() to return JS strings.");
    case "datetime64":
    case "timedelta64":
      return new BigInt64Array(buffer);
    case "bool":
      return new Uint8Array(buffer);
    case "int8":
      return new Int8Array(buffer);
    case "int16":
      return new Int16Array(buffer);
    case "float16":
      return new (float16Constructor())(buffer);
    case "uint8":
      return new Uint8Array(buffer);
    case "uint16":
      return new Uint16Array(buffer);
    case "int32":
      return new Int32Array(buffer);
    case "int64":
      return new BigInt64Array(buffer);
    case "uint32":
      return new Uint32Array(buffer);
    case "uint64":
      return new BigUint64Array(buffer);
    case "complex64":
    case "float32":
      return new Float32Array(buffer);
    case "complex128":
    case "float64":
      return new Float64Array(buffer);
  }
}

// ---------------------------------------------------------------------------
// Contexts

export interface ContextOptions {
  maxConcurrentJobs: number;
  maxQueueLength: number;
  /** Bytes. */
  cpuMemoryBudget: number | bigint;
  /** Bytes. Use 0 for a CPU-only context. */
  gpuMemoryBudget: number | bigint;
}

export interface ExecutionMeasurement {
  id: number;
  kind: "compute" | "write" | "copy" | "file_read" | "file_write";
  operation: string;
  device: Device;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  queueTimeMs: number;
  executionTimeMs: number;
}

export interface MemoryMeasurement {
  limit: bigint;
  used: bigint;
  peak: bigint;
  cumulativeReserved: bigint;
  rejected: number;
}

export interface ContextMetrics {
  executions: ExecutionMeasurement[];
  totalSubmitted: number;
  totalSucceeded: number;
  totalFailed: number;
  totalCancelled: number;
  totalRejected: number;
  cpuMemory: MemoryMeasurement;
  gpuMemory: MemoryMeasurement;
}

export interface ArrayOptions<
  D extends ComputableDType,
  S extends Shape,
> extends DTypeMetadata {
  dtype: D;
  shape?: S;
}

function invalidOptions(message: string): NMathError {
  return new NMathError(
    "NMATH_CONTEXT_INVALID_OPTIONS",
    `NMATH_CONTEXT_INVALID_OPTIONS: ${message} Pass positive maxConcurrentJobs and non-negative queue and memory limits.`,
  );
}

function budget(value: unknown, name: string): bigint {
  if (typeof value === "bigint" && value >= 0n) {
    return value;
  }
  if (typeof value === "number" && Number.isSafeInteger(value) && value >= 0) {
    return BigInt(value);
  }
  throw invalidOptions(`${name} must be a non-negative integer.`);
}

function count(value: unknown, name: string, min: number): number {
  if (
    typeof value !== "number" ||
    !Number.isInteger(value) ||
    value < min ||
    value > 0xffff_ffff
  ) {
    throw invalidOptions(`${name} must be an integer of at least ${min}.`);
  }
  return value;
}

function inferShape(data: unknown): number[] {
  const shape: number[] = [];
  let cursor: unknown = data;
  while (Array.isArray(cursor)) {
    shape.push(cursor.length);
    cursor = cursor[0];
  }
  return shape;
}

/** Flattens nested input in C order and rejects ragged branches. */
function flatten(
  data: unknown,
  shape: readonly number[],
  axis: number,
  out: unknown[],
): void {
  if (axis === shape.length) {
    if (Array.isArray(data)) {
      throw raggedInput();
    }
    out.push(data);
    return;
  }
  if (!Array.isArray(data) || data.length !== shape[axis]) {
    throw raggedInput();
  }
  for (const item of data) {
    flatten(item, shape, axis + 1, out);
  }
}

function raggedInput(): NMathError {
  return new NMathError(
    "NMATH_SHAPE_MISMATCH",
    "NMATH_SHAPE_MISMATCH: Nested input is ragged. Make every nested array at the same depth the same length.",
  );
}

function itemSize(dtype: ComputableDType): number {
  switch (dtype) {
    case "complex128":
      return 16;
    case "bool":
    case "int8":
    case "uint8":
      return 1;
    case "int16":
    case "uint16":
    case "float16":
      return 2;
    case "int32":
    case "uint32":
    case "float32":
      return 4;
    default:
      return 8;
  }
}

function product(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/** @internal */
export function validatedShape(shape: readonly number[]): number[] {
  const copy = [...shape];
  if (
    copy.some(
      (dimension) =>
        typeof dimension !== "number" ||
        !Number.isInteger(dimension) ||
        dimension < 0 ||
        dimension > 0xffff_ffff,
    )
  ) {
    throw new NMathError(
      "NMATH_SHAPE_MISMATCH",
      "NMATH_SHAPE_MISMATCH: Each dimension must be a finite integer from 0 through 4294967295. Correct the shape before creating an array or expression.",
    );
  }
  return copy;
}

function toBytes(
  dtype: ComputableDType,
  values: ArrayLike<unknown>,
  metadata: DTypeMetadata = {},
): Buffer {
  if (dtype === "str") {
    const width = metadata.stringWidth ?? 1;
    const data = new Uint32Array(values.length * width);
    for (let i = 0; i < values.length; i++) {
      const value = values[i];
      if (typeof value !== "string")
        throw new TypeError("String storage requires JS strings.");
      let j = 0;
      for (const c of value) {
        const cp = c.codePointAt(0)!;
        if (cp >= 0xd800 && cp <= 0xdfff)
          throw new TypeError(
            "Unpaired UTF-16 surrogates cannot be stored as Unicode scalars.",
          );
        if (j >= width)
          throw new RangeError(
            "String input exceeds the selected width. Use an explicit cast to truncate.",
          );
        data[i * width + j++] = cp;
      }
    }
    return Buffer.from(data.buffer);
  }
  if (dtype === "datetime64" || dtype === "timedelta64") {
    const ticks = new BigInt64Array(values.length);
    for (let i = 0; i < values.length; i++) {
      const value = values[i];
      if (value instanceof Temporal) {
        if (value.dtype !== dtype || value.unit !== (metadata.unit ?? "ns"))
          throw new TypeError(
            "Temporal scalar kind and unit must match array metadata. Convert units explicitly.",
          );
        ticks[i] = value.ticks;
      } else if (
        typeof value === "bigint" &&
        value >= -(1n << 63n) &&
        value < 1n << 63n
      )
        ticks[i] = value;
      else
        throw new TypeError(
          "Temporal input requires exact int64 ticks or Temporal scalars.",
        );
    }
    return Buffer.from(ticks.buffer);
  }

  if (dtype !== "complex64" && dtype !== "complex128") {
    for (let i = 0; i < values.length; i++) {
      if (values[i] instanceof Complex)
        throw new TypeError(
          "Complex input requires a complex dtype. Convert explicitly before creating real arrays.",
        );
    }
  }
  const length = values.length;
  let typed: ArrayBufferView;
  switch (dtype) {
    case "float16":
      typed = float16Constructor().from({ length }, (_, i) =>
        Number(values[i]),
      );
      break;
    case "complex64":
    case "complex128": {
      const parts =
        dtype === "complex64"
          ? new Float32Array(length * 2)
          : new Float64Array(length * 2);
      for (let i = 0; i < length; i++) {
        const value = values[i];
        if (!(value instanceof Complex))
          throw new TypeError(
            "Complex array elements must be Complex scalars. Use a floating typed array for interleaved components.",
          );
        parts[2 * i] = value.real;
        parts[2 * i + 1] = value.imag;
      }
      typed = parts;
      break;
    }
    case "bool":
      typed = Uint8Array.from({ length }, (_, i) => (values[i] ? 1 : 0));
      break;
    case "int64":
      typed = BigInt64Array.from({ length }, (_, i) =>
        BigInt(values[i] as bigint),
      );
      break;
    case "uint64":
      typed = BigUint64Array.from({ length }, (_, i) =>
        BigInt(values[i] as bigint),
      );
      break;
    default:
      typed = new {
        int8: Int8Array,
        int16: Int16Array,
        uint8: Uint8Array,
        uint16: Uint16Array,
        int32: Int32Array,
        uint32: Uint32Array,
        float32: Float32Array,
        float64: Float64Array,
      }[dtype](Array.from({ length }, (_, i) => Number(values[i])));
  }
  return Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength);
}

/** An explicit execution context. There is no implicit default context. */
export class NMathContext {
  /** @internal */
  readonly nativeContext: InstanceType<typeof binding.NativeContext>;

  /** @internal */
  constructor(options: ContextOptions) {
    if (typeof options !== "object" || options === null) {
      throw invalidOptions("Context options are required.");
    }
    this.nativeContext = native(
      () =>
        new binding.NativeContext({
          maxConcurrentJobs: count(
            options.maxConcurrentJobs,
            "maxConcurrentJobs",
            1,
          ),
          maxQueueLength: count(options.maxQueueLength, "maxQueueLength", 0),
          cpuMemoryBudget: budget(options.cpuMemoryBudget, "cpuMemoryBudget"),
          gpuMemoryBudget: budget(options.gpuMemoryBudget, "gpuMemoryBudget"),
        }),
    );
  }

  get closed(): boolean {
    return this.nativeContext.closed;
  }

  /**
   * @internal Fails with `NMATH_MEMORY_BUDGET_EXCEEDED` before a creation helper builds
   * input buffers for `elements` values that would not fit the CPU budget.
   */
  ensureCpuBudget(elements: number, dtype: ComputableDType): void {
    native(() =>
      this.nativeContext.checkCpuBudget(
        BigInt(Math.max(0, elements) * itemSize(dtype)),
      ),
    );
  }

  /** Creates an array from a typed array or nested JS arrays. */
  array<D extends ComputableDType, const S extends Shape = number[]>(
    data: TypedArrayOf<D> | Nested<ScalarOf<D>>,
    options: ArrayOptions<D, S>,
  ): NMathArray<D, S> {
    if (options.dtype === "str") {
      let width = 1;
      const visit = (v: unknown): void => {
        if (Array.isArray(v)) {
          for (const x of v) visit(x);
        } else if (typeof v === "string") {
          let n = 0;
          for (const _ of v) n++;
          width = Math.max(width, n);
        } else throw new TypeError("String storage requires JS strings.");
      };
      visit(data);
      options = { ...options, stringWidth: options.stringWidth ?? width };
      descriptor(options.dtype, options);
      if (width > options.stringWidth!)
        throw new RangeError("String input exceeds the selected width.");
      native(() =>
        this.nativeContext.checkCpuBudget(
          BigInt(product(inferShape(data))) * BigInt(options.stringWidth!) * 4n,
        ),
      );
    }
    let values: ArrayLike<unknown>;
    let shape: number[];
    const explicitShape = options.shape
      ? validatedShape(options.shape)
      : undefined;
    if (
      options.dtype === "float16" &&
      typeof Float16Array !== "undefined" &&
      data instanceof Float16Array
    ) {
      this.ensureCpuBudget(data.length, options.dtype);
      const array = native(() =>
        this.nativeContext.array(
          options.shape ? [...options.shape] : [data.length],
          options.dtype,
          Buffer.from(data.buffer, data.byteOffset, data.byteLength),
        ),
      );
      return new NMathArray(array, this);
    }
    if (
      ArrayBuffer.isView(data) &&
      (options.dtype === "complex64" || options.dtype === "complex128")
    ) {
      const Expected =
        options.dtype === "complex64" ? Float32Array : Float64Array;
      if (!(data instanceof Expected) || data.length % 2 !== 0) {
        throw new TypeError(
          "Complex typed input must contain real/imaginary pairs in the matching floating typed array.",
        );
      }
      this.ensureCpuBudget(data.length / 2, options.dtype);
      const shape = explicitShape ?? validatedShape([data.length / 2]);
      const array = native(() =>
        this.nativeContext.array(
          shape,
          options.dtype,
          Buffer.from(data.buffer, data.byteOffset, data.byteLength),
        ),
      );
      return new NMathArray(array, this);
    }
    if (ArrayBuffer.isView(data)) {
      values = data as unknown as ArrayLike<unknown>;
      shape = explicitShape ?? validatedShape([values.length]);
      this.ensureCpuBudget(values.length, options.dtype);
    } else {
      const inferred = inferShape(data);
      validatedShape(inferred);
      if (options.dtype !== "str")
        this.ensureCpuBudget(product(inferred), options.dtype);
      const flat: unknown[] = [];
      flatten(data, inferred, 0, flat);
      values = flat;
      shape = explicitShape ?? inferred;
    }
    const array = native(() =>
      this.nativeContext.array(
        shape,
        descriptor(options.dtype, options),
        toBytes(options.dtype, values, options),
      ),
    );
    return new NMathArray(array, this);
  }

  /** Creates float16 storage from exact IEEE-754 binary16 encodings. */
  arrayFromFloat16Bits<const S extends Shape = number[]>(
    bits: Uint16Array,
    options: { shape?: S } = {},
  ): NMathArray<"float16", S> {
    if (!(bits instanceof Uint16Array))
      throw new TypeError("Binary16 bits must be a Uint16Array.");
    const shape = validatedShape(options.shape ?? [bits.length]);
    this.ensureCpuBudget(bits.length, "float16");
    const array = native(() =>
      this.nativeContext.array(
        shape,
        "float16",
        Buffer.from(bits.buffer, bits.byteOffset, bits.byteLength),
      ),
    );
    return new NMathArray(array, this);
  }

  full<D extends ComputableDType, const S extends Shape>(
    shape: S,
    value: ScalarOf<D>,
    dtype: D,
    metadata: DTypeMetadata = {},
  ): NMathArray<D, S> {
    const checkedShape = validatedShape(shape);
    const size = product(checkedShape);
    if (dtype === "str" || dtype === "datetime64" || dtype === "timedelta64") {
      if (dtype === "str")
        metadata = {
          ...metadata,
          stringWidth:
            metadata.stringWidth ??
            (typeof value === "string" ? [...value].length || 1 : 1),
        };
      if (value instanceof Temporal)
        metadata = { unit: value.unit, ...metadata };
      const desc = descriptor(dtype, metadata);
      native(() =>
        this.nativeContext.checkCpuBudget(
          BigInt(size) *
            BigInt(dtype === "str" ? metadata.stringWidth! * 4 : 8),
        ),
      );
      const values = { length: size } as ArrayLike<unknown>;
      const repeated = new Proxy(values, {
        get: (_target, key) => (key === "length" ? size : value),
      });
      return new NMathArray(
        native(() =>
          this.nativeContext.array(
            checkedShape,
            desc,
            toBytes(dtype, repeated, metadata),
          ),
        ),
        this,
      );
    }
    this.ensureCpuBudget(size, dtype);
    // Fill a typed array directly instead of building a JS array of `size` values.
    const typed = filled(dtype, size, value);
    const array = native(() =>
      this.nativeContext.array(
        checkedShape,
        dtype,
        Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength),
      ),
    );
    return new NMathArray(array, this);
  }

  zeros<D extends ComputableDType, const S extends Shape>(
    shape: S,
    dtype: D,
  ): NMathArray<D, S> {
    return this.full(shape, zeroOf(dtype), dtype);
  }

  /** NumPy-compatible random generator (PCG64 seeded through SeedSequence). */
  generator(seed: number | bigint): NMathGenerator {
    const value = typeof seed === "bigint" ? seed : BigInt(seed);
    return new NMathGenerator(
      native(() => this.nativeContext.generator(value)),
      this,
    );
  }

  /** Copies an array from another context into this one, charged to this context. */
  async copy<D extends ComputableDType, S extends Shape>(
    array: NMathArray<D, S>,
    options: SignalOptions = {},
  ): Promise<NMathArray<D, S>> {
    const source = array.nativeArray;
    const copy = await withSignal(options.signal, (cancel) =>
      this.nativeContext.copyFrom(source, cancel),
    );
    return new NMathArray(copy, this);
  }

  /** Reads an NPY file. */
  async load(
    path: string,
    options: SignalOptions = {},
  ): Promise<NMathArray<ComputableDType, number[]>> {
    const array = await withSignal(options.signal, (cancel) =>
      this.nativeContext.loadNpy(path, cancel),
    );
    return new NMathArray(array, this);
  }

  readText(path: string, format: TextFormat, options?: TextReadOptions) {
    return readText(this, path, format, options);
  }
  writeText(
    path: string,
    format: TextFormat,
    columns: ReadonlyMap<string, Column>,
    validity?: Validity,
    options?: TextWriteOptions,
  ) {
    return writeText(this, path, format, columns, validity, options);
  }

  readColumnar(
    path: string,
    format: ColumnarFormat,
    options: ColumnarReadOptions = {},
  ) {
    return readColumnar(this, path, format, options);
  }
  writeColumnar(
    path: string,
    format: ColumnarFormat,
    columns: ReadonlyMap<string, Column>,
    validity: Validity = new Map(),
    options: ColumnarWriteOptions = {},
  ) {
    return writeColumnar(this, path, format, columns, validity, options);
  }

  /** Writes named arrays to an exact NPZ path in Map insertion order. */
  saveNpz(
    path: string,
    arrays: ReadonlyMap<string, NMathArray<any, any>>,
    options: SignalOptions & { overwrite?: boolean; compressed?: boolean } = {},
  ): Promise<void> {
    const entries = [...arrays];
    return withSignal(options.signal, (cancel) =>
      this.nativeContext.saveNpz(
        entries.map(([name]) => name),
        entries.map(([, array]) => array.nativeArray),
        path,
        options.compressed ?? false,
        options.overwrite ?? false,
        cancel,
      ),
    );
  }

  /** Reads an NPZ archive. Rank-zero members remain arrays. */
  async loadNpz(
    path: string,
    options: SignalOptions = {},
  ): Promise<Map<string, NMathArray<ComputableDType, number[]>>> {
    const entries = await withSignal(options.signal, (cancel) =>
      this.nativeContext.loadNpz(path, cancel),
    );
    return new Map(
      entries.map(({ name, array }) => [name, new NMathArray(array, this)]),
    );
  }

  metrics(): ContextMetrics {
    return this.nativeContext.metrics() as ContextMetrics;
  }

  /** Rejects new work, cancels queued and running work, and waits for cleanup. */
  close(): Promise<void> {
    return nativeAsync(() => this.nativeContext.close());
  }

  [Symbol.asyncDispose](): Promise<void> {
    return this.close();
  }
}

function filled(
  dtype: ComputableDType,
  size: number,
  value: unknown,
): ArrayBufferView {
  if (
    value instanceof Complex &&
    dtype !== "complex64" &&
    dtype !== "complex128"
  ) {
    throw new TypeError(
      "Complex values require a complex dtype. Convert explicitly first.",
    );
  }
  switch (dtype) {
    case "str":
    case "datetime64":
    case "timedelta64":
      throw new TypeError("Use metadata-aware array creation.");
    case "float16":
      return new (float16Constructor())(size).fill(Number(value));
    case "complex64":
    case "complex128": {
      const z = value instanceof Complex ? value : new Complex(Number(value));
      const parts =
        dtype === "complex64"
          ? new Float32Array(size * 2)
          : new Float64Array(size * 2);
      for (let i = 0; i < size; i++) {
        parts[2 * i] = z.real;
        parts[2 * i + 1] = z.imag;
      }
      return parts;
    }
    case "bool":
      return new Uint8Array(size).fill(value ? 1 : 0);
    case "int64":
      return new BigInt64Array(size).fill(BigInt(value as bigint));
    case "uint64":
      return new BigUint64Array(size).fill(BigInt(value as bigint));
    default:
      return new {
        int8: Int8Array,
        int16: Int16Array,
        uint8: Uint8Array,
        uint16: Uint16Array,
        int32: Int32Array,
        uint32: Uint32Array,
        float32: Float32Array,
        float64: Float64Array,
      }[dtype](size).fill(Number(value));
  }
}

function zeroOf<D extends ComputableDType>(dtype: D): ScalarOf<D> {
  return (
    dtype === "str"
      ? ""
      : dtype === "datetime64" || dtype === "timedelta64"
        ? new Temporal(0n, "ns", dtype)
        : dtype === "complex64" || dtype === "complex128"
          ? new Complex(0)
          : dtype === "int64" || dtype === "uint64"
            ? 0n
            : dtype === "bool"
              ? false
              : 0
  ) as ScalarOf<D>;
}

export { NMathGenerator } from "./random.js";

/** Creates an explicit execution context. All four limits are required. */
export function createContext(options: ContextOptions): NMathContext {
  return new NMathContext(options);
}

export type { DType };

export type {
  TextReadOptions,
  TextWriteOptions,
  TextFormat,
  TextDType,
  TextTable,
} from "./io-text.js";
