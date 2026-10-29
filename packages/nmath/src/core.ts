// SPDX-License-Identifier: Apache-2.0
import { NMathError, native, nativeAsync } from "./errors.js";
import { binding } from "./native.js";
import type {
  NativeArray,
  NativeExpr,
  NativeGenerator,
  NativeSelector,
} from "./native-binding.generated.js";
import type {
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
    `NMATH_DTYPE_UNSUPPORTED: Operands must be NMath values, numbers, bigints or booleans. Convert the operand to a supported dtype explicitly.`,
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

  add<D2 extends ComputableDType, S2 extends Shape>(
    other: Lazy<D2, S2> & BroadcastCheck<S, S2>,
  ): Expr<Promote<D, D2>, Broadcast<S, S2>>;
  add<W extends WeakScalar>(other: W): Expr<PromoteWeak<D, W>, S>;
  add(other: Operand): Expr<any, any> {
    return this.binaryOp("add", other);
  }

  sub<D2 extends ComputableDType, S2 extends Shape>(
    other: Lazy<D2, S2> & BroadcastCheck<S, S2>,
  ): Expr<Promote<D, D2>, Broadcast<S, S2>>;
  sub<W extends WeakScalar>(other: W): Expr<PromoteWeak<D, W>, S>;
  sub(other: Operand): Expr<any, any> {
    return this.binaryOp("sub", other);
  }

  mul<D2 extends ComputableDType, S2 extends Shape>(
    other: Lazy<D2, S2> & BroadcastCheck<S, S2>,
  ): Expr<Promote<D, D2>, Broadcast<S, S2>>;
  mul<W extends WeakScalar>(other: W): Expr<PromoteWeak<D, W>, S>;
  mul(other: Operand): Expr<any, any> {
    return this.binaryOp("mul", other);
  }

  div<D2 extends ComputableDType, S2 extends Shape>(
    other: Lazy<D2, S2> & BroadcastCheck<S, S2>,
  ): Expr<TrueDivide<Promote<D, D2>>, Broadcast<S, S2>>;
  div<W extends WeakScalar>(other: W): Expr<TrueDivide<PromoteWeak<D, W>>, S>;
  div(other: Operand): Expr<any, any> {
    return this.binaryOp("div", other);
  }

  mod<D2 extends ComputableDType, S2 extends Shape>(
    other: Lazy<D2, S2> & BroadcastCheck<S, S2>,
  ): Expr<Promote<D, D2>, Broadcast<S, S2>>;
  mod<W extends WeakScalar>(other: W): Expr<PromoteWeak<D, W>, S>;
  mod(other: Operand): Expr<any, any> {
    return this.binaryOp("mod", other);
  }

  pow<D2 extends ComputableDType, S2 extends Shape>(
    other: Lazy<D2, S2> & BroadcastCheck<S, S2>,
  ): Expr<Promote<D, D2>, Broadcast<S, S2>>;
  pow<W extends WeakScalar>(other: W): Expr<PromoteWeak<D, W>, S>;
  pow(other: Operand): Expr<any, any> {
    return this.binaryOp("pow", other);
  }

  neg(): Expr<D, S> {
    return new Expr(
      native(() => binding.NativeExpr.unary("neg", this.nativeExpr())),
      this.context,
    );
  }

  pos(): Expr<D, S> {
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
    axis?: A,
  ): Expr<SumDType<D>, ReduceShape<S, A>> {
    return this.reduceOp("sum", axis);
  }

  mean<A extends number | undefined = undefined>(
    axis?: A,
  ): Expr<TrueDivide<D>, ReduceShape<S, A>> {
    return this.reduceOp("mean", axis);
  }

  min<A extends number | undefined = undefined>(
    axis?: A,
  ): Expr<D, ReduceShape<S, A>> {
    return this.reduceOp("min", axis);
  }

  max<A extends number | undefined = undefined>(
    axis?: A,
  ): Expr<D, ReduceShape<S, A>> {
    return this.reduceOp("max", axis);
  }

  matmul<D2 extends ComputableDType, S2 extends Shape>(
    other: Lazy<D2, S2> & MatMulCheck<S, S2>,
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
  astype<D2 extends ComputableDType>(dtype: D2): Expr<D2, S> {
    return new Expr(
      native(() => binding.NativeExpr.cast(this.nativeExpr(), dtype)),
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
      return native(() => array.item([]));
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
    return native(() => this.#array.item(checked)) as ScalarOf<D>;
  }

  /** Copies the elements into a typed array in C order. */
  toTypedArray(): TypedArrayOf<D> {
    const bytes = native(() => this.#array.toBytes());
    const copy = new Uint8Array(bytes).buffer;
    return typedArrayFor(this.dtype, copy) as TypedArrayOf<D>;
  }

  /** Copies the elements into nested JS arrays. */
  toArray(): Nested<ScalarOf<D>> {
    const flat = Array.from(
      this.toTypedArray() as ArrayLike<number | bigint>,
      (v) => (this.dtype === "bool" ? v !== 0 : v),
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

function typedArrayFor(dtype: ComputableDType, buffer: ArrayBuffer) {
  switch (dtype) {
    case "bool":
      return new Uint8Array(buffer);
    case "int32":
      return new Int32Array(buffer);
    case "int64":
      return new BigInt64Array(buffer);
    case "uint32":
      return new Uint32Array(buffer);
    case "uint64":
      return new BigUint64Array(buffer);
    case "float32":
      return new Float32Array(buffer);
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

export interface ArrayOptions<D extends ComputableDType, S extends Shape> {
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

function flatten(data: unknown, out: unknown[]): void {
  if (Array.isArray(data)) {
    for (const item of data) {
      flatten(item, out);
    }
  } else {
    out.push(data);
  }
}

function toBytes(dtype: ComputableDType, values: ArrayLike<unknown>): Buffer {
  const length = values.length;
  let typed: ArrayBufferView;
  switch (dtype) {
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

  /** Creates an array from a typed array or nested JS arrays. */
  array<D extends ComputableDType, const S extends Shape = number[]>(
    data: TypedArrayOf<D> | Nested<ScalarOf<D>>,
    options: ArrayOptions<D, S>,
  ): NMathArray<D, S> {
    let values: ArrayLike<unknown>;
    let shape: number[];
    if (ArrayBuffer.isView(data)) {
      values = data as unknown as ArrayLike<unknown>;
      shape = options.shape ? [...options.shape] : [values.length];
    } else {
      const flat: unknown[] = [];
      flatten(data, flat);
      values = flat;
      shape = options.shape ? [...options.shape] : inferShape(data);
    }
    const array = native(() =>
      this.nativeContext.array(
        shape,
        options.dtype,
        toBytes(options.dtype, values),
      ),
    );
    return new NMathArray(array, this);
  }

  full<D extends ComputableDType, const S extends Shape>(
    shape: S,
    value: ScalarOf<D>,
    dtype: D,
  ): NMathArray<D, S> {
    const size = shape.reduce((a: number, b: number) => a * b, 1);
    return this.array<D, S>(
      Array.from({ length: size }, () => value) as Nested<ScalarOf<D>>,
      { dtype, shape },
    );
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

function zeroOf<D extends ComputableDType>(dtype: D): ScalarOf<D> {
  return (
    dtype === "int64" || dtype === "uint64" ? 0n : dtype === "bool" ? false : 0
  ) as ScalarOf<D>;
}

export class NMathGenerator {
  readonly #generator: NativeGenerator;
  readonly context: NMathContext;

  /** @internal */
  constructor(generator: NativeGenerator, context: NMathContext) {
    this.#generator = generator;
    this.context = context;
  }

  /**
   * `Generator.random(shape)` as a lazy float64 expression. State advances only when a
   * computation that uses it succeeds; jobs sharing a generator run in request order.
   */
  random<const S extends Shape>(shape: S): Expr<"float64", S> {
    return new Expr(
      binding.NativeExpr.random(this.#generator, [...shape]),
      this.context,
    );
  }
}

/** Creates an explicit execution context. All four limits are required. */
export function createContext(options: ContextOptions): NMathContext {
  return new NMathContext(options);
}

export type { DType };
