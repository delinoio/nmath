// SPDX-License-Identifier: Apache-2.0
// Generated napi-rs declarations; keep synchronized with native/native.d.ts.

export declare class NativeArray {
  get dtype(): string
  get dtypeDescriptor(): string
  get shape(): Array<number>
  get contextId(): number
  get disposed(): boolean
  /** Copies the visible elements in C order using native byte order. */
  toBytes(): Buffer
  /**
   * Reads one element. 64-bit integers are returned as bigint so they stay
   * exact.
   */
  item(index: Array<number>): number | bigint | boolean | NativeComplex | string | NativeTemporal
  sharesStorage(other: NativeArray): boolean
  /**
   * Releases this handle. Storage stays alive while other views or running
   * jobs use it.
   */
  dispose(): void
}

/** A cancellation handle the TypeScript layer connects to an `AbortSignal`. */
export declare class NativeCancellation {
  constructor()
  cancel(): void
  get cancelled(): boolean
}

export declare class NativeContext {
  constructor(options: NativeContextOptions)
  get id(): number
  get closed(): boolean
  /**
   * Fails with `NMATH_MEMORY_BUDGET_EXCEEDED` when `bytes` more would not
   * fit the CPU budget now. Creation helpers call it before they build
   * input buffers.
   */
  checkCpuBudget(bytes: bigint): void
  array(shape: Array<number>, dtype: string, data: Buffer): NativeArray
  generator(seed: bigint): NativeGenerator
  compute(expr: NativeExpr, device: string, signal?: NativeCancellation | undefined | null): Promise<NativeArray>
  assign(target: NativeArray, selectors: Array<NativeSelector>, value: NativeExpr, signal?: NativeCancellation | undefined | null): Promise<void>
  copyFrom(source: NativeArray, signal?: NativeCancellation | undefined | null): Promise<NativeArray>
  saveNpy(array: NativeArray, path: string, overwrite: boolean, signal?: NativeCancellation | undefined | null): Promise<void>
  loadNpy(path: string, signal?: NativeCancellation | undefined | null): Promise<NativeArray>
  saveNpz(names: Array<string>, arrays: Array<NativeArray>, path: string, compressed: boolean, overwrite: boolean, signal?: NativeCancellation | undefined | null): Promise<void>
  loadNpz(path: string, signal?: NativeCancellation | undefined | null): Promise<Array<{ name: string; array: NativeArray }>>
  /**
   * Closes the context on a helper thread and resolves when cleanup
   * finishes.
   */
  close(): Promise<void>
  metrics(): NativeMetrics
  readColumnar(path: string, fileFormat: string, signal?: NativeCancellation | undefined | null): Promise<NativeColumnarResult>
  writeColumnar(path: string, fileFormat: string, names: Array<string>, values: Array<NativeExpr>, validity: Array<NativeExpr>, nullable: Array<boolean>, overwrite: boolean, signal?: NativeCancellation | undefined | null): Promise<void>
  readText(path: string, kind: string, names: Array<string>, dtypes: Array<string>, signal?: NativeCancellation | undefined | null): Promise<NativeTextTable>
  writeText(path: string, kind: string, names: Array<string>, values: Array<NativeExpr>, validity: Array<NativeExpr>, nullable: Array<boolean>, overwrite: boolean, signal?: NativeCancellation | undefined | null): Promise<void>
}

export declare class NativeExpr {
  static input(array: NativeArray): NativeExpr
  static textScalar(value: string): NativeExpr
  static temporalScalar(ticks: bigint, dtype: string): NativeExpr
  static temporalMilliseconds(ticks: bigint, unit: string): bigint
  static complexScalar(real: number, imag: number): NativeExpr
  static floatScalar(value: number): NativeExpr
  static intScalar(value: bigint): NativeExpr
  static boolScalar(value: boolean): NativeExpr
  static binary(op: string, lhs: NativeExpr, rhs: NativeExpr): NativeExpr
  static unary(op: string, input: NativeExpr): NativeExpr
  static reduce(op: string, input: NativeExpr, axes?: Array<number> | undefined | null): NativeExpr
  static variance(op: string, input: NativeExpr, axis: number | undefined | null, ddof: number): NativeExpr
  static scan(op: string, input: NativeExpr, axis?: number | undefined | null): NativeExpr
  static transpose(input: NativeExpr, axes?: Array<number> | undefined | null): NativeExpr
  static reshape(input: NativeExpr, shape: Array<number>, copy?: boolean | undefined | null): NativeExpr
  static matmul(lhs: NativeExpr, rhs: NativeExpr): NativeExpr
  static linalg(op: string, input: NativeExpr, rhs?: NativeExpr | undefined | null): NativeExpr
  static polynomial(coefficients: NativeExpr, points: NativeExpr, ascending: boolean, tensor: boolean): NativeExpr
  static polynomialDerivative(coefficients: NativeExpr, ascending: boolean, count: number, axis: number, scale: number | NativeComplex): NativeExpr
  static fft(op: string, input: NativeExpr, n: number | undefined | null, axis: number, norm: string): NativeExpr
  static index(input: NativeExpr, selectors: Array<NativeSelector>): NativeExpr
  static cast(input: NativeExpr, dtype: string): NativeExpr
  static random(generator: NativeGenerator, shape: Array<number>): NativeExpr
  static randomDistribution(generator: NativeGenerator, shape: Array<number>, operation: string, dtype: string, first: string, second: string, endpoint: boolean): NativeExpr
  get dtypeDescriptor(): string
  get dtype(): string
  get shape(): Array<number>
  get contextId(): number | null
}

export declare class NativeGenerator {}

export declare class NativeTable {
  static source(values: Array<NativeExpr>, validity: Array<NativeExpr>, nullable: Array<boolean>): NativeTable
  select(positions: Array<number>): NativeTable
  filter(position: number): NativeTable
  sort(position: number, descending: boolean, nullsLast: boolean): NativeTable
  join(other: NativeTable, leftKey: number, rightKey: number, left: boolean, nullsEqual: boolean): NativeTable
  group(key: number, value: number, operation: string, dropNulls: boolean, pandas: boolean): NativeTable
  compute(context: NativeContext, signal?: NativeCancellation | undefined | null): Promise<NativeTableResult>
}

export interface NativeColumnarResult {
  names: Array<string>
  values: Array<NativeArray>
  valid: Array<NativeArray>
  nullable: Array<boolean>
  rows: number
}

export interface NativeComplex {
  real: number
  imag: number
}

export interface NativeContextOptions {
  maxConcurrentJobs: number
  maxQueueLength: number
  cpuMemoryBudget: bigint
  gpuMemoryBudget: bigint
}

export interface NativeExecution {
  id: number
  kind: string
  operation: string
  device: string
  status: string
  queueTimeMs: number
  executionTimeMs: number
}

export interface NativeMemoryUsage {
  limit: bigint
  used: bigint
  peak: bigint
  cumulativeReserved: bigint
  rejected: number
}

export interface NativeMetrics {
  executions: Array<NativeExecution>
  totalSubmitted: number
  totalSucceeded: number
  totalFailed: number
  totalCancelled: number
  totalRejected: number
  cpuMemory: NativeMemoryUsage
  gpuMemory: NativeMemoryUsage
}

export interface NativeNpzEntry {
  name: string
  array: NativeArray
}

export interface NativeSelector {
  /** `index`, `slice`, `indices` or `mask`. */
  kind: string
  index?: number
  start?: number
  stop?: number
  step?: number
  indices?: Array<number>
  mask?: Array<boolean>
}

export interface NativeTableResult {
  values: Array<NativeArray>
  valid: Array<NativeArray>
  rows: number
}

export interface NativeTemporal {
  ticks: bigint
  dtype: string
  unit: string
}

export interface NativeTextTable {
  names: Array<string>
  values: Array<NativeArray>
  valid: Array<NativeArray>
}
