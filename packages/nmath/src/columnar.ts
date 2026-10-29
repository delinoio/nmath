// SPDX-License-Identifier: Apache-2.0
import { NMathArray, type NMathContext } from "./core.js";
import { nativeAsync, NMathError } from "./errors.js";
import { binding } from "./native.js";
import type {
  NativeArray,
  NativeCancellation,
  NativeExpr,
} from "./native-binding.generated.js";
import { validateColumns, type Column, type Validity } from "./frame.js";

export type ColumnarFormat = "arrow" | "parquet";
export interface ColumnarReadOptions {
  signal?: AbortSignal;
}
export interface ColumnarWriteOptions extends ColumnarReadOptions {
  overwrite?: boolean;
}
export interface ColumnarTable {
  readonly columns: ReadonlyMap<string, NMathArray<any, any>>;
  readonly validity: ReadonlyMap<string, NMathArray<"bool", any>>;
  readonly rows: number;
}
interface NativeColumnarContext {
  readColumnar(
    path: string,
    format: ColumnarFormat,
    signal: NativeCancellation,
  ): Promise<{
    names: string[];
    values: NativeArray[];
    valid: NativeArray[];
    nullable: boolean[];
    rows: number;
  }>;
  writeColumnar(
    path: string,
    format: ColumnarFormat,
    names: string[],
    values: NativeExpr[],
    valid: NativeExpr[],
    nullable: boolean[],
    overwrite: boolean,
    signal: NativeCancellation,
  ): Promise<void>;
}
async function run<T>(
  signal: AbortSignal | undefined,
  work: (cancel: NativeCancellation) => Promise<T>,
): Promise<T> {
  const cancel = new binding.NativeCancellation();
  const abort = () => cancel.cancel();
  if (signal?.aborted) abort();
  signal?.addEventListener("abort", abort, { once: true });
  try {
    return await nativeAsync(() => work(cancel));
  } finally {
    signal?.removeEventListener("abort", abort);
  }
}
export async function readColumnar(
  context: NMathContext,
  path: string,
  format: ColumnarFormat,
  options: ColumnarReadOptions = {},
): Promise<ColumnarTable> {
  const result = await run(options.signal, (cancel) =>
    (context.nativeContext as unknown as NativeColumnarContext).readColumnar(
      path,
      format,
      cancel,
    ),
  );
  const columns = new Map<string, NMathArray<any, any>>();
  const validity = new Map<string, NMathArray<"bool", any>>();
  for (let i = 0; i < result.names.length; i++) {
    columns.set(result.names[i], new NMathArray(result.values[i], context));
    if (result.nullable[i])
      validity.set(result.names[i], new NMathArray(result.valid[i], context));
    else result.valid[i].dispose();
  }
  return { columns, validity, rows: result.rows };
}
export async function writeColumnar(
  context: NMathContext,
  path: string,
  format: ColumnarFormat,
  columns: ReadonlyMap<string, Column>,
  validity: Validity = new Map(),
  options: ColumnarWriteOptions = {},
): Promise<void> {
  if (columns.size && validateColumns(columns, validity) !== context)
    throw new NMathError(
      "NMATH_CONTEXT_MISMATCH",
      "File columns must use the execution context. Copy them explicitly.",
    );
  if (!columns.size && validity.size)
    throw new NMathError(
      "NMATH_IO_FORMAT",
      "An empty schema cannot have validity columns.",
    );
  const names = [...columns.keys()];
  return run(options.signal, (cancel) =>
    (context.nativeContext as unknown as NativeColumnarContext).writeColumnar(
      path,
      format,
      names,
      names.map((n) => columns.get(n)!.nativeExpr()),
      names
        .filter((n) => validity.has(n))
        .map((n) => validity.get(n)!.nativeExpr()),
      names.map((n) => validity.has(n)),
      options.overwrite ?? false,
      cancel,
    ),
  );
}
/** pandas index metadata is not silently dropped by file adaptation. */
export function requireDefaultIndex(index: readonly (string | number)[]): void {
  if (index.some((value, i) => value !== i))
    throw new NMathError(
      "NMATH_DTYPE_UNSUPPORTED",
      "Columnar output cannot preserve this pandas index. Reset it explicitly before writing.",
    );
}

/** Frame constructors currently derive row counts from their columns. */
export function requireFrameRows(table: ColumnarTable): void {
  if (table.rows > 0 && table.columns.size === 0)
    throw new NMathError(
      "NMATH_DTYPE_UNSUPPORTED",
      "Nonempty zero-column tables cannot be represented by this DataFrame. Use readColumnar to retain the explicit row count.",
    );
}
