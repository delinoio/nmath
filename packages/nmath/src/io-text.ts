// SPDX-License-Identifier: Apache-2.0
import { NMathArray, type NMathContext } from "./core.js";
import { binding } from "./native.js";
import { nativeAsync } from "./errors.js";
import type { Column, Validity } from "./frame.js";
import type { ComputableDType, Shape } from "./types.js";

export type TextFormat = "csv" | "json";
export type TextDType = Exclude<ComputableDType, "complex64" | "complex128">;
export interface TextReadOptions {
  /** Complete ordered schema. Required to preserve narrow and all-null dtypes. */
  schema?: ReadonlyMap<string, TextDType>;
  signal?: AbortSignal;
}
export interface TextWriteOptions {
  overwrite?: boolean;
  signal?: AbortSignal;
}
export interface TextTable {
  columns: Map<string, NMathArray<TextDType, Shape>>;
  validity: Map<string, NMathArray<"bool", Shape>>;
}
async function cancellable<T>(
  signal: AbortSignal | undefined,
  run: (cancel: InstanceType<typeof binding.NativeCancellation>) => Promise<T>,
): Promise<T> {
  const cancel = new binding.NativeCancellation();
  const abort = () => cancel.cancel();
  if (signal?.aborted) abort();
  signal?.addEventListener("abort", abort, { once: true });
  try {
    return await nativeAsync(() => run(cancel));
  } finally {
    signal?.removeEventListener("abort", abort);
  }
}
export async function readText(
  context: NMathContext,
  path: string,
  format: TextFormat,
  options: TextReadOptions = {},
): Promise<TextTable> {
  const schema = [...(options.schema ?? [])];
  const result = await cancellable(options.signal, (cancel) =>
    context.nativeContext.readText(
      path,
      format,
      schema.map(([n]) => n),
      schema.map(([, d]) => d),
      cancel,
    ),
  );
  return {
    columns: new Map(
      result.names.map((n, i) => [
        n,
        new NMathArray<TextDType, Shape>(result.values[i], context),
      ]),
    ),
    validity: new Map(
      result.names.map((n, i) => [
        n,
        new NMathArray<"bool", Shape>(result.valid[i], context),
      ]),
    ),
  };
}
export function writeText(
  context: NMathContext,
  path: string,
  format: TextFormat,
  columns: ReadonlyMap<string, Column>,
  validity: Validity = new Map(),
  options: TextWriteOptions = {},
): Promise<void> {
  for (const name of validity.keys())
    if (!columns.has(name))
      throw new TypeError("Validity columns must exist in the table.");
  const entries = [...columns];
  return cancellable(options.signal, (cancel) =>
    context.nativeContext.writeText(
      path,
      format,
      entries.map(([n]) => n),
      entries.map(([, v]) => v.nativeExpr()),
      entries
        .filter(([n]) => validity.has(n))
        .map(([n]) => validity.get(n)!.nativeExpr()),
      entries.map(([n]) => validity.has(n)),
      options.overwrite ?? false,
      cancel,
    ),
  );
}
