// SPDX-License-Identifier: Apache-2.0
import type { NMathContext } from "./core.js";
import * as pandas from "./pandas.js";
import * as polars from "./polars.js";
import {
  readText,
  writeText,
  type TextFormat,
  type TextReadOptions,
  type TextWriteOptions,
} from "./io-text.js";
export type { TextReadOptions, TextWriteOptions } from "./io-text.js";
export async function readPandas(
  context: NMathContext,
  path: string,
  format: TextFormat,
  options?: TextReadOptions,
): Promise<pandas.DataFrame> {
  const result = await readText(context, path, format, options);
  return new pandas.DataFrame(result.columns, { validity: result.validity });
}
export async function readPolars(
  context: NMathContext,
  path: string,
  format: TextFormat,
  options?: TextReadOptions,
): Promise<polars.DataFrame> {
  const result = await readText(context, path, format, options);
  return new polars.DataFrame(result.columns, result.validity);
}
export function writeFrame(
  context: NMathContext,
  frame: pandas.DataFrame | polars.DataFrame,
  path: string,
  format: TextFormat,
  options: TextWriteOptions & { dropIndex?: boolean } = {},
): Promise<void> {
  if (
    frame instanceof pandas.DataFrame &&
    frame.index.some((v, i) => v !== i) &&
    options.dropIndex !== true
  )
    throw new TypeError(
      "Text record output cannot preserve a custom index. Use dropIndex:true to authorize label loss.",
    );
  return writeText(
    context,
    path,
    format,
    frame.columnMap,
    frame.validity,
    options,
  );
}
