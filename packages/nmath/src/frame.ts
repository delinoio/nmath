// SPDX-License-Identifier: Apache-2.0
// Shared column storage for the pandas and Polars entry points. Columns are lazy 1-D
// NMath values; computing a frame materializes every column in one context.

import {
  type ComputeOptions,
  Lazy,
  NMathArray,
  type NMathContext,
  type Operand,
} from "./core.js";
import { NMathError } from "./errors.js";
import type { ComputableDType, Shape } from "./types.js";

// Columns are loosely typed: frames track dtypes at runtime, not per column in the type.
export type Column = Lazy<any, any>;

function shapeError(message: string): NMathError {
  return new NMathError(
    "NMATH_SHAPE_MISMATCH",
    `NMATH_SHAPE_MISMATCH: ${message} Check the operand shapes against the operation requirements.`,
  );
}

export function ensureColumn(name: string, values: Column): Column {
  if (values.shape.length !== 1) {
    throw shapeError(
      `Column '${name}' must be one-dimensional, got shape [${values.shape.join(", ")}].`,
    );
  }
  return values;
}

export function ensureSameLength(columns: ReadonlyMap<string, Column>): number {
  let length: number | undefined;
  for (const [name, column] of columns) {
    const n = column.shape[0];
    if (length !== undefined && n !== length) {
      throw shapeError(
        `Column '${name}' has length ${n}, but other columns have length ${length}.`,
      );
    }
    length = n;
  }
  return length ?? 0;
}

export async function computeColumns(
  columns: ReadonlyMap<string, Column>,
  options: ComputeOptions,
): Promise<Map<string, NMathArray<ComputableDType, Shape>>> {
  const entries = await Promise.all(
    [...columns].map(async ([name, column]) => {
      if (column instanceof NMathArray) {
        return [name, column] as const;
      }
      const result = await column.compute({ ...options, device: "cpu" });
      if (!(result instanceof NMathArray)) {
        throw shapeError(`Column '${name}' computed to a scalar.`);
      }
      return [name, result] as const;
    }),
  );
  return new Map(entries);
}

export function columnContext(
  columns: ReadonlyMap<string, Column>,
): NMathContext | undefined {
  for (const column of columns.values()) {
    if (column.context) {
      return column.context;
    }
  }
  return undefined;
}

export type { Operand };
