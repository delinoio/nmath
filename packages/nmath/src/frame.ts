// SPDX-License-Identifier: Apache-2.0
import {
  type ComputeOptions,
  Lazy,
  NMathArray,
  type NMathContext,
  type Operand,
} from "./core.js";
import { NMathError, native, nativeAsync } from "./errors.js";
import { binding } from "./native.js";
import type {
  NativeArray,
  NativeCancellation,
  NativeContext,
  NativeExpr,
} from "./native-binding.generated.js";
import type { ComputableDType, Shape } from "./types.js";
export type Column = Lazy<any, any>;
export type Aggregate = "sum" | "mean" | "min" | "max" | "count";
export type Validity = ReadonlyMap<string, Column>;
export function tableError(
  message = "Invalid table arguments. Check the schema and operation options.",
): NMathError {
  return new NMathError(
    "NMATH_INDEX_INVALID",
    `NMATH_INDEX_INVALID: ${message}`,
  );
}
function shapeError(): NMathError {
  return new NMathError(
    "NMATH_SHAPE_MISMATCH",
    "NMATH_SHAPE_MISMATCH: Table columns must have equal one-dimensional shapes. Check the schema.",
  );
}
export function ensureColumn(_name: string, values: Column): Column {
  if (values.shape.length !== 1) throw shapeError();
  return values;
}
export function ensureSameLength(columns: ReadonlyMap<string, Column>): number {
  let length: number | undefined;
  for (const col of columns.values()) {
    ensureColumn("", col);
    if (length !== undefined && col.shape[0] !== length) throw shapeError();
    length = col.shape[0];
  }
  columnContext(columns);
  return length ?? 0;
}
export function columnContext(
  columns: ReadonlyMap<string, Column>,
): NMathContext | undefined {
  let context: NMathContext | undefined;
  for (const col of columns.values())
    if (col.context) {
      if (context && context !== col.context)
        throw new NMathError(
          "NMATH_CONTEXT_MISMATCH",
          "NMATH_CONTEXT_MISMATCH: Table inputs must use one context. Copy inputs explicitly.",
        );
      context = col.context;
    }
  return context;
}
// This narrow interface isolates the binding addition from the generated shared file.
interface NativeTable {
  select(positions: number[]): NativeTable;
  filter(position: number): NativeTable;
  sort(position: number, descending: boolean, nullsLast: boolean): NativeTable;
  join(
    other: NativeTable,
    leftKey: number,
    rightKey: number,
    left: boolean,
    nullsEqual: boolean,
  ): NativeTable;
  group(
    key: number,
    value: number,
    op: string,
    dropNulls: boolean,
    pandas: boolean,
  ): NativeTable;
  compute(
    context: NativeContext,
    signal: NativeCancellation,
  ): Promise<{ values: NativeArray[]; valid: NativeArray[]; rows: number }>;
}
const tables = binding as unknown as {
  NativeTable: {
    source(
      values: NativeExpr[],
      validity: NativeExpr[],
      nullable: boolean[],
    ): NativeTable;
  };
};
export function validateColumns(
  columns: ReadonlyMap<string, Column>,
  validity: Validity = new Map(),
): NMathContext {
  ensureSameLength(columns);
  const context = columnContext(
    new Map(
      [...columns.values(), ...validity.values()].map((v, i) => [String(i), v]),
    ),
  );
  if (!context)
    throw tableError(
      "A nonempty table requires an explicit context. Supply context-owned columns.",
    );
  for (const [name, mask] of validity)
    if (
      !columns.has(name) ||
      mask.dtype !== "bool" ||
      mask.shape.length !== 1 ||
      mask.shape[0] !== columns.get(name)!.shape[0]
    )
      throw shapeError();
  return context;
}
export class TablePlan {
  constructor(
    readonly plan: NativeTable,
    readonly names: readonly string[],
    readonly context: NMathContext,
    readonly nullable: ReadonlySet<string> = new Set(),
  ) {}
  static source(
    columns: ReadonlyMap<string, Column>,
    validity: Validity = new Map(),
  ): TablePlan {
    const context = validateColumns(columns, validity);
    const names = [...columns.keys()];
    return new TablePlan(
      native(() =>
        tables.NativeTable.source(
          [...columns.values()].map((c) => c.nativeExpr()),
          names
            .filter((n) => validity.has(n))
            .map((n) => validity.get(n)!.nativeExpr()),
          names.map((n) => validity.has(n)),
        ),
      ),
      names,
      context,
      new Set(validity.keys()),
    );
  }
  position(name: string): number {
    const n = this.names.indexOf(name);
    if (n < 0) throw tableError();
    return n;
  }
  select(names: readonly string[]): TablePlan {
    if (!names.length || new Set(names).size !== names.length)
      throw tableError();
    return new TablePlan(
      native(() => this.plan.select(names.map((n) => this.position(n)))),
      [...names],
      this.context,
      new Set(names.filter((n) => this.nullable.has(n))),
    );
  }
  filter(name: string): TablePlan {
    return new TablePlan(
      native(() => this.plan.filter(this.position(name))),
      this.names,
      this.context,
      this.nullable,
    );
  }
  sort(name: string, descending = false, nullsLast = true): TablePlan {
    return new TablePlan(
      native(() => this.plan.sort(this.position(name), descending, nullsLast)),
      this.names,
      this.context,
      this.nullable,
    );
  }
  join(
    other: TablePlan,
    key: string,
    options: {
      rightKey?: string;
      how?: "inner" | "left";
      nullsEqual?: boolean;
      suffix?: string;
    } = {},
  ): TablePlan {
    if (this.context !== other.context)
      throw new NMathError(
        "NMATH_CONTEXT_MISMATCH",
        "NMATH_CONTEXT_MISMATCH: Table inputs must use one context. Copy inputs explicitly.",
      );
    if (options.how && options.how !== "inner" && options.how !== "left")
      throw tableError();
    const leftKey = this.position(key);
    const rightKey = other.position(options.rightKey ?? key);
    const right = other.names.map((n) =>
      this.names.includes(n) ? n + (options.suffix ?? "_right") : n,
    );
    const coalesce = (options.rightKey ?? key) === key;
    const positions = [
      ...this.names.map((_, i) => i),
      ...other.names.flatMap((_, i) =>
        coalesce && i === rightKey ? [] : [this.names.length + i],
      ),
    ];
    const names = positions.map((i) =>
      i < this.names.length ? this.names[i] : right[i - this.names.length],
    );
    if (!names.length || new Set(names).size !== names.length)
      throw tableError();
    return new TablePlan(
      native(() => {
        const joined = this.plan.join(
          other.plan,
          leftKey,
          rightKey,
          options.how === "left",
          options.nullsEqual ?? false,
        );
        return coalesce ? joined.select(positions) : joined;
      }),
      names,
      this.context,
      new Set([
        ...this.nullable,
        ...other.names.flatMap((name, i) =>
          coalesce && i === rightKey
            ? []
            : options.how === "left" || other.nullable.has(name)
              ? [right[i]]
              : [],
        ),
      ]),
    );
  }
  group(
    key: string,
    value: string,
    op: Aggregate,
    dropNulls = false,
    pandas = false,
  ): TablePlan {
    if (key === value) throw tableError();
    return new TablePlan(
      native(() =>
        this.plan.group(
          this.position(key),
          this.position(value),
          op,
          dropNulls,
          pandas,
        ),
      ),
      [key, value],
      this.context,
      new Set([
        ...(this.nullable.has(key) ? [key] : []),
        ...(["min", "max", "mean"].includes(op) ? [value] : []),
      ]),
    );
  }
  async compute(options: ComputeOptions = {}): Promise<{
    columns: Map<string, NMathArray<ComputableDType, Shape>>;
    validity: Map<string, Column>;
  }> {
    if (options.device && options.device !== "cpu")
      throw new NMathError(
        "NMATH_GPU_UNSUPPORTED_OPERATION",
        "NMATH_GPU_UNSUPPORTED_OPERATION: Tables support CPU execution only. Select cpu.",
      );
    const cancel = new binding.NativeCancellation();
    const abort = () => cancel.cancel();
    if (options.signal?.aborted) abort();
    options.signal?.addEventListener("abort", abort, { once: true });
    try {
      const result = await nativeAsync(() =>
        this.plan.compute(this.context.nativeContext, cancel),
      );
      result.valid.forEach((mask, i) => {
        if (!this.nullable.has(this.names[i])) mask.dispose();
      });
      return {
        columns: new Map(
          this.names.map((n, i) => [
            n,
            new NMathArray(result.values[i], this.context),
          ]),
        ),
        validity: new Map(
          this.names.flatMap((n, i) =>
            this.nullable.has(n)
              ? [[n, new NMathArray(result.valid[i], this.context)] as const]
              : [],
          ),
        ),
      };
    } finally {
      options.signal?.removeEventListener("abort", abort);
    }
  }
}
export async function computeColumns(
  columns: ReadonlyMap<string, Column>,
  options: ComputeOptions,
): Promise<Map<string, NMathArray<ComputableDType, Shape>>> {
  if (!columns.size) return new Map();
  return (await TablePlan.source(columns).compute(options)).columns;
}
export type { Operand };
