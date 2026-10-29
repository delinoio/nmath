// SPDX-License-Identifier: Apache-2.0
import type { NMathContext } from "./core.js";
import {
  readPolars,
  writeFrame,
  type TextReadOptions,
  type TextWriteOptions,
} from "./io-text-frames.js";
import {
  readColumnar,
  writeColumnar,
  requireDefaultIndex,
  type ColumnarReadOptions,
  type ColumnarWriteOptions,
} from "./columnar.js";
// Polars-style entry point (subset). Polars operations always run on the CPU. The
// compatibility inventory in README.md lists which Polars 2.1.0 APIs exist.

import type { ComputeOptions, Operand } from "./core.js";
import {
  type Column,
  type Aggregate,
  type Validity,
  TablePlan,
  validateColumns,
  tableError,
  computeColumns,
  ensureColumn,
  ensureSameLength,
} from "./frame.js";
import * as pandas from "./pandas.js";
import type { ComputableDType } from "./types.js";

/** A named column, as `pl.Series(name, values)`. */
export class Series {
  readonly name: string;
  readonly values: Column;

  readonly validity?: Column;
  constructor(name: string, values: Column, validity?: Column) {
    this.name = name;
    this.values = ensureColumn(name, values);
    this.validity = validity;
    if (validity)
      validateColumns(new Map([[name, values]]), new Map([[name, validity]]));
  }

  get dtype(): ComputableDType {
    return this.values.dtype;
  }

  len(): number {
    return this.values.shape[0];
  }

  toPandas(): pandas.Series {
    return new pandas.Series(this.values, this.name, undefined, this.validity);
  }
  alias(name: string): Series {
    return new Series(name, this.values, this.validity);
  }
}

/** A column expression, as `pl.col(name)`. Evaluated against a frame. */
export class ColumnExpr {
  readonly #build: (frame: ReadonlyMap<string, Column>) => Column;
  readonly outputName: string;

  private constructor(
    build: (frame: ReadonlyMap<string, Column>) => Column,
    outputName: string,
  ) {
    this.#build = build;
    this.outputName = outputName;
  }

  static col(name: string): ColumnExpr {
    return new ColumnExpr((frame) => {
      const column = frame.get(name);
      if (!column) {
        throw tableError();
      }
      return column;
    }, name);
  }

  #binary(
    op: "add" | "sub" | "mul" | "div" | "mod" | "pow",
    other: ColumnExpr | Operand,
  ): ColumnExpr {
    return new ColumnExpr((frame) => {
      const right = other instanceof ColumnExpr ? other.evaluate(frame) : other;
      return this.evaluate(frame).binaryOp(op, right);
    }, this.outputName);
  }

  add(other: ColumnExpr | Operand) {
    return this.#binary("add", other);
  }
  sub(other: ColumnExpr | Operand) {
    return this.#binary("sub", other);
  }
  mul(other: ColumnExpr | Operand) {
    return this.#binary("mul", other);
  }
  truediv(other: ColumnExpr | Operand) {
    return this.#binary("div", other);
  }
  mod(other: ColumnExpr | Operand) {
    return this.#binary("mod", other);
  }
  pow(other: ColumnExpr | Operand) {
    return this.#binary("pow", other);
  }

  alias(name: string): ColumnExpr {
    return new ColumnExpr((frame) => this.evaluate(frame), name);
  }

  /** @internal */
  evaluate(frame: ReadonlyMap<string, Column>): Column {
    return this.#build(frame);
  }
}

export const col = ColumnExpr.col;

/** An eager frame. */
export class DataFrame {
  async write_parquet(
    context: NMathContext,
    path: string,
    options: ColumnarWriteOptions = {},
  ): Promise<void> {
    return writeColumnar(
      context,
      path,
      "parquet",
      this.columnMap,
      this.validity,
      options,
    );
  }

  async write_ipc(
    context: NMathContext,
    path: string,
    options: ColumnarWriteOptions = {},
  ): Promise<void> {
    return writeColumnar(
      context,
      path,
      "arrow",
      this.columnMap,
      this.validity,
      options,
    );
  }

  readonly #columns: Map<string, Column>;
  readonly validity: Validity;

  constructor(
    columns: readonly Series[] | Map<string, Column>,
    validity?: Validity,
  ) {
    this.#columns =
      columns instanceof Map
        ? new Map(columns)
        : new Map(columns.map((s) => [s.name, s.values]));
    ensureSameLength(this.#columns);
    if (
      !(columns instanceof Map) &&
      new Set(columns.map((s) => s.name)).size !== columns.length
    )
      throw tableError();
    this.validity = new Map(
      validity ??
        (columns instanceof Map
          ? []
          : columns
              .filter((s) => s.validity)
              .map((s) => [s.name, s.validity!])),
    );
    if (this.#columns.size) validateColumns(this.#columns, this.validity);
  }

  writeCsv(
    context: NMathContext,
    path: string,
    options?: TextWriteOptions & { dropIndex?: boolean },
  ) {
    return writeFrame(context, this, path, "csv", options);
  }
  writeJson(
    context: NMathContext,
    path: string,
    options?: TextWriteOptions & { dropIndex?: boolean },
  ) {
    return writeFrame(context, this, path, "json", options);
  }

  get columns(): string[] {
    return [...this.#columns.keys()];
  }

  get height(): number {
    return ensureSameLength(this.#columns);
  }

  get width(): number {
    return this.#columns.size;
  }

  getColumn(name: string): Series {
    const column = this.#columns.get(name);
    if (!column) {
      throw tableError();
    }
    return new Series(name, column, this.validity.get(name));
  }

  toPandas(): pandas.DataFrame {
    return new pandas.DataFrame(new Map(this.#columns), {
      validity: this.validity,
    });
  }
  lazy(): LazyFrame {
    return new LazyFrame(this.#columns, [], this.validity);
  }

  /** @internal */
  get columnMap(): ReadonlyMap<string, Column> {
    return this.#columns;
  }
}

type Step =
  | { kind: "select"; exprs: ColumnExpr[] }
  | { kind: "with_columns"; exprs: ColumnExpr[] };

/** A lazy query. `collect()` computes every output column on the CPU. */
export class LazyFrame {
  readonly #source: ReadonlyMap<string, Column>;
  readonly #steps: readonly Step[];
  readonly #validity: Validity;
  readonly #table?: TablePlan;

  /** @internal */
  constructor(
    source: ReadonlyMap<string, Column>,
    steps: readonly Step[],
    validity: Validity = new Map(),
    table?: TablePlan,
  ) {
    this.#source = source;
    this.#steps = steps;
    this.#validity = validity;
    this.#table = table;
  }

  select(...exprs: (ColumnExpr | string)[]): LazyFrame {
    if (this.#table || this.#validity.size) {
      if (exprs.some((e) => typeof e !== "string"))
        throw tableError(
          "Computed projections after row operations are pending. Select columns by name.",
        );
      return this.#next(this.#native().select(exprs as string[]));
    }
    return new LazyFrame(
      this.#source,
      [
        ...this.#steps,
        {
          kind: "select",
          exprs: exprs.map((e) => (typeof e === "string" ? col(e) : e)),
        },
      ],
      this.#validity,
    );
  }

  withColumns(...exprs: ColumnExpr[]): LazyFrame {
    if (this.#table || this.#validity.size)
      throw tableError(
        "Computed projections after nullable or row operations are pending. Apply supported projections first.",
      );
    return new LazyFrame(
      this.#source,
      [...this.#steps, { kind: "with_columns", exprs }],
      this.#validity,
    );
  }

  #plan(): Map<string, Column> {
    let frame = new Map(this.#source);
    for (const step of this.#steps) {
      const outputs = step.exprs.map(
        (e) => [e.outputName, e.evaluate(frame)] as const,
      );
      frame =
        step.kind === "select"
          ? new Map(outputs)
          : new Map([...frame, ...outputs]);
    }
    return frame;
  }

  #native(): TablePlan {
    if (this.#table) return this.#table;
    if (this.#validity.size && this.#steps.length)
      throw tableError(
        "Nullable expression projections are pending. Use named selection after a row operation.",
      );
    return TablePlan.source(this.#plan(), this.#validity);
  }
  #next(plan: TablePlan): LazyFrame {
    return new LazyFrame(this.#source, [], this.#validity, plan);
  }
  filter(column: string): LazyFrame {
    return this.#next(this.#native().filter(column));
  }
  sort(
    column: string,
    options: { descending?: boolean; nullsLast?: boolean } = {},
  ): LazyFrame {
    return this.#next(
      this.#native().sort(
        column,
        options.descending ?? false,
        options.nullsLast ?? false,
      ),
    );
  }
  join(
    other: LazyFrame,
    key: string,
    options: {
      how?: "inner" | "left";
      rightKey?: string;
      nullsEqual?: boolean;
      suffix?: string;
    } = {},
  ): LazyFrame {
    return this.#next(this.#native().join(other.#native(), key, options));
  }
  groupBy(key: string, value: string, op: Aggregate): LazyFrame {
    return this.#next(this.#native().group(key, value, op));
  }
  async collect(options: ComputeOptions = {}): Promise<DataFrame> {
    if (!this.#source.size && !this.#table) return new DataFrame(new Map());
    const out = await this.#native().compute(options);
    return new DataFrame(out.columns, out.validity);
  }
}

export function read_csv(
  context: NMathContext,
  path: string,
  options?: TextReadOptions,
) {
  return readPolars(context, path, "csv", options);
}
export function read_json(
  context: NMathContext,
  path: string,
  options?: TextReadOptions,
) {
  return readPolars(context, path, "json", options);
}

export async function read_parquet(
  context: NMathContext,
  path: string,
  options: ColumnarReadOptions = {},
): Promise<DataFrame> {
  const result = await readColumnar(context, path, "parquet", options);
  return new DataFrame(new Map(result.columns), result.validity);
}

export async function read_ipc(
  context: NMathContext,
  path: string,
  options: ColumnarReadOptions = {},
): Promise<DataFrame> {
  const result = await readColumnar(context, path, "arrow", options);
  return new DataFrame(new Map(result.columns), result.validity);
}
