// SPDX-License-Identifier: Apache-2.0
// Polars-style entry point (subset). Polars operations always run on the CPU. The
// compatibility inventory in README.md lists which Polars 2.1.0 APIs exist.

import type { ComputeOptions, Operand } from "./core.js";
import {
  type Column,
  computeColumns,
  ensureColumn,
  ensureSameLength,
} from "./frame.js";
import type { ComputableDType } from "./types.js";

/** A named column, as `pl.Series(name, values)`. */
export class Series {
  readonly name: string;
  readonly values: Column;

  constructor(name: string, values: Column) {
    this.name = name;
    this.values = ensureColumn(name, values);
  }

  get dtype(): ComputableDType {
    return this.values.dtype;
  }

  len(): number {
    return this.values.shape[0];
  }

  alias(name: string): Series {
    return new Series(name, this.values);
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
        throw new RangeError(`Column '${name}' does not exist.`);
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
  readonly #columns: Map<string, Column>;

  constructor(columns: readonly Series[] | Map<string, Column>) {
    this.#columns =
      columns instanceof Map
        ? new Map(columns)
        : new Map(columns.map((s) => [s.name, s.values]));
    ensureSameLength(this.#columns);
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
      throw new RangeError(`Column '${name}' does not exist.`);
    }
    return new Series(name, column);
  }

  lazy(): LazyFrame {
    return new LazyFrame(this.#columns, []);
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

  /** @internal */
  constructor(source: ReadonlyMap<string, Column>, steps: readonly Step[]) {
    this.#source = source;
    this.#steps = steps;
  }

  select(...exprs: (ColumnExpr | string)[]): LazyFrame {
    return new LazyFrame(this.#source, [
      ...this.#steps,
      {
        kind: "select",
        exprs: exprs.map((e) => (typeof e === "string" ? col(e) : e)),
      },
    ]);
  }

  withColumns(...exprs: ColumnExpr[]): LazyFrame {
    return new LazyFrame(this.#source, [
      ...this.#steps,
      { kind: "with_columns", exprs },
    ]);
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

  async collect(options: ComputeOptions = {}): Promise<DataFrame> {
    return new DataFrame(await computeColumns(this.#plan(), options));
  }
}
