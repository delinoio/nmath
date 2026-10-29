// SPDX-License-Identifier: Apache-2.0
// pandas-style entry point (subset). pandas operations always run on the CPU. The
// compatibility inventory in README.md lists which pandas 3.1.0 APIs exist.

import {
  type ComputeOptions,
  type Expr,
  type Indices,
  type Lazy,
  type Mask,
  NMathArray,
  type Operand,
  type Slice,
} from "./core.js";
import {
  type Column,
  computeColumns,
  ensureColumn,
  ensureSameLength,
} from "./frame.js";
import type { ComputableDType, Shape } from "./types.js";

function unwrap(value: Series | Operand): Operand {
  return value instanceof Series ? value.values : value;
}

/** A named one-dimensional column. Arithmetic is lazy and aligned by position. */
export class Series {
  readonly name: string;
  readonly values: Column;

  constructor(values: Column, name = "") {
    this.values = ensureColumn(name, values);
    this.name = name;
  }

  get dtype(): ComputableDType {
    return this.values.dtype;
  }

  get length(): number {
    return this.values.shape[0];
  }

  add(other: Series | Operand): Series {
    return new Series(this.values.binaryOp("add", unwrap(other)), this.name);
  }
  sub(other: Series | Operand): Series {
    return new Series(this.values.binaryOp("sub", unwrap(other)), this.name);
  }
  mul(other: Series | Operand): Series {
    return new Series(this.values.binaryOp("mul", unwrap(other)), this.name);
  }
  div(other: Series | Operand): Series {
    return new Series(this.values.binaryOp("div", unwrap(other)), this.name);
  }
  mod(other: Series | Operand): Series {
    return new Series(this.values.binaryOp("mod", unwrap(other)), this.name);
  }
  pow(other: Series | Operand): Series {
    return new Series(this.values.binaryOp("pow", unwrap(other)), this.name);
  }

  sum() {
    return this.values.sum();
  }
  mean() {
    return this.values.mean();
  }
  min() {
    return this.values.min();
  }
  max() {
    return this.values.max();
  }

  /**
   * Positional selection (`Series.iloc`). An integer returns a lazy scalar expression;
   * slices, `indices` and `mask` return a Series.
   */
  iloc(selector: number): Expr<ComputableDType, []>;
  iloc(selector: Slice | Indices | Mask): Series;
  iloc(
    selector: number | Slice | Indices | Mask,
  ): Series | Expr<ComputableDType, []> {
    if (typeof selector === "number") {
      // Columns are 1-D, so an integer selector drops the only axis.
      return (this.values as Lazy<ComputableDType, [number]>).get(selector);
    }
    return new Series(this.values.get(selector), this.name);
  }

  rename(name: string): Series {
    return new Series(this.values, name);
  }

  /** Materializes the column. */
  async compute(options: ComputeOptions = {}): Promise<Series> {
    const [values] = (
      await computeColumns(new Map([[this.name, this.values]]), options)
    ).values();
    return new Series(values, this.name);
  }

  /** Explicit conversion to an NMath array; the name is not kept on arrays. */
  async toArray(
    options: ComputeOptions = {},
  ): Promise<NMathArray<ComputableDType, Shape>> {
    return (await this.compute(options)).values as NMathArray<
      ComputableDType,
      Shape
    >;
  }
}

/** Columns keyed by name with a shared length. */
export class DataFrame {
  readonly #columns: Map<string, Column>;

  constructor(
    columns: Record<string, Column | Series> | Map<string, Column | Series>,
  ) {
    const entries =
      columns instanceof Map ? [...columns] : Object.entries(columns);
    this.#columns = new Map(
      entries.map(([name, col]) => [
        name,
        ensureColumn(name, col instanceof Series ? col.values : col),
      ]),
    );
    ensureSameLength(this.#columns);
  }

  get columns(): string[] {
    return [...this.#columns.keys()];
  }

  get shape(): [number, number] {
    return [ensureSameLength(this.#columns), this.#columns.size];
  }

  get dtypes(): Record<string, ComputableDType> {
    return Object.fromEntries(
      [...this.#columns].map(([name, col]) => [name, col.dtype]),
    );
  }

  /** `df[name]`. */
  get(name: string): Series {
    const column = this.#columns.get(name);
    if (!column) {
      throw new RangeError(`Column '${name}' does not exist.`);
    }
    return new Series(column, name);
  }

  /** `df.assign(**columns)`: returns a new frame. */
  assign(columns: Record<string, Series | Column>): DataFrame {
    const next = new Map<string, Column | Series>(this.#columns);
    for (const [name, col] of Object.entries(columns)) {
      next.set(name, col);
    }
    return new DataFrame(next);
  }

  /** `df[[...names]]`. */
  select(names: readonly string[]): DataFrame {
    return new DataFrame(new Map(names.map((n) => [n, this.get(n)])));
  }

  /** Column sums, as pandas `df.sum()`. */
  sum(): Record<string, Expr<ComputableDType, []>> {
    return Object.fromEntries(
      [...this.#columns].map(([name, col]) => [name, col.sum()]),
    );
  }

  async compute(options: ComputeOptions = {}): Promise<DataFrame> {
    return new DataFrame(await computeColumns(this.#columns, options));
  }
}
