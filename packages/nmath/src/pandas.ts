import type { NMathContext } from "./core.js";
import {
  readPandas,
  writeFrame,
  type TextReadOptions,
  type TextWriteOptions,
} from "./io-text-frames.js";
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
  indices,
} from "./core.js";
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
import * as polars from "./polars.js";
import type { ComputableDType, Shape } from "./types.js";

export type Label = string | number;
function unwrap(value: Series | Operand, source: Series): Operand {
  if (source.validity)
    throw tableError(
      "Nullable arithmetic is pending. Use table operations that preserve masks.",
    );
  if (!(value instanceof Series)) return value;
  if (source.validity || value.validity)
    throw tableError(
      "Nullable arithmetic is pending. Use table operations that preserve masks.",
    );
  if (
    source.index.length === value.index.length &&
    source.index.every((label, i) => label === value.index[i])
  )
    return value.values;
  if (
    source.index.length !== value.index.length ||
    new Set(value.index).size !== value.index.length
  )
    throw tableError(
      "This arithmetic requires unique matching labels. Align labels explicitly.",
    );
  const positions = source.index.map((label) => value.index.indexOf(label));
  if (positions.some((i) => i < 0))
    throw tableError(
      "Outer label alignment is pending. Supply matching labels.",
    );
  return value.values.get(indices(positions));
}

/** A named numeric column with explicit labels. Matching unique labels align before arithmetic. */
export class Series {
  readonly name: string;
  readonly values: Column;

  readonly index: readonly Label[];
  readonly validity?: Column;

  constructor(
    values: Column,
    name = "",
    index?: readonly Label[],
    validity?: Column,
  ) {
    this.values = ensureColumn(name, values);
    this.name = name;
    this.index = Object.freeze(
      index ? [...index] : Array.from({ length: values.shape[0] }, (_, i) => i),
    );
    this.validity = validity;
    if (this.index.length !== values.shape[0]) throw tableError();
    if (validity)
      validateColumns(new Map([[name, values]]), new Map([[name, validity]]));
  }

  get dtype(): ComputableDType {
    return this.values.dtype;
  }

  get length(): number {
    return this.values.shape[0];
  }

  add(other: Series | Operand): Series {
    return new Series(
      this.values.binaryOp("add", unwrap(other, this)),
      this.name,
      this.index,
    );
  }
  sub(other: Series | Operand): Series {
    return new Series(
      this.values.binaryOp("sub", unwrap(other, this)),
      this.name,
      this.index,
    );
  }
  mul(other: Series | Operand): Series {
    return new Series(
      this.values.binaryOp("mul", unwrap(other, this)),
      this.name,
      this.index,
    );
  }
  div(other: Series | Operand): Series {
    return new Series(
      this.values.binaryOp("div", unwrap(other, this)),
      this.name,
      this.index,
    );
  }
  mod(other: Series | Operand): Series {
    return new Series(
      this.values.binaryOp("mod", unwrap(other, this)),
      this.name,
      this.index,
    );
  }
  pow(other: Series | Operand): Series {
    return new Series(
      this.values.binaryOp("pow", unwrap(other, this)),
      this.name,
      this.index,
    );
  }

  sum() {
    if (this.validity)
      throw tableError(
        "Nullable Series reductions are pending. Use groupBy with a validity mask.",
      );
    return this.values.sum();
  }
  mean() {
    if (this.validity)
      throw tableError(
        "Nullable Series reductions are pending. Use groupBy with a validity mask.",
      );
    return this.values.mean();
  }
  min() {
    if (this.validity)
      throw tableError(
        "Nullable Series reductions are pending. Use groupBy with a validity mask.",
      );
    return this.values.min();
  }
  max() {
    if (this.validity)
      throw tableError(
        "Nullable Series reductions are pending. Use groupBy with a validity mask.",
      );
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
    if (this.validity && typeof selector === "number")
      throw tableError(
        "A scalar expression cannot preserve a null mask. Select a one-row Series.",
      );
    if (typeof selector === "number") {
      // Columns are 1-D, so an integer selector drops the only axis.
      return (this.values as Lazy<ComputableDType, [number]>).get(selector);
    }
    const values = this.values.get(selector);
    let selected: number[];
    if (selector.kind === "indices")
      selected = selector.values.map((i) => (i < 0 ? i + this.length : i));
    else if (selector.kind === "mask")
      selected = selector.values.flatMap((v, i) => (v ? [i] : []));
    else {
      const step = selector.step ?? 1;
      const bound = (v: number | undefined, fallback: number) =>
        v === undefined
          ? fallback
          : Math.max(
              step < 0 ? -1 : 0,
              Math.min(
                step < 0 ? this.length - 1 : this.length,
                v < 0 ? v + this.length : v,
              ),
            );
      const start = bound(selector.start, step < 0 ? this.length - 1 : 0),
        stop = bound(selector.stop, step < 0 ? -1 : this.length);
      selected = [];
      for (let i = start; step < 0 ? i > stop : i < stop; i += step)
        selected.push(i);
    }
    return new Series(
      values,
      this.name,
      selected.map((i) => this.index[i]),
      this.validity?.get(selector),
    );
  }

  loc(label: Label): Series {
    const positions = this.index.flatMap((v, i) => (v === label ? [i] : []));
    if (!positions.length) throw tableError();
    return new Series(
      this.values.get(indices(positions)),
      this.name,
      positions.map((i) => this.index[i]),
      this.validity?.get(indices(positions)),
    );
  }

  rename(name: string): Series {
    return new Series(this.values, name, this.index, this.validity);
  }

  /** Materializes the column. */
  async compute(options: ComputeOptions = {}): Promise<Series> {
    const out = await TablePlan.source(
      new Map([[this.name, this.values]]),
      this.validity ? new Map([[this.name, this.validity]]) : undefined,
    ).compute(options);
    return new Series(
      out.columns.get(this.name)!,
      this.name,
      this.index,
      this.validity ? out.validity.get(this.name) : undefined,
    );
  }

  toPolars(options: { dropIndex?: boolean } = {}): polars.Series {
    if (!options.dropIndex && this.index.some((v, i) => v !== i))
      throw tableError(
        "Polars has no label index. Set dropIndex explicitly to discard labels.",
      );
    return new polars.Series(this.name, this.values, this.validity);
  }
  toColumn(): {
    name: string;
    values: Column;
    validity?: Column;
    index: readonly Label[];
  } {
    return {
      name: this.name,
      values: this.values,
      validity: this.validity,
      index: this.index,
    };
  }

  /** Explicit conversion to an NMath array; the name is not kept on arrays. */
  async toArray(
    options: ComputeOptions = {},
  ): Promise<NMathArray<ComputableDType, Shape>> {
    if (this.validity)
      throw tableError(
        "Array conversion cannot preserve null masks. Use toColumn().",
      );
    return (await this.compute(options)).values as NMathArray<
      ComputableDType,
      Shape
    >;
  }
}

/** Columns keyed by name with a shared length. */
export class DataFrame {
  readonly #columns: Map<string, Column>;
  readonly validity: Validity;
  readonly index: readonly Label[];

  constructor(
    columns: Record<string, Column | Series> | Map<string, Column | Series>,
    options: { index?: readonly Label[]; validity?: Validity } = {},
  ) {
    const entries =
      columns instanceof Map ? [...columns] : Object.entries(columns);
    this.#columns = new Map(
      entries.map(([name, col]) => [
        name,
        ensureColumn(name, col instanceof Series ? col.values : col),
      ]),
    );
    const length = ensureSameLength(this.#columns);
    const series = entries.filter(
      (entry): entry is [string, Series] => entry[1] instanceof Series,
    );
    this.index = Object.freeze([
      ...(options.index ??
        series[0]?.[1].index ??
        Array.from({ length }, (_, i) => i)),
    ]);
    if (
      this.index.length !== length ||
      series.some(([, s]) =>
        s.index.some((label, i) => label !== this.index[i]),
      )
    )
      throw tableError(
        "Frame labels must match. Align Series explicitly before construction.",
      );
    this.validity = new Map(
      options.validity ??
        series.filter(([, s]) => s.validity).map(([n, s]) => [n, s.validity!]),
    );
    if (this.#columns.size) validateColumns(this.#columns, this.validity);
  }

  toCsv(
    context: NMathContext,
    path: string,
    options?: TextWriteOptions & { dropIndex?: boolean },
  ) {
    return writeFrame(context, this, path, "csv", options);
  }
  toJson(
    context: NMathContext,
    path: string,
    options?: TextWriteOptions & { dropIndex?: boolean },
  ) {
    return writeFrame(context, this, path, "json", options);
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
      throw tableError();
    }
    return new Series(column, name, this.index, this.validity.get(name));
  }

  /** `df.assign(**columns)`: returns a new frame. */
  assign(columns: Record<string, Series | Column>): DataFrame {
    const next = new Map<string, Column | Series>(this.#columns);
    for (const [name, col] of Object.entries(columns)) {
      next.set(name, col);
    }
    const masks = new Map(this.validity);
    for (const [name, col] of Object.entries(columns)) {
      masks.delete(name);
      if (col instanceof Series && col.validity) masks.set(name, col.validity);
    }
    return new DataFrame(next, { index: this.index, validity: masks });
  }

  /** `df[[...names]]`. */
  select(names: readonly string[]): DataFrame {
    return new DataFrame(new Map(names.map((n) => [n, this.get(n)])));
  }

  /** Column sums, as pandas `df.sum()`. */
  sum(): Record<string, Expr<ComputableDType, []>> {
    if (this.validity.size)
      throw tableError("Nullable frame reductions are pending. Use groupBy.");
    return Object.fromEntries(
      [...this.#columns].map(([name, col]) => [name, col.sum()]),
    );
  }

  toPolars(options: { dropIndex?: boolean } = {}): polars.DataFrame {
    if (!options.dropIndex && this.index.some((v, i) => v !== i))
      throw tableError(
        "Polars has no label index. Set dropIndex explicitly to discard labels.",
      );
    return new polars.DataFrame(new Map(this.#columns), this.validity);
  }
  get columnMap(): ReadonlyMap<string, Column> {
    return this.#columns;
  }
  async compute(options: ComputeOptions = {}): Promise<DataFrame> {
    if (!this.#columns.size) return new DataFrame({});
    const out = await TablePlan.source(this.#columns, this.validity).compute(
      options,
    );
    return new DataFrame(out.columns, {
      index: this.index,
      validity: out.validity,
    });
  }
  async #rows(
    operation: (plan: TablePlan) => TablePlan,
    options: ComputeOptions,
  ): Promise<DataFrame> {
    const ctx = validateColumns(this.#columns, this.validity);
    let internal = "";
    while (this.#columns.has(internal)) internal += "_";
    const ordinal = ctx.array(
      Array.from({ length: this.index.length }, (_, i) => BigInt(i)),
      { dtype: "uint64" },
    );
    try {
      const out = await operation(
        TablePlan.source(
          new Map([...this.#columns, [internal, ordinal]]),
          this.validity,
        ),
      ).compute(options);
      const rows = out.columns.get(internal)!;
      const labels = (rows.toArray() as bigint[]).map(
        (v) => this.index[Number(v)],
      );
      rows.dispose();
      out.columns.delete(internal);
      (
        out.validity.get(internal) as NMathArray<any, any> | undefined
      )?.dispose();
      out.validity.delete(internal);
      return new DataFrame(out.columns, {
        index: labels,
        validity: out.validity,
      });
    } finally {
      ordinal.dispose();
    }
  }
  filter(maskColumn: string, options: ComputeOptions = {}): Promise<DataFrame> {
    return this.#rows((p) => p.filter(maskColumn), options);
  }
  sortValues(
    by: string,
    options: ComputeOptions & { ascending?: boolean; nullsLast?: boolean } = {},
  ): Promise<DataFrame> {
    return this.#rows(
      (p) => p.sort(by, options.ascending === false, options.nullsLast ?? true),
      options,
    );
  }
  async merge(
    other: DataFrame,
    key: string,
    options: ComputeOptions & {
      how?: "inner" | "left";
      rightKey?: string;
      suffix?: string;
    } = {},
  ): Promise<DataFrame> {
    const out = await TablePlan.source(this.#columns, this.validity)
      .join(TablePlan.source(other.#columns, other.validity), key, {
        ...options,
        nullsEqual: true,
      })
      .compute(options);
    return new DataFrame(out.columns, { validity: out.validity });
  }
  async groupBy(
    key: string,
    value: string,
    op: Aggregate,
    options: ComputeOptions & { dropna?: boolean; sort?: boolean } = {},
  ): Promise<DataFrame> {
    let plan = TablePlan.source(this.#columns, this.validity).group(
      key,
      value,
      op,
      options.dropna ?? true,
      true,
    );
    if (options.sort !== false) plan = plan.sort(key);
    const out = await plan.compute(options);
    return new DataFrame(out.columns, { validity: out.validity });
  }
}

export function read_csv(
  context: NMathContext,
  path: string,
  options?: TextReadOptions,
) {
  return readPandas(context, path, "csv", options);
}
export function read_json(
  context: NMathContext,
  path: string,
  options?: TextReadOptions,
) {
  return readPandas(context, path, "json", options);
}
