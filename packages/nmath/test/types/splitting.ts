// SPDX-License-Identifier: Apache-2.0
import { Complex, createContext, type Expr } from "../../src/index.js";
import {
  array_split,
  diff,
  dsplit,
  hsplit,
  repeat,
  roll,
  split,
  tile,
  vsplit,
  type TileShape,
} from "../../src/numpy.js";

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;
declare function exact<T extends true>(): void;

const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});
const v = c.zeros([6], "int16");
const m = c.zeros([4, 6], "float32");
const t = c.zeros([2, 3, 4], "complex64");
const s = c.zeros([], "bool");
const text = c.zeros([4], "str");
const when = c.zeros([3], "datetime64");
declare const n: number;
declare const axes: number[];
const unknownRank = c.array([1, 2], { dtype: "uint8" }).reshape(axes);

const halves: readonly Expr<"int16", [number]>[] = split(v, 2);
const columns: readonly Expr<"float32", [4, number]>[] = split(m, [1, 3], 1);
const lastAxis: readonly Expr<"complex64", [2, 3, number]>[] = array_split(
  t,
  3,
  -1,
);
const dynamicAxis: readonly Expr<"float32", [number, number]>[] = split(
  m,
  2,
  n,
);
const anyRank: readonly Expr<"uint8", number[]>[] = split(unknownRank, 2);
const horizontal: readonly Expr<"float32", [4, number]>[] = hsplit(m, 2);
const horizontalVector: readonly Expr<"int16", [number]>[] = hsplit(v, 3);
const vertical: readonly Expr<"float32", [number, 6]>[] = vsplit(m, 2);
const depth: readonly Expr<"complex64", [2, 3, number]>[] = dsplit(t, 2);
const strings: readonly Expr<"str", [number]>[] = split(text, 2);

const flatRepeat: Expr<"int16", [number]> = repeat(v, 2);
const rowRepeat: Expr<"float32", [number, 6]> = repeat(m, [1, 2, 0, 1], 0);
const columnRepeat: Expr<"float32", [4, number]> = repeat(m, 3, -1);
const scalarRepeat: Expr<"bool", [number]> = repeat(s, 4);
const scalarAxisRepeat: Expr<"bool", [number]> = repeat(s, 4, 0);
const scalarLastRepeat: Expr<"bool", [number]> = repeat(s, [2], -1);

const tiled: Expr<"int16", [number]> = tile(v, 2);
const tiledUp: Expr<"int16", [number, number]> = tile(v, [2, 2]);
const tiledKeep: Expr<"complex64", [number, number, number]> = tile(t, [2, 1]);
const tiledScalar: Expr<"bool", [number, number]> = tile(s, [2, 3]);
exact<Equal<TileShape<[2], [1, 1, 1]>, [number, number, number]>>();
exact<Equal<TileShape<[2, 3], number[]>, number[]>>();

const rolled: Expr<"float32", [4, 6]> = roll(m, 1);
const rolledAxis: Expr<"float32", [4, 6]> = roll(m, [1, -2], [0, 1]);
const rolledText: Expr<"str", [4]> = roll(text, -1, 0);

const widened: Expr<"float64", [number]> = diff(v, 1, -1, { prepend: 0 });
const bigints: Expr<"int64", [number]> = diff(v, 1, -1, { append: 0n });
const kept: Expr<"int16", [number]> = diff(v, 1, -1, { prepend: true });
const complexEdge: Expr<"complex128", [number, number]> = diff(m, 1, 0, {
  prepend: new Complex(0, 1),
});
const arrayEdge: Expr<"float32", [number, number]> = diff(m, 1, 1, {
  prepend: c.zeros([4, 1], "int8"),
  append: c.zeros([4, 2], "float32"),
});
const dates: Expr<"timedelta64", [number]> = diff(when, 1, -1, {
  prepend: c.zeros([1], "datetime64"),
});

// @ts-expect-error the split axis is out of range
split(m, 2, 2);
// @ts-expect-error hsplit rejects 0-d input
hsplit(s, 1);
// @ts-expect-error vsplit requires at least two dimensions
vsplit(v, 2);
// @ts-expect-error dsplit requires at least three dimensions
dsplit(m, 2);
// @ts-expect-error the repeat axis is out of range
repeat(m, 2, -3);
// @ts-expect-error a 0-d input repeats as 1-D, so axis 1 is out of range
repeat(s, 2, 1);
// @ts-expect-error the roll axis is out of range
roll(m, 1, [0, 2]);
// @ts-expect-error a number edge cannot join datetimes
diff(when, 1, -1, { prepend: 0 });
// @ts-expect-error string edges are not accepted
diff(v, 1, -1, { append: "a" });

void [
  halves,
  columns,
  lastAxis,
  dynamicAxis,
  anyRank,
  horizontal,
  horizontalVector,
  vertical,
  depth,
  strings,
  flatRepeat,
  rowRepeat,
  columnRepeat,
  scalarRepeat,
  scalarAxisRepeat,
  scalarLastRepeat,
  tiled,
  tiledUp,
  tiledKeep,
  tiledScalar,
  rolled,
  rolledAxis,
  rolledText,
  widened,
  bigints,
  kept,
  complexEdge,
  arrayEdge,
  dates,
];
