// SPDX-License-Identifier: Apache-2.0
import { createContext, type Expr } from "../../src/index.js";
import {
  append,
  atleast_1d,
  atleast_2d,
  atleast_3d,
  column_stack,
  concatenate,
  dstack,
  hstack,
  stack,
  vstack,
  type ConcatenateShape,
  type JoinDType,
  type StackShape,
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
const m = c.zeros([2, 3], "int32");
const f = c.zeros([4, 3], "float32");
const v = c.zeros([3], "int8");
const s = c.zeros([], "bool");
const t = c.zeros([2, 3, 4], "complex64");
const text = c.zeros([2], "str");
const when = c.zeros([2], "datetime64");
declare const n: number;
declare const axes: number[];
const dynamic = c.zeros([n, 3], "float64");
const unknownRank = c.array([1, 2], { dtype: "uint8" }).reshape(axes);
declare const list: Expr<"float32", [2, 3]>[];

const rows: Expr<"float64", [number, 3]> = concatenate([m, f]);
const cols: Expr<"int32", [2, number]> = concatenate([m, m], 1);
const last: Expr<"int32", [2, number]> = concatenate([m, m], -1);
const deep: Expr<"complex64", [2, 3, number]> = concatenate([t, t], 2);
const flat: Expr<"float64", [number]> = concatenate([m, f, v], null);
const dynamicAxis: Expr<"int32", [number, number]> = concatenate([m, m], n);
const fromList: Expr<"float32", [number, 3]> = concatenate(list);
const unknown: Expr<"uint8", number[]> = concatenate([unknownRank]);
const strings: Expr<"str", [number]> = concatenate([text, text]);
const dates: Expr<"datetime64", [number]> = concatenate([when, when]);
exact<Equal<ConcatenateShape<[typeof m], 1>, [2, number]>>();
exact<Equal<ConcatenateShape<[typeof m], 2>, never>>();
exact<Equal<ConcatenateShape<[typeof m], -3>, never>>();

// Promotion follows NumPy's main-dtype sequence, not a pairwise fold.
const i8 = c.zeros([2], "int8");
const u8 = c.zeros([2], "uint8");
const i16 = c.zeros([2], "int16");
const u16 = c.zeros([2], "uint16");
const half = c.zeros([2], "float16");
declare const either: Expr<"int8" | "float16", [2]>;
exact<Equal<JoinDType<[typeof i8, typeof u8, typeof half]>, "float16">>();
exact<Equal<JoinDType<[typeof half, typeof u8, typeof i8]>, "float16">>();
exact<Equal<JoinDType<[typeof i16, typeof u16, typeof half]>, "float32">>();
exact<Equal<JoinDType<[typeof either, typeof u8]>, "int16" | "float16">>();
exact<Equal<JoinDType<[typeof when, typeof i8]>, never>>();
const mainHalf: Expr<"float16", [number]> = concatenate([i8, u8, half]);

const stacked: Expr<"int32", [2, 2, 3]> = stack([m, m]);
const stackedLast: Expr<"int32", [2, 3, 3]> = stack([m, m, m], -1);
const stackedMiddle: Expr<"float64", [2, 2, 3]> = stack([m, dynamic], 1);
const stackedDynamic: Expr<"int8", [number, number]> = stack([v, v], n);
const stackedList: Expr<"float32", [number, 2, 3]> = stack(list);
const stackedScalars: Expr<"bool", [3]> = stack([s, s, s]);
exact<Equal<StackShape<[typeof v], 2>, never>>();
declare const frontOrBack: 0 | 1;
declare const backOrFront: -1 | -2;
exact<Equal<StackShape<[typeof v, typeof v], 0 | 1>, [2, 3] | [3, 2]>>();
exact<Equal<StackShape<[typeof v, typeof v], -1 | -2>, [3, 2] | [2, 3]>>();
exact<Equal<StackShape<[typeof v], 0 | 2>, [1, 3]>>();
const stackedEither: Expr<"int8", [2, 3] | [3, 2]> = stack([v, v], frontOrBack);
const stackedEitherBack: Expr<"int8", [2, 3] | [3, 2]> = stack(
  [v, v],
  backOrFront,
);

const vertical: Expr<"int32", [number, 3]> = vstack([v, m]);
const verticalScalars: Expr<"bool", [number, 1]> = vstack([s, s]);
const horizontal: Expr<"int8", [number]> = hstack([v, v]);
const horizontalMatrix: Expr<"float32", [4, number]> = hstack([f, f]);
const depth: Expr<"int32", [2, 3, number]> = dstack([m, m]);
const depthVector: Expr<"int8", [1, 3, number]> = dstack([v, v]);
const columns: Expr<"int32", [3, number]> = column_stack([v, m]);
const scalarColumns: Expr<"bool", [1, number]> = column_stack([s, s]);

const appended: Expr<"float64", [number]> = append(m, f);
const appendedRows: Expr<"float64", [number, 3]> = append(m, f, 0);

const one: Expr<"bool", [1]> = atleast_1d(s);
const kept: Expr<"int32", [2, 3]> = atleast_1d(m);
const row: Expr<"int8", [1, 3]> = atleast_2d(v);
const both = atleast_2d(s, v);
exact<
  Equal<typeof both, readonly [Expr<"bool", [1, 1]>, Expr<"int8", [1, 3]>]>
>();
const cube: Expr<"int8", [1, 3, 1]> = atleast_3d(v);
const slab: Expr<"int32", [2, 3, 1]> = atleast_3d(m);
const solid: Expr<"complex64", [2, 3, 4]> = atleast_3d(t);
const dynamicCube: Expr<"uint8", number[]> = atleast_3d(unknownRank);

// @ts-expect-error strings and numbers cannot be joined
concatenate([text, v]);
// @ts-expect-error datetimes and timedeltas cannot be joined
concatenate([when, c.zeros([2], "timedelta64")]);
// @ts-expect-error zero-dimensional arrays cannot be concatenated
const scalarJoin: Expr<"bool", number[]> = concatenate([s, s]);
// @ts-expect-error the concatenation axis is out of range
const outOfRange: Expr<"int32", number[]> = concatenate([m, m], 2);
// @ts-expect-error the stack axis is out of range
const stackOutOfRange: Expr<"int8", number[]> = stack([v, v], -3);
// @ts-expect-error vstack rejects strings with numbers
vstack([text, v]);
// @ts-expect-error append rejects strings with numbers
append(text, v);
// @ts-expect-error operands must be NMath values
concatenate([m, [1, 2, 3]]);

void [
  stackedEither,
  stackedEitherBack,
  mainHalf,
  rows,
  cols,
  last,
  deep,
  flat,
  dynamicAxis,
  fromList,
  unknown,
  strings,
  dates,
  stacked,
  stackedLast,
  stackedMiddle,
  stackedDynamic,
  stackedList,
  stackedScalars,
  vertical,
  verticalScalars,
  horizontal,
  horizontalMatrix,
  depth,
  depthVector,
  columns,
  scalarColumns,
  appended,
  appendedRows,
  one,
  kept,
  row,
  cube,
  slab,
  solid,
  dynamicCube,
  scalarJoin,
  outOfRange,
  stackOutOfRange,
];
