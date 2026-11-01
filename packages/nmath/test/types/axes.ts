// SPDX-License-Identifier: Apache-2.0
import { createContext, type Expr } from "../../src/index.js";
import {
  expand_dims,
  flip,
  fliplr,
  flipud,
  moveaxis,
  ravel,
  squeeze,
  swapaxes,
} from "../../src/numpy.js";

const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});
const a = c.zeros([1, 3, 1, 4], "complex64");
const b = c.zeros([2, 3, 4], "int64");
const v = c.zeros([5], "float32");
const s = c.zeros([], "bool");
declare const n: number;
declare const axes: number[];
const dynamic = c.zeros([n, 3], "float64");
const unknownRank = c.array([1, 2], { dtype: "uint8" }).reshape(axes);

const squeezed: Expr<"complex64", [3, 4]> = squeeze(a);
const squeezed0: Expr<"complex64", [3, 1, 4]> = squeeze(a, 0);
const squeezedNeg: Expr<"complex64", [1, 3, 4]> = squeeze(a, -2);
const squeezedList: Expr<"complex64", [3, 4]> = squeeze(a, [0, -2]);
const squeezedNone: Expr<"complex64", [1, 3, 1, 4]> = squeeze(a, []);
const squeezedScalar: Expr<"bool", []> = squeeze(s, -1);
const squeezedDynamic: Expr<"float64", number[]> = squeeze(dynamic);
const squeezedDynamicAxis: Expr<"complex64", number[]> = squeeze(a, axes);
const squeezedUnknownLength: Expr<"float64", [3]> = squeeze(dynamic, 0);

const expanded: Expr<"int64", [1, 2, 3, 4]> = expand_dims(b, 0);
const expandedEnd: Expr<"int64", [2, 3, 4, 1]> = expand_dims(b, -1);
const expandedList: Expr<"int64", [1, 2, 1, 3, 4, 1]> = expand_dims(
  b,
  [0, 2, -1],
);
const expandedScalar: Expr<"bool", [1, 1]> = expand_dims(s, [0, 1]);
const expandedDynamic: Expr<"int64", number[]> = expand_dims(b, n);

const swapped: Expr<"int64", [4, 3, 2]> = swapaxes(b, 0, -1);
const swappedSame: Expr<"int64", [2, 3, 4]> = swapaxes(b, 1, 1);
const swappedPartial: Expr<"float64", [3, number]> = swapaxes(dynamic, 0, 1);
const swappedDynamic: Expr<"int64", [number, number, number]> = swapaxes(
  b,
  n,
  0,
);

const moved: Expr<"int64", [3, 4, 2]> = moveaxis(b, 0, -1);
const movedBack: Expr<"int64", [4, 2, 3]> = moveaxis(b, -1, 0);
const movedList: Expr<"int64", [4, 3, 2]> = moveaxis(b, [0, 1], [-1, -2]);
const movedMany: Expr<"int64", [3, 4, 2]> = moveaxis(b, [0, 1, 2], [2, 0, 1]);
const movedDynamic: Expr<"int64", [number, number, number]> = moveaxis(
  b,
  axes,
  [0],
);
const movedUnknownRank: Expr<"uint8", number[]> = moveaxis(unknownRank, 0, 1);

const raveled: Expr<"int64", [number]> = ravel(b);
const raveledScalar: Expr<"bool", [number]> = ravel(s, "C");

const flipped: Expr<"int64", [2, 3, 4]> = flip(b);
const flippedAxis: Expr<"int64", [2, 3, 4]> = flip(b, [0, -1]);
const flippedDynamic: Expr<"float64", [number, 3]> = flip(dynamic, n);
const flippedLr: Expr<"int64", [2, 3, 4]> = fliplr(b);
const flippedUd: Expr<"float32", [5]> = flipud(v);
const flippedUnknown: Expr<"uint8", number[]> = fliplr(unknownRank);

// @ts-expect-error a literal axis of length 3 cannot be squeezed
squeeze(a, 1);
// @ts-expect-error squeeze axes must be in range
squeeze(a, 4);
// @ts-expect-error squeeze axes cannot repeat
squeeze(a, [0, -4]);
// @ts-expect-error expand_dims axes normalize against the output rank
expand_dims(b, 4);
// @ts-expect-error expand_dims axes cannot repeat after normalization
expand_dims(b, [0, -5]);
// @ts-expect-error swapaxes axes must be in range
swapaxes(b, 0, 3);
// @ts-expect-error moveaxis source and destination lengths must match
moveaxis(b, [0, 1], [2]);
// @ts-expect-error moveaxis destination axes cannot repeat
moveaxis(b, [0, 1], [2, -1]);
// @ts-expect-error flip axes must be integers
flip(b, 0.5);
// @ts-expect-error flip axes cannot repeat
flip(b, [1, -2]);
// @ts-expect-error fliplr requires at least two dimensions
fliplr(v);
// @ts-expect-error flipud requires at least one dimension
flipud(s);
// @ts-expect-error ravel supports C order only
ravel(b, "F");

void [
  squeezed,
  squeezed0,
  squeezedNeg,
  squeezedList,
  squeezedNone,
  squeezedScalar,
  squeezedDynamic,
  squeezedDynamicAxis,
  squeezedUnknownLength,
  expanded,
  expandedEnd,
  expandedList,
  expandedScalar,
  expandedDynamic,
  swapped,
  swappedSame,
  swappedPartial,
  swappedDynamic,
  moved,
  movedBack,
  movedList,
  movedMany,
  movedDynamic,
  movedUnknownRank,
  raveled,
  raveledScalar,
  flipped,
  flippedAxis,
  flippedDynamic,
  flippedLr,
  flippedUd,
  flippedUnknown,
];
