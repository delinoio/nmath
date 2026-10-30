// SPDX-License-Identifier: Apache-2.0
import {
  createContext,
  type Expr,
  type TransposeShape,
  type ReshapeShape,
} from "../../src/index.js";
import { transpose, reshape } from "../../src/numpy.js";
const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});
const a = c.zeros([2, 3, 4], "complex64");
const reversed: Expr<"complex64", [4, 3, 2]> = a.transpose();
const permuted: Expr<"complex64", [4, 2, 3]> = transpose(a, [-1, 0, 1]);
const exact: Expr<"complex64", [6, 4]> = reshape(a, [6, -1]);
const direct: Expr<"complex64", [8, 3]> = a.reshape([-1, 3], { copy: true });
const scalar: Expr<"float32", []> = c.zeros([1], "float32").reshape([]);
const rank0: Expr<"float32", []> = scalar.transpose([]);
const zero: Expr<"float32", [0, 2]> = c
  .zeros([0, 3], "float32")
  .reshape([-1, 2]);
const large: Expr<"float32", [128, 128]> = c
  .zeros([256, 64], "float32")
  .reshape([128, -1]);
const largeExact: Expr<"float32", [1024, 1024]> = c
  .zeros([2048, 512], "float32")
  .reshape([1024, -1]);
declare const n: number;
const dynamic = c.zeros([n, 3, 4], "float32");
const partlyKnown: Expr<"float32", [4, number, 3]> = dynamic.transpose([
  2, 0, 1,
]);
const dynamicReshape: Expr<"float32", [number, 4]> = dynamic.reshape([-1, 4]);
const preserveTarget: Expr<"float32", [number, 4]> = dynamic.reshape([n, 4]);
declare const axes: number[];
const dynamicPermutation: Expr<"float32", [number, number, number]> =
  dynamic.transpose(axes);
declare const shape: number[];
const dynamicTarget: Expr<"complex64", number[]> = a.reshape(shape);
// @ts-expect-error known total size differs
a.reshape([5, 5]);
// @ts-expect-error inferred dimension cannot divide the original size
reshape(a, [5, -1]);
// @ts-expect-error only one dimension may be inferred
a.reshape([-1, -1]);
// @ts-expect-error negative dimensions other than -1 are unsupported
a.reshape([-2, 12]);
// @ts-expect-error dimensions must be integers
a.reshape([1.5, 16]);
// @ts-expect-error zero times inferred dimension is ambiguous
c.zeros([0], "float32").reshape([0, -1]);
// @ts-expect-error a nonempty array cannot reshape to an empty one
a.reshape([0, 24]);
// @ts-expect-error a permutation cannot repeat axes
transpose(a, [0, 0, 2]);
// @ts-expect-error axis count must match rank
a.transpose([0, 1]);
// @ts-expect-error axis must be in range
a.transpose([-12, 0, 1]);
// @ts-expect-error a positive out-of-range axis fails too
a.transpose([3, 0, 1]);
// @ts-expect-error axes must be integers
a.transpose([0.5, 1, 2]);
// @ts-expect-error only C traversal is supported
a.reshape([24], { order: "F" });
void [
  reversed,
  permuted,
  exact,
  direct,
  scalar,
  rank0,
  zero,
  large,
  largeExact,
  partlyKnown,
  dynamicReshape,
  preserveTarget,
  dynamicPermutation,
  dynamicTarget,
];
type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;
type Assert<T extends true> = T;
type Arithmetic = Assert<Equal<ReshapeShape<[123, 456], [12, -1]>, [12, 4674]>>;
type Invalid = Assert<Equal<TransposeShape<[2, 3], [-3, 0]>, never>>;
type Empty = Assert<Equal<ReshapeShape<[0], [-1]>, [0]>>;
