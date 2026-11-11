// SPDX-License-Identifier: Apache-2.0
// Open tuple shape declarations are checked by TypeScript and never executed.
import {
  createContext,
  type Broadcast,
  type ComputeResult,
  type Expr,
  type NMathArray,
  type ReduceShape,
} from "../../src/index.js";
import * as np from "../../src/numpy.js";
import * as nts from "../../src/nts-runtime.js";

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;
declare function exact<T extends true>(): void;

type Prefix = readonly [2, ...number[]];
type Suffix = readonly [...number[], 3];
type Mixed = readonly [2, ...number[], 3];

exact<Equal<Broadcast<Prefix, readonly [3]>, number[]>>();
exact<Equal<Broadcast<readonly [3], Prefix>, number[]>>();
exact<Equal<Broadcast<[2, ...number[]], [3]>, number[]>>();
exact<Equal<Broadcast<[3], [2, ...number[]]>, number[]>>();
exact<Equal<Broadcast<Suffix, readonly [3]>, number[]>>();
exact<Equal<Broadcast<readonly [3], Suffix>, number[]>>();
exact<Equal<Broadcast<Mixed, readonly [3]>, number[]>>();
exact<Equal<Broadcast<readonly [3], Mixed>, number[]>>();
exact<Equal<Broadcast<readonly [2, 3], readonly [3]>, [2, 3]>>();
exact<Equal<Broadcast<readonly [2, number], readonly [3]>, [2, 3]>>();
exact<Equal<Broadcast<readonly [2], readonly [3]>, never>>();

exact<Equal<ReduceShape<Prefix, 1>, number[]>>();
exact<Equal<ReduceShape<Suffix, 1>, number[]>>();
exact<Equal<ReduceShape<Mixed, 1>, number[]>>();
exact<Equal<ReduceShape<Prefix, undefined>, []>>();
exact<Equal<ReduceShape<Suffix, undefined>, []>>();
exact<Equal<ReduceShape<Mixed, undefined>, []>>();
exact<Equal<ReduceShape<Prefix, undefined, true>, number[]>>();
exact<Equal<ReduceShape<readonly [2, 3, 4], 1>, [2, 4]>>();
exact<Equal<ReduceShape<readonly [2, 3, 4], undefined>, []>>();

declare const mutablePrefix: [2, ...number[]];
declare const readonlyPrefix: Prefix;
declare const suffix: Suffix;
declare const mixed: Mixed;
const ctx = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 1,
  cpuMemoryBudget: 1024,
  gpuMemoryBudget: 0,
});
const vector = ctx.zeros([3] as const, "float32");
const mutable = ctx.zeros(mutablePrefix, "float32");
const readonlyArray = ctx.zeros(readonlyPrefix, "float32");

exact<Equal<typeof mutable, NMathArray<"float32", [2, ...number[]]>>>();
exact<Equal<typeof readonlyArray, NMathArray<"float32", Prefix>>>();

const left = mutable.add(vector);
const right = vector.add(mutable);
const readonlyLeft = readonlyArray.add(vector);
const numpyLeft = np.add(readonlyArray, vector);
const helperLeft = nts.add(readonlyArray, vector);
exact<Equal<typeof left, Expr<"float32", number[]>>>();
exact<Equal<typeof right, Expr<"float32", number[]>>>();
exact<Equal<typeof readonlyLeft, Expr<"float32", number[]>>>();
exact<Equal<typeof numpyLeft, Expr<"float32", number[]>>>();
exact<Equal<typeof helperLeft, Expr<"float32", number[]>>>();
exact<
  Equal<
    ComputeResult<"float32", number[]>,
    number | NMathArray<"float32", number[]>
  >
>();

const reduced = readonlyArray.sum(1);
const allAxes = readonlyArray.sum();
const numpyReduced = np.sum(readonlyArray, 1);
const numpyAllAxes = np.sum(readonlyArray);
const suffixReduced = ctx.zeros(suffix, "float32").sum(1);
const mixedReduced = ctx.zeros(mixed, "float32").sum(1);
exact<Equal<typeof reduced, Expr<"float32", number[]>>>();
exact<Equal<typeof allAxes, Expr<"float32", []>>>();
exact<Equal<typeof numpyReduced, Expr<"float32", number[]>>>();
exact<Equal<typeof numpyAllAxes, Expr<"float32", []>>>();
exact<Equal<typeof suffixReduced, Expr<"float32", number[]>>>();
exact<Equal<typeof mixedReduced, Expr<"float32", number[]>>>();

const fixed = ctx.zeros([2, 3] as const, "float32");
const fixedBroadcast = fixed.add(vector);
const fixedReduction = fixed.sum(1);
exact<Equal<typeof fixedBroadcast, Expr<"float32", [2, 3]>>>();
exact<Equal<typeof fixedReduction, Expr<"float32", [2]>>>();
// @ts-expect-error incompatible fixed shapes still reject broadcasting
ctx.zeros([2] as const, "float32").add(vector);
