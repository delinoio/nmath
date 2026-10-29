// SPDX-License-Identifier: Apache-2.0
// Type fixtures checked by `pnpm typecheck` with TypeScript 7.0.2. Never executed.
import {
  add,
  createContext,
  div,
  indices,
  slice,
  type Expr,
  type NMathArray,
} from "../../src/index.js";
import * as np from "../../src/numpy.js";
import * as nts from "../../src/nts-runtime.js";
import * as pd from "../../src/pandas.js";

declare function expectType<T>(value: T): void;
type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;
declare function assertTrue<T extends true>(): void;

const ctx = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 1,
  cpuMemoryBudget: 1024,
  gpuMemoryBudget: 0,
});
const m = ctx.zeros([2, 3], "float32");
const v = ctx.zeros([3], "int32");
const col = ctx.zeros([2, 1], "float64");

// Dtype promotion follows NumPy.
const mv = m.add(v);
assertTrue<Equal<typeof mv, Expr<"float64", [2, 3]>>>();
const scaled = m.mul(2);
assertTrue<Equal<typeof scaled, Expr<"float32", readonly [2, 3]>>>();
const iw = v.add(1n);
assertTrue<Equal<(typeof iw)["dtype"], "int32">>();
const iwDiv = v.div(2n);
const fw = v.add(1.5);
assertTrue<Equal<(typeof fw)["dtype"], "float64">>();
assertTrue<Equal<(typeof iwDiv)["dtype"], "float64">>();

// Broadcasting.
const b = m.sub(col);
assertTrue<Equal<(typeof b)["shape"], [2, 3]>>();
// @ts-expect-error shapes [2, 3] and [4] do not broadcast
m.add(ctx.zeros([4], "float32"));

// Reductions and matmul.
const total = m.sum();
assertTrue<Equal<(typeof total)["shape"], []>>();
const s0 = m.sum(0);
assertTrue<Equal<(typeof s0)["shape"], [3]>>();
const vtotal = v.sum();
assertTrue<Equal<(typeof vtotal)["dtype"], "int64">>();
const mm = m.matmul(ctx.zeros([3, 5], "float32"));
assertTrue<Equal<(typeof mm)["shape"], [2, 5]>>();
// @ts-expect-error core dimensions 3 and 4 do not match
m.matmul(ctx.zeros([4, 5], "float32"));

// Indexing keeps known information.
const row = m.get(1);
assertTrue<Equal<(typeof row)["shape"], [3]>>();
const sl = m.get(slice(0, 1), indices([0, 2]));
assertTrue<Equal<(typeof sl)["shape"], [number, number]>>();

// Dynamic shapes keep dtype and fall back to runtime validation.
declare const dyn: NMathArray<"float64", number[]>;
const d = dyn.add(m);
assertTrue<Equal<(typeof d)["dtype"], "float64">>();
assertTrue<Equal<(typeof d)["shape"], number[]>>();

// compute() returns JS scalars for rank 0 and bigint for 64-bit integers.
async function results() {
  expectType<number>(await m.sum().compute());
  expectType<bigint>(await v.sum().compute());
  expectType<NMathArray<"float32", readonly [2, 3]>>(await m.neg().compute());
}

// The `.nts` runtime requires narrowing for values that may or may not be NMath.
declare const maybe: number | NMathArray<"float64", [2]>;
// @ts-expect-error a union of NMath and JS values must be narrowed first
nts.add(maybe, 1);
// @ts-expect-error ordinary JavaScript must not call compiler helpers
nts.add(1, 2);
// @ts-expect-error ordinary JavaScript must not call compiler helpers
nts.add("a", 1);
expectType<Expr<any, any>>(nts.add(m, 1));
void results;
async function assignment() {
  expectType<number>(await nts.set(m, [0, 1], 2));
}
void assignment;

// Integer Series.iloc yields a scalar expression; other selectors keep the Series.
declare const series: pd.Series;
const first = series.iloc(0);
assertTrue<Equal<(typeof first)["shape"], []>>();
expectType<pd.Series>(series.iloc(slice(1)));

// Functional arithmetic keeps dtype and shape like the methods, NumPy aliases included.
const fmv = add(m, v);
assertTrue<Equal<typeof fmv, Expr<"float64", [2, 3]>>>();
const fscaled = np.multiply(2, m);
assertTrue<Equal<typeof fscaled, Expr<"float32", readonly [2, 3]>>>();
const fdiv = div(v, 2n);
assertTrue<Equal<(typeof fdiv)["dtype"], "float64">>();
const ftrue = np.trueDivide(v, v);
assertTrue<Equal<(typeof ftrue)["dtype"], "float64">>();
// @ts-expect-error shapes [2] and [3] do not broadcast
add(ctx.zeros([2], "float32"), ctx.zeros([3], "float32"));
// @ts-expect-error shapes [2, 3] and [4] do not broadcast
np.subtract(m, ctx.zeros([4], "float32"));

// Small integer results preserve width except for NumPy promotion and reductions.
const i8 = ctx.zeros([2], "int8");
const i16 = ctx.zeros([2], "int16");
const u8 = ctx.zeros([2], "uint8");
const u16 = ctx.zeros([2], "uint16");
expectType<Int8Array>(i8.toTypedArray());
expectType<Int16Array>(i16.toTypedArray());
expectType<Uint8Array>(u8.toTypedArray());
expectType<Uint16Array>(u16.toTypedArray());
const mixed8 = i8.add(u8);
assertTrue<Equal<typeof mixed8, Expr<"int16", [2]>>>();
const mixed16 = i16.add(u16);
assertTrue<Equal<typeof mixed16, Expr<"int32", [2]>>>();
expectType<Expr<"float32", [2]>>(u16.add(ctx.zeros([2], "float32")));
expectType<Expr<"int8", readonly [2]>>(i8.add(1n));
expectType<Expr<"float64", readonly [2]>>(i8.add(1));
expectType<Expr<"int64", []>>(i8.sum());
expectType<Expr<"uint64", []>>(u16.sum());
expectType<number>(i8.item(0));

const mixedFunctional = add(i8, u8);
assertTrue<Equal<typeof mixedFunctional, Expr<"int16", [2]>>>();

const boolValues = ctx.zeros([2], "bool");
const boolPower = boolValues.pow(boolValues);
const boolRemainder = boolValues.mod(true);
assertTrue<Equal<(typeof boolPower)["dtype"], "int8">>();
assertTrue<Equal<(typeof boolRemainder)["dtype"], "int8">>();
assertTrue<Equal<(typeof boolPower)["shape"], [2]>>();
assertTrue<Equal<(typeof boolRemainder)["shape"], readonly [2]>>();
// @ts-expect-error NumPy rejects unary positive for bool arrays.
boolValues.pos();
// @ts-expect-error NumPy rejects unary negative for bool arrays.
boolValues.neg();
void boolPower;
void boolRemainder;
