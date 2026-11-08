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

// arange keeps the float64 fallback when an explicit dtype may be undefined.
declare const optionalInt32DType: "int32" | undefined;
declare const optionalUint32DType: "uint32" | undefined;
declare const optionalFloat32DType: "float32" | undefined;
const optionalInt32Range = np.arange(ctx, 0, 2, 1, optionalInt32DType);
const optionalUint32Range = np.arange(ctx, 0, 2, 1, optionalUint32DType);
const optionalFloat32Range = np.arange(ctx, 0, 2, 1, optionalFloat32DType);
assertTrue<Equal<typeof optionalInt32Range.dtype, "int32" | "float64">>();
assertTrue<Equal<typeof optionalUint32Range.dtype, "uint32" | "float64">>();
assertTrue<Equal<typeof optionalFloat32Range.dtype, "float32" | "float64">>();
assertTrue<Equal<typeof optionalInt32Range.shape, [number]>>();
expectType<Int32Array | Float64Array>(optionalInt32Range.toTypedArray());
expectType<Uint32Array | Float64Array>(optionalUint32Range.toTypedArray());
expectType<Float32Array | Float64Array>(optionalFloat32Range.toTypedArray());
// @ts-expect-error the undefined branch exports Float64Array
expectType<Int32Array>(optionalInt32Range.toTypedArray());
// @ts-expect-error the undefined branch exports Float64Array
expectType<Uint32Array>(optionalUint32Range.toTypedArray());
// @ts-expect-error the undefined branch exports Float64Array
expectType<Float32Array>(optionalFloat32Range.toTypedArray());
const defaultRange = np.arange(ctx, 0, 2);
const undefinedRange = np.arange(ctx, 0, 2, 1, undefined);
const int32Range = np.arange(ctx, 0, 2, 1, "int32");
const uint32Range = np.arange(ctx, 0, 2, 1, "uint32");
const float32Range = np.arange(ctx, 0, 2, 1, "float32");
const float64Range = np.arange(ctx, 0, 2, 1, "float64");
assertTrue<Equal<typeof defaultRange.dtype, "float64">>();
assertTrue<Equal<typeof undefinedRange.dtype, "float64">>();
assertTrue<Equal<typeof int32Range.dtype, "int32">>();
assertTrue<Equal<typeof uint32Range.dtype, "uint32">>();
assertTrue<Equal<typeof float32Range.dtype, "float32">>();
assertTrue<Equal<typeof float64Range.dtype, "float64">>();
// @ts-expect-error bool is not supported by arange
np.arange(ctx, 0, 2, 1, "bool");
// @ts-expect-error int64 is not supported by arange
np.arange(ctx, 0, 2, 1, "int64");

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
const numpySum = np.sum(v);
const numpyMean = np.mean(v);
const numpyMin = np.min(v);
const numpyMax = np.max(v);
assertTrue<Equal<typeof numpySum, Expr<"int64", []>>>();
assertTrue<Equal<typeof numpyMean, Expr<"float64", []>>>();
assertTrue<Equal<typeof numpyMin, Expr<"int32", []>>>();
assertTrue<Equal<typeof numpyMax, Expr<"int32", []>>>();

// Literal axes keep the remaining dimensions and each operation's dtype.
const numpyRows = ctx.zeros([2, 3] as const, "float32");
assertTrue<Equal<typeof numpyRows, NMathArray<"float32", readonly [2, 3]>>>();
const sumAxis0 = np.sum(numpyRows, 0);
const meanAxis1 = np.mean(numpyRows, 1);
const minAxis0 = np.min(numpyRows, 0);
const maxAxis1 = np.max(numpyRows, 1);
assertTrue<Equal<typeof sumAxis0, Expr<"float32", [3]>>>();
assertTrue<Equal<typeof meanAxis1, Expr<"float32", [2]>>>();
assertTrue<Equal<typeof minAxis0, Expr<"float32", [3]>>>();
assertTrue<Equal<typeof maxAxis1, Expr<"float32", [2]>>>();
declare const dynamicAxis: number;
const dynamicSum = np.sum(numpyRows, dynamicAxis);
assertTrue<Equal<typeof dynamicSum, Expr<"float32", number[]>>>();

// NumPy unary wrappers retain the dtype and shape of arrays and expressions.
declare const floatUnaryArray: NMathArray<"float32", readonly [2, 3]>;
declare const floatUnaryExpr: Expr<"float32", readonly [2, 3]>;
declare const complexUnaryArray: NMathArray<"complex64", []>;
declare const complexUnaryExpr: Expr<"complex64", []>;
declare const durationUnaryArray: NMathArray<"timedelta64", number[]>;
declare const durationUnaryExpr: Expr<"timedelta64", number[]>;
assertTrue<
  Equal<
    ReturnType<typeof floatUnaryArray.neg>,
    Expr<"float32", readonly [2, 3]>
  >
>();
assertTrue<
  Equal<
    ReturnType<typeof floatUnaryArray.pos>,
    Expr<"float32", readonly [2, 3]>
  >
>();
const unaryFloatNegativeArray = np.negative(floatUnaryArray);
const unaryFloatPositiveArray = np.positive(floatUnaryArray);
const unaryFloatNegativeExpr = np.negative(floatUnaryExpr);
const unaryFloatPositiveExpr = np.positive(floatUnaryExpr);
assertTrue<
  Equal<typeof unaryFloatNegativeArray, Expr<"float32", readonly [2, 3]>>
>();
assertTrue<
  Equal<typeof unaryFloatPositiveArray, Expr<"float32", readonly [2, 3]>>
>();
assertTrue<
  Equal<typeof unaryFloatNegativeExpr, Expr<"float32", readonly [2, 3]>>
>();
assertTrue<
  Equal<typeof unaryFloatPositiveExpr, Expr<"float32", readonly [2, 3]>>
>();
const unaryComplexNegativeArray = np.negative(complexUnaryArray);
const unaryComplexPositiveArray = np.positive(complexUnaryArray);
const unaryComplexNegativeExpr = np.negative(complexUnaryExpr);
const unaryComplexPositiveExpr = np.positive(complexUnaryExpr);
assertTrue<Equal<typeof unaryComplexNegativeArray, Expr<"complex64", []>>>();
assertTrue<Equal<typeof unaryComplexPositiveArray, Expr<"complex64", []>>>();
assertTrue<Equal<typeof unaryComplexNegativeExpr, Expr<"complex64", []>>>();
assertTrue<Equal<typeof unaryComplexPositiveExpr, Expr<"complex64", []>>>();
const unaryDurationNegativeArray = np.negative(durationUnaryArray);
const unaryDurationPositiveArray = np.positive(durationUnaryArray);
const unaryDurationNegativeExpr = np.negative(durationUnaryExpr);
const unaryDurationPositiveExpr = np.positive(durationUnaryExpr);
assertTrue<
  Equal<typeof unaryDurationNegativeArray, Expr<"timedelta64", number[]>>
>();
assertTrue<
  Equal<typeof unaryDurationPositiveArray, Expr<"timedelta64", number[]>>
>();
assertTrue<
  Equal<typeof unaryDurationNegativeExpr, Expr<"timedelta64", number[]>>
>();
assertTrue<
  Equal<typeof unaryDurationPositiveExpr, Expr<"timedelta64", number[]>>
>();

declare const boolUnary: NMathArray<"bool", [2]>;
declare const stringUnary: NMathArray<"str", [2]>;
declare const datetimeUnary: NMathArray<"datetime64", [2]>;
// @ts-expect-error NumPy negative rejects bool arrays.
np.negative(boolUnary);
// @ts-expect-error NumPy positive rejects bool arrays.
np.positive(boolUnary);
// @ts-expect-error NumPy negative rejects string arrays.
np.negative(stringUnary);
// @ts-expect-error NumPy positive rejects string arrays.
np.positive(stringUnary);
// @ts-expect-error NumPy negative rejects datetime arrays.
np.negative(datetimeUnary);
// @ts-expect-error NumPy positive rejects datetime arrays.
np.positive(datetimeUnary);
// @ts-expect-error Direct negative rejects bool arrays.
boolUnary.neg();
// @ts-expect-error Direct positive rejects bool arrays.
boolUnary.pos();
// @ts-expect-error Direct negative rejects string arrays.
stringUnary.neg();
// @ts-expect-error Direct positive rejects string arrays.
stringUnary.pos();
// @ts-expect-error Direct negative rejects datetime arrays.
datetimeUnary.neg();
// @ts-expect-error Direct positive rejects datetime arrays.
datetimeUnary.pos();

// Compiler unary helpers retain supported dtype and shape information.
const helperFloatNegative = nts.neg(floatUnaryArray);
const helperFloatPositive = nts.pos(floatUnaryExpr);
const helperIntegerNegative = nts.neg(v);
const helperComplexNegative = nts.neg(complexUnaryExpr);
const helperComplexPositive = nts.pos(complexUnaryArray);
const helperDurationNegative = nts.neg(durationUnaryArray);
const helperDurationPositive = nts.pos(durationUnaryExpr);
assertTrue<
  Equal<typeof helperFloatNegative, Expr<"float32", readonly [2, 3]>>
>();
assertTrue<
  Equal<typeof helperFloatPositive, Expr<"float32", readonly [2, 3]>>
>();
assertTrue<Equal<typeof helperIntegerNegative, Expr<"int32", [3]>>>();
assertTrue<Equal<typeof helperComplexNegative, Expr<"complex64", []>>>();
assertTrue<Equal<typeof helperComplexPositive, Expr<"complex64", []>>>();
assertTrue<
  Equal<typeof helperDurationNegative, Expr<"timedelta64", number[]>>
>();
assertTrue<
  Equal<typeof helperDurationPositive, Expr<"timedelta64", number[]>>
>();
// @ts-expect-error Compiler negative rejects bool arrays.
nts.neg(boolUnary);
// @ts-expect-error Compiler positive rejects bool arrays.
nts.pos(boolUnary);
// @ts-expect-error Compiler negative rejects string arrays.
nts.neg(stringUnary);
// @ts-expect-error Compiler positive rejects string arrays.
nts.pos(stringUnary);
// @ts-expect-error Compiler negative rejects datetime arrays.
nts.neg(datetimeUnary);
// @ts-expect-error Compiler positive rejects datetime arrays.
nts.pos(datetimeUnary);

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
  expectType<bigint>(await numpySum.compute());
  expectType<number>(await numpyMean.compute());
  expectType<number>(await numpyMin.compute());
  expectType<number>(await numpyMax.compute());
  // @ts-expect-error rank-zero sum returns a bigint, not an array
  (await numpySum.compute()).toArray();
  // @ts-expect-error rank-zero mean returns a number, not an array
  (await numpyMean.compute()).toArray();
  // @ts-expect-error rank-zero min returns a number, not an array
  (await numpyMin.compute()).toArray();
  // @ts-expect-error rank-zero max returns a number, not an array
  (await numpyMax.compute()).toArray();
  expectType<Expr<"int64", []>>(np.sum(v, undefined));
  expectType<Expr<"float64", []>>(np.mean(v, undefined));
  expectType<Expr<"int32", []>>(np.min(v, undefined));
  expectType<Expr<"int32", []>>(np.max(v, undefined));
  expectType<Expr<"float32", number[]>>(np.sum(numpyRows, dynamicAxis));
  expectType<Expr<"float32", number[]>>(np.mean(numpyRows, dynamicAxis));
  expectType<Expr<"float32", number[]>>(np.min(numpyRows, dynamicAxis));
  expectType<Expr<"float32", number[]>>(np.max(numpyRows, dynamicAxis));
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
