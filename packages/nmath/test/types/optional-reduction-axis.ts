// SPDX-License-Identifier: Apache-2.0
// Optional reduction axes are checked by TypeScript and never executed.
import { NMathArray, type Expr, type ReduceShape } from "../../src/index.js";
import * as np from "../../src/numpy.js";

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;
declare function exact<T extends true>(): void;
declare function expectType<T>(value: T): void;

type Rows = readonly [2, 3];
type OptionalAxisShape = [] | [3];
type OptionalAxisResult = Expr<"float32", OptionalAxisShape>;
declare const readonlyRows: NMathArray<"float32", Rows>;
declare const mutableRows: NMathArray<"float32", [2, 3]>;

exact<Equal<ReduceShape<Rows, 0 | undefined>, OptionalAxisShape>>();
exact<Equal<ReduceShape<Rows, 1 | undefined>, [] | [2]>>();
exact<Equal<ReduceShape<Rows, 0 | 1 | undefined>, [] | [2] | [3]>>();
exact<Equal<ReduceShape<Rows, 0 | undefined, true>, [1, 1] | [1, 3]>>();

function optionalAxis(axis: 0 | undefined) {
  const coreSum = readonlyRows.sum(axis);
  const coreMean = readonlyRows.mean(axis);
  const coreMin = readonlyRows.min(axis);
  const coreMax = readonlyRows.max(axis);
  const numpySum = np.sum(readonlyRows, axis);
  const numpyMean = np.mean(readonlyRows, axis);
  const numpyMin = np.min(readonlyRows, axis);
  const numpyMax = np.max(readonlyRows, axis);
  const mutableSum = mutableRows.sum(axis);
  exact<Equal<typeof coreSum, OptionalAxisResult>>();
  exact<Equal<typeof coreMean, OptionalAxisResult>>();
  exact<Equal<typeof coreMin, OptionalAxisResult>>();
  exact<Equal<typeof coreMax, OptionalAxisResult>>();
  exact<Equal<typeof numpySum, OptionalAxisResult>>();
  exact<Equal<typeof numpyMean, OptionalAxisResult>>();
  exact<Equal<typeof numpyMin, OptionalAxisResult>>();
  exact<Equal<typeof numpyMax, OptionalAxisResult>>();
  exact<Equal<typeof mutableSum, OptionalAxisResult>>();

  async function computed() {
    const result = await coreSum.compute();
    const numpyResult = await numpySum.compute();
    expectType<number | NMathArray<"float32", [3]>>(result);
    expectType<number | NMathArray<"float32", [3]>>(numpyResult);
    // @ts-expect-error the all-axis branch returns a scalar
    const arrayOnly: NMathArray<"float32", [3]> = result;
    // @ts-expect-error array methods require narrowing
    result.toTypedArray();
    if (result instanceof NMathArray) {
      expectType<NMathArray<"float32", [3]>>(result);
    } else {
      expectType<number>(result);
    }
    return arrayOnly;
  }
  return computed;
}

declare const axis1: 1 | undefined;
declare const eitherAxis: 0 | 1 | undefined;
const axis1Sum = mutableRows.sum(axis1);
const axis1Mean = np.mean(mutableRows, axis1);
const eitherMin = readonlyRows.min(eitherAxis);
const eitherMax = np.max(readonlyRows, eitherAxis);
exact<Equal<typeof axis1Sum, Expr<"float32", [] | [2]>>>();
exact<Equal<typeof axis1Mean, Expr<"float32", [] | [2]>>>();
exact<Equal<typeof eitherMin, Expr<"float32", [] | [2] | [3]>>>();
exact<Equal<typeof eitherMax, Expr<"float32", [] | [2] | [3]>>>();

declare const vector: NMathArray<"float32", readonly [3]>;
declare const optionalVectorAxis: 0 | undefined;
const vectorSum = vector.sum(optionalVectorAxis);
const vectorMax = np.max(vector, optionalVectorAxis);
exact<Equal<typeof vectorSum, Expr<"float32", []>>>();
exact<Equal<typeof vectorMax, Expr<"float32", []>>>();

const omittedSum = readonlyRows.sum();
const undefinedMean = readonlyRows.mean(undefined);
const omittedMin = np.min(readonlyRows);
const undefinedMax = np.max(readonlyRows, undefined);
const fixedSum = readonlyRows.sum(0);
const fixedMean = readonlyRows.mean(1);
const fixedMin = np.min(readonlyRows, 0);
const fixedMax = np.max(readonlyRows, 1);
exact<Equal<typeof omittedSum, Expr<"float32", []>>>();
exact<Equal<typeof undefinedMean, Expr<"float32", []>>>();
exact<Equal<typeof omittedMin, Expr<"float32", []>>>();
exact<Equal<typeof undefinedMax, Expr<"float32", []>>>();
exact<Equal<typeof fixedSum, Expr<"float32", [3]>>>();
exact<Equal<typeof fixedMean, Expr<"float32", [2]>>>();
exact<Equal<typeof fixedMin, Expr<"float32", [3]>>>();
exact<Equal<typeof fixedMax, Expr<"float32", [2]>>>();
declare const dynamicAxis: number;
const dynamicSum = readonlyRows.sum(dynamicAxis);
const dynamicMean = np.mean(readonlyRows, dynamicAxis);
exact<Equal<typeof dynamicSum, Expr<"float32", number[]>>>();
exact<Equal<typeof dynamicMean, Expr<"float32", number[]>>>();

declare const integers: NMathArray<"int32", Rows>;
const integerSum = integers.sum();
const integerMean = np.mean(integers, 0);
exact<Equal<typeof integerSum, Expr<"int64", []>>>();
exact<Equal<typeof integerMean, Expr<"float64", [3]>>>();
declare const strings: NMathArray<"str", Rows>;
// @ts-expect-error string arrays do not support sums
strings.sum(0);
// @ts-expect-error string arrays do not support NumPy means
np.mean(strings, 0);

void optionalAxis;
