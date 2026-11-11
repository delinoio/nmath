// SPDX-License-Identifier: Apache-2.0
import {
  Complex,
  NMathArray,
  Temporal,
  type ComputeResult,
  type Expr,
} from "../../src/index.js";

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;
declare function exact<T extends true>(): void;
declare function expectType<T>(value: T): void;

exact<
  Equal<
    ComputeResult<"float64", number[]>,
    number | NMathArray<"float64", number[]>
  >
>();
exact<
  Equal<
    ComputeResult<"float64", readonly number[]>,
    number | NMathArray<"float64", readonly number[]>
  >
>();
exact<Equal<ComputeResult<"float64", []>, number>>();
exact<Equal<ComputeResult<"float64", readonly []>, number>>();
exact<
  Equal<ComputeResult<"float64", [number]>, NMathArray<"float64", [number]>>
>();
exact<Equal<ComputeResult<"float64", [2, 3]>, NMathArray<"float64", [2, 3]>>>();
exact<
  Equal<
    ComputeResult<"float64", readonly [number, ...number[]]>,
    NMathArray<"float64", readonly [number, ...number[]]>
  >
>();
exact<
  Equal<ComputeResult<"float64", [] | [2]>, number | NMathArray<"float64", [2]>>
>();

exact<
  Equal<
    ComputeResult<"int64", number[]>,
    bigint | NMathArray<"int64", number[]>
  >
>();
exact<
  Equal<ComputeResult<"bool", number[]>, boolean | NMathArray<"bool", number[]>>
>();
exact<
  Equal<ComputeResult<"str", number[]>, string | NMathArray<"str", number[]>>
>();
exact<
  Equal<
    ComputeResult<"complex64", number[]>,
    Complex | NMathArray<"complex64", number[]>
  >
>();
exact<
  Equal<
    ComputeResult<"datetime64", number[]>,
    Temporal<"datetime64"> | NMathArray<"datetime64", number[]>
  >
>();
exact<Equal<ComputeResult<"int64", []>, bigint>>();
exact<Equal<ComputeResult<"bool", []>, boolean>>();
exact<Equal<ComputeResult<"str", []>, string>>();
exact<Equal<ComputeResult<"complex128", []>, Complex>>();
exact<Equal<ComputeResult<"timedelta64", []>, Temporal<"timedelta64">>>();

declare const dynamic: Expr<"float64", number[]>;
declare const readonlyDynamic: Expr<"float64", readonly number[]>;

async function narrowResults() {
  const result = await dynamic.compute();
  // @ts-expect-error an unknown-rank result may be a scalar
  result.toArray();
  if (result instanceof NMathArray) {
    expectType<NMathArray<"float64", number[]>>(result);
    result.toArray();
  } else {
    expectType<number>(result);
  }

  const readonlyResult = await readonlyDynamic.compute();
  // @ts-expect-error a readonly unknown-rank result may be a scalar
  readonlyResult.toArray();
  if (readonlyResult instanceof NMathArray) {
    expectType<NMathArray<"float64", readonly number[]>>(readonlyResult);
  } else {
    expectType<number>(readonlyResult);
  }
}

void narrowResults;
