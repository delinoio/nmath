// SPDX-License-Identifier: Apache-2.0
import type { NMathArray } from "../../src/core.js";
import type { LinalgDType, SolveDType } from "../../src/linalg.js";
import { linalg } from "../../src/numpy.js";

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;
type Assert<T extends true> = T;

type AllowedDTypes =
  | "bool"
  | "int8"
  | "int16"
  | "int32"
  | "int64"
  | "uint8"
  | "uint16"
  | "uint32"
  | "uint64"
  | "float32"
  | "float64"
  | "complex64"
  | "complex128";
type AllowedDTypesCheck = Assert<Equal<LinalgDType, AllowedDTypes>>;

// This reference follows the native result_dtype cases for concrete operands.
type ExpectedSolve<
  A extends LinalgDType,
  B extends LinalgDType,
> = A extends "complex128"
  ? "complex128"
  : B extends "complex128"
    ? "complex128"
    : A extends "complex64"
      ? B extends "float32" | "complex64"
        ? "complex64"
        : "complex128"
      : B extends "complex64"
        ? A extends "float32"
          ? "complex64"
          : "complex128"
        : A extends "float32"
          ? B extends "float32"
            ? "float32"
            : "float64"
          : "float64";

// The mapped type checks every one of the 13 x 13 concrete operand pairs.
type ConcretePairChecks = {
  [A in LinalgDType]: {
    [B in LinalgDType]: Equal<SolveDType<A, B>, ExpectedSolve<A, B>>;
  }[LinalgDType];
}[LinalgDType];
type AllConcretePairsCheck = Assert<Equal<ConcretePairChecks, true>>;

type FloatUncertainty = Assert<
  Equal<SolveDType<"float32" | "float64", "float32">, "float32" | "float64">
>;
type ReversedFloatUncertainty = Assert<
  Equal<SolveDType<"float32", "float32" | "float64">, "float32" | "float64">
>;
type ComplexUncertainty = Assert<
  Equal<
    SolveDType<"complex64" | "complex128", "float32">,
    "complex64" | "complex128"
  >
>;
type FloatOrComplex = Assert<
  Equal<SolveDType<"float32" | "complex64", "float32">, "float32" | "complex64">
>;
type ExclusiveInputs = Assert<
  Equal<SolveDType<"float64" | "complex64", "float32">, "float64" | "complex64">
>;
type BothInputsUncertain = Assert<
  Equal<
    SolveDType<"float32" | "complex64", "float32" | "float64">,
    "float32" | "float64" | "complex64" | "complex128"
  >
>;

declare const uncertainMatrix: NMathArray<
  "float32" | "float64",
  readonly [2, 2]
>;
declare const float32Matrix: NMathArray<"float32", readonly [2, 2]>;
declare const float32Rhs: NMathArray<"float32", readonly [2]>;
declare const uncertainRhs: NMathArray<"float32" | "float64", readonly [2]>;
declare const openMatrix: NMathArray<
  "float32" | "float64",
  readonly [2, ...number[]]
>;

const result = linalg.solve(uncertainMatrix, float32Rhs);
const reversed = linalg.solve(float32Matrix, uncertainRhs);
const openResult = linalg.solve(openMatrix, float32Rhs);
type ResultDType = Assert<Equal<typeof result.dtype, "float32" | "float64">>;
type ReversedDType = Assert<
  Equal<typeof reversed.dtype, "float32" | "float64">
>;
type ResultShape = Assert<Equal<typeof result.shape, [2]>>;
type OpenResultShape = Assert<Equal<typeof openResult.shape, number[]>>;
type ComputedResult = Awaited<ReturnType<typeof result.compute>>;
type ComputeDType = Assert<
  Equal<ComputedResult["dtype"], "float32" | "float64">
>;
type ComputedStorage = ReturnType<ComputedResult["toTypedArray"]>;
type StorageTypes = Assert<Equal<ComputedStorage, Float32Array | Float64Array>>;

async function checkPublicStorage() {
  const out = await result.compute();
  const storage: Float32Array | Float64Array = out.toTypedArray();
  // @ts-expect-error a float32 solve result cannot promise Float64Array storage
  const guaranteedDouble: Float64Array = out.toTypedArray();
  void [storage, guaranteedDouble];
}

declare const float16Matrix: NMathArray<"float16", readonly [2, 2]>;
// @ts-expect-error float16 remains unsupported by solve
linalg.solve(float16Matrix, float32Rhs);
void [result, reversed, openResult, checkPublicStorage];
