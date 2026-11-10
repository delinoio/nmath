// SPDX-License-Identifier: Apache-2.0
import type { Expr, NMathContext } from "../../src/core.js";
type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;
declare const ctx: NMathContext;
const rng = ctx.generator(42);
const f32: Expr<"float32", readonly [3]> = rng.random([3] as const, "float32");
const f64: Expr<"float64", readonly [3]> = rng.random([3] as const);
const integers: Expr<"uint64", readonly [2, 3]> = rng.integers(
  0n,
  2n ** 64n,
  [2, 3] as const,
  { dtype: "uint64" },
);
const normal: Expr<"float32", readonly [3]> = rng.standard_normal(
  [3] as const,
  "float32",
);
const scalar: Promise<bigint> = rng.integers(0, 10, [] as const).compute();
const integerOptions: { dtype?: "int32" } = {};
const optionalInteger = rng.integers(0, 10, [2] as const, integerOptions);
const optionalIntegerDtype: Equal<
  typeof optionalInteger.dtype,
  "int32" | "int64"
> = true;
const emptyOptions = rng.integers(0, 10, [2] as const, {});
const emptyOptionsDtype: Equal<typeof emptyOptions.dtype, "int64"> = true;
const undefinedIntegerDtype = rng.integers(0, 10, [2] as const, {
  dtype: undefined,
});
const undefinedIntegerDtypeResult: Equal<
  typeof undefinedIntegerDtype.dtype,
  "int64"
> = true;
const explicitInteger = rng.integers(0, 10, [2] as const, {
  dtype: "int32",
});
const explicitIntegerDtype: Equal<typeof explicitInteger.dtype, "int32"> = true;
const explicitDefaultInteger = rng.integers(0, 10, [2] as const, {
  dtype: "int64",
});
const explicitDefaultIntegerDtype: Equal<
  typeof explicitDefaultInteger.dtype,
  "int64"
> = true;

function optionalFloat(dtype: "float32" | undefined) {
  const random = rng.random([2] as const, dtype);
  const standardNormal = rng.standard_normal([2] as const, dtype);
  const randomDtype: Equal<typeof random.dtype, "float32" | "float64"> = true;
  const normalDtype: Equal<typeof standardNormal.dtype, "float32" | "float64"> =
    true;
  const randomShape: readonly [2] = random.shape;
  const normalShape: readonly [2] = standardNormal.shape;
  void [randomDtype, normalDtype, randomShape, normalShape];
  return [random, standardNormal];
}
const undefinedRandom = rng.random([2] as const, undefined);
const undefinedNormal = rng.standard_normal([2] as const, undefined);
const undefinedRandomDtype: Equal<typeof undefinedRandom.dtype, "float64"> =
  true;
const undefinedNormalDtype: Equal<typeof undefinedNormal.dtype, "float64"> =
  true;
const explicitDefaultRandom = rng.random([2] as const, "float64");
const explicitDefaultNormal = rng.standard_normal([2] as const, "float64");
const explicitDefaultRandomDtype: Equal<
  typeof explicitDefaultRandom.dtype,
  "float64"
> = true;
const explicitDefaultNormalDtype: Equal<
  typeof explicitDefaultNormal.dtype,
  "float64"
> = true;

async function storageRequiresNarrowing(dtype: "float32" | undefined) {
  const integers = await rng
    .integers(0, 10, [2] as const, integerOptions)
    .compute();
  // @ts-expect-error the optional integer dtype can select BigInt64Array
  const int32Storage: Int32Array = integers.toTypedArray();
  const [random, normal] = optionalFloat(dtype);
  for (const expr of [random, normal]) {
    const result = await expr.compute();
    // @ts-expect-error the optional float dtype can select Float64Array
    const float32Storage: Float32Array = result.toTypedArray();
    void float32Storage;
  }
  void int32Storage;
}
// @ts-expect-error random accepts only float32 and float64
rng.random([3], "int32");
// @ts-expect-error integers does not accept floating dtypes
rng.integers(0, 10, [3], { dtype: "float64" });
// @ts-expect-error scalar parameters do not broadcast arrays
rng.normal([0, 1], 1, [2]);
void [
  f32,
  f64,
  integers,
  normal,
  scalar,
  optionalIntegerDtype,
  emptyOptionsDtype,
  undefinedIntegerDtypeResult,
  explicitIntegerDtype,
  explicitDefaultIntegerDtype,
  undefinedRandomDtype,
  undefinedNormalDtype,
  explicitDefaultRandomDtype,
  explicitDefaultNormalDtype,
  storageRequiresNarrowing,
];
