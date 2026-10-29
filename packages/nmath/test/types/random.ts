// SPDX-License-Identifier: Apache-2.0
import type { Expr, NMathContext } from "../../src/core.js";
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
// @ts-expect-error random accepts only float32 and float64
rng.random([3], "int32");
// @ts-expect-error integers does not accept floating dtypes
rng.integers(0, 10, [3], { dtype: "float64" });
// @ts-expect-error scalar parameters do not broadcast arrays
rng.normal([0, 1], 1, [2]);
void [f32, f64, integers, normal, scalar];
