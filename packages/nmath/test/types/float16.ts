// SPDX-License-Identifier: Apache-2.0
import {
  createContext,
  add,
  Complex,
  type Expr,
  type Promote,
  type TypedArrayOf,
} from "../../src/index.js";
import * as runtime from "../../src/nts-runtime.js";
declare function expect<T>(x: T): void;
type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;
declare function exact<T extends true>(): void;
const ctx = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 1,
  cpuMemoryBudget: 1024,
  gpuMemoryBudget: 0,
});
const a = ctx.array([1, 2], { dtype: "float16", shape: [2] });
const weak = a.add(1),
  small = a.add(ctx.zeros([2], "int8")),
  medium = a.add(ctx.zeros([2], "int16")),
  wide = a.add(ctx.zeros([2], "int32"));
exact<Equal<typeof weak.dtype, "float16">>();
exact<Equal<typeof small.dtype, "float16">>();
exact<Equal<typeof medium.dtype, "float32">>();
exact<Equal<typeof wide.dtype, "float64">>();
exact<Equal<Promote<"float16", "complex64">, "complex64">>();
expect<Float16Array>(a.toTypedArray());
expect<TypedArrayOf<"float16">>(new Float16Array(2));
expect<Uint16Array>(a.toFloat16Bits());
expect<Promise<number>>(a.sum().compute());
expect<Expr<"float16", readonly [2]>>(runtime.add(a, 1));
expect<Expr<"float16", readonly [2]>>(add(1, a));
expect<Expr<"complex64", readonly [2]>>(a.add(new Complex(1)));
expect<Expr<"float16", []>>(a.matmul(a));
ctx.arrayFromFloat16Bits(new Uint16Array([0x3c00]));
// @ts-expect-error incompatible known dimensions remain invalid
a.add(ctx.zeros([3], "float16"));
// @ts-expect-error bit export requires float16 storage
ctx.zeros([2], "uint16").toFloat16Bits();
