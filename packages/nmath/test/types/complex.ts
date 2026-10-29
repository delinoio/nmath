// SPDX-License-Identifier: Apache-2.0
import {
  Complex,
  createContext,
  add,
  div,
  mod,
  type Expr,
  type ScalarOf,
  type TypedArrayOf,
} from "../../src/index.js";

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
const a = ctx.array([new Complex(1, 2)], { dtype: "complex64", shape: [1] });
const b = ctx.zeros([1], "complex128");
const small = ctx.zeros([1], "int16"),
  big = ctx.zeros([1], "int32");
const f = ctx.zeros([1], "float32");
const z = new Complex(1, -2);
const mixed = a.add(small),
  wide = a.add(big),
  weak = f.add(z),
  inverse = add(z, f);
exact<Equal<typeof mixed.dtype, "complex64">>();
exact<Equal<typeof wide.dtype, "complex128">>();
exact<Equal<typeof weak.dtype, "complex64">>();
exact<Equal<typeof inverse.dtype, "complex64">>();
expect<Expr<"complex128", [1]>>(a.add(b));
expect<Expr<"complex64", readonly [1]>>(a.add(2));
expect<Expr<"complex64", readonly [1]>>(a.div(2n));
expect<Expr<"complex64", readonly [1]>>(div(z, f));
expect<Promise<Complex>>(a.sum().compute());
expect<Promise<Complex>>(a.mean().compute());
expect<Promise<Complex>>(a.min().compute());
expect<Complex>(a.item(0));
expect<Float32Array>(a.toTypedArray());
expect<Float64Array>(b.toTypedArray());
expect<ScalarOf<"complex64">>(z);
expect<TypedArrayOf<"complex128">>(new Float64Array(2));
ctx.array(new Float32Array([1, 2]), { dtype: "complex64" });
ctx.full([2], z, "complex128");
// @ts-expect-error complex array elements require the dedicated scalar type
ctx.array([1, 2], { dtype: "complex128" });
// @ts-expect-error complex results do not implicitly lose the imaginary part
expect<Promise<number>>(a.sum().compute());
// @ts-expect-error modulo is undefined for complex arrays
a.mod(a);
// @ts-expect-error modulo is undefined for complex weak scalars
f.mod(z);
// @ts-expect-error modulo is undefined for complex operands
mod(f, a);
// @ts-expect-error modulo is undefined for complex weak scalars
mod(z, f);
