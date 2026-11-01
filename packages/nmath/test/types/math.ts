// SPDX-License-Identifier: Apache-2.0
import { createContext, type Expr } from "../../src/index.js";
import {
  abs,
  conjugate,
  floor,
  isnan,
  reciprocal,
  sign,
  signbit,
  sin,
  sqrt,
  square,
} from "../../src/numpy.js";

const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});

const i8 = c.zeros([2, 3], "int8");
const i16 = c.zeros([2, 3], "int16");
const i32 = c.zeros([2, 3], "int32");
const f32 = c.zeros([2, 3], "float32");
const flags = c.zeros([2, 3], "bool");
const complex = c.zeros([2, 3], "complex128");

const small: Expr<"float16", readonly [2, 3]> = sqrt(i8);
const medium: Expr<"float32", readonly [2, 3]> = sin(i16);
const wide: Expr<"float64", readonly [2, 3]> = sqrt(i32);
const single: Expr<"float32", readonly [2, 3]> = sin(f32);
const fromBool: Expr<"float16", readonly [2, 3]> = sqrt(flags);
const absolute: Expr<"int8", readonly [2, 3]> = abs(i8);
const squared: Expr<"int8", readonly [2, 3]> = square(flags);
const floored: Expr<"bool", readonly [2, 3]> = floor(flags);
const signs: Expr<"int32", readonly [2, 3]> = sign(i32);
const inverse: Expr<"float32", readonly [2, 3]> = reciprocal(f32);
const conj: Expr<"complex128", readonly [2, 3]> = conjugate(complex);
const conjBool: Expr<"int8", readonly [2, 3]> = conjugate(flags);
const nan: Expr<"bool", readonly [2, 3]> = isnan(complex);
const negative: Expr<"bool", readonly [2, 3]> = signbit(f32);

// @ts-expect-error complex transcendental functions remain targets
sqrt(complex);
// @ts-expect-error signbit is not defined for complex values
signbit(complex);
// @ts-expect-error integer reciprocal is not supported
reciprocal(i32);
// @ts-expect-error NumPy has no bool sign loop
sign(flags);
// @ts-expect-error string arrays are rejected
sqrt(c.array(["a"], { dtype: "str" }));
// @ts-expect-error int16 uses the float32 loop
const wrong: Expr<"float64", readonly [2, 3]> = sin(i16);

void [small, medium, wide, single, fromBool, absolute, squared, floored];
void [signs, inverse, conj, conjBool, nan, negative, wrong];
