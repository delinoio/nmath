// SPDX-License-Identifier: Apache-2.0
import { createContext, Complex, type Expr } from "../../src/index.js";
import {
  arctan2,
  copysign,
  floor_divide,
  fmod,
  hypot,
  maximum,
} from "../../src/numpy.js";

const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});

const i8 = c.zeros([2, 3], "int8");
const u8 = c.zeros([3], "uint8");
const f32 = c.zeros([2, 3], "float32");
const flags = c.zeros([2, 3], "bool");
const z = c.zeros([3], "complex64");

const most: Expr<"int16", readonly [2, 3]> = maximum(i8, u8);
const complexMost: Expr<"complex64", readonly [3]> = maximum(
  z,
  new Complex(1, 0),
);
const signs: Expr<"float16", readonly [2, 3]> = copysign(i8, u8);
const angles: Expr<"float32", readonly [2, 3]> = arctan2(f32, i8);
const weak: Expr<"float32", readonly [2, 3]> = hypot(f32, 2);
const weakInt: Expr<"float16", readonly [2, 3]> = hypot(i8, 2n);
const quotient: Expr<"int8", readonly [2, 3]> = floor_divide(flags, flags);
const remainder: Expr<"int16", readonly [2, 3]> = fmod(i8, u8);

// @ts-expect-error float-valued functions reject complex operands
arctan2(z, z);
// @ts-expect-error fmod rejects complex operands
fmod(z, z);
// @ts-expect-error incompatible static shapes
maximum(i8, c.zeros([4], "int8"));

void [most, complexMost, signs, angles, weak, weakInt, quotient, remainder];
