// SPDX-License-Identifier: Apache-2.0
import { createContext, type Expr } from "../../src/index.js";
import {
  bitwise_and,
  bitwise_xor,
  invert,
  left_shift,
  logical_or,
} from "../../src/numpy.js";

const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});

const i8 = c.zeros([2, 3], "int8");
const u8 = c.zeros([3], "uint8");
const flags = c.zeros([2, 3], "bool");
const f64 = c.zeros([2, 3], "float64");

const promoted: Expr<"int16", readonly [2, 3]> = bitwise_and(i8, u8);
const boolean: Expr<"bool", readonly [2, 3]> = bitwise_xor(flags, flags);
const weak: Expr<"int8", readonly [2, 3]> = bitwise_and(i8, 3n);
const shiftedBool: Expr<"int8", readonly [2, 3]> = left_shift(flags, flags);
const inverted: Expr<"uint8", readonly [3]> = invert(u8);
const logical: Expr<"bool", readonly [2, 3]> = logical_or(f64, i8);

// @ts-expect-error float operands have no bitwise loop
bitwise_and(f64, f64);
// @ts-expect-error a JS number is a weak float
bitwise_and(i8, 3);
// @ts-expect-error int64 with uint64 promotes to float64
bitwise_and(c.zeros([2], "int64"), c.zeros([2], "uint64"));
// @ts-expect-error invert rejects floats
invert(f64);

void [promoted, boolean, weak, shiftedBool, inverted, logical];
