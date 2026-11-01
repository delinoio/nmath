// SPDX-License-Identifier: Apache-2.0
import { createContext, Complex, type Expr } from "../../src/index.js";
import { less, where } from "../../src/numpy.js";

const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});

const ints = c.zeros([2, 3], "int16");
const floats = c.zeros([3], "float32");
const mask = less(ints, 2);
const same: Expr<"int16", readonly [2, 3]> = where(mask, ints, ints);
const promoted: Expr<"float32", readonly [2, 3]> = where(mask, ints, floats);
const broadcast: Expr<"float64", readonly [2, 3]> = where(
  c.zeros([2, 1], "bool"),
  c.zeros([3], "float64"),
  c.zeros([1, 3], "float32"),
);
const numericCondition: Expr<"int16", readonly [2, 3]> = where(
  c.zeros([2, 3], "float64"),
  ints,
  ints,
);
const weakRight: Expr<"int16", readonly [2, 3]> = where(mask, ints, 7n);
const weakLeft: Expr<"int16", readonly [2, 3]> = where(mask, 7n, ints);
// JS numbers are weak float scalars, so integer branches promote to float64.
const weakFloat: Expr<"float64", readonly [2, 3]> = where(mask, ints, 7);
const weakComplex: Expr<"complex64", readonly [3]> = where(
  c.zeros([3], "bool"),
  floats,
  new Complex(0, 1),
);
const bothWeak: Expr<"float64", readonly [2, 3]> = where(mask, 1n, 2.5);
const bothBool: Expr<"bool", readonly [2, 3]> = where(mask, true, false);

const strings = c.array(["a"], { dtype: "str" });
// @ts-expect-error string branches are not supported
where(mask, strings, strings);
// @ts-expect-error string conditions are not supported
where(strings, floats, floats);
// @ts-expect-error incompatible static shapes between condition and x
where(mask, c.zeros([4], "float32"), floats);
// @ts-expect-error incompatible static shapes with y
where(mask, ints, c.zeros([4], "int16"));
// @ts-expect-error where promotes branches, it does not return bool
const wrong: Expr<"bool", readonly [2, 3]> = where(mask, ints, ints);

void same;
void promoted;
void broadcast;
void numericCondition;
void weakRight;
void weakLeft;
void weakFloat;
void weakComplex;
void bothWeak;
void bothBool;
void wrong;
