// SPDX-License-Identifier: Apache-2.0
import { createContext, Complex, type Expr } from "../../src/index.js";
import {
  equal,
  greater,
  greater_equal,
  less,
  less_equal,
  not_equal,
} from "../../src/numpy.js";

const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});

const ints = c.zeros([2, 3], "int16");
const same: Expr<"bool", readonly [2, 3]> = equal(ints, ints);
const ordered: Expr<"bool", readonly [2, 3]> = less(ints, ints);
const weak: Expr<"bool", readonly [2, 3]> = greater(ints, 2);
const weakBigint: Expr<"bool", readonly [2, 3]> = least(ints);
const rows = c.zeros([2, 1], "float32");
const columns = c.zeros([3], "float64");
const broadcast: Expr<"bool", readonly [2, 3]> = not_equal(rows, columns);
const complex = c.zeros([2, 3], "complex64");
const complexEqual: Expr<"bool", readonly [2, 3]> = equal(complex, complex);
const complexWeak: Expr<"bool", readonly [2, 3]> = equal(
  complex,
  new Complex(1, 2),
);
const dynamic = c.zeros([1, 3], "float32");
const dynamicCompare: Expr<"bool", readonly [2, 3]> = less_equal(
  c.zeros([2, 1], "float32"),
  c.zeros([3], "float32"),
);

const strings = c.array(["a"], { dtype: "str" });
// @ts-expect-error string comparisons are not supported
equal(strings, strings);
// @ts-expect-error ordering comparisons reject complex operands
less(complex, complex);
// @ts-expect-error ordering comparisons reject complex weak scalars
greater(complex, new Complex(1, 2));
// @ts-expect-error incompatible static shapes
equal(c.zeros([2, 3], "float32"), c.zeros([4], "float32"));
// @ts-expect-error comparisons return bool, not the operand dtype
const wrong: Expr<"int16", [2, 3]> = equal(ints, ints);

void same;
void ordered;
void weak;
void weakBigint;
void broadcast;
void complexEqual;
void complexWeak;
void dynamicCompare;
void dynamic;
void wrong;

function least(x: typeof ints): Expr<"bool", readonly [2, 3]> {
  return greater_equal(x, 2n);
}
