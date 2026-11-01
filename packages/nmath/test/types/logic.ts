// SPDX-License-Identifier: Apache-2.0
import { Complex, createContext, type Expr } from "../../src/index.js";
import {
  all,
  allclose,
  any,
  array_equal,
  array_equiv,
  count_nonzero,
  isclose,
} from "../../src/numpy.js";

const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});

const a = c.zeros([2, 3, 4], "float32");
const flags = c.zeros([2, 3, 4], "bool");
const z = c.zeros([3, 1], "complex128");
const row = c.zeros([4], "int32");
const text = c.array(["a"], { dtype: "str", shape: [1] });
const when = c.zeros([2], "datetime64");

const anyAll: Expr<"bool", []> = any(a);
const allAxis: Expr<"bool", [2, 4]> = all(flags, 1);
const kept: Expr<"bool", [1, 3, 1]> = any(a, [0, 2], { keepdims: true });
const dynamic: Expr<"bool", number[]> = all(a, -1);
const count: Expr<"int64", []> = count_nonzero(a);
const countKept: Expr<"int64", [2, 1, 4]> = count_nonzero(flags, 1, {
  keepdims: true,
});
const close: Expr<"bool", [3, 4]> = isclose(z, row, { rtol: 1e-3 });
const closeScalar: Expr<"bool", [4]> = isclose(row, 2);
const scalarFirst: Expr<"bool", [3, 1]> = isclose(new Complex(1, 0), z);
const together: Expr<"bool", []> = allclose(a, 1, { equal_nan: true });
const same: Expr<"bool", []> = array_equal(a, row, { equal_nan: true });
const equiv: Expr<"bool", []> = array_equiv(a, row);
const equivScalar: Expr<"bool", []> = array_equiv(row, 0n);

// @ts-expect-error any rejects str input
any(text);
// @ts-expect-error all rejects temporal input
all(when);
// @ts-expect-error count_nonzero rejects str input
count_nonzero(text);
// @ts-expect-error isclose rejects str input
isclose(text, a);
// @ts-expect-error isclose rejects shapes that do not broadcast
isclose(a, c.zeros([5], "float64"));
// @ts-expect-error array_equal rejects temporal input
array_equal(when, when);
// @ts-expect-error any returns bool
const wrongDType: Expr<"int64", []> = any(a);
// @ts-expect-error count_nonzero keeps reduced axes with keepdims
const wrongShape: Expr<"int64", [2, 4]> = count_nonzero(a, 1, {
  keepdims: true,
});

void [anyAll, allAxis, kept, dynamic, count, countKept, close];
void [closeScalar, scalarFirst, together, same, equiv, equivScalar];
void [wrongDType, wrongShape];
