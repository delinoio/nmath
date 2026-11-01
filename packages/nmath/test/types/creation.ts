// SPDX-License-Identifier: Apache-2.0
import { createContext, type NMathArray } from "../../src/index.js";
import {
  empty,
  empty_like,
  eye,
  full_like,
  identity,
  linspace,
  logspace,
  ones_like,
  zeros_like,
} from "../../src/numpy.js";

const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});

const defaults: NMathArray<"float64", [50]> = linspace(c, 0, 1);
const five: NMathArray<"float64", [5]> = linspace(c, 0, 1, 5);
const dynamic: NMathArray<"float32", [number]> = linspace(
  c,
  0,
  1,
  5 as number,
  {
    dtype: "float32",
  },
);
const integers: NMathArray<"int64", [4]> = linspace(c, 0, 9, 4, {
  dtype: "int64",
  endpoint: false,
});
const [samples, step]: readonly [NMathArray<"float64", [3]>, number] = linspace(
  c,
  0,
  1,
  3,
  { retstep: true },
);
const logs: NMathArray<"float64", [4]> = logspace(c, 2, 3, 4, { base: 2 });
const logInts: NMathArray<"int32", [number]> = logspace(c, 0, 3, undefined, {
  dtype: "int32",
});

const square: NMathArray<"float64", [3, 3]> = eye(c, 3);
const wide: NMathArray<"bool", [2, 4]> = eye(c, 2, 4, { k: 1, dtype: "bool" });
const id: NMathArray<"complex128", [2, 2]> = identity(c, 2, "complex128");
const blank: NMathArray<"uint8", [2, 3]> = empty(c, [2, 3], "uint8");

const source = c.zeros([2, 3], "float32");
const z: NMathArray<"float32", readonly [2, 3]> = zeros_like(source);
const o: NMathArray<"int64", readonly [2, 3]> = ones_like(source, "int64");
const f: NMathArray<"float32", readonly [2, 3]> = full_like(source, 2.5);
const fi: NMathArray<"int64", readonly [2, 3]> = full_like(source, 2n, "int64");
const e: NMathArray<"float32", readonly [2, 3]> = empty_like(source);
const fromExpr: NMathArray<"float32", readonly [2, 3]> = zeros_like(
  source.add(1),
);

// @ts-expect-error linspace does not create string arrays.
linspace(c, 0, 1, 3, { dtype: "str" });
// @ts-expect-error eye does not create temporal arrays.
eye(c, 2, 2, { dtype: "datetime64" });
// @ts-expect-error identity takes a dtype, not an options object.
identity(c, 2, { dtype: "float32" });
// @ts-expect-error the fill value must match the result dtype.
full_like(source, 2n);
// @ts-expect-error 64-bit fills require bigint values.
full_like(source, 2, "int64");
// @ts-expect-error logspace endpoints are JS numbers.
logspace(c, 0n, 1);

void [defaults, five, dynamic, integers, samples, step, logs, logInts];
void [square, wide, id, blank, z, o, f, fi, e, fromExpr];
