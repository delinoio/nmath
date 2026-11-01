// SPDX-License-Identifier: Apache-2.0
import { createContext, type Expr } from "../../src/index.js";
import { linalg } from "../../src/numpy.js";

const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});

const square = c.zeros([2, 3, 3], "float32");
const tall = c.zeros([5, 3], "complex128");
const ints = c.zeros([4, 4], "int32");

const chol: Expr<"float32", readonly [2, 3, 3]> = linalg.cholesky(square);
const [q, r] = linalg.qr(tall);
const qShape: Expr<"complex128", [5, number]> = q;
const rShape: Expr<"complex128", [number, 3]> = r;
const complete: Expr<"complex128", [5, 5]> = linalg.qr(tall, "complete").Q;
const onlyR: Expr<"complex128", [number, 3]> = linalg.qr(tall, "r");
const eig = linalg.eigh(square, "U");
const values: Expr<"float32", [2, 3]> = eig.eigenvalues;
const vectors: Expr<"float32", readonly [2, 3, 3]> = eig[1];
const real: Expr<"float64", [5]> = linalg.eigvalsh(
  c.zeros([5, 5], "complex128"),
);
const full = linalg.svd(tall);
const u: Expr<"complex128", [5, 5]> = full.U;
const s: Expr<"float64", [number]> = full.S;
const vh: Expr<"complex128", [3, 3]> = full[2];
const reduced: Expr<"complex128", [5, number]> = linalg.svd(tall, {
  full_matrices: false,
}).U;
const sv: Expr<"float64", [number]> = linalg.svd(tall, { compute_uv: false });
const sv2: Expr<"float64", [number]> = linalg.svdvals(ints);
const det = linalg.slogdet(ints);
const sign: Expr<"float64", []> = det.sign;
const logabs: Expr<"float64", []> = det[1];

// @ts-expect-error cholesky requires square matrices
linalg.cholesky(tall);
// @ts-expect-error eigh requires square matrices
linalg.eigh(tall);
// @ts-expect-error float16 linear algebra is rejected
linalg.svdvals(c.zeros([2, 2], "float16"));

void [chol, qShape, rShape, complete, onlyR, values, vectors, real];
void [u, s, vh, reduced, sv, sv2, sign, logabs];
