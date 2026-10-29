// SPDX-License-Identifier: Apache-2.0
import { Expr, type Lazy } from "./core.js";
import { native } from "./errors.js";
import { binding } from "./native.js";
import type { Broadcast, Shape } from "./types.js";

import type { NumericalDType } from "./fft.js";

export type LinalgDType = Exclude<NumericalDType, "float16">;
export type LinalgResult<D extends LinalgDType> = D extends
  "float32" | "complex64" | "complex128"
  ? D
  : "float64";
type EqualDim<A extends number, B extends number> = number extends A | B
  ? true
  : A extends B
    ? true
    : false;
type Square<S extends Shape> = number[] extends S
  ? unknown
  : S extends readonly [
        ...Shape,
        infer M extends number,
        infer N extends number,
      ]
    ? EqualDim<M, N> extends true
      ? unknown
      : never
    : never;
export type DetShape<S extends Shape> = S extends readonly [
  ...infer B extends Shape,
  number,
  number,
]
  ? B
  : number[];
export type SolveShape<A extends Shape, B extends Shape> = A extends readonly [
  ...infer AB extends Shape,
  number,
  infer N extends number,
]
  ? B extends readonly [number]
    ? [...AB, N]
    : B extends readonly [
          ...infer BB extends Shape,
          number,
          infer K extends number,
        ]
      ? Broadcast<AB, BB> extends infer Batch extends Shape
        ? [...Batch, N, K]
        : never
      : number[]
  : number[];
type SolveCheck<A extends Shape, B extends Shape> = number[] extends A | B
  ? unknown
  : A extends readonly [...Shape, number, infer N extends number]
    ? B extends
        | readonly [infer M extends number]
        | readonly [...Shape, infer M extends number, number]
      ? EqualDim<N, M> extends true
        ? [SolveShape<A, B>] extends [never]
          ? never
          : unknown
        : never
      : never
    : never;
export type SolveDType<A extends LinalgDType, B extends LinalgDType> =
  A | B extends "float32" | "complex64"
  ? "complex64" extends A | B
    ? "complex64"
    : "float32"
  : Extract<A | B, "complex64" | "complex128"> extends never
    ? "float64"
    : "complex128";

export function inv<D extends LinalgDType, S extends Shape>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D } & Square<S>,
): Expr<LinalgResult<D>, S> {
  return new Expr(
    native(() => binding.NativeExpr.linalg("inv", a.nativeExpr())),
    a.context,
  );
}
export function det<D extends LinalgDType, S extends Shape>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D } & Square<S>,
): Expr<LinalgResult<D>, DetShape<S>> {
  return new Expr(
    native(() => binding.NativeExpr.linalg("det", a.nativeExpr())),
    a.context,
  );
}
export function solve<
  A extends LinalgDType,
  AS extends Shape,
  B extends LinalgDType,
  BS extends Shape,
>(
  a: Lazy<NoInfer<A>, AS> & { readonly dtype: A } & Square<AS>,
  b: Lazy<NoInfer<B>, BS> & { readonly dtype: B } & SolveCheck<AS, BS>,
): Expr<SolveDType<A, B>, SolveShape<AS, BS>> {
  return new Expr(
    native(() =>
      binding.NativeExpr.linalg("solve", a.nativeExpr(), b.nativeExpr()),
    ),
    a.context ?? b.context,
  );
}
