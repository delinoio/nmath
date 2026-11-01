// SPDX-License-Identifier: Apache-2.0
import { Expr, type Lazy } from "./core.js";
import { NMathError } from "./errors.js";
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

/** Real dtype of a factorization result: eigenvalues, singular values, log-determinants. */
export type LinalgRealResult<D extends LinalgDType> =
  LinalgResult<D> extends "complex64"
    ? "float32"
    : LinalgResult<D> extends "complex128"
      ? "float64"
      : LinalgResult<D>;
type Batch<S extends Shape> = S extends readonly [
  ...infer B extends Shape,
  number,
  number,
]
  ? B
  : number[];
type Rows<S extends Shape> = S extends readonly [
  ...Shape,
  infer M extends number,
  number,
]
  ? M
  : number;
type Cols<S extends Shape> = S extends readonly [
  ...Shape,
  number,
  infer N extends number,
]
  ? N
  : number;
type Matrix<S extends Shape> = number[] extends S
  ? unknown
  : S extends readonly [...Shape, number, number]
    ? unknown
    : never;
type Arg<D extends LinalgDType, S extends Shape> = Lazy<NoInfer<D>, S> & {
  readonly dtype: D;
};
type Out<S extends Shape, T extends Shape> = number[] extends S ? number[] : T;

/** A NumPy-style named tuple: indexable and destructurable, with named fields. */
function named<const K extends readonly string[], V extends readonly unknown[]>(
  keys: K,
  values: V,
): V & {
  readonly [I in keyof K & `${number}` as K[I] & string]: V[I & keyof V];
} {
  const out = [...values] as unknown as Record<string, unknown>;
  keys.forEach((key, i) => {
    out[key] = values[i];
  });
  return Object.freeze(out) as never;
}

function factor(op: string, a: Lazy<any, any>): Expr<any, any> {
  return new Expr(
    native(() => binding.NativeExpr.linalg(op, a.nativeExpr())),
    a.context,
  );
}

function uplo(value: "L" | "U" | undefined): boolean {
  if (value === undefined || value === "L") return false;
  if (value === "U") return true;
  throw new NMathError(
    "NMATH_INDEX_INVALID",
    'NMATH_INDEX_INVALID: UPLO must be "L" or "U".',
  );
}

function flag(value: unknown, name: string, fallback: boolean): boolean {
  if (value === undefined) return fallback;
  if (typeof value !== "boolean") {
    throw new NMathError(
      "NMATH_INDEX_INVALID",
      `NMATH_INDEX_INVALID: ${name} must be a boolean.`,
    );
  }
  return value;
}

/** Cholesky factor from the lower triangle. `upper` returns its conjugate transpose. */
export function cholesky<D extends LinalgDType, S extends Shape>(
  a: Arg<D, S> & Square<S>,
  options: { readonly upper?: boolean } = {},
): Expr<LinalgResult<D>, S> {
  return factor(
    flag(options.upper, "upper", false) ? "cholesky_upper" : "cholesky",
    a,
  );
}

export type QRResult<
  D extends LinalgDType,
  Q extends Shape,
  R extends Shape,
> = readonly [Expr<LinalgResult<D>, Q>, Expr<LinalgResult<D>, R>] & {
  readonly Q: Expr<LinalgResult<D>, Q>;
  readonly R: Expr<LinalgResult<D>, R>;
};

/** Householder QR. Modes: `"reduced"` (default), `"complete"` and `"r"`. */
export function qr<D extends LinalgDType, S extends Shape>(
  a: Arg<D, S> & Matrix<S>,
  mode?: "reduced",
): QRResult<
  D,
  Out<S, [...Batch<S>, Rows<S>, number]>,
  Out<S, [...Batch<S>, number, Cols<S>]>
>;
export function qr<D extends LinalgDType, S extends Shape>(
  a: Arg<D, S> & Matrix<S>,
  mode: "complete",
): QRResult<
  D,
  Out<S, [...Batch<S>, Rows<S>, Rows<S>]>,
  Out<S, [...Batch<S>, Rows<S>, Cols<S>]>
>;
export function qr<D extends LinalgDType, S extends Shape>(
  a: Arg<D, S> & Matrix<S>,
  mode: "r",
): Expr<LinalgResult<D>, Out<S, [...Batch<S>, number, Cols<S>]>>;
export function qr(
  a: Lazy<any, any>,
  mode: "reduced" | "complete" | "r" = "reduced",
): unknown {
  if (mode === "r") return factor("qr_r", a);
  if (mode !== "reduced" && mode !== "complete") {
    throw new NMathError(
      "NMATH_INDEX_INVALID",
      'NMATH_INDEX_INVALID: qr mode must be "reduced", "complete" or "r". The LAPACK-specific "raw" mode is not supported.',
    );
  }
  const suffix = mode === "complete" ? "_complete" : "";
  return named(
    ["Q", "R"],
    [factor(`qr_q${suffix}`, a), factor(`qr_r${suffix}`, a)],
  );
}

export type EighResult<D extends LinalgDType, S extends Shape> = readonly [
  Expr<LinalgRealResult<D>, Out<S, [...Batch<S>, Cols<S>]>>,
  Expr<LinalgResult<D>, S>,
] & {
  readonly eigenvalues: Expr<
    LinalgRealResult<D>,
    Out<S, [...Batch<S>, Cols<S>]>
  >;
  readonly eigenvectors: Expr<LinalgResult<D>, S>;
};

/** Hermitian eigendecomposition read from one triangle (`UPLO`, default `"L"`). */
export function eigh<D extends LinalgDType, S extends Shape>(
  a: Arg<D, S> & Square<S>,
  UPLO?: "L" | "U",
): EighResult<D, S> {
  const upper = uplo(UPLO) ? "_upper" : "";
  return named(
    ["eigenvalues", "eigenvectors"],
    [factor(`eigvalsh${upper}`, a), factor(`eigh_vectors${upper}`, a)],
  ) as never;
}

/** Ascending eigenvalues of a Hermitian matrix read from one triangle. */
export function eigvalsh<D extends LinalgDType, S extends Shape>(
  a: Arg<D, S> & Square<S>,
  UPLO?: "L" | "U",
): Expr<LinalgRealResult<D>, Out<S, [...Batch<S>, Cols<S>]>> {
  return factor(uplo(UPLO) ? "eigvalsh_upper" : "eigvalsh", a);
}

export type SVDResult<
  D extends LinalgDType,
  U extends Shape,
  S extends Shape,
  V extends Shape,
> = readonly [
  Expr<LinalgResult<D>, U>,
  Expr<LinalgRealResult<D>, S>,
  Expr<LinalgResult<D>, V>,
] & {
  readonly U: Expr<LinalgResult<D>, U>;
  readonly S: Expr<LinalgRealResult<D>, S>;
  readonly Vh: Expr<LinalgResult<D>, V>;
};
export interface SVDOptions {
  /** Square `U` and `Vh` (NumPy default `true`). */
  readonly full_matrices?: boolean;
  /** Return only singular values when `false`. */
  readonly compute_uv?: boolean;
  /**
   * Accepted for NumPy compatibility. The general algorithm runs either way,
   * so the factors are a valid SVD but can differ in sign from NumPy's
   * eigh-based result.
   */
  readonly hermitian?: boolean;
}

/** Singular value decomposition by one-sided Jacobi rotations. */
export function svd<D extends LinalgDType, S extends Shape>(
  a: Arg<D, S> & Matrix<S>,
  options: SVDOptions & { readonly compute_uv: false },
): Expr<LinalgRealResult<D>, Out<S, [...Batch<S>, number]>>;
export function svd<D extends LinalgDType, S extends Shape>(
  a: Arg<D, S> & Matrix<S>,
  options?: SVDOptions & {
    readonly full_matrices?: true;
    readonly compute_uv?: true;
  },
): SVDResult<
  D,
  Out<S, [...Batch<S>, Rows<S>, Rows<S>]>,
  Out<S, [...Batch<S>, number]>,
  Out<S, [...Batch<S>, Cols<S>, Cols<S>]>
>;
export function svd<D extends LinalgDType, S extends Shape>(
  a: Arg<D, S> & Matrix<S>,
  options: SVDOptions & {
    readonly full_matrices: false;
    readonly compute_uv?: true;
  },
): SVDResult<
  D,
  Out<S, [...Batch<S>, Rows<S>, number]>,
  Out<S, [...Batch<S>, number]>,
  Out<S, [...Batch<S>, number, Cols<S>]>
>;
export function svd(a: Lazy<any, any>, options: SVDOptions = {}): unknown {
  for (const key of Object.keys(options)) {
    if (!["full_matrices", "compute_uv", "hermitian"].includes(key)) {
      throw new NMathError(
        "NMATH_INDEX_INVALID",
        `NMATH_INDEX_INVALID: Unknown svd option '${key}'.`,
      );
    }
  }
  flag(options.hermitian, "hermitian", false);
  if (!flag(options.compute_uv, "compute_uv", true)) return factor("svd_s", a);
  const full = flag(options.full_matrices, "full_matrices", true)
    ? "_full"
    : "";
  return named(
    ["U", "S", "Vh"],
    [factor(`svd_u${full}`, a), factor("svd_s", a), factor(`svd_vh${full}`, a)],
  );
}

/** Descending singular values. */
export function svdvals<D extends LinalgDType, S extends Shape>(
  a: Arg<D, S> & Matrix<S>,
): Expr<LinalgRealResult<D>, Out<S, [...Batch<S>, number]>> {
  return factor("svd_s", a);
}

export type SlogdetResult<D extends LinalgDType, S extends Shape> = readonly [
  Expr<LinalgResult<D>, DetShape<S>>,
  Expr<LinalgRealResult<D>, DetShape<S>>,
] & {
  readonly sign: Expr<LinalgResult<D>, DetShape<S>>;
  readonly logabsdet: Expr<LinalgRealResult<D>, DetShape<S>>;
};

/** Sign and natural log of the absolute determinant. Singular matrices give `(0, -inf)`. */
export function slogdet<D extends LinalgDType, S extends Shape>(
  a: Arg<D, S> & Square<S>,
): SlogdetResult<D, S> {
  return named(
    ["sign", "logabsdet"],
    [factor("slogdet_sign", a), factor("slogdet_logabs", a)],
  ) as never;
}
