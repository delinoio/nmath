// SPDX-License-Identifier: Apache-2.0
import { Complex, Expr, Lazy, slice } from "./core.js";
import { NMathError } from "./errors.js";
import { native } from "./errors.js";
import { binding } from "./native.js";
import type { Broadcast, NumericDType, Shape } from "./types.js";

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

// ---------------------------------------------------------------------------
// Norms and derived matrix functions

export type NormOrder = number | "fro" | "nuc";

function nativeNorm(
  a: Lazy<any, any>,
  kind: string,
  order?: number,
): Expr<any, any> {
  return new Expr(
    native(() => binding.NativeExpr.norm(a.nativeExpr(), kind, order)),
    a.context,
  );
}

function invalidOrder(message: string): never {
  throw new NMathError(
    "NMATH_INDEX_INVALID",
    `NMATH_INDEX_INVALID: ${message}`,
  );
}

function vectorKind(ord: NormOrder | undefined): [string, number?] {
  if (ord === undefined || ord === 2) return ["vector_two"];
  if (ord === Infinity) return ["vector_inf"];
  if (ord === -Infinity) return ["vector_neg_inf"];
  if (ord === 0) return ["vector_zero"];
  if (ord === 1) return ["vector_one"];
  if (typeof ord === "number" && !Number.isNaN(ord)) return ["vector_p", ord];
  return invalidOrder("Invalid norm order for vectors.");
}

function matrixKind(ord: NormOrder | undefined): string {
  switch (ord) {
    case undefined:
    case "fro":
      return "fro";
    case "nuc":
      return "nuc";
    case 1:
      return "matrix_one";
    case -1:
      return "matrix_neg_one";
    case 2:
      return "matrix_two";
    case -2:
      return "matrix_neg_two";
    case Infinity:
      return "matrix_inf";
    case -Infinity:
      return "matrix_neg_inf";
    default:
      return invalidOrder("Invalid norm order for matrices.");
  }
}

/**
 * NumPy `linalg.norm`. Without `axis`, `ord` undefined gives the 2-norm of the
 * flattened input; otherwise the input must be 1-D (vector) or 2-D (matrix).
 * One axis selects a vector norm and two axes a matrix norm.
 */
export function norm<D extends LinalgDType, S extends Shape>(
  x: Arg<D, S>,
  ord?: NormOrder,
  axis?: number | readonly [number] | readonly [number, number],
  keepdims = false,
): Expr<LinalgRealResult<D>, number[]> {
  if (typeof keepdims !== "boolean")
    invalidOrder("keepdims must be a boolean.");
  const rank = x.shape.length;
  let axes: number[];
  if (axis === undefined) {
    if (ord === undefined) {
      const flat = new Expr<any, any>(
        native(() =>
          binding.NativeExpr.reshape(
            x.nativeExpr(),
            [x.shape.reduce((n: number, d: number) => n * d, 1)],
            undefined,
          ),
        ),
        x.context,
      );
      const value = nativeNorm(flat, "vector_two");
      return keepdims
        ? reshapeTo(
            value,
            x.shape.map(() => 1),
          )
        : value;
    }
    if (rank !== 1 && rank !== 2) {
      throw new NMathError(
        "NMATH_SHAPE_MISMATCH",
        "NMATH_SHAPE_MISMATCH: Improper number of dimensions to norm. Pass axis for arrays with more than two dimensions.",
      );
    }
    axes = rank === 1 ? [0] : [0, 1];
  } else {
    axes = typeof axis === "number" ? [axis] : [...axis];
  }
  if (axes.length < 1 || axes.length > 2)
    invalidOrder("axis must select one or two axes.");
  const normalized = axes.map((a) => {
    if (!Number.isSafeInteger(a)) invalidOrder("An axis must be an integer.");
    const v = a < 0 ? a + rank : a;
    if (v < 0 || v >= rank) {
      throw new NMathError(
        "NMATH_INDEX_OUT_OF_BOUNDS",
        `NMATH_INDEX_OUT_OF_BOUNDS: Axis ${a} is out of bounds for an array of dimension ${rank}.`,
      );
    }
    return v;
  });
  if (normalized.length === 2 && normalized[0] === normalized[1]) {
    invalidOrder("Duplicate axes given.");
  }
  const rest = x.shape.map((_, i) => i).filter((i) => !normalized.includes(i));
  const order = [...rest, ...normalized];
  const moved = order.every((a, i) => a === i)
    ? x
    : new Expr<any, any>(
        native(() => binding.NativeExpr.transpose(x.nativeExpr(), order)),
        x.context,
      );
  const value =
    normalized.length === 1
      ? nativeNorm(moved, ...vectorKind(ord))
      : nativeNorm(moved, matrixKind(ord));
  return keepdims
    ? reshapeTo(
        value,
        x.shape.map((d, i) => (normalized.includes(i) ? 1 : d)),
      )
    : value;
}

function reshapeTo(value: Expr<any, any>, shape: number[]): Expr<any, any> {
  return new Expr(
    native(() =>
      binding.NativeExpr.reshape(value.nativeExpr(), shape, undefined),
    ),
    value.context,
  );
}

function swapLast(a: Expr<any, any>): Expr<any, any> {
  const rank = a.shape.length;
  const order = [...Array(rank).keys()];
  [order[rank - 2], order[rank - 1]] = [order[rank - 1], order[rank - 2]];
  return new Expr(
    native(() => binding.NativeExpr.transpose(a.nativeExpr(), order)),
    a.context,
  );
}

function requireMatrix(a: Lazy<any, any>, square: boolean): void {
  const s = a.shape;
  if (s.length < 2 || (square && s[s.length - 1] !== s[s.length - 2])) {
    throw new NMathError(
      "NMATH_SHAPE_MISMATCH",
      square
        ? "NMATH_SHAPE_MISMATCH: The final two axes must form square matrices."
        : "NMATH_SHAPE_MISMATCH: The input must have at least two dimensions.",
    );
  }
}

const EPS: Record<string, number> = {
  float32: 2 ** -23,
  float64: Number.EPSILON,
};

export interface MatrixRankOptions {
  /** Absolute threshold. Mutually exclusive with `rtol`. */
  readonly tol?: number;
  /** Relative threshold; NumPy's default is `max(M, N) * eps`. */
  readonly rtol?: number;
  /** Accepted for NumPy compatibility; the general SVD runs either way. */
  readonly hermitian?: boolean;
}

/** Number of singular values above the threshold, as `int64`. */
export function matrix_rank<D extends LinalgDType, S extends Shape>(
  A: Arg<D, S> & Matrix<S>,
  options: MatrixRankOptions = {},
): Expr<"int64", Out<S, Batch<S>>> {
  requireMatrix(A, false);
  if (options.tol !== undefined && options.rtol !== undefined) {
    invalidOrder("tol and rtol can't be both set.");
  }
  flag(options.hermitian, "hermitian", false);
  const s = factor("svd_s", A);
  const m = A.shape[A.shape.length - 2];
  const n = A.shape[A.shape.length - 1];
  let threshold: Lazy<any, any> | number;
  if (options.tol !== undefined) {
    threshold = options.tol;
  } else if (Math.min(m, n) === 0) {
    // NumPy uses max(..., initial=0), so an empty spectrum has rank 0.
    threshold = 0;
  } else {
    const rtol = options.rtol ?? Math.max(m, n) * EPS[s.dtype];
    threshold = s.max(-1, { keepdims: true }).binaryOp("mul", rtol);
  }
  return s.binaryOp("gt", threshold).sum(-1) as never;
}

export interface PinvOptions {
  /** Relative cutoff for small singular values (NumPy default `1e-15`). */
  readonly rcond?: number;
  /** NumPy 2 alias of `rcond`; `null` selects `max(M, N) * eps`. */
  readonly rtol?: number | null;
  /** Accepted for NumPy compatibility; the general SVD runs either way. */
  readonly hermitian?: boolean;
}

type PinvShape<S extends Shape> = Out<S, [...Batch<S>, Cols<S>, Rows<S>]>;

/** Moore-Penrose pseudo-inverse through the reduced SVD, as NumPy computes it. */
export function pinv<D extends LinalgDType, S extends Shape>(
  a: Arg<D, S> & Matrix<S>,
  options: PinvOptions = {},
): Expr<LinalgResult<D>, PinvShape<S>> {
  requireMatrix(a, false);
  flag(options.hermitian, "hermitian", false);
  const m = a.shape[a.shape.length - 2];
  const n = a.shape[a.shape.length - 1];
  if (options.rcond !== undefined && options.rtol !== undefined) {
    invalidOrder("rtol and rcond can't be both set.");
  }
  const conj = new Expr<any, any>(
    native(() => binding.NativeExpr.math("conjugate", a.nativeExpr())),
    a.context,
  );
  const u = factor("svd_u", conj);
  const s = factor("svd_s", conj);
  const vt = factor("svd_vh", conj);
  if (m === 0 || n === 0) {
    // NumPy returns an empty (..., n, m) array.
    return reshapeTo(swapLast(u), [...a.shape.slice(0, -2), n, m]) as never;
  }
  const rcond =
    options.rcond ??
    (options.rtol === undefined
      ? 1e-15
      : options.rtol === null
        ? Math.max(m, n) * EPS[s.dtype]
        : options.rtol);
  const cutoff = s.max(-1, { keepdims: true }).binaryOp("mul", rcond);
  const large = s.binaryOp("gt", cutoff);
  const inverse = large.whereOp(s.reversedBinaryOp("div", 1), 0);
  const scaled = readShaped(inverse).binaryOp("mul", swapLast(u));
  return swapLast(vt).matmul(scaled as never) as never;
}

/** `s[..., newaxis]`. */
function readShaped(s: Expr<any, any>): Expr<any, any> {
  return reshapeTo(s, [...s.shape, 1]);
}

/**
 * Condition number from singular values: `p` undefined or 2 gives
 * `s_max / s_min`, -2 gives the reciprocal. NaN results become infinity
 * unless the input contains NaN, as in NumPy.
 */
export function cond<D extends LinalgDType, S extends Shape>(
  x: Arg<D, S> & Matrix<S>,
  p?: 2 | -2,
): Expr<LinalgRealResult<D>, Out<S, Batch<S>>> {
  requireMatrix(x, false);
  if (p !== undefined && p !== 2 && p !== -2) {
    invalidOrder(
      "cond supports p = 2 and -2. Other orders depend on NumPy's error-ignoring inverse and remain targets.",
    );
  }
  const m = x.shape[x.shape.length - 2];
  const n = x.shape[x.shape.length - 1];
  if (m === 0 || n === 0) {
    throw new NMathError(
      "NMATH_SHAPE_MISMATCH",
      "NMATH_SHAPE_MISMATCH: cond is not defined on empty arrays.",
    );
  }
  const s = factor("svd_s", x);
  const k = Math.min(m, n);
  const first = s.get(
    ...([...Array(s.shape.length - 1)].map(() => slice()) as never[]),
    0 as never,
  );
  const last = s.get(
    ...([...Array(s.shape.length - 1)].map(() => slice()) as never[]),
    (k - 1) as never,
  );
  const ratio =
    p === -2 ? last.binaryOp("div", first) : first.binaryOp("div", last);
  const ratioNan = new Expr<any, any>(
    native(() => binding.NativeExpr.math("isnan", ratio.nativeExpr())),
    x.context,
  );
  const inputNan = new Expr<any, any>(
    native(() => binding.NativeExpr.math("isnan", x.nativeExpr())),
    x.context,
  ).max([-2, -1]);
  const replace = ratioNan.binaryOp(
    "logical_and",
    new Expr<any, any>(
      native(() =>
        binding.NativeExpr.math("logical_not", inputNan.nativeExpr()),
      ),
      x.context,
    ),
  );
  return replace.whereOp(Infinity, ratio) as never;
}

function identityValues(dtype: string, n: number): unknown[] {
  return Array.from({ length: n * n }, (_, i) => {
    const one = i % (n + 1) === 0;
    if (dtype === "bool") return one;
    if (dtype === "int64" || dtype === "uint64") return one ? 1n : 0n;
    if (dtype === "complex64" || dtype === "complex128") {
      return new Complex(one ? 1 : 0, 0);
    }
    return one ? 1 : 0;
  });
}

/** Integer matrix power with NumPy's multiplication order; negative powers invert first. */
export function matrix_power<
  D extends NumericDType,
  S extends Shape,
  const N extends number,
>(
  a: Lazy<NoInfer<D>, S> & { readonly dtype: D } & Square<S>,
  n: N,
): Expr<
  `${N}` extends `-${string}`
    ? D extends LinalgDType
      ? LinalgResult<D>
      : never
    : D,
  S
> {
  requireMatrix(a, true);
  if (!Number.isSafeInteger(n))
    invalidOrder("The exponent must be an integer.");
  if (n === 0) {
    const size = a.shape[a.shape.length - 1];
    const context = a.context;
    if (context === undefined)
      invalidOrder("matrix_power requires a contextual array.");
    const eye = context.array(
      identityValues(a.dtype, size) as never,
      {
        dtype: a.dtype,
        shape: [size, size],
      } as never,
    );
    return context
      .zeros(a.shape as never, a.dtype as never)
      .binaryOp("add", eye as never) as never;
  }
  let base: Lazy<any, any> = n < 0 ? factor("inv", a) : a;
  let power = Math.abs(n);
  if (power === 1) return asExpr(base) as never;
  if (power === 2) return base.matmul(base as never) as never;
  if (power === 3)
    return base.matmul(base as never).matmul(base as never) as never;
  let z: Lazy<any, any> | undefined;
  let result: Lazy<any, any> | undefined;
  while (power > 0) {
    z = z === undefined ? base : z.matmul(z as never);
    const bit = power % 2;
    power = Math.floor(power / 2);
    if (bit) result = result === undefined ? z : result.matmul(z as never);
  }
  return result as never;
}

/** A lazy view of `a` as an expression. */
function asExpr(a: Lazy<any, any>): Expr<any, any> {
  return a instanceof Expr ? a : new Expr(a.nativeExpr(), a.context);
}
