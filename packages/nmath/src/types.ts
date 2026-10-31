/// <reference lib="esnext.float16" preserve="true" />
import type { Complex, Temporal } from "./core.js";
// SPDX-License-Identifier: Apache-2.0
// Type-level dtype promotion and shape inference. Static rules mirror the native runtime;
// dimensions typed as `number` are dynamic and validated at runtime.

/** Every dtype identifier. */
export type DType =
  | "bool"
  | "int8"
  | "int16"
  | "int32"
  | "int64"
  | "uint8"
  | "uint16"
  | "uint32"
  | "uint64"
  | "float16"
  | "float32"
  | "float64"
  | "complex64"
  | "complex128"
  | "str"
  | "datetime64"
  | "timedelta64";

/** Dtypes with CPU kernels in this build. */
export type ComputableDType =
  | "str"
  | "datetime64"
  | "timedelta64"
  | "bool"
  | "int8"
  | "int16"
  | "uint8"
  | "uint16"
  | "int32"
  | "int64"
  | "uint32"
  | "uint64"
  | "float16"
  | "float32"
  | "float64"
  | "complex64"
  | "complex128";

export type ComplexDType = "complex64" | "complex128";

export type FloatDType = "float16" | "float32" | "float64";
export type IntDType =
  | "int8"
  | "int16"
  | "uint8"
  | "uint16"
  | "int32"
  | "int64"
  | "uint32"
  | "uint64";

export type Shape = readonly number[];

/** JS representation of one element of dtype `D`. */
export type ScalarOf<D extends DType> = D extends "str"
  ? string
  : D extends TemporalDType
    ? Temporal<D>
    : D extends ComplexDType
      ? Complex
      : D extends "int64" | "uint64"
        ? bigint
        : D extends "bool"
          ? boolean
          : number;

/** Typed array used to exchange data of dtype `D`. */
export type TemporalDType = "datetime64" | "timedelta64";
export type NumericDType = Exclude<ComputableDType, "str" | TemporalDType>;
export type TypedArrayOf<D extends ComputableDType> = D extends "str"
  ? never
  : D extends TemporalDType
    ? BigInt64Array
    : D extends "float16"
      ? Float16Array
      : D extends "int8"
        ? Int8Array
        : D extends "int16"
          ? Int16Array
          : D extends "uint16"
            ? Uint16Array
            : D extends "bool" | "uint8"
              ? Uint8Array
              : D extends "int32"
                ? Int32Array
                : D extends "int64"
                  ? BigInt64Array
                  : D extends "uint32"
                    ? Uint32Array
                    : D extends "uint64"
                      ? BigUint64Array
                      : D extends "float32" | "complex64"
                        ? Float32Array
                        : Float64Array;

// NumPy 2.5.4 result_type; tested against the recorded integer dtype fixture.
type PromoteTable = {
  float16: {
    bool: "float16";
    int8: "float16";
    uint8: "float16";
    int16: "float32";
    uint16: "float32";
    int32: "float64";
    uint32: "float64";
    int64: "float64";
    uint64: "float64";
    float16: "float16";
    float32: "float32";
    float64: "float64";
  };
  bool: {
    float16: "float16";
    bool: "bool";
    int8: "int8";
    int16: "int16";
    int32: "int32";
    int64: "int64";
    uint8: "uint8";
    uint16: "uint16";
    uint32: "uint32";
    uint64: "uint64";
    float32: "float32";
    float64: "float64";
  };
  int8: {
    float16: "float16";
    bool: "int8";
    int8: "int8";
    int16: "int16";
    int32: "int32";
    int64: "int64";
    uint8: "int16";
    uint16: "int32";
    uint32: "int64";
    uint64: "float64";
    float32: "float32";
    float64: "float64";
  };
  int16: {
    float16: "float32";
    bool: "int16";
    int8: "int16";
    int16: "int16";
    int32: "int32";
    int64: "int64";
    uint8: "int16";
    uint16: "int32";
    uint32: "int64";
    uint64: "float64";
    float32: "float32";
    float64: "float64";
  };
  int32: {
    float16: "float64";
    bool: "int32";
    int8: "int32";
    int16: "int32";
    int32: "int32";
    int64: "int64";
    uint8: "int32";
    uint16: "int32";
    uint32: "int64";
    uint64: "float64";
    float32: "float64";
    float64: "float64";
  };
  int64: {
    float16: "float64";
    bool: "int64";
    int8: "int64";
    int16: "int64";
    int32: "int64";
    int64: "int64";
    uint8: "int64";
    uint16: "int64";
    uint32: "int64";
    uint64: "float64";
    float32: "float64";
    float64: "float64";
  };
  uint8: {
    float16: "float16";
    bool: "uint8";
    int8: "int16";
    int16: "int16";
    int32: "int32";
    int64: "int64";
    uint8: "uint8";
    uint16: "uint16";
    uint32: "uint32";
    uint64: "uint64";
    float32: "float32";
    float64: "float64";
  };
  uint16: {
    float16: "float32";
    bool: "uint16";
    int8: "int32";
    int16: "int32";
    int32: "int32";
    int64: "int64";
    uint8: "uint16";
    uint16: "uint16";
    uint32: "uint32";
    uint64: "uint64";
    float32: "float32";
    float64: "float64";
  };
  uint32: {
    float16: "float64";
    bool: "uint32";
    int8: "int64";
    int16: "int64";
    int32: "int64";
    int64: "int64";
    uint8: "uint32";
    uint16: "uint32";
    uint32: "uint32";
    uint64: "uint64";
    float32: "float64";
    float64: "float64";
  };
  uint64: {
    float16: "float64";
    bool: "uint64";
    int8: "float64";
    int16: "float64";
    int32: "float64";
    int64: "float64";
    uint8: "uint64";
    uint16: "uint64";
    uint32: "uint64";
    uint64: "uint64";
    float32: "float64";
    float64: "float64";
  };
  float32: {
    float16: "float32";
    bool: "float32";
    int8: "float32";
    int16: "float32";
    int32: "float64";
    int64: "float64";
    uint8: "float32";
    uint16: "float32";
    uint32: "float64";
    uint64: "float64";
    float32: "float32";
    float64: "float64";
  };
  float64: {
    float16: "float64";
    bool: "float64";
    int8: "float64";
    int16: "float64";
    int32: "float64";
    int64: "float64";
    uint8: "float64";
    uint16: "float64";
    uint32: "float64";
    uint64: "float64";
    float32: "float64";
    float64: "float64";
  };
};

/** NumPy `result_type` for two array dtypes. */
export type Promote<
  A extends ComputableDType,
  B extends ComputableDType,
> = A extends "str"
  ? B extends "str"
    ? "str"
    : never
  : A extends TemporalDType
    ? B extends TemporalDType
      ? A extends "datetime64"
        ? "datetime64"
        : B
      : never
    : B extends "str" | TemporalDType
      ? never
      : A extends ComplexDType
        ? ComplexPromote<A, B>
        : B extends ComplexDType
          ? ComplexPromote<B, A>
          : A extends keyof PromoteTable
            ? B extends keyof PromoteTable[A]
              ? PromoteTable[A][B]
              : never
            : never;

type ComplexPromote<
  A extends ComplexDType,
  B extends ComputableDType,
> = A extends "complex128"
  ? "complex128"
  : B extends "complex128" | "float64" | "int32" | "uint32" | "int64" | "uint64"
    ? "complex128"
    : "complex64";

/** Weak JS scalar kinds: `number` is a Python float, `bigint` a Python int. */
export type WeakScalar =
  number | bigint | boolean | Complex | string | Temporal;

export type PromoteWeak<
  D extends ComputableDType,
  S extends WeakScalar,
> = S extends string
  ? D extends "str"
    ? "str"
    : never
  : S extends Temporal<infer K>
    ? Promote<D, K>
    : D extends "str"
      ? never
      : D extends TemporalDType
        ? S extends bigint
          ? D
          : never
        : S extends Complex
          ? D extends ComplexDType
            ? D
            : D extends "float16" | "float32"
              ? "complex64"
              : "complex128"
          : S extends number
            ? D extends FloatDType | ComplexDType
              ? D
              : "float64"
            : S extends bigint
              ? D extends "bool"
                ? "int64"
                : D
              : D;

/** Boolean remainder and power use the NumPy int8 loop. */
export type BoolArithmetic<D extends ComputableDType> = D extends "bool"
  ? "int8"
  : D;

export type TrueDivide<D extends ComputableDType> = D extends
  FloatDType | ComplexDType
  ? D
  : "float64";

export type SumDType<D extends ComputableDType> = D extends
  "bool" | "int8" | "int16" | "int32"
  ? "int64"
  : D extends "uint8" | "uint16" | "uint32"
    ? "uint64"
    : D;

type Reverse<T extends Shape, Acc extends number[] = []> = T extends readonly [
  infer H extends number,
  ...infer R extends number[],
]
  ? Reverse<R, [H, ...Acc]>
  : Acc;

type IsLiteral<N extends number> = number extends N ? false : true;

type BroadcastDim<A extends number, B extends number> = A extends 1
  ? B
  : B extends 1
    ? A
    : IsLiteral<A> extends true
      ? IsLiteral<B> extends true
        ? A extends B
          ? A
          : never
        : A
      : B;

type BroadcastReversed<
  A extends number[],
  B extends number[],
  Acc extends number[] = [],
> = A extends [infer HA extends number, ...infer RA extends number[]]
  ? B extends [infer HB extends number, ...infer RB extends number[]]
    ? [BroadcastDim<HA, HB>] extends [never]
      ? never
      : BroadcastReversed<RA, RB, [BroadcastDim<HA, HB>, ...Acc]>
    : BroadcastReversed<RA, [], [HA, ...Acc]>
  : B extends [infer HB extends number, ...infer RB extends number[]]
    ? BroadcastReversed<[], RB, [HB, ...Acc]>
    : Acc;

/** Broadcast result shape, or `never` when static dimensions conflict. */
export type Broadcast<A extends Shape, B extends Shape> = number[] extends A
  ? number[]
  : number[] extends B
    ? number[]
    : BroadcastReversed<Reverse<A>, Reverse<B>>;

/** Makes an argument unassignable when shapes cannot broadcast statically. */
export type BroadcastCheck<A extends Shape, B extends Shape> = [
  Broadcast<A, B>,
] extends [never]
  ? { readonly "shapes do not broadcast": [A, B] }
  : unknown;

type DropAt<
  S extends readonly number[],
  I extends number,
  Acc extends number[] = [],
> = S extends readonly [infer H extends number, ...infer R extends number[]]
  ? Acc["length"] extends I
    ? [...Acc, ...R]
    : DropAt<R, I, [...Acc, H]>
  : Acc;

/** Numeric dtypes that allow ordering comparisons. */
export type OrderedDType = Exclude<NumericDType, ComplexDType>;

/** NumPy variance result dtype: float16 stays, float32/complex64 become float32. */
export type VarianceDType<D extends ComputableDType> = D extends "float16"
  ? "float16"
  : D extends "float32" | "complex64"
    ? "float32"
    : "float64";

/** The same shape with mutable tuple members. */
export type MutableShape<S extends Shape> = S extends readonly [
  ...infer D extends number[],
]
  ? D
  : S;

/** Shape after reducing over one non-negative literal axis. */
export type ReduceShape<S extends Shape, A extends number | undefined> = [
  A,
] extends [undefined]
  ? []
  : number[] extends S
    ? number[]
    : A extends number
      ? IsLiteral<A> extends true
        ? `${A}` extends `-${string}`
          ? number[]
          : DropAt<S, A>
        : number[]
      : number[];

/** Matrix product shape with broadcast batch axes and vector promotion. */
export type MatMulShape<
  A extends Shape,
  B extends Shape,
> = A extends readonly []
  ? never
  : B extends readonly []
    ? never
    : number[] extends A | B
      ? number[]
      : A extends readonly [
            ...infer AB extends Shape,
            infer N extends number,
            infer K extends number,
          ]
        ? B extends readonly [
            ...infer BB extends Shape,
            infer K2 extends number,
            infer M extends number,
          ]
          ? Compatible<K, K2> extends true
            ? Broadcast<AB, BB> extends infer Batch extends Shape
              ? [...Batch, N, M]
              : never
            : never
          : B extends readonly [infer K2 extends number]
            ? Compatible<K, K2> extends true
              ? [...AB, N]
              : never
            : number[]
        : A extends readonly [infer K extends number]
          ? B extends readonly [
              ...infer BB extends Shape,
              infer K2 extends number,
              infer M extends number,
            ]
            ? Compatible<K, K2> extends true
              ? [...BB, M]
              : never
            : B extends readonly [infer K2 extends number]
              ? Compatible<K, K2> extends true
                ? []
                : never
              : number[]
          : number[];

type Compatible<A extends number, B extends number> =
  IsLiteral<A> extends true
    ? IsLiteral<B> extends true
      ? A extends B
        ? true
        : false
      : true
    : true;

export type MatMulCheck<A extends Shape, B extends Shape> = [
  MatMulShape<Readonly<A>, Readonly<B>>,
] extends [never]
  ? { readonly "matmul dimensions do not match": [Readonly<A>, Readonly<B>] }
  : unknown;

/** Shape after indexing: an integer drops an axis; other selectors keep a dynamic axis. */
export type IndexShape<
  S extends Shape,
  Sel extends readonly unknown[],
> = number[] extends S
  ? number[]
  : Sel extends readonly [infer H, ...infer R]
    ? S extends readonly [number, ...infer SR extends number[]]
      ? H extends number
        ? IndexShape<SR, R>
        : [number, ...IndexShape<SR, R>]
      : never
    : S extends readonly number[]
      ? [...S]
      : never;

/** Operation-specific temporal results, distinct from general promotion. */
export type BinaryDType<
  O extends string,
  A extends ComputableDType,
  B extends ComputableDType,
> = O extends "mod"
  ? A extends ComplexDType
    ? never
    : B extends ComplexDType
      ? never
      : BinaryNonComplex<O, A, B>
  : BinaryNonComplex<O, A, B>;
type BinaryNonComplex<
  O extends string,
  A extends ComputableDType,
  B extends ComputableDType,
> = A extends "str"
  ? O extends "add"
    ? B extends "str"
      ? "str"
      : never
    : never
  : B extends "str"
    ? never
    : A extends TemporalDType
      ? B extends TemporalDType
        ? O extends "add"
          ? A extends "datetime64"
            ? B extends "datetime64"
              ? never
              : "datetime64"
            : B
          : O extends "sub"
            ? A extends "datetime64"
              ? B extends "datetime64"
                ? "timedelta64"
                : "datetime64"
              : B extends "timedelta64"
                ? "timedelta64"
                : never
            : O extends "div"
              ? A extends "timedelta64"
                ? B extends "timedelta64"
                  ? "float64"
                  : never
                : never
              : O extends "mod"
                ? A extends "timedelta64"
                  ? B extends "timedelta64"
                    ? "timedelta64"
                    : never
                  : never
                : never
        : A extends "timedelta64"
          ? B extends IntDType
            ? O extends "mul" | "div"
              ? A
              : never
            : never
          : never
      : B extends TemporalDType
        ? B extends "timedelta64"
          ? A extends IntDType
            ? O extends "mul"
              ? B
              : never
            : never
          : never
        : O extends "div"
          ? TrueDivide<Promote<A, B>>
          : O extends "mod" | "pow"
            ? BoolArithmetic<Promote<A, B>>
            : Promote<A, B>;
export type BinaryCheck<
  O extends string,
  A extends ComputableDType,
  B extends ComputableDType,
> = [BinaryDType<O, A, B>] extends [never] ? never : unknown;
export type WeakDType<W extends WeakScalar> =
  W extends Temporal<infer K>
    ? K
    : W extends string
      ? "str"
      : W extends bigint
        ? "int64"
        : W extends boolean
          ? "bool"
          : W extends Complex
            ? "complex128"
            : "float64";
export type WeakBinary<
  O extends string,
  D extends ComputableDType,
  W extends WeakScalar,
> = D extends "str" | TemporalDType
  ? BinaryDType<O, D, WeakDType<W>>
  : W extends string | Temporal
    ? BinaryDType<O, D, WeakDType<W>>
    : O extends "div"
      ? TrueDivide<PromoteWeak<D, W>>
      : O extends "mod" | "pow"
        ? BoolArithmetic<PromoteWeak<D, W>>
        : PromoteWeak<D, W>;
