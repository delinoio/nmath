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
  | "float32"
  | "float64"
  | "complex64"
  | "complex128"
  | "str"
  | "datetime64"
  | "timedelta64";

/** Dtypes with CPU kernels in this build. */
export type ComputableDType =
  "bool" | "int32" | "int64" | "uint32" | "uint64" | "float32" | "float64";

export type FloatDType = "float32" | "float64";
export type IntDType = "int32" | "int64" | "uint32" | "uint64";

export type Shape = readonly number[];

/** JS representation of one element of dtype `D`. */
export type ScalarOf<D extends DType> = D extends "int64" | "uint64"
  ? bigint
  : D extends "bool"
    ? boolean
    : number;

/** Typed array used to exchange data of dtype `D`. */
export type TypedArrayOf<D extends ComputableDType> = D extends "bool"
  ? Uint8Array
  : D extends "int32"
    ? Int32Array
    : D extends "int64"
      ? BigInt64Array
      : D extends "uint32"
        ? Uint32Array
        : D extends "uint64"
          ? BigUint64Array
          : D extends "float32"
            ? Float32Array
            : Float64Array;

type PromoteTable = {
  bool: {
    bool: "bool";
    int32: "int32";
    int64: "int64";
    uint32: "uint32";
    uint64: "uint64";
    float32: "float32";
    float64: "float64";
  };
  int32: {
    bool: "int32";
    int32: "int32";
    int64: "int64";
    uint32: "int64";
    uint64: "float64";
    float32: "float64";
    float64: "float64";
  };
  int64: {
    bool: "int64";
    int32: "int64";
    int64: "int64";
    uint32: "int64";
    uint64: "float64";
    float32: "float64";
    float64: "float64";
  };
  uint32: {
    bool: "uint32";
    int32: "int64";
    int64: "int64";
    uint32: "uint32";
    uint64: "uint64";
    float32: "float64";
    float64: "float64";
  };
  uint64: {
    bool: "uint64";
    int32: "float64";
    int64: "float64";
    uint32: "uint64";
    uint64: "uint64";
    float32: "float64";
    float64: "float64";
  };
  float32: {
    bool: "float32";
    int32: "float64";
    int64: "float64";
    uint32: "float64";
    uint64: "float64";
    float32: "float32";
    float64: "float64";
  };
  float64: {
    bool: "float64";
    int32: "float64";
    int64: "float64";
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
> = PromoteTable[A][B];

/** Weak JS scalar kinds: `number` is a Python float, `bigint` a Python int. */
export type WeakScalar = number | bigint | boolean;

export type PromoteWeak<
  D extends ComputableDType,
  S extends WeakScalar,
> = S extends number
  ? D extends FloatDType
    ? D
    : "float64"
  : S extends bigint
    ? D extends "bool"
      ? "int64"
      : D
    : D;

export type TrueDivide<D extends ComputableDType> = D extends FloatDType
  ? D
  : "float64";

export type SumDType<D extends ComputableDType> = D extends "bool" | "int32"
  ? "int64"
  : D extends "uint32"
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

/** `matmul` result shape for 1-D and 2-D operands. */
export type MatMulShape<A extends Shape, B extends Shape> = A extends readonly [
  infer N extends number,
  infer K extends number,
]
  ? B extends readonly [infer K2 extends number, infer M extends number]
    ? Compatible<K, K2> extends true
      ? [N, M]
      : never
    : B extends readonly [infer K2 extends number]
      ? Compatible<K, K2> extends true
        ? [N]
        : never
      : number[]
  : A extends readonly [infer K extends number]
    ? B extends readonly [infer K2 extends number, infer M extends number]
      ? Compatible<K, K2> extends true
        ? [M]
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
  MatMulShape<A, B>,
] extends [never]
  ? { readonly "matmul core dimensions do not match": [A, B] }
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
