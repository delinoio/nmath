// SPDX-License-Identifier: Apache-2.0
// Decimal type arithmetic avoids allocating tuples as large as array sizes.
// Scientific notation and dynamic dimensions retain runtime validation.
import type { Shape } from "./types.js";
type Digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9";
type Units = {
  "0": [];
  "1": [0];
  "2": [0, 0];
  "3": [0, 0, 0];
  "4": [0, 0, 0, 0];
  "5": [0, 0, 0, 0, 0];
  "6": [0, 0, 0, 0, 0, 0];
  "7": [0, 0, 0, 0, 0, 0, 0];
  "8": [0, 0, 0, 0, 0, 0, 0, 0];
  "9": [0, 0, 0, 0, 0, 0, 0, 0, 0];
};
type Ten = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
type ReverseText<
  S extends string,
  R extends string = "",
> = S extends `${infer H}${infer T}` ? ReverseText<T, `${H}${R}`> : R;
type Trim<S extends string> = S extends `0${infer T}`
  ? T extends ""
    ? "0"
    : Trim<T>
  : S;
type SumDigit<A extends Digit, B extends Digit, C extends 0[] = []> = [
  ...Units[A],
  ...Units[B],
  ...C,
] extends infer T extends 0[]
  ? T extends [...Ten, ...infer R extends 0[]]
    ? [`${R["length"]}`, [0]]
    : [`${T["length"]}`, []]
  : never;
type AddReverse<
  A extends string,
  B extends string,
  C extends 0[] = [],
  R extends string = "",
> = A extends `${infer H extends Digit}${infer T}`
  ? B extends `${infer J extends Digit}${infer U}`
    ? SumDigit<H, J, C> extends [
        infer D extends string,
        infer Carry extends 0[],
      ]
      ? AddReverse<T, U, Carry, `${D}${R}`>
      : never
    : AddReverse<A, "0", C, R>
  : B extends `${infer J extends Digit}${infer U}`
    ? SumDigit<"0", J, C> extends [
        infer D extends string,
        infer Carry extends 0[],
      ]
      ? AddReverse<"", U, Carry, `${D}${R}`>
      : never
    : Trim<C extends [] ? (R extends "" ? "0" : R) : `1${R}`>;
type Add<A extends string, B extends string> = AddReverse<
  ReverseText<A>,
  ReverseText<B>
>;
type TimesDigit<
  A extends string,
  N extends 0[],
  R extends string = "0",
> = N extends [0, ...infer T extends 0[]] ? TimesDigit<A, T, Add<R, A>> : R;
type Multiply<
  A extends string,
  B extends string,
  R extends string = "0",
> = B extends `${infer H extends Digit}${infer T}`
  ? Multiply<A, T, Add<`${R}0`, TimesDigit<A, Units[H]>>>
  : Trim<R>;
type DigitGte<A extends Digit, B extends Digit> = Units[A] extends [
  ...Units[B],
  ...0[],
]
  ? true
  : false;
type SameLengthGte<
  A extends string,
  B extends string,
> = A extends `${infer H extends Digit}${infer T}`
  ? B extends `${infer J extends Digit}${infer U}`
    ? H extends J
      ? SameLengthGte<T, U>
      : DigitGte<H, J>
    : true
  : true;
type LengthCompare<
  A extends string,
  B extends string,
  FullA extends string = A,
  FullB extends string = B,
> = A extends `${infer _H}${infer T}`
  ? B extends `${infer _J}${infer U}`
    ? LengthCompare<T, U, FullA, FullB>
    : true
  : B extends ""
    ? SameLengthGte<FullA, FullB>
    : false;
type SubDigit<A extends Digit, B extends Digit, Borrow extends 0[]> = [
  ...Units[B],
  ...Borrow,
] extends infer Need extends 0[]
  ? Units[A] extends [...Need, ...infer R extends 0[]]
    ? [`${R["length"]}`, []]
    : [...Ten, ...Units[A]] extends [...Need, ...infer R extends 0[]]
      ? [`${R["length"]}`, [0]]
      : never
  : never;
type SubReverse<
  A extends string,
  B extends string,
  Borrow extends 0[] = [],
  R extends string = "",
> = A extends `${infer H extends Digit}${infer T}`
  ? B extends `${infer J extends Digit}${infer U}`
    ? SubDigit<H, J, Borrow> extends [
        infer D extends string,
        infer Next extends 0[],
      ]
      ? SubReverse<T, U, Next, `${D}${R}`>
      : never
    : SubReverse<A, "0", Borrow, R>
  : Trim<R extends "" ? "0" : R>;
type Subtract<A extends string, B extends string> = SubReverse<
  ReverseText<A>,
  ReverseText<B>
>;
type QuotientDigit<A extends string, B extends string, N extends 0[] = []> =
  LengthCompare<A, B> extends true
    ? QuotientDigit<Subtract<A, B>, B, [...N, 0]>
    : [`${N["length"]}`, A];
type Divide<
  A extends string,
  B extends string,
  Remainder extends string = "0",
  Q extends string = "",
> = B extends "0"
  ? never
  : A extends `${infer H extends Digit}${infer T}`
    ? QuotientDigit<Trim<`${Remainder}${H}`>, B> extends [
        infer D extends string,
        infer R extends string,
      ]
      ? Divide<T, B, R, `${Q}${D}`>
      : never
    : Remainder extends "0"
      ? Trim<Q>
      : never;

type Product<S extends Shape, R extends string = "1"> = 0 extends S[number]
  ? number extends S[number]
    ? ProductNonzero<S, R>
    : "0"
  : ProductNonzero<S, R>;
type ProductNonzero<
  S extends Shape,
  R extends string,
> = number extends S["length"]
  ? string
  : S extends readonly [infer H extends number, ...infer T extends Shape]
    ? number extends H
      ? string
      : `${H}` extends `${bigint}`
        ? Product<T, Multiply<R, `${H}`>>
        : string
    : R;
type ValidDimensions<
  R extends Shape,
  Seen extends boolean = false,
> = R extends readonly [infer H extends number, ...infer T extends Shape]
  ? number extends H
    ? ValidDimensions<T, Seen>
    : H extends -1
      ? Seen extends true
        ? false
        : ValidDimensions<T, true>
      : `${H}` extends `-${string}`
        ? false
        : `${H}` extends `${bigint}`
          ? ValidDimensions<T, Seen>
          : false
  : true;
type WithoutInference<R extends Shape> = R extends readonly [
  infer H extends number,
  ...infer T extends Shape,
]
  ? [H extends -1 ? 1 : H, ...WithoutInference<T>]
  : number extends R["length"]
    ? number[]
    : [];
type Inferred<S extends Shape, R extends Shape> =
  Product<S> extends infer P extends string
    ? Product<WithoutInference<R>> extends infer Q extends string
      ? Q extends "0"
        ? never
        : string extends P | Q
          ? number
          : Divide<P, Q> extends infer D
            ? [D] extends [never]
              ? never
              : D extends `${infer N extends number}`
                ? N
                : number
            : never
      : never
    : never;
type ReplaceInference<R extends Shape, N extends number> = {
  -readonly [K in keyof R]: R[K] extends -1 ? N : R[K];
};
export type ReshapeShape<S extends Shape, R extends Shape> =
  ValidDimensions<R> extends false
    ? never
    : number extends R["length"]
      ? number[]
      : -1 extends R[number]
        ? number extends R[number]
          ? ReplaceInference<R, number>
          : [Inferred<S, R>] extends [never]
            ? never
            : ReplaceInference<R, Inferred<S, R>>
        : string extends Product<S> | Product<R>
          ? ReplaceInference<R, number>
          : Product<S> extends Product<R>
            ? ReplaceInference<R, number>
            : never;
export type ReshapeCheck<S extends Shape, R extends Shape> = [
  ReshapeShape<S, R>,
] extends [never]
  ? never
  : unknown;

type RankTuple<S extends Shape> = S extends readonly [
  number,
  ...infer T extends Shape,
]
  ? [0, ...RankTuple<T>]
  : [];
type NormalizeAxis<
  S extends Shape,
  A extends number,
> = `${A}` extends `${bigint}`
  ? `${A}` extends `-${infer P extends number}`
    ? LengthCompare<`${S["length"]}`, `${P}`> extends true
      ? Subtract<`${S["length"]}`, `${P}`> extends `${infer N extends number}`
        ? N
        : never
      : never
    : A
  : never;
type At<
  S extends Shape,
  A extends number,
  I extends 0[] = [],
> = S extends readonly [infer H extends number, ...infer T extends Shape]
  ? I["length"] extends A
    ? H
    : At<T, A, [...I, 0]>
  : never;
type Permute<
  S extends Shape,
  A extends Shape,
  Seen extends number = never,
> = A extends readonly [infer H extends number, ...infer T extends Shape]
  ? NormalizeAxis<S, H> extends infer N extends number
    ? [N] extends [never]
      ? never
      : N extends Seen
        ? never
        : [At<S, N>] extends [never]
          ? never
          : Permute<S, T, Seen | N> extends infer Rest extends Shape
            ? [At<S, N>, ...Rest]
            : never
    : never
  : [];
type Reverse<S extends Shape> = S extends readonly [
  infer H extends number,
  ...infer T extends Shape,
]
  ? [...Reverse<T>, H]
  : [];
type DynamicDims<S extends Shape> = { -readonly [K in keyof S]: number };
export type TransposeShape<
  S extends Shape,
  A extends Shape | undefined = undefined,
> = number extends S["length"]
  ? number[]
  : A extends Shape
    ? number extends A["length"]
      ? DynamicDims<S>
      : number extends A[number]
        ? DynamicDims<S>
        : A["length"] extends RankTuple<S>["length"]
          ? Permute<S, A>
          : never
    : Reverse<S>;
export type TransposeCheck<S extends Shape, A extends Shape | undefined> = [
  TransposeShape<S, A>,
] extends [never]
  ? never
  : unknown;

/** Nonnegative integer literals reject statically; dynamic values validate at runtime. */
export type DerivativeOrder<M extends number> = number extends M
  ? unknown
  : `${M}` extends `-${string}`
    ? never
    : `${M}` extends `${bigint}`
      ? unknown
      : never;
type DerivativeLength<
  N extends number,
  M extends number,
  Asc extends boolean,
> = number extends N | M
  ? number
  : M extends 0
    ? N
    : LengthCompare<`${M}`, `${N}`> extends true
      ? Asc extends true
        ? N extends 0
          ? 0
          : 1
        : 0
      : Subtract<`${N}`, `${M}`> extends `${infer R extends number}`
        ? R
        : number;
type DerivativeAxis<
  S extends Shape,
  M extends number,
  A extends number,
  Asc extends boolean,
  I extends 0[] = [],
> = S extends readonly [infer H extends number, ...infer T extends Shape]
  ? I["length"] extends A
    ? [DerivativeLength<H, M, Asc>, ...T]
    : DerivativeAxis<T, M, A, Asc, [...I, 0]> extends infer R extends Shape
      ? [R] extends [never]
        ? never
        : [H, ...R]
      : never
  : never;
type DerivativeKnownShape<
  S extends Shape,
  M extends number,
  A extends number,
  Asc extends boolean,
> = number extends S["length"]
  ? number[]
  : number extends A
    ? DynamicDims<S>
    : NormalizeAxis<S, A> extends infer Axis extends number
      ? [Axis] extends [never]
        ? never
        : DerivativeAxis<S, M, Axis, Asc>
      : never;
export type DerivativeShape<
  S extends Shape,
  M extends number = 1,
  A extends number = 0,
  Asc extends boolean = true,
> = M extends M
  ? A extends A
    ? [DerivativeOrder<M>] extends [never]
      ? never
      : Asc extends true
        ? DerivativeKnownShape<S extends readonly [] ? [1] : S, M, A, Asc>
        : S extends readonly []
          ? never
          : number extends S["length"]
            ? number[]
            : S extends readonly [number]
              ? DerivativeKnownShape<S, M, 0, Asc>
              : never
    : never
  : never;

type ActiveDerivative<
  S extends Shape,
  M extends number,
  A extends number,
> = M extends 0
  ? false
  : number extends S["length"] | M | A
    ? boolean
    : NormalizeAxis<S, A> extends infer Axis extends number
      ? At<S, Axis> extends infer N extends number
        ? number extends N
          ? boolean
          : LengthCompare<`${M}`, `${N}`> extends true
            ? false
            : true
        : boolean
      : boolean;
export type DerivativeActive<
  S extends Shape,
  M extends number,
  A extends number,
> = M extends M
  ? A extends A
    ? ActiveDerivative<S extends readonly [] ? [1] : S, M, A>
    : never
  : never;
