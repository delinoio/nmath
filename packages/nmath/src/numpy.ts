// SPDX-License-Identifier: Apache-2.0
// NumPy-style entry point. Functions take an explicit context where NumPy would use global
// state. Names follow NumPy in camelCase. The compatibility inventory in README.md lists
// which NumPy 2.5.4 APIs exist.

import {
  type ArrayOptions,
  Lazy,
  type NMathArray,
  type NMathContext,
  type NMathGenerator,
  type Operand,
} from "./core.js";
import { add, div, mod, mul, pow, sub } from "./index.js";
import type {
  ComputableDType,
  ScalarOf,
  Shape,
  TypedArrayOf,
} from "./types.js";

export { indices, mask, slice } from "./core.js";

type Nested<T> = T | readonly Nested<T>[];

export function array<
  D extends ComputableDType,
  const S extends Shape = number[],
>(
  context: NMathContext,
  data: TypedArrayOf<D> | Nested<ScalarOf<D>>,
  options: ArrayOptions<D, S>,
): NMathArray<D, S> {
  return context.array(data, options);
}

export function zeros<D extends ComputableDType, const S extends Shape>(
  context: NMathContext,
  shape: S,
  dtype: D,
): NMathArray<D, S> {
  return context.zeros(shape, dtype);
}

export function ones<D extends ComputableDType, const S extends Shape>(
  context: NMathContext,
  shape: S,
  dtype: D,
): NMathArray<D, S> {
  const one = (
    dtype === "int64" || dtype === "uint64" ? 1n : dtype === "bool" ? true : 1
  ) as ScalarOf<D>;
  return context.full(shape, one, dtype);
}

export function full<D extends ComputableDType, const S extends Shape>(
  context: NMathContext,
  shape: S,
  value: ScalarOf<D>,
  dtype: D,
): NMathArray<D, S> {
  return context.full(shape, value, dtype);
}

/** `np.arange(start, stop, step)` for numeric dtypes other than 64-bit integers. */
export function arange<D extends "int32" | "uint32" | "float32" | "float64">(
  context: NMathContext,
  start: number,
  stop: number,
  step = 1,
  dtype: D = "float64" as D,
): NMathArray<D, [number]> {
  if (step === 0) {
    throw new RangeError("arange step must not be zero.");
  }
  const length = Math.max(0, Math.ceil((stop - start) / step));
  context.ensureCpuBudget(length, dtype);
  const values = Array.from({ length }, (_, i) => start + i * step);
  return context.array(values as never, { dtype, shape: [length] as [number] });
}

export const subtract = sub;
export const multiply = mul;
export const divide = div;
export const trueDivide = div;
export const remainder = mod;
export const power = pow;
export { add, mod };

export function negative<T extends Lazy<ComputableDType, Shape>>(x: T) {
  return x.neg();
}
export function positive<T extends Lazy<ComputableDType, Shape>>(x: T) {
  return x.pos();
}
export function sum(x: Lazy<ComputableDType, Shape>, axis?: number) {
  return x.sum(axis);
}
export function mean(x: Lazy<ComputableDType, Shape>, axis?: number) {
  return x.mean(axis);
}
export function min(x: Lazy<ComputableDType, Shape>, axis?: number) {
  return x.min(axis);
}
export function max(x: Lazy<ComputableDType, Shape>, axis?: number) {
  return x.max(axis);
}
export function matmul(
  a: Lazy<ComputableDType, Shape>,
  b: Lazy<ComputableDType, Shape>,
) {
  return a.matmul(b as never);
}

/** `np.save`: writes NPY without overwriting unless requested. */
export function save(
  path: string,
  x: NMathArray<ComputableDType, Shape>,
  options: { overwrite?: boolean; signal?: AbortSignal } = {},
): Promise<void> {
  return x.save(path, options);
}

/** `np.load` for NPY files. */
export function load(
  context: NMathContext,
  path: string,
  options: { signal?: AbortSignal } = {},
) {
  return context.load(path, options);
}

export const random = {
  /** `np.random.default_rng(seed)`. */
  defaultRng(context: NMathContext, seed: number | bigint): NMathGenerator {
    return context.generator(seed);
  },
};

export type { Operand };

/** Explicit JS adaptation of NumPy positional and keyword archive members. */
export interface NpzInputs {
  positional?: readonly NMathArray<any, any>[];
  named?: ReadonlyMap<string, NMathArray<any, any>>;
}

function saveArchive(
  context: NMathContext,
  path: string,
  inputs: NpzInputs,
  compressed: boolean,
  options: { overwrite?: boolean; signal?: AbortSignal },
): Promise<void> {
  const arrays = new Map(inputs.named);
  for (const [i, array] of (inputs.positional ?? []).entries()) {
    const name = `arr_${i}`;
    if (arrays.has(name)) {
      return Promise.reject(
        new TypeError(`Duplicate NPZ member name: ${name}.`),
      );
    }
    arrays.set(name, array);
  }
  return context.saveNpz(path.endsWith(".npz") ? path : `${path}.npz`, arrays, {
    ...options,
    compressed,
  });
}

export function savez(
  context: NMathContext,
  path: string,
  inputs: NpzInputs = {},
  options: { overwrite?: boolean; signal?: AbortSignal } = {},
): Promise<void> {
  return saveArchive(context, path, inputs, false, options);
}

export function savez_compressed(
  context: NMathContext,
  path: string,
  inputs: NpzInputs = {},
  options: { overwrite?: boolean; signal?: AbortSignal } = {},
): Promise<void> {
  return saveArchive(context, path, inputs, true, options);
}

/** Explicit NPZ load; preserves names safely in a Map and scalars as arrays. */
export function loadNpz(
  context: NMathContext,
  path: string,
  options: { signal?: AbortSignal } = {},
) {
  return context.loadNpz(path, options);
}
