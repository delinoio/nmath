// SPDX-License-Identifier: Apache-2.0
import { Expr, validatedShape, type NMathContext } from "./core.js";
import { native } from "./errors.js";
import { binding } from "./native.js";
import type { NativeGenerator } from "./native-binding.generated.js";
import type { Shape } from "./types.js";

type FloatDType = "float32" | "float64";
type IntegerDType =
  | "bool"
  | "int8"
  | "int16"
  | "int32"
  | "int64"
  | "uint8"
  | "uint16"
  | "uint32"
  | "uint64";
type RandomDType = FloatDType | IntegerDType;

function integer(value: number | bigint): string {
  if (typeof value !== "bigint" && !Number.isSafeInteger(value)) {
    throw new TypeError(
      "Integer bounds must be safe integers or bigint values.",
    );
  }
  return String(value);
}

function real(value: number): string {
  if (typeof value !== "number")
    throw new TypeError("Distribution parameters must be numbers.");
  return Object.is(value, -0) ? "-0" : String(value);
}

/** PCG64 with scalar distribution parameters and lazy, independent executions. */
export class NMathGenerator {
  readonly #generator: NativeGenerator;
  readonly context: NMathContext;

  /** @internal */
  constructor(generator: NativeGenerator, context: NMathContext) {
    this.#generator = generator;
    this.context = context;
  }

  private expression<D extends RandomDType, S extends Shape>(
    operation: string,
    shape: S,
    dtype: D,
    first = "0",
    second = "0",
    endpoint = false,
  ): Expr<D, S> {
    const checkedShape = validatedShape(shape);
    return new Expr(
      native(() =>
        binding.NativeExpr.randomDistribution(
          this.#generator,
          checkedShape,
          operation,
          dtype,
          first,
          second,
          endpoint,
        ),
      ),
      this.context,
    );
  }

  random<const S extends Shape, D extends FloatDType = "float64">(
    shape: S,
    dtype: D = "float64" as D,
  ): Expr<D, S> {
    return this.expression("random", shape, dtype);
  }

  integers<const S extends Shape, D extends IntegerDType = "int64">(
    low: number | bigint,
    high: number | bigint | undefined,
    shape: S,
    options: { dtype?: D; endpoint?: boolean } = {},
  ): Expr<D, S> {
    if (high === undefined) {
      high = low;
      low = 0;
    }
    return this.expression(
      "integers",
      shape,
      options.dtype ?? ("int64" as D),
      integer(low),
      integer(high),
      options.endpoint ?? false,
    );
  }

  uniform<const S extends Shape>(
    low: number,
    high: number,
    shape: S,
  ): Expr<"float64", S> {
    return this.expression("uniform", shape, "float64", real(low), real(high));
  }

  standard_normal<const S extends Shape, D extends FloatDType = "float64">(
    shape: S,
    dtype: D = "float64" as D,
  ): Expr<D, S> {
    return this.expression("standard_normal", shape, dtype);
  }

  normal<const S extends Shape>(
    loc: number,
    scale: number,
    shape: S,
  ): Expr<"float64", S> {
    return this.expression("normal", shape, "float64", real(loc), real(scale));
  }

  exponential<const S extends Shape>(
    scale: number,
    shape: S,
  ): Expr<"float64", S> {
    return this.expression("exponential", shape, "float64", real(scale));
  }
}
