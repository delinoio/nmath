// SPDX-License-Identifier: Apache-2.0

/** Stable NMath error codes. */
export const ERROR_CODES = [
  "NMATH_CONTEXT_INVALID_OPTIONS",
  "NMATH_CONTEXT_MISMATCH",
  "NMATH_CONTEXT_CLOSED",
  "NMATH_DISPOSED",
  "NMATH_SCHED_QUEUE_FULL",
  "NMATH_CANCELLED",
  "NMATH_MEMORY_BUDGET_EXCEEDED",
  "NMATH_SHAPE_BROADCAST",
  "NMATH_SHAPE_MISMATCH",
  "NMATH_LINALG_SINGULAR",
  "NMATH_DTYPE_UNSUPPORTED",
  "NMATH_INDEX_OUT_OF_BOUNDS",
  "NMATH_INDEX_INVALID",
  "NMATH_GPU_UNAVAILABLE",
  "NMATH_GPU_UNSUPPORTED_OPERATION",
  "NMATH_GPU_DEVICE_LOST",
  "NMATH_GPU_OUT_OF_MEMORY",
  "NMATH_IO_EXISTS",
  "NMATH_IO_FORMAT",
  "NMATH_IO_MODIFIED",
  "NMATH_IO_FAILED",
  "NMATH_NATIVE_LOAD",
] as const;

export type ErrorCode = (typeof ERROR_CODES)[number];

/** An NMath failure with a stable `code` and an English message with recovery guidance. */
export class NMathError extends Error {
  readonly code: ErrorCode;

  constructor(code: ErrorCode, message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "NMathError";
    this.code = code;
  }
}

const CODE_PREFIX = /^(NMATH_[A-Z0-9_]+): /;

/** Converts an error thrown by the native binding into an `NMathError`. */
export function translateError(error: unknown): unknown {
  if (error instanceof NMathError || !(error instanceof Error)) {
    return error;
  }
  const match = CODE_PREFIX.exec(error.message);
  if (match && (ERROR_CODES as readonly string[]).includes(match[1])) {
    return new NMathError(match[1] as ErrorCode, error.message, {
      cause: error,
    });
  }
  return error;
}

/** Runs `fn` and rethrows native errors as `NMathError`. */
export function native<T>(fn: () => T): T {
  try {
    return fn();
  } catch (error) {
    throw translateError(error);
  }
}

/** Awaits a native promise and rethrows native errors as `NMathError`. */
export async function nativeAsync<T>(promise: () => Promise<T>): Promise<T> {
  try {
    return await native(promise);
  } catch (error) {
    throw translateError(error);
  }
}
