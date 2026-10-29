// SPDX-License-Identifier: Apache-2.0
// Development-only differential runner. It reads fixtures and never launches Python.
import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { createContext, indices, mask, slice } from "../../dist/index.js";
import * as np from "../../dist/numpy.js";

const fixtureRoot = new URL("../../compatibility/", import.meta.url);
const read = (path) =>
  JSON.parse(readFileSync(new URL(path, fixtureRoot), "utf8"));
const criteria = read("criteria.json");
const context = () =>
  createContext({
    maxConcurrentJobs: 1,
    maxQueueLength: 16,
    cpuMemoryBudget: 1n << 28n,
    gpuMemoryBudget: 0n,
  });

function encode(value, dtype) {
  if (dtype === "bool") return Boolean(value);
  if (/^u?int/.test(dtype)) return String(value);
  const buffer = Buffer.alloc(dtype === "float32" ? 4 : 8);
  if (dtype === "float32") buffer.writeFloatBE(value);
  else buffer.writeDoubleBE(value);
  return buffer.toString("hex");
}

function input(ctx, spec) {
  if (spec.kind)
    return spec.kind === "bigint" ? BigInt(spec.scalar) : spec.scalar;
  const convert = (x) =>
    Array.isArray(x)
      ? x.map(convert)
      : /^(u?int64)$/.test(spec.dtype)
        ? BigInt(x)
        : spec.dtype === "bool"
          ? Boolean(x)
          : Number(x);
  return ctx.array(convert(spec.data), { dtype: spec.dtype });
}

function compare(expr, value, expected) {
  assert.equal(expr.dtype, expected.dtype, "dtype");
  assert.deepEqual([...expr.shape], expected.shape, "shape");
  const flat = expected.shape.length ? value.toArray().flat(Infinity) : [value];
  assert.equal(flat.length, expected.values.length, "element count");
  flat.forEach((x, i) => {
    const actual = encode(x, expected.dtype);
    const reference = expected.values[i];
    // The contract requires missing-value semantics, not a platform-specific NaN payload.
    if (expected.dtype.startsWith("float") && Number.isNaN(x)) {
      const b = Buffer.from(reference, "hex");
      assert.ok(
        Number.isNaN(b.length === 4 ? b.readFloatBE() : b.readDoubleBE()),
        `NaN at ${i}`,
      );
    } else assert.equal(actual, reference, `element ${i}`);
  });
}

const outcomes = [];
for (const fixture of read("fixtures/primitive.json").cases) {
  const ctx = context();
  try {
    let expression, value, caught;
    try {
      const args = fixture.inputs.map((spec) => input(ctx, spec));
      if (fixture.operation === "index") {
        const selectors = fixture.selectors.map((s) =>
          typeof s === "number"
            ? s
            : s.slice
              ? slice(...s.slice.map((x) => x ?? undefined))
              : s.indices
                ? indices(s.indices)
                : mask(s.mask),
        );
        expression = args[0].get(...selectors);
      } else expression = np[fixture.operation](...args);
      value = await expression.compute();
      if (
        fixture.sharesStorage !== undefined &&
        fixture.sharesStorage !== null
      ) {
        assert.equal(
          value.sharesStorage(args[0]),
          fixture.sharesStorage,
          "view/copy",
        );
      }
    } catch (error) {
      caught = error;
    }
    if (fixture.expectedError) {
      // An arbitrary loader/assertion error must not satisfy an oracle rejection case.
      assert.ok(
        caught &&
          caught.code ===
            {
              UFuncTypeError: "NMATH_DTYPE_UNSUPPORTED",
              TypeError: "NMATH_DTYPE_UNSUPPORTED",
              IndexError: "NMATH_INDEX_OUT_OF_BOUNDS",
              ValueError: "NMATH_SHAPE_MISMATCH",
            }[fixture.expectedError],
        `expected categorized native rejection (${fixture.expectedError})`,
      );
    } else {
      if (caught) throw caught;
      compare(expression, value, fixture.expected);
    }
    outcomes.push({
      id: fixture.id,
      outcome: "pass",
      acceptance: criteria[fixture.criterion].acceptance,
    });
  } catch (error) {
    outcomes.push({
      id: fixture.id,
      outcome: "fail",
      acceptance: criteria[fixture.criterion].acceptance,
      error: error.message,
    });
  } finally {
    await ctx.close();
  }
}
for (const fixture of read("fixtures/mutation.json").cases) {
  const ctx = context();
  try {
    const base = ctx.array(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { dtype: "int32" },
    );
    const expression = base.get(
      fixture.kind === "basic" ? slice() : indices([1, 0]),
      1,
    );
    const selected = await expression.compute();
    await selected.set([0], 99n);
    compare(base, base, fixture.baseAfter);
    compare(selected, selected, fixture.selectedAfter);
    outcomes.push({ id: fixture.id, outcome: "pass", acceptance: "required" });
  } catch (error) {
    outcomes.push({
      id: fixture.id,
      outcome: "fail",
      acceptance: "required",
      error: error.message,
    });
  } finally {
    await ctx.close();
  }
}
for (const fixture of read("fixtures/random.json").cases) {
  const ctx = context();
  try {
    const rng = ctx.generator(BigInt(fixture.seed));
    for (const call of fixture.calls) {
      const expression = rng.random(call.shape);
      // Cancellation must not consume state before the successful call.
      const controller = new AbortController();
      controller.abort();
      await assert.rejects(expression.compute({ signal: controller.signal }), {
        code: "NMATH_CANCELLED",
      });
      compare(expression, await expression.compute(), call.expected);
    }
    const repeat = ctx.generator(BigInt(fixture.seed));
    const expression = repeat.random([3]);
    compare(expression, await expression.compute(), fixture.calls[0].expected);
    // A fresh expression and a reused expression must consume the same next values.
    const fresh = ctx.generator(BigInt(fixture.seed));
    await fresh.random([3]).compute();
    assert.deepEqual(
      (await expression.compute()).toArray(),
      (await fresh.random([3]).compute()).toArray(),
    );
    outcomes.push({ id: fixture.id, outcome: "pass", acceptance: "required" });
  } catch (error) {
    outcomes.push({
      id: fixture.id,
      outcome: "fail",
      acceptance: "required",
      error: error.message,
    });
  } finally {
    await ctx.close();
  }
}
const report = {
  schemaVersion: 1,
  sourceRevision: execFileSync("git", ["rev-parse", "HEAD"], {
    encoding: "utf8",
  }).trim(),
  node: process.version,
  platform: process.platform,
  arch: process.arch,
  limitations: [
    "No GPU validation",
    "Raw RNG state is not exposed by NMath",
    "Floating diagnostic success is not numerical acceptance",
  ],
  passed: outcomes.filter((x) => x.outcome === "pass").length,
  failed: outcomes.filter((x) => x.outcome === "fail").length,
  outcomes,
};
const output = process.argv[2];
if (output) writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
console.log(
  JSON.stringify({
    passed: report.passed,
    failed: report.failed,
    report: output ?? null,
  }),
);
if (report.failed) process.exitCode = 1;
