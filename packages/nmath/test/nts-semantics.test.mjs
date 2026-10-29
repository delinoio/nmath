// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { SourceMap } from "node:module";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { build, check } from "../dist/compiler.js";

function fixture(files) {
  const root = mkdtempSync(join(tmpdir(), "nmath-semantic-"));
  mkdirSync(join(root, "node_modules", "@delinoio"), { recursive: true });
  symlinkSync(
    fileURLToPath(new URL("../", import.meta.url)),
    join(root, "node_modules", "@delinoio", "nmath"),
    "dir",
  );
  mkdirSync(join(root, "node_modules", "@types"));
  symlinkSync(
    fileURLToPath(new URL("../node_modules/@types/node", import.meta.url)),
    join(root, "node_modules", "@types", "node"),
    "dir",
  );
  writeFileSync(join(root, "package.json"), '{"type":"module"}');
  writeFileSync(
    join(root, "tsconfig.json"),
    JSON.stringify({
      compilerOptions: {
        strict: true,
        target: "es2024",
        module: "nodenext",
        moduleResolution: "nodenext",
        outDir: "out",
        types: ["node"],
        lib: ["es2024", "esnext.disposable"],
      },
      include: ["*.nts"],
    }),
  );
  for (const [name, text] of Object.entries(files))
    writeFileSync(join(root, name), text);
  return {
    root,
    tsconfig: join(root, "tsconfig.json"),
    close: () => rmSync(root, { recursive: true, force: true }),
  };
}
const context = `import { createContext, NMathArray, isNMath, type Expr } from "@delinoio/nmath";
const ctx = createContext({ maxConcurrentJobs: 1, maxQueueLength: 8, cpuMemoryBudget: 1048576n, gpuMemoryBudget: 0n });
`;

test("ordinary JS side effects, receivers and synchronous writes survive compilation", async () => {
  const source = `
class NMathArray { [index: number]: number; }
const fake = new NMathArray();
fake[0] = 42;
if (fake[0] !== 42) throw new Error("user-defined names must not overload");
const events: string[] = [];
const values = [1, 2];
function target() { events.push("target"); return values; }
function key() { events.push("key"); return 0; }
function rhs() { events.push("rhs"); return 7; }
function sync() { return target()[key()] = rhs(); }
const result = sync();
const methods = { value: 3, run() { events.push("method"); return this.value; } };
const method = methods["run"]();
const coercible = { [Symbol.toPrimitive]() { events.push("coerce"); return 4; } };
// Type assertions describe the runtime primitive coercion without changing it.
const sum = (coercible as unknown as number) + (events.push("right"), 2);
const keys = { [Symbol.toPrimitive]() { events.push("property-key"); return "x"; } };
const object: Record<string, number> = {};
object[keys as unknown as string] = (events.push("property-rhs"), 5);
const ordinary = JSON.stringify({ events, values, result, method, sum, object });
if (ordinary !== '{"events":["target","key","rhs","method","right","coerce","property-rhs","property-key"],"values":[7,2],"result":7,"method":3,"sum":6,"object":{"x":5}}') throw new Error(ordinary);
export { ordinary };
`;
  const f = fixture({ "main.nts": source });
  try {
    const result = await build(f);
    assert.equal(result.ok, true, JSON.stringify(result.diagnostics));
    const js = readFileSync(join(f.root, "out/main.js"), "utf8");
    assert.doesNotMatch(js, /__nmath|nts-runtime/);
    execFileSync(process.execPath, [join(f.root, "out/main.js")]);
  } finally {
    f.close();
  }
});

test("imports, aliases, inferred chains, narrowing and writes use precise NMath operations", async () => {
  const f = fixture({
    "producer.nts": `import type { NMathArray as ArrayType } from "@delinoio/nmath";
type Vector = ArrayType<"float64", readonly [2]>;
export function twice(value: Vector) { return value * 2; }
`,
    "main.nts":
      context +
      `import { twice as scale } from "./producer.nts";
const a = ctx.array([1, 2], { dtype: "float64", shape: [2] as const });
export const chained = scale(a) + -a;
const typed: Expr<"float64", [2]> = chained;
export const scalar = chained[0];
const scalarType: Expr<"float64", []> = scalar;
const events: string[] = [];
function target() { events.push("target"); return a; }
function key() { events.push("key"); return 0; }
function rhs() { events.push("rhs"); return 9 as const; }
async function write() {
  const assigned: 9 = (target()[key()] = rhs());
  events.push("after");
  if (assigned !== 9 || a.item(0) !== 9) throw new Error("write result");
}
await write();
if (events.join() !== "target,key,rhs,after") throw new Error(events.join());
function narrowed(value: NMathArray<"float64", readonly [2]> | number) {
  if (typeof value === "number") return value + 1;
  return value + 1;
}
const lazy = narrowed(a);
if (!isNMath(lazy)) throw new Error("narrowing");
const actual = await chained.compute();
const expected = await a.mul(2).sub(a).compute();
if (JSON.stringify(actual.toArray()) !== JSON.stringify(expected.toArray())) throw new Error("arithmetic");
if (await scalar.compute() !== 9) throw new Error("scalar read");
if (narrowed(2) !== 3) throw new Error("ordinary branch");
const rhsExpr = a[1];
const assignedExpr: typeof rhsExpr = (a[0] = rhsExpr);
if (assignedExpr !== rhsExpr || await assignedExpr.compute() !== 2) throw new Error("expression write result");
await ctx.close();
`,
  });
  try {
    const result = await build(f);
    assert.equal(result.ok, true, JSON.stringify(result.diagnostics));
    const declarations = readFileSync(join(f.root, "out/main.d.ts"), "utf8");
    assert.match(declarations, /chained:.*Expr<"float64", \[2\]>/);
    assert.match(declarations, /scalar:.*Expr<"float64", \[\]>/);
    const js = readFileSync(join(f.root, "out/main.js"), "utf8");
    const payload = JSON.parse(
      readFileSync(join(f.root, "out/main.js.map"), "utf8"),
    );
    const generated = js.slice(0, js.indexOf("chained, 0"));
    assert.ok(js.includes("chained, 0"));
    const generatedLines = generated.split("\n");
    const entry = new SourceMap(payload).findEntry(
      generatedLines.length - 1,
      generatedLines.at(-1).length,
    );
    const original = readFileSync(join(f.root, "main.nts"), "utf8");
    const originalLines = original
      .slice(0, original.indexOf("chained[0]"))
      .split("\n");
    assert.equal(entry.originalLine, originalLines.length - 1);
    assert.equal(entry.originalColumn, originalLines.at(-1).length);
    assert.equal(payload.sourcesContent[0], original);

    execFileSync(process.execPath, [join(f.root, "out/main.js")]);
  } finally {
    f.close();
  }
});

test("ambiguous unions and unsupported NMath operations report original locations", async () => {
  for (const [expression, code] of [
    ["value + 1", "NMATH_NTS_AMBIGUOUS_TYPE"],
    ["value[0]", "NMATH_NTS_AMBIGUOUS_TYPE"],
    ["value[0] = 2", "NMATH_NTS_AMBIGUOUS_TYPE"],
    ["array < array", "NMATH_NTS_UNSUPPORTED_SYNTAX"],
    ["array & array", "NMATH_NTS_UNSUPPORTED_SYNTAX"],
    ["array += array", "NMATH_NTS_UNSUPPORTED_SYNTAX"],
    ["array[0]++", "NMATH_NTS_UNSUPPORTED_SYNTAX"],
    ["array?.[0]", "NMATH_NTS_UNSUPPORTED_SYNTAX"],
    ["delete array[0]", "NMATH_NTS_UNSUPPORTED_SYNTAX"],
  ]) {
    const f = fixture({
      "main.nts": `import type { NMathArray } from "@delinoio/nmath";
declare let array: NMathArray<"float64", [2]>;
declare let value: typeof array | number[];
${expression};
`,
    });
    try {
      const result = await check(f);
      assert.equal(result.ok, false, expression);
      assert.ok(
        result.diagnostics.some((d) => d.code === code && d.line === 4),
        JSON.stringify(result.diagnostics),
      );
    } finally {
      f.close();
    }
  }
});

test("only NMath writes require async context", async () => {
  const f = fixture({
    "main.nts": `import type { NMathArray } from "@delinoio/nmath";
export function ordinary(a: number[]) { return a[0] = 4; }
export function native(a: NMathArray<"float64", [2]>) { return a[0] = 4; }
`,
  });
  try {
    const result = await check(f);
    assert.equal(result.ok, false);
    assert.ok(
      result.diagnostics.some(
        (d) => d.code === "NMATH_NTS_ASSIGN_REQUIRES_ASYNC" && d.line === 3,
      ),
      JSON.stringify(result.diagnostics),
    );
    assert.ok(result.diagnostics.every((d) => d.line !== 2));
  } finally {
    f.close();
  }
});

test("selector wrappers, all arithmetic operators and generic narrowing agree with explicit APIs", async () => {
  const f = fixture({
    "main.nts":
      context +
      `import { slice, indices, mask, sub, div } from "@delinoio/nmath";
const a = ctx.array([[1, 2], [3, 4]], { dtype: "float64", shape: [2, 2] as const });
export const row = a[[0, slice()]];
export const column = a[[slice(), 1]];
const gathered = a[indices([1, 0])];
const masked = a[mask([true, false])];
const expressions = [a + 2, 2 - a, a * 2, 2 / a, a % 2, a ** 2, -a, +a];
const explicit = [a.add(2), sub(2, a), a.mul(2), div(2, a), a.mod(2), a.pow(2), a.neg(), a.pos()];
for (let i = 0; i < expressions.length; i++) {
  const result = await expressions[i].compute();
  const expected = await explicit[i].compute();
  if (JSON.stringify(result.toArray()) !== JSON.stringify(expected.toArray())) throw new Error("operator " + i);
}
if (JSON.stringify((await row.compute()).toArray()) !== "[1,2]") throw new Error("row");
if (JSON.stringify((await column.compute()).toArray()) !== "[2,4]") throw new Error("column");
if (JSON.stringify((await gathered.compute()).toArray()) !== "[[3,4],[1,2]]") throw new Error("indices");
if (JSON.stringify((await masked.compute()).toArray()) !== "[[1,2]]") throw new Error("mask");
function generic<T extends NMathArray<"float64", readonly [2, 2]>>(v: T) { return v + 1; }
function instance(v: typeof a | number) { if (v instanceof NMathArray) return v + 1; return v + 1; }
function discriminated(v: { kind: "array", data: typeof a } | { kind: "number", data: number }) {
  if (v.kind === "array") return v.data + 1;
  return v.data + 1;
}
if (!isNMath(generic(a)) || !isNMath(instance(a)) || !isNMath(discriminated({ kind: "array", data: a }))) throw new Error("inference");
if (instance(2) !== 3 || discriminated({ kind: "number", data: 3 }) !== 4) throw new Error("ordinary narrowing");
await ctx.close();
`,
  });
  try {
    const result = await build(f);
    assert.equal(result.ok, true, JSON.stringify(result.diagnostics));
    execFileSync(process.execPath, [join(f.root, "out/main.js")]);
  } finally {
    f.close();
  }
});
