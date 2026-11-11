// SPDX-License-Identifier: Apache-2.0
import { createContext } from "../../src/index.js";
import { read_csv, read_json } from "../../src/pandas.js";
import * as pl from "../../src/polars.js";
import type { TextDType, TextTable } from "../../src/core.js";

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;
declare function exact<T extends true>(): void;

const supportedDTypes = [
  "bool",
  "int8",
  "int16",
  "int32",
  "int64",
  "uint8",
  "uint16",
  "uint32",
  "uint64",
  "float32",
  "float64",
] as const;
exact<Equal<TextDType, (typeof supportedDTypes)[number]>>();
declare const textTable: TextTable;
exact<
  Equal<
    NonNullable<ReturnType<typeof textTable.columns.get>>["dtype"],
    TextDType
  >
>();
const ctx = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 1,
  cpuMemoryBudget: 1000000n,
  gpuMemoryBudget: 0n,
});
const schema = new Map<string, "uint64" | "bool">([
  ["id", "uint64"],
  ["ok", "bool"],
]);
const table = await ctx.readText("input.csv", "csv", { schema });
await ctx.writeText("out.json", "json", table.columns, table.validity, {
  overwrite: true,
});
const frame = await read_csv(ctx, "input.csv", { schema });
await frame.toJson(ctx, "out.json");
await (await read_json(ctx, "out.json", { schema })).toCsv(ctx, "again.csv");
await (
  await pl.read_csv(ctx, "input.csv", { schema })
).writeJson(ctx, "polars.json");
for (const dtype of supportedDTypes) {
  const options = { schema: new Map([["value", dtype]]) };
  ctx.readText("input.csv", "csv", options);
  read_csv(ctx, "input.csv", options);
  read_json(ctx, "input.json", options);
  pl.read_csv(ctx, "input.csv", options);
  pl.read_json(ctx, "input.json", options);
}

const strOptions = { schema: new Map<string, "str">([["value", "str"]]) };
// @ts-expect-error Text readers do not accept string schemas.
ctx.readText("input.csv", "csv", strOptions);
// @ts-expect-error The pandas CSV adapter shares the text reader schema.
read_csv(ctx, "input.csv", strOptions);
// @ts-expect-error The pandas JSON adapter shares the text reader schema.
read_json(ctx, "input.json", strOptions);
// @ts-expect-error The Polars CSV adapter shares the text reader schema.
pl.read_csv(ctx, "input.csv", strOptions);
// @ts-expect-error The Polars JSON adapter shares the text reader schema.
pl.read_json(ctx, "input.json", strOptions);

const datetimeOptions = {
  schema: new Map<string, "datetime64">([["value", "datetime64"]]),
};
// @ts-expect-error Text readers do not accept datetime schemas.
ctx.readText("input.csv", "csv", datetimeOptions);
// @ts-expect-error The pandas CSV adapter shares the text reader schema.
read_csv(ctx, "input.csv", datetimeOptions);
// @ts-expect-error The pandas JSON adapter shares the text reader schema.
read_json(ctx, "input.json", datetimeOptions);
// @ts-expect-error The Polars CSV adapter shares the text reader schema.
pl.read_csv(ctx, "input.csv", datetimeOptions);
// @ts-expect-error The Polars JSON adapter shares the text reader schema.
pl.read_json(ctx, "input.json", datetimeOptions);

const timedeltaOptions = {
  schema: new Map<string, "timedelta64">([["value", "timedelta64"]]),
};
// @ts-expect-error Text readers do not accept timedelta schemas.
ctx.readText("input.csv", "csv", timedeltaOptions);
// @ts-expect-error The pandas CSV adapter shares the text reader schema.
read_csv(ctx, "input.csv", timedeltaOptions);
// @ts-expect-error The pandas JSON adapter shares the text reader schema.
read_json(ctx, "input.json", timedeltaOptions);
// @ts-expect-error The Polars CSV adapter shares the text reader schema.
pl.read_csv(ctx, "input.csv", timedeltaOptions);
// @ts-expect-error The Polars JSON adapter shares the text reader schema.
pl.read_json(ctx, "input.json", timedeltaOptions);

const float16Options = {
  schema: new Map<string, "float16">([["value", "float16"]]),
};
// @ts-expect-error Text readers do not accept float16 schemas.
ctx.readText("input.csv", "csv", float16Options);
// @ts-expect-error The pandas CSV adapter shares the text reader schema.
read_csv(ctx, "input.csv", float16Options);
// @ts-expect-error The pandas JSON adapter shares the text reader schema.
read_json(ctx, "input.json", float16Options);
// @ts-expect-error The Polars CSV adapter shares the text reader schema.
pl.read_csv(ctx, "input.csv", float16Options);
// @ts-expect-error The Polars JSON adapter shares the text reader schema.
pl.read_json(ctx, "input.json", float16Options);
// @ts-expect-error Text table files do not accept complex column schemas.
ctx.readText("input.csv", "csv", { schema: new Map([["x", "complex128"]]) });
// @ts-expect-error Formats are an explicit finite union.
ctx.readText("input.csv", "xml");
