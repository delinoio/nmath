// SPDX-License-Identifier: Apache-2.0
import { createContext } from "../../src/index.js";
import { read_csv, read_json } from "../../src/pandas.js";
import * as pl from "../../src/polars.js";
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
// @ts-expect-error Text table files do not accept complex column schemas.
ctx.readText("input.csv", "csv", { schema: new Map([["x", "complex128"]]) });
// @ts-expect-error Formats are an explicit finite union.
ctx.readText("input.csv", "xml");
