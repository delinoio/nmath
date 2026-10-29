// SPDX-License-Identifier: Apache-2.0
import { createContext } from "../../src/index.js";
import { read_parquet, read_feather, DataFrame } from "../../src/pandas.js";
import { read_ipc } from "../../src/polars.js";
const ctx = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 100000000n,
  gpuMemoryBudget: 0n,
});
const column = ctx.array([1n], { dtype: "uint64" });
const columns = new Map([["value", column]]);
const write: Promise<void> = ctx.writeColumnar("file", "parquet", columns);
const pandas: Promise<DataFrame> = read_parquet(ctx, "file");
void write;
void pandas;
void read_feather(ctx, "file");
void read_ipc(ctx, "file");
// @ts-expect-error formats are closed identifiers
ctx.readColumnar("file", "csv");
// @ts-expect-error overwrite must be boolean
ctx.writeColumnar("file", "arrow", columns, new Map(), { overwrite: "yes" });
