// SPDX-License-Identifier: Apache-2.0
import { createContext } from "../../src/index.js";
import { DataFrame as PandasFrame } from "../../src/pandas.js";
import {
  DataFrame as PolarsFrame,
  Series,
  type LazyFrame,
} from "../../src/polars.js";

const c = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 2,
  cpuMemoryBudget: 1048576n,
  gpuMemoryBudget: 0n,
});

const pandasFrame = new PandasFrame({
  key: c.zeros([3], "int64"),
  sub: c.zeros([3], "int64"),
  value: c.zeros([3], "int64"),
});
const uniquePivot: Promise<PandasFrame> = pandasFrame.pivot(
  "key",
  "sub",
  "value",
);
const summedPivot: Promise<PandasFrame> = pandasFrame.pivotTable(
  "key",
  "sub",
  "value",
  { aggfunc: "sum" },
);
const meanPivot: Promise<PandasFrame> = pandasFrame.pivotTable(
  "key",
  "sub",
  "value",
);

const polarsFrame = new PolarsFrame([
  new Series("key", c.zeros([3], "int64")),
  new Series("sub", c.zeros([3], "int64")),
  new Series("value", c.zeros([3], "int64")),
]);
const lazy: LazyFrame = polarsFrame.pivot("sub", "key", "value", "sum");
const collected: Promise<PolarsFrame> = lazy.collect();
const uniqueLazy: LazyFrame = polarsFrame.pivot("sub", "key", "value");

// @ts-expect-error pivotTable aggregation must be a supported aggregate
pandasFrame.pivotTable("key", "sub", "value", { aggfunc: "median" });
// @ts-expect-error pivot aggregation must be a supported aggregate
polarsFrame.pivot("sub", "key", "value", "median");

void uniquePivot;
void summedPivot;
void meanPivot;
void lazy;
void collected;
void uniqueLazy;
