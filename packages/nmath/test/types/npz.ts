// SPDX-License-Identifier: Apache-2.0
import {
  createContext,
  type ComputableDType,
  type NMathArray,
} from "../../src/index.js";
import { savez, savez_compressed, loadNpz } from "../../src/numpy.js";
const ctx = createContext({
  maxConcurrentJobs: 1,
  maxQueueLength: 1,
  cpuMemoryBudget: 1000000n,
  gpuMemoryBudget: 0n,
});
const a = ctx.zeros([2] as const, "complex64");
void savez(ctx, "archive", { positional: [a], named: new Map([["named", a]]) });
void ctx.saveNpz("archive", new Map([["a", a]]));
void savez_compressed(
  ctx,
  "archive",
  {},
  { overwrite: true, signal: new AbortController().signal },
);
void loadNpz(ctx, "archive.npz").then((result) => {
  const member: NMathArray<ComputableDType, number[]> | undefined =
    result.get("named");
  return member;
});
// @ts-expect-error Named entries use a Map, not object properties.
void savez(ctx, "archive", { named: { a } });
// @ts-expect-error Positional inputs must be arrays.
void savez(ctx, "archive", { positional: [42] });
