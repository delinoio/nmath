// SPDX-License-Identifier: Apache-2.0
import { createContext } from "../dist/index.js";

export function context(overrides = {}) {
  return createContext({
    maxConcurrentJobs: 2,
    maxQueueLength: 8,
    cpuMemoryBudget: 1n << 30n,
    gpuMemoryBudget: 0n,
    ...overrides,
  });
}
