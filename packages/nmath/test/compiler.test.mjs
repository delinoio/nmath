// SPDX-License-Identifier: Apache-2.0
// End-to-end `.nts` workflow: compile with nmath-compiler, then run the output against the
// real package. Requires `pnpm build:compiler`.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import { build, check } from "../dist/compiler.js";

const packageRoot = fileURLToPath(new URL("../", import.meta.url));

function project(files) {
  const dir = mkdtempSync(join(tmpdir(), "nmath-nts-"));
  mkdirSync(join(dir, "node_modules", "@delinoio"), { recursive: true });
  symlinkSync(
    packageRoot,
    join(dir, "node_modules", "@delinoio", "nmath"),
    "dir",
  );
  mkdirSync(join(dir, "node_modules", "@types"));
  symlinkSync(
    join(packageRoot, "node_modules", "@types", "node"),
    join(dir, "node_modules", "@types", "node"),
    "dir",
  );
  writeFileSync(join(dir, "package.json"), JSON.stringify({ type: "module" }));
  writeFileSync(
    join(dir, "tsconfig.json"),
    JSON.stringify({
      compilerOptions: {
        target: "es2024",
        module: "nodenext",
        moduleResolution: "nodenext",
        strict: true,
        outDir: "out",
        rootDir: "src",
        types: ["node"],
        lib: ["es2024", "esnext.disposable"],
      },
      include: ["src"],
    }),
  );
  mkdirSync(join(dir, "src"));
  for (const [name, text] of Object.entries(files)) {
    writeFileSync(join(dir, "src", name), text);
  }
  return dir;
}

const MAIN = `import { createContext } from "@delinoio/nmath";
import { scale } from "./scale.nts";

const ctx = createContext({ maxConcurrentJobs: 1, maxQueueLength: 4, cpuMemoryBudget: 1n << 20n, gpuMemoryBudget: 0n });
const a = ctx.array([1, 2, 3], { dtype: "float64" });
const b = scale(a) + a ** 2;
a[0] = 10;
const result = await (b - -a).compute();
console.log(JSON.stringify(result.toArray()));
console.log(1 + 2);
await ctx.close();
`;

const SCALE = `import type { NMathArray } from "@delinoio/nmath";
export function scale(x: NMathArray<"float64", number[]>) {
  return x * 2;
}
`;

test(
  "check, build and run a multi-file .nts project",
  { skip: !existsSync(join(packageRoot, "native")) },
  async () => {
    const dir = project({ "main.nts": MAIN, "scale.nts": SCALE });
    try {
      const tsconfig = join(dir, "tsconfig.json");
      const checked = await check({ tsconfig });
      assert.deepEqual(checked.diagnostics, []);
      assert.equal(checked.ok, true);
      const built = await build({ tsconfig });
      assert.equal(built.ok, true, JSON.stringify(built.diagnostics));
      const mainJs = join(dir, "out", "main.js");
      assert.ok(built.outputFiles.some((f) => f.endsWith("main.js")));
      assert.ok(built.outputFiles.some((f) => f.endsWith("main.d.ts")));
      assert.match(readFileSync(mainJs, "utf8"), /from "\.\/scale\.js"/);
      const output = execFileSync(process.execPath, [mainJs], {
        cwd: dir,
        encoding: "utf8",
      });
      // b = 2a + a**2 with a = [1, 2, 3] frozen when the job starts after a[0] = 10:
      // a = [10, 2, 3] -> b = [120, 8, 15]; b - -a = [130, 10, 18]
      assert.deepEqual(output.trim().split("\n"), ["[130,10,18]", "3"]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  },
);

test(
  "type errors are reported at original .nts positions and keep old output",
  { skip: !existsSync(join(packageRoot, "native")) },
  async () => {
    const dir = project({ "main.nts": `const n: number = "x";\nexport {};\n` });
    try {
      mkdirSync(join(dir, "out"));
      writeFileSync(join(dir, "out", "main.js"), "old");
      const result = await build({ tsconfig: join(dir, "tsconfig.json") });
      assert.equal(result.ok, false);
      const diagnostic = result.diagnostics.find((d) =>
        d.file?.endsWith("main.nts"),
      );
      assert.equal(diagnostic.code, "NMATH_COMPILER_TYPE_ERROR");
      assert.match(diagnostic.file, /main\.nts$/);
      assert.equal(diagnostic.line, 1);
      assert.deepEqual(result.outputFiles, []);
      assert.equal(readFileSync(join(dir, "out", "main.js"), "utf8"), "old");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  },
);

test(
  "the CLI wrapper forwards arguments and exit codes",
  { skip: !existsSync(join(packageRoot, "native")) },
  () => {
    const bin = join(packageRoot, "bin", "nmath-compiler.mjs");
    assert.throws(
      () => execFileSync(process.execPath, [bin, "nope"], { stdio: "pipe" }),
      (error) => error.status === 2,
    );
    void pathToFileURL;
  },
);
