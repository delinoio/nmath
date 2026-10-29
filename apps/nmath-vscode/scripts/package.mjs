// SPDX-License-Identifier: Apache-2.0
import { execFileSync } from "node:child_process";
import { copyFileSync, readFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";
const root = fileURLToPath(new URL("../", import.meta.url));
const bundle = process.env.NMATH_LICENSE_BUNDLE;
if (!bundle || !readFileSync(bundle, "utf8").trim())
  throw new Error(
    "NMATH_LICENSE_BUNDLE must contain reviewed third-party terms and notices for the bundled extension dependencies.",
  );
mkdirSync(join(root, "dist"), { recursive: true });
copyFileSync(bundle, join(root, "dist/THIRD_PARTY_LICENSES.txt"));
await build({
  entryPoints: [join(root, "src/extension.ts")],
  outfile: join(root, "dist/extension.js"),
  bundle: true,
  platform: "node",
  format: "cjs",
  target: "node20",
  external: ["vscode"],
  sourcemap: false,
});
const require = createRequire(import.meta.url);
const vsce = join(
  dirname(require.resolve("@vscode/vsce/package.json")),
  "vsce",
);
execFileSync(
  process.execPath,
  [
    vsce,
    "package",
    "--no-dependencies",
    "--out",
    join(root, "dist/nmath-vscode.vsix"),
  ],
  { cwd: root, stdio: "inherit" },
);
