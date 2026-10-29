// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";

import resolve from "../dist/resolve.js";

const { resolveCompilerPath, startFailureMessage } = resolve;

test("an explicit setting wins", () => {
  assert.equal(
    resolveCompilerPath({
      setting: " /opt/nmath-compiler ",
      workspaceFolders: ["/w"],
      platform: "linux",
      exists: () => true,
    }),
    "/opt/nmath-compiler",
  );
});

test("the first workspace-local binary is used next", () => {
  const local = join("/b", "node_modules", ".bin", "nmath-compiler");
  assert.equal(
    resolveCompilerPath({
      setting: "",
      workspaceFolders: ["/a", "/b"],
      platform: "darwin",
      exists: (f) => f === local,
    }),
    local,
  );
  const win = join("/a", "node_modules", ".bin", "nmath-compiler.cmd");
  assert.equal(
    resolveCompilerPath({
      setting: "",
      workspaceFolders: ["/a"],
      platform: "win32",
      exists: (f) => f === win,
    }),
    win,
  );
});

test("PATH is the fallback", () => {
  assert.equal(
    resolveCompilerPath({
      setting: "",
      workspaceFolders: ["/a"],
      platform: "linux",
      exists: () => false,
    }),
    "nmath-compiler",
  );
});

test("start failures explain recovery in English", () => {
  const message = startFailureMessage("nmath-compiler", "spawn ENOENT");
  assert.match(message, /could not start with 'nmath-compiler': spawn ENOENT/);
  assert.match(message, /nmath\.compilerPath/);
});

test("the manifest associates .nts files and uses standard UI only", () => {
  const manifest = JSON.parse(
    readFileSync(new URL("../package.json", import.meta.url), "utf8"),
  );
  const language = manifest.contributes.languages.find((l) => l.id === "nts");
  assert.deepEqual(language.extensions, [".nts"]);
  assert.deepEqual(manifest.activationEvents, ["onLanguage:nts"]);
  assert.equal(manifest.contributes.views, undefined);
  assert.equal(manifest.contributes.viewsContainers, undefined);
  assert.equal(manifest.capabilities.untrustedWorkspaces.supported, false);
  assert.equal(manifest.license, "Apache-2.0");
});
