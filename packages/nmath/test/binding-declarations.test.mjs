// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("src/native-binding.generated.ts matches the generated binding declarations", () => {
  const generated = readFileSync(
    new URL("../native/native.d.ts", import.meta.url),
    "utf8",
  );
  const copy = readFileSync(
    new URL("../src/native-binding.generated.ts", import.meta.url),
    "utf8",
  );
  // Compare declarations only: napi-rs rewraps doc comments with Rust formatting.
  const body = (text) =>
    text
      .slice(text.indexOf("export declare class NativeArray"))
      .replace(/\/\*\*[\s\S]*?\*\//g, "")
      .replace(/\s+/g, " ")
      .trim();
  assert.equal(body(copy), body(generated));
});
