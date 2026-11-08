// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
  mkdirSync,
  existsSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const adapter = fileURLToPath(
  new URL(
    "../../.agents/skills/before-and-after/scripts/adapters/blob.sh",
    import.meta.url,
  ),
);

function fixture(t) {
  const directory = mkdtempSync(join(tmpdir(), "blob-adapter-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const bin = join(directory, "bin");
  mkdirSync(bin);
  const curl = join(bin, "curl");
  writeFileSync(
    curl,
    '#!/bin/bash\nprintf "%s\\n" "$@" > "$BLOB_TEST_ARGS"\nprintf "%s" "$BLOB_TEST_RESPONSE"\n',
    { mode: 0o755 },
  );
  const input = join(directory, "input file.png");
  writeFileSync(input, "fixture only");
  const args = join(directory, "curl-args");
  const run = (response) =>
    spawnSync("bash", [adapter, input], {
      encoding: "utf8",
      env: {
        ...process.env,
        PATH: `${bin}${delimiter}${process.env.PATH}`,
        BLOB_UPLOAD_URL: "https://never-contacted.invalid/upload",
        BLOB_TEST_RESPONSE: response,
        BLOB_TEST_ARGS: args,
      },
    });
  return { directory, input, args, run };
}

test("Blob adapter decodes JSON URL escapes and preserves plain URLs", (t) => {
  const f = fixture(t);
  const expected = "https://example.invalid/image.png?a=1&b=2";
  for (const response of [
    expected,
    JSON.stringify({ url: expected }),
    '{"url":"https:\\/\\/example.invalid\\/image.png?a=1\\u0026b=2"}',
    '{"url":"https:\\u002f\\u002fexample.invalid/\\u0069mage.png?a=1\\u0026b=2"}',
  ]) {
    const result = f.run(response);
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout, `${expected}\n`);
    assert.deepEqual(readFileSync(f.args, "utf8").trimEnd().split("\n"), [
      "-s",
      "-X",
      "POST",
      "-F",
      `file=@${f.input}`,
      "https://never-contacted.invalid/upload",
    ]);
  }
});

test("Blob adapter rejects malformed, missing, non-string, and non-HTTP URLs", (t) => {
  const f = fixture(t);
  for (const response of [
    '{"url":',
    "{}",
    '{"url":null}',
    '{"url":12}',
    '{"url":[]}',
    '{"url":"file:///tmp/image.png"}',
  ]) {
    const result = f.run(response);
    assert.notEqual(result.status, 0, response);
    assert.equal(result.stdout, "", response);
  }
});

test("Blob adapter treats response text as data", (t) => {
  const f = fixture(t);
  const marker = join(f.directory, "executed");
  const url = `https://example.invalid/image.png?value=$(touch ${marker})`;
  const response = JSON.stringify({ url }).replace("$", "\\u0024");
  const result = f.run(response);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout, `${url}\n`);
  assert.equal(existsSync(marker), false);
});
