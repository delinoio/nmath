// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
  chmodSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const capture = fileURLToPath(
  new URL(
    "../../.agents/skills/before-and-after/scripts/capture.sh",
    import.meta.url,
  ),
);

function fixture(t) {
  const dir = mkdtempSync(join(tmpdir(), "before-and-after-capture-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const bin = join(dir, "bin");
  mkdirSync(bin);
  const log = join(dir, "calls.jsonl");
  const spy = join(bin, "agent-browser");
  writeFileSync(
    spy,
    '#!/usr/bin/env node\nconst args = process.argv.slice(2);\nrequire("node:fs").appendFileSync(process.env.CAPTURE_SPY_LOG, JSON.stringify(args) + "\\n");\nif (args[0] === process.env.CAPTURE_SPY_FAIL) process.exit(23);\n',
  );
  chmodSync(spy, 0o755);

  const url = "https://example.invalid/a path?name=hello world";
  const output = join(dir, "images with spaces", "whole page.png");
  const run = (args = [], fail = "") => {
    const result = spawnSync("bash", [capture, url, output, ...args], {
      encoding: "utf8",
      env: {
        ...process.env,
        PATH: [bin, process.env.PATH].join(delimiter),
        CAPTURE_SPY_LOG: log,
        CAPTURE_SPY_FAIL: fail,
      },
    });
    const calls = readFileSync(log, "utf8")
      .trimEnd()
      .split("\n")
      .map((line) => JSON.parse(line));
    return { result, calls };
  };
  return { url, output, run };
}

for (const options of [[], ["--full"]]) {
  test(`full-page capture dispatches supported commands (${options.join(" ") || "default"})`, (t) => {
    const f = fixture(t);
    const { result, calls } = f.run(options);
    assert.equal(result.status, 0, result.stderr);
    assert.deepEqual(calls, [
      ["open", f.url],
      ["wait", "--load", "networkidle"],
      ["set", "viewport", "1500", "1000"],
      ["screenshot", f.output, "--full"],
    ]);
  });
}

test("element capture keeps selector and output as separate arguments", (t) => {
  const f = fixture(t);
  const selector = "section.hero > .card";
  const { result, calls } = f.run(["--element", selector]);
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(calls, [
    ["open", f.url],
    ["wait", "--load", "networkidle"],
    ["screenshot", selector, f.output],
  ]);
});

test("a browser failure stops capture and retains its exit status", (t) => {
  const f = fixture(t);
  const { result, calls } = f.run([], "open");
  assert.equal(result.status, 23, result.stderr);
  assert.deepEqual(calls, [["open", f.url]]);
});
