// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
  chmodSync,
  copyFileSync,
  existsSync,
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

const root = fileURLToPath(new URL("../..", import.meta.url));
const scriptDir = join(root, ".agents/skills/before-and-after/scripts");
const gistAdapter = join(scriptDir, "adapters/gist.sh");
const uploadHelper = join(scriptDir, "upload-and-copy.sh");

function fixture(t) {
  const cwd = mkdtempSync(join(tmpdir(), "gist-adapter-"));
  t.after(() => rmSync(cwd, { recursive: true, force: true }));
  const scripts = join(cwd, "scripts");
  mkdirSync(join(scripts, "adapters"), { recursive: true });
  const fixtureAdapter = join(scripts, "adapters/gist.sh");
  const fixtureUploadHelper = join(scripts, "upload-and-copy.sh");
  copyFileSync(gistAdapter, fixtureAdapter);
  copyFileSync(uploadHelper, fixtureUploadHelper);
  chmodSync(fixtureAdapter, 0o755);
  chmodSync(fixtureUploadHelper, 0o755);
  const bin = join(cwd, "bin");
  mkdirSync(bin);
  const gh = join(bin, "gh");
  writeFileSync(
    gh,
    `#!/bin/bash
printf '%s\\n' "$1" >> "$GH_CALLS"
case "$1" in
  gist)
    printf '%s\\0' "$@" > "$GH_CREATE_ARGS"
    if [[ "$GH_MODE" == "create-fail" ]]; then
      echo "creation failed" >&2
      exit 7
    fi
    echo "https://gist.github.com/test/abc123"
    ;;
  api)
    printf '%s\\0' "$@" > "$GH_API_ARGS"
    if [[ "$GH_MODE" == "lookup-fail" ]]; then
      echo "lookup failed" >&2
      exit 9
    fi
    echo "https://gist.githubusercontent.com/test/abc123/raw/sample"
    ;;
  *) exit 2 ;;
esac
`,
  );
  chmodSync(gh, 0o755);
  const pnpm = join(bin, "pnpm");
  writeFileSync(
    pnpm,
    `#!/bin/bash
printf '%s\\n' "$@" >> "$PNPM_CALLS"
exit 127
`,
  );
  chmodSync(pnpm, 0o755);
  const paths = {
    calls: join(cwd, "gh-calls"),
    createArgs: join(cwd, "gh-create-args"),
    apiArgs: join(cwd, "gh-api-args"),
    pnpmCalls: join(cwd, "pnpm-calls"),
  };
  const env = {
    ...process.env,
    PATH: [bin, process.env.PATH].join(delimiter),
    GH_CALLS: paths.calls,
    GH_CREATE_ARGS: paths.createArgs,
    GH_API_ARGS: paths.apiArgs,
    PNPM_CALLS: paths.pnpmCalls,
    GH_MODE: "success",
  };
  const file = (name) => {
    const path = join(cwd, name);
    writeFileSync(path, "disposable text fixture\n");
    return path;
  };
  const run = (script, args, overrides = {}) =>
    spawnSync(script, args, {
      cwd,
      env: { ...env, ...overrides },
      encoding: "utf8",
    });
  const calls = () =>
    existsSync(paths.calls)
      ? readFileSync(paths.calls, "utf8").trim().split("\n")
      : [];
  const args = (path) =>
    readFileSync(path).toString("utf8").split("\0").slice(0, -1);
  return {
    adapter: fixtureAdapter,
    uploadHelper: fixtureUploadHelper,
    file,
    run,
    calls,
    args,
    paths,
  };
}

test("Gist adapter returns the raw URL without clibox and preserves a spaced filename", (t) => {
  const f = fixture(t);
  const file = f.file("sample file.txt");
  const result = f.run(f.adapter, [file]);

  assert.equal(result.status, 0, result.stderr);
  assert.equal(
    result.stdout,
    "https://gist.githubusercontent.com/test/abc123/raw/sample\n",
  );
  assert.deepEqual(f.calls(), ["gist", "api"]);
  assert.deepEqual(f.args(f.paths.createArgs), [
    "gist",
    "create",
    file,
    "--public",
  ]);
  assert.deepEqual(f.args(f.paths.apiArgs), [
    "api",
    "gists/abc123",
    "--jq",
    '.files["sample file.txt"].raw_url',
  ]);
  assert.equal(existsSync(f.paths.pnpmCalls), false);
});

test("Gist adapter propagates creation and lookup failures without a raw URL", (t) => {
  for (const [mode, expectedCalls] of [
    ["create-fail", ["gist"]],
    ["lookup-fail", ["gist", "api"]],
  ]) {
    const f = fixture(t);
    const result = f.run(f.adapter, [f.file(`${mode}.txt`)], { GH_MODE: mode });

    assert.notEqual(result.status, 0);
    assert.equal(result.stdout, "");
    assert.deepEqual(f.calls(), expectedCalls);
    assert.equal(existsSync(f.paths.pnpmCalls), false);
  }
});

test("upload helper still invokes the Gist adapter and keeps clipboard fallback", (t) => {
  const f = fixture(t);
  const result = f.run(
    f.uploadHelper,
    [f.file("before.txt"), f.file("after.txt")],
    { IMAGE_ADAPTER: "gist" },
  );

  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(f.calls(), ["gist", "api", "gist", "api"]);
  assert.match(
    result.stdout,
    /Before URL: https:\/\/gist\.githubusercontent\.com\/test\/abc123\/raw\/sample/,
  );
  assert.match(
    result.stdout,
    /After URL: https:\/\/gist\.githubusercontent\.com\/test\/abc123\/raw\/sample/,
  );
  assert.match(result.stdout, /clipboard unavailable/);
  assert.equal(existsSync(f.paths.pnpmCalls), true);
});

const capture = fileURLToPath(
  new URL(
    "../../.agents/skills/before-and-after/scripts/capture.sh",
    import.meta.url,
  ),
);
const blobAdapter = fileURLToPath(
  new URL(
    "../../.agents/skills/before-and-after/scripts/adapters/blob.sh",
    import.meta.url,
  ),
);

function captureFixture(t) {
  const directory = mkdtempSync(join(tmpdir(), "before-and-after-capture-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const bin = join(directory, "bin");
  mkdirSync(bin);
  const log = join(directory, "calls.jsonl");
  const spy = join(bin, "agent-browser");
  writeFileSync(
    spy,
    '#!/usr/bin/env node\nconst args = process.argv.slice(2);\nrequire("node:fs").appendFileSync(process.env.CAPTURE_SPY_LOG, JSON.stringify(args) + "\\n");\nif (args[0] === process.env.CAPTURE_SPY_FAIL) process.exit(23);\n',
  );
  chmodSync(spy, 0o755);

  const url = "https://example.invalid/a path?name=hello world";
  const output = join(directory, "images with spaces", "whole page.png");
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

function blobFixture(t) {
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
    spawnSync("bash", [blobAdapter, input], {
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

for (const options of [[], ["--full"]]) {
  test(`full-page capture dispatches supported commands (${options.join(" ") || "default"})`, (t) => {
    const f = captureFixture(t);
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
  const f = captureFixture(t);
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
  const f = captureFixture(t);
  const { result, calls } = f.run([], "open");
  assert.equal(result.status, 23, result.stderr);
  assert.deepEqual(calls, [["open", f.url]]);
});

test("Blob adapter decodes JSON URL escapes and preserves plain URLs", (t) => {
  const f = blobFixture(t);
  const expected = "https://example.invalid/image.png?a=1&b=2";
  for (const response of [
    expected,
    JSON.stringify({ url: expected }),
    '{"url":"https:\\/\\/example.invalid/image.png?a=1\\u0026b=2"}',
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
  const f = blobFixture(t);
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
  const f = blobFixture(t);
  const marker = join(f.directory, "executed");
  const url = `https://example.invalid/image.png?value=$(touch ${marker})`;
  const response = JSON.stringify({ url }).replace("$", "\\u0024");
  const result = f.run(response);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout, `${url}\n`);
  assert.equal(existsSync(marker), false);
});
