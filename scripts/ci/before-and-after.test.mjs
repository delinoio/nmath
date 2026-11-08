// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
  chmodSync,
  copyFileSync,
  existsSync,
  mkdtempSync,
  mkdirSync,
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
const adapter = join(scriptDir, "adapters/gist.sh");
const uploadHelper = join(scriptDir, "upload-and-copy.sh");

function fixture(t) {
  const cwd = mkdtempSync(join(tmpdir(), "gist-adapter-"));
  t.after(() => rmSync(cwd, { recursive: true, force: true }));
  const scripts = join(cwd, "scripts");
  mkdirSync(join(scripts, "adapters"), { recursive: true });
  const fixtureAdapter = join(scripts, "adapters/gist.sh");
  const fixtureUploadHelper = join(scripts, "upload-and-copy.sh");
  copyFileSync(adapter, fixtureAdapter);
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
