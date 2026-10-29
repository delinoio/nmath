// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, posix, win32 } from "node:path";
import test from "node:test";
import {
  archiveNotices,
  bundleRelativePath,
  render,
  verifyBundle,
  sourceInputPaths,
  verifySourceInputs,
} from "../../packages/nmath/scripts/license-bundle.mjs";

const digest = (text) => createHash("sha256").update(text).digest("hex");
function fixture(t) {
  const stage = mkdtempSync(join(tmpdir(), "license-fixture-"));
  t.after(() => rmSync(stage, { recursive: true, force: true }));
  mkdirSync(join(stage, "texts"));
  const terms = "Synthetic license terms. Not actual artifact acceptance.\r\n";
  const sha256 = digest(terms);
  writeFileSync(join(stage, "texts", sha256), terms);
  const inventory = {
    schemaVersion: 1,
    kind: "actual-source-inventory",
    revision: "a".repeat(40),
    locks: Object.fromEntries(
      sourceInputPaths().map((file) => [
        file,
        digest(readFileSync(new URL("../../" + file, import.meta.url))),
      ]),
    ),
    components: [
      {
        id: "synthetic:component",
        declaredLicense: "fixture",
        notices: [{ path: "LICENSE", sha256 }],
      },
    ],
  };
  const save = () =>
    writeFileSync(join(stage, "inventory.json"), JSON.stringify(inventory));
  save();
  return {
    stage,
    inventory,
    save,
    terms,
    sha256,
    output: join(stage, "bundle.txt"),
  };
}

test("synthetic fixtures: deterministic render preserves original notice bytes and remains pending", (t) => {
  const f = fixture(t);
  const first = render(f.stage, f.output);
  const bytes = readFileSync(f.output);
  assert.ok(bytes.includes(Buffer.from(f.terms)));
  assert.deepEqual(render(f.stage, f.output), first);
  assert.deepEqual(readFileSync(f.output), bytes);
  assert.throws(() => verifyBundle(f.output), /reviewed actual-source/);
  assert.equal(
    verifyBundle(f.output, { allowPending: true }).reviewStatus,
    "pending",
  );
});
test("synthetic fixtures: missing and changed notices fail closed", (t) => {
  const f = fixture(t);
  writeFileSync(join(f.stage, "texts", f.sha256), "changed");
  assert.throws(() => render(f.stage, f.output), /digest mismatch/);
  rmSync(join(f.stage, "texts", f.sha256));
  assert.throws(() => render(f.stage, f.output), /ENOENT/);
  f.inventory.components[0].notices = [];
  f.save();
  assert.throws(() => render(f.stage, f.output), /Missing license/);
});
test("synthetic fixtures: dependency change invalidates inventory and review", (t) => {
  const f = fixture(t);
  f.inventory.locks["Cargo.lock"] = "0".repeat(64);
  f.save();
  assert.throws(() => render(f.stage, f.output), /digest mismatch/);
});
test("synthetic fixtures: pending review and altered bundle cannot pass release verification", (t) => {
  const f = fixture(t);
  const review = join(f.stage, "review.json");
  writeFileSync(
    review,
    JSON.stringify({
      inventorySha256: digest(readFileSync(join(f.stage, "inventory.json"))),
      status: "pending",
    }),
  );
  assert.throws(() => render(f.stage, f.output, review), /review incomplete/);
  render(f.stage, f.output);
  writeFileSync(f.output, "altered");
  assert.throws(
    () => verifyBundle(f.output, { allowPending: true }),
    /digest mismatch/,
  );
});
test("synthetic archives: preserve nested notices; refuse absent full terms and invalid UTF-8", (t) => {
  const f = fixture(t);
  const source = join(f.stage, "source");
  mkdirSync(join(source, "nested"), { recursive: true });
  writeFileSync(join(source, "LICENSE"), "synthetic full terms");
  writeFileSync(join(source, "nested/NOTICE.txt"), "synthetic attribution");
  const archive = join(f.stage, "source.tar");
  const pack = () => execFileSync("tar", ["-cf", archive, "-C", source, "."]);
  pack();
  assert.equal(archiveNotices(archive).length, 2);
  rmSync(join(source, "LICENSE"));
  pack();
  assert.throws(() => archiveNotices(archive), /Missing license terms/);
  writeFileSync(join(source, "LICENSE"), Buffer.from([0xff]));
  pack();
  assert.throws(() => archiveNotices(archive), /encoded data/);
});

for (const [paths, root] of [
  [posix, "/work/package"],
  [win32, "C:\\work\\package"],
]) {
  test(`bundle input containment uses ${paths.sep} and portable archive members`, () => {
    assert.equal(
      bundleRelativePath(root, paths.join(root, "lib", "index.js"), paths),
      "lib/index.js",
    );
    assert.equal(bundleRelativePath(root, root, paths), null);
    assert.equal(
      bundleRelativePath(root, paths.join(root, "..", "outside.js"), paths),
      null,
    );
    assert.equal(
      bundleRelativePath(root, root + "-other" + paths.sep + "index.js", paths),
      null,
    );
    const src = paths.join(root, "src");
    assert.equal(
      bundleRelativePath(src, paths.join(src, "nested", "extension.ts"), paths),
      "nested/extension.ts",
    );
    assert.equal(
      bundleRelativePath(
        src,
        paths.join(root, "src-other", "extension.ts"),
        paths,
      ),
      null,
    );
  });
}
test("Windows bundle containment rejects a different drive or UNC share", () => {
  assert.equal(bundleRelativePath("C:/work", "D:/work/file.js", win32), null);
  assert.equal(
    bundleRelativePath(
      "//server/share/pkg",
      "//server/other/pkg/file.js",
      win32,
    ),
    null,
  );
  assert.equal(
    bundleRelativePath(
      "//server/share/pkg",
      "//server/share/pkg/lib/file.js",
      win32,
    ),
    "lib/file.js",
  );
});

test("source membership rejects added build scripts and notices before and after staging", (t) => {
  const dir = mkdtempSync(join(tmpdir(), "license-membership-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  execFileSync("git", ["init", "--quiet", dir]);
  for (const file of sourceInputPaths(dir)) {
    mkdirSync(join(dir, file, ".."), { recursive: true });
    writeFileSync(join(dir, file), "fixture");
  }
  const locks = Object.fromEntries(
    sourceInputPaths(dir).map((file) => [
      file,
      digest(readFileSync(join(dir, file))),
    ]),
  );
  verifySourceInputs(locks, dir);
  for (const name of ["build.rs", "IMPORTED-NOTICE.txt"]) {
    const file = `crates/nmath-core/${name}`;
    mkdirSync(join(dir, "crates/nmath-core"), { recursive: true });
    writeFileSync(join(dir, file), "new input");
    assert.throws(() => verifySourceInputs(locks, dir), /membership changed/);
    execFileSync("git", ["-C", dir, "add", file]);
    assert.throws(() => verifySourceInputs(locks, dir), /membership changed/);
    execFileSync("git", ["-C", dir, "rm", "--cached", file]);
    rmSync(join(dir, file));
    verifySourceInputs(locks, dir);
  }
});
test("render and bundle verification reject incomplete source inventories", (t) => {
  const f = fixture(t);
  render(f.stage, f.output);
  const receipt = JSON.parse(readFileSync(`${f.output}.json`, "utf8"));
  delete receipt.locks["Cargo.toml"];
  writeFileSync(`${f.output}.json`, JSON.stringify(receipt));
  assert.throws(
    () => verifyBundle(f.output, { allowPending: true }),
    /membership changed/,
  );
  delete f.inventory.locks["Cargo.toml"];
  f.save();
  assert.throws(() => render(f.stage, f.output), /membership changed/);
});
