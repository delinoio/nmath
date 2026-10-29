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
import { join } from "node:path";
import test from "node:test";
import {
  archiveNotices,
  render,
  verifyBundle,
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
    locks: {
      "Cargo.lock": digest(
        readFileSync(new URL("../../Cargo.lock", import.meta.url)),
      ),
    },
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
