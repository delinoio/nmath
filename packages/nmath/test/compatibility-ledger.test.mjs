// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, statSync } from "node:fs";
import test from "node:test";

const root = new URL("../compatibility/", import.meta.url);
const read = (path) => JSON.parse(readFileSync(new URL(path, root), "utf8"));
const manifest = read("inventory/manifest.generated.json");
const rows = new Map();
for (const shard of manifest.shards) {
  const data = readFileSync(new URL(`inventory/${shard.path}`, root));
  assert.equal(createHash("sha256").update(data).digest("hex"), shard.sha256);
  const values = data.toString().trimEnd().split("\n").map(JSON.parse);
  assert.equal(values.length, shard.rows);
  for (const row of values) {
    assert.ok(!rows.has(row.name), `duplicate ${row.name}`);
    rows.set(row.name, row);
  }
}

test("ledger retains provenance, pending obligations and resolvable aliases", () => {
  const counts = {};
  for (const row of rows.values()) {
    counts[row.library] = (counts[row.library] ?? 0) + 1;
    assert.equal(row.version, manifest.versions[row.library]);
    assert.equal(row.supportTarget, row.exclusion === null, row.name);
    assert.ok(row.boundaryEvidence.length, row.name);
    assert.ok(
      row.semantics.reference.startsWith("https://github.com/"),
      row.name,
    );
    for (const field of [
      "arguments",
      "dtypePlatform",
      "viewCopyMutation",
      "missingValues",
      "verification",
    ]) {
      assert.ok(row[field].status, `${row.name}: ${field}`);
    }
    if (row.aliasOf) {
      assert.ok(
        rows.has(row.aliasOf),
        `${row.name}: dangling alias ${row.aliasOf}`,
      );
      assert.notEqual(row.aliasOf, row.name);
      // Alias chains must terminate, including platform-conditional aliases.
      const visited = new Set([row.name]);
      let alias = row.aliasOf;
      while (alias) {
        assert.ok(!visited.has(alias), `${row.name}: alias cycle`);
        visited.add(alias);
        alias = rows.get(alias).aliasOf;
      }
    }
    if (row.status === "excluded") assert.ok(row.exclusion);
    if (!row.counterpart && !row.exclusion) assert.equal(row.status, "pending");
    assert.notEqual(
      row.verification.status,
      "verified",
      "Fixture generation cannot certify native behavior",
    );
  }
  assert.deepEqual(counts, manifest.counts);
  assert.equal(manifest.acceptanceStatus, "incomplete");
  assert.deepEqual(manifest.unresolvedNames, []);
});

test("public accessors, aliases, optional families and conditional dtypes remain inventoried", () => {
  for (const name of [
    "numpy.ndarray.__getitem__",
    "numpy.int32.sum",
    "numpy.float64.sum",
    "numpy.matlib.int32.sum",
    "numpy.add.reduce",
    "numpy.random.Generator.random",
    "numpy.polynomial.Polynomial.deriv",
    "numpy.float96",
    "numpy.complex192",
    "pandas.Series.str.contains",
    "pandas.Series.dt.year",
    "pandas.api.types.is_bool_dtype",
    "pandas.tseries.offsets.BusinessDay",
    "pandas.io.formats.style.Styler.background_gradient",
    "polars.Expr.str.contains",
    "polars.Expr.arr.sum",
    "polars.Series.list.len",
    "polars.testing.parametric.dataframes",
    "polars.plugins.register_plugin_function",
  ]) {
    assert.ok(rows.has(name), name);
    assert.equal(rows.get(name).supportTarget, true, name);
  }
  assert.equal(rows.has("numpy.register_dlpack_dtype"), false);
  assert.equal(
    rows.get("numpy.dtypes.register_dlpack_dtype").exclusion,
    "Python-only extension/serialization APIs",
  );
  assert.equal(rows.get("pandas.read_excel").exclusion, "Excel");
  assert.equal(rows.get("polars.read_ods").supportTarget, true);
});

test("fixture IDs and numerical criteria have explicit obligations", () => {
  const criteria = read("criteria.json");
  const fixtures = ["primitive", "random", "mutation"].flatMap(
    (name) => read(`fixtures/${name}.json`).cases,
  );
  const ids = new Set(fixtures.map((fixture) => fixture.id));
  assert.equal(ids.size, fixtures.length);
  for (const fixture of fixtures)
    assert.ok(criteria[fixture.criterion], fixture.id);
  for (const row of rows.values()) {
    for (const id of row.verification.fixtureIds) assert.ok(ids.has(id), id);
  }
  assert.equal(
    criteria["floating-diagnostic"].acceptance,
    "pending-upstream-criterion",
  );
  assert.equal("rtol" in criteria["floating-diagnostic"], false);
  for (const name of readdirSync(new URL("inventory/", root))) {
    assert.ok(
      statSync(new URL(`inventory/${name}`, root)).size < 512 * 1024,
      `${name}: shard exceeds LFS threshold`,
    );
  }
});
