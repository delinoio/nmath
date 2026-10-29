// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import {
  mkdtempSync,
  readFileSync,
  rmSync,
  existsSync,
  readdirSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import * as np from "../dist/numpy.js";
import { context } from "./helpers.mjs";

const fixtures = new URL(
  "../../../crates/nmath-core/tests/fixtures/io-npz/",
  import.meta.url,
);
const expected = JSON.parse(
  readFileSync(new URL("complex-expected.json", fixtures)),
);
const hex = (array) => {
  const typed = array.toTypedArray();
  return Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength).toString(
    "hex",
  );
};

for (const compression of ["stored", "deflated"]) {
  test(`NPZ ${compression}: exact pinned NumPy complex bytes in both directions`, async () => {
    const ctx = context();
    const dir = mkdtempSync(join(tmpdir(), "nmath-npz-complex-"));
    try {
      const input = join(dir, "input.npz");
      writeFileSync(
        input,
        readFileSync(new URL(`complex-${compression}.npz`, fixtures)),
      );
      const arrays = await np.loadNpz(ctx, input);
      assert.ok(arrays instanceof Map);
      assert.deepEqual(
        [...arrays.keys()],
        expected.map((e) => e.name),
      );
      for (const entry of expected) {
        const a = arrays.get(entry.name);
        assert.equal(a.dtype, entry.dtype);
        assert.deepEqual(a.shape, entry.shape);
        assert.equal(hex(a), entry.hex);
      }
      const path = join(dir, "roundtrip");
      const save = compression === "stored" ? np.savez : np.savez_compressed;
      await save(ctx, path, { named: arrays });
      assert.ok(existsSync(`${path}.npz`));
      const again = await ctx.loadNpz(`${path}.npz`);
      for (const entry of expected)
        assert.equal(hex(again.get(entry.name)), entry.hex);
      if (process.env.NMATH_NPZ_ORACLE_OUTPUT) {
        writeFileSync(
          join(
            process.env.NMATH_NPZ_ORACLE_OUTPUT,
            `node-complex-${compression}.npz`,
          ),
          readFileSync(`${path}.npz`),
        );
      }
      for (const a of [...arrays.values(), ...again.values()]) a.dispose();
      assert.equal(ctx.metrics().cpuMemory.used, 0n);
    } finally {
      await ctx.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });
}

test("NPZ Map order, positional names, scalar arrays, empty archives and overwrite", async () => {
  const ctx = context();
  const dir = mkdtempSync(join(tmpdir(), "nmath-npz-api-"));
  try {
    const a = ctx.array([1n, 2n], { dtype: "int64" });
    const scalar = ctx.array(true, { dtype: "bool", shape: [] });
    const path = join(dir, "mixed.npz");
    await np.savez(ctx, path, {
      named: new Map([
        ["__proto__", a],
        ["한글", scalar],
      ]),
      positional: [a],
    });
    const loaded = await ctx.loadNpz(path);
    assert.deepEqual([...loaded.keys()], ["__proto__", "한글", "arr_0"]);
    assert.deepEqual(loaded.get("한글").shape, []);
    assert.equal(loaded.get("한글").item(), true);
    const original = readFileSync(path);
    await assert.rejects(np.savez(ctx, path), { code: "NMATH_IO_EXISTS" });
    assert.deepEqual(readFileSync(path), original);
    await assert.rejects(
      np.savez(ctx, join(dir, "collision"), {
        named: new Map([["arr_0", a]]),
        positional: [a],
      }),
      TypeError,
    );
    assert.ok(!existsSync(join(dir, "collision.npz")));
    await np.savez_compressed(ctx, path, {}, { overwrite: true });
    assert.equal((await ctx.loadNpz(path)).size, 0);
    const exact = join(dir, "no-suffix");
    await ctx.saveNpz(exact, new Map());
    assert.ok(existsSync(exact));
    assert.ok(!existsSync(`${exact}.npz`));
    for (const x of loaded.values()) x.dispose();
    a.dispose();
    scalar.dispose();
  } finally {
    await ctx.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("NPZ rejects context mismatch, unsafe names, malformed input and pre-cancellation without output changes", async () => {
  const ctx = context();
  const other = context();
  const dir = mkdtempSync(join(tmpdir(), "nmath-npz-failures-"));
  try {
    const a = other.array([1], { dtype: "int32" });
    const path = join(dir, "archive.npz");
    await assert.rejects(ctx.saveNpz(path, new Map([["a", a]])), {
      code: "NMATH_CONTEXT_MISMATCH",
    });
    const local = ctx.array([1], { dtype: "int32" });
    for (const name of ["", "../escape", "a/b", "a\\b"]) {
      await assert.rejects(ctx.saveNpz(path, new Map([[name, local]])), {
        code: "NMATH_IO_FORMAT",
      });
    }
    await ctx.saveNpz(path, new Map([["a", local]]));
    const bytes = readFileSync(path);
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(
      ctx.saveNpz(path, new Map(), {
        overwrite: true,
        signal: controller.signal,
      }),
      { code: "NMATH_CANCELLED" },
    );
    await assert.rejects(ctx.loadNpz(path, { signal: controller.signal }), {
      code: "NMATH_CANCELLED",
    });
    assert.deepEqual(readFileSync(path), bytes);
    const invalid = join(dir, "invalid.npz");
    writeFileSync(invalid, "invalid");
    await assert.rejects(ctx.loadNpz(invalid), { code: "NMATH_IO_FORMAT" });
    assert.deepEqual(readdirSync(dir).sort(), ["archive.npz", "invalid.npz"]);
    a.dispose();
    local.dispose();
  } finally {
    await Promise.all([ctx.close(), other.close()]);
    rmSync(dir, { recursive: true, force: true });
  }
});
