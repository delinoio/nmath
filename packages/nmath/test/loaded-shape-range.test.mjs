// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { context } from "./helpers.mjs";

function npy(shape, values = []) {
  const tuple =
    shape.length === 0
      ? ""
      : `${shape.join(", ")}${shape.length === 1 ? "," : ""}`;
  const text = `{'descr': '|u1', 'fortran_order': False, 'shape': (${tuple}), }`;
  const padding = (64 - ((10 + Buffer.byteLength(text) + 1) % 64)) % 64;
  const header = Buffer.from(`${text}${" ".repeat(padding)}\n`);
  const prefix = Buffer.from([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, 0, 0]);
  prefix.writeUInt16LE(header.length, 8);
  return Buffer.concat([prefix, header, Buffer.from(values)]);
}

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++)
      crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function npz(entries) {
  const local = [];
  const central = [];
  let offset = 0;
  for (const [name, data] of entries) {
    const filename = Buffer.from(`${name}.npy`);
    const checksum = crc32(data);
    const member = Buffer.alloc(30 + filename.length);
    member.writeUInt32LE(0x04034b50, 0);
    member.writeUInt16LE(20, 4);
    member.writeUInt32LE(checksum, 14);
    member.writeUInt32LE(data.length, 18);
    member.writeUInt32LE(data.length, 22);
    member.writeUInt16LE(filename.length, 26);
    filename.copy(member, 30);
    local.push(member, data);

    const directory = Buffer.alloc(46 + filename.length);
    directory.writeUInt32LE(0x02014b50, 0);
    directory.writeUInt16LE(20, 4);
    directory.writeUInt16LE(20, 6);
    directory.writeUInt32LE(checksum, 16);
    directory.writeUInt32LE(data.length, 20);
    directory.writeUInt32LE(data.length, 24);
    directory.writeUInt16LE(filename.length, 28);
    directory.writeUInt32LE(offset, 42);
    filename.copy(directory, 46);
    central.push(directory);
    offset += member.length + data.length;
  }
  const directoryLength = central.reduce((sum, entry) => sum + entry.length, 0);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(directoryLength, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...local, ...central, end]);
}

test("NPY and NPZ reject dimensions outside the JS shape range without retaining arrays", async () => {
  const ctx = context();
  const dir = mkdtempSync(join(tmpdir(), "nmath-loaded-shape-"));
  try {
    const ordinary = npy([2, 3], [1, 2, 3, 4, 5, 6]);
    const scalar = npy([], [7]);
    const maximum = npy([0, 4294967295]);
    const ordinaryPath = join(dir, "ordinary.npy");
    const scalarPath = join(dir, "scalar.npy");
    const maximumPath = join(dir, "maximum.npy");
    const maximumArchivePath = join(dir, "maximum.npz");
    writeFileSync(ordinaryPath, ordinary);
    writeFileSync(scalarPath, scalar);
    writeFileSync(maximumPath, maximum);
    writeFileSync(maximumArchivePath, npz([["maximum", maximum]]));

    for (const dimension of [4294967296, 4294967297]) {
      const wide = npy([0, dimension]);
      const npyPath = join(dir, `${dimension}.npy`);
      const npzPath = join(dir, `${dimension}.npz`);
      writeFileSync(npyPath, wide);
      writeFileSync(
        npzPath,
        npz([
          ["ordinary", ordinary],
          ["wide", wide],
        ]),
      );
      await assert.rejects(ctx.load(npyPath), { code: "NMATH_IO_FORMAT" });
      assert.equal(ctx.metrics().cpuMemory.used, 0n);
      await assert.rejects(ctx.loadNpz(npzPath), { code: "NMATH_IO_FORMAT" });
      assert.equal(ctx.metrics().cpuMemory.used, 0n);
    }

    const maxArray = await ctx.load(maximumPath);
    assert.deepEqual(maxArray.shape, [0, 4294967295]);
    assert.deepEqual(maxArray.transpose().shape, [4294967295, 0]);
    assert.equal(maxArray.dtype, "uint8");
    assert.equal(maxArray.size, 0);
    maxArray.dispose();
    const maxArchive = await ctx.loadNpz(maximumArchivePath);
    const maxMember = maxArchive.get("maximum");
    assert.deepEqual(maxMember.shape, [0, 4294967295]);
    assert.equal(maxMember.dtype, "uint8");
    maxMember.dispose();

    const scalarArray = await ctx.load(scalarPath);
    assert.deepEqual(scalarArray.shape, []);
    assert.equal(scalarArray.item(), 7);
    scalarArray.dispose();
    const ordinaryArray = await ctx.load(ordinaryPath);
    assert.deepEqual(ordinaryArray.shape, [2, 3]);
    assert.deepEqual(ordinaryArray.toArray(), [
      [1, 2, 3],
      [4, 5, 6],
    ]);
    ordinaryArray.dispose();
    assert.equal(ctx.metrics().cpuMemory.used, 0n);
  } finally {
    await ctx.close();
    rmSync(dir, { recursive: true, force: true });
  }
});
