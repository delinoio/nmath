// SPDX-License-Identifier: Apache-2.0
import { execFileSync } from "node:child_process";
import {
  chmodSync,
  copyFileSync,
  cpSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const root = fileURLToPath(new URL("../../../", import.meta.url));
export const targets = JSON.parse(
  readFileSync(join(root, "packaging/nmath/targets.json"), "utf8"),
);
const source = join(root, "packages/nmath");
const manifest = JSON.parse(readFileSync(join(source, "package.json"), "utf8"));
export function targetFor(triple) {
  const target = targets.find((entry) => entry.triple === triple);
  if (!target) throw new Error(`Unsupported distribution target: ${triple}`);
  return target;
}
export function platformManifest(target, version = manifest.version) {
  return {
    name: `@delinoio/nmath-${target.triple}`,
    version,
    description: `NMath native addon and compiler for ${target.triple}.`,
    license: "Apache-2.0",
    repository: manifest.repository,
    engines: manifest.engines,
    os: [target.os],
    cpu: [target.cpu],
    ...(target.libc ? { libc: [target.libc] } : {}),
    main: `nmath.${target.triple}.node`,
    publishConfig: {
      executableFiles: [
        `nmath-compiler.${target.triple}${target.os === "win32" ? ".exe" : ""}`,
      ],
    },
    files: [
      "*.node",
      "nmath-compiler.*",
      "LICENSE",
      "NOTICE",
      "THIRD_PARTY_LICENSES.txt",
    ],
  };
}
function licenses(destination, licenseBundle) {
  if (!licenseBundle || !readFileSync(licenseBundle, "utf8").trim()) {
    throw new Error(
      "Supply NMATH_LICENSE_BUNDLE with reviewed full third-party license terms and notices.",
    );
  }
  copyFileSync(licenseBundle, join(destination, "THIRD_PARTY_LICENSES.txt"));
  for (const name of ["LICENSE", "NOTICE"])
    copyFileSync(join(source, name), join(destination, name));
}
export function stagePlatform({ triple, artifacts, output, licenseBundle }) {
  const target = targetFor(triple);
  const destination = join(output, triple);
  rmSync(destination, { recursive: true, force: true });
  mkdirSync(destination, { recursive: true });
  const exe = target.os === "win32" ? ".exe" : "";
  copyFileSync(
    join(artifacts, target.library),
    join(destination, `nmath.${triple}.node`),
  );
  const compiler = join(destination, `nmath-compiler.${triple}${exe}`);
  copyFileSync(join(artifacts, `nmath-compiler${exe}`), compiler);
  chmodSync(compiler, 0o755);
  licenses(destination, licenseBundle);
  writeFileSync(
    join(destination, "package.json"),
    JSON.stringify(platformManifest(target), null, 2) + "\n",
  );
  return destination;
}
export function stagePublic(output, licenseBundle) {
  const destination = join(output, "nmath");
  rmSync(destination, { recursive: true, force: true });
  mkdirSync(destination, { recursive: true });
  for (const name of ["dist", "bin", "README.md"])
    cpSync(join(source, name), join(destination, name), { recursive: true });
  licenses(destination, licenseBundle);
  const { scripts, devDependencies, napi, ...published } = manifest;
  published.files = [
    "dist",
    "bin",
    "README.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_LICENSES.txt",
  ];
  published.optionalDependencies = Object.fromEntries(
    targets.map((t) => [`@delinoio/nmath-${t.triple}`, manifest.version]),
  );
  writeFileSync(
    join(destination, "package.json"),
    JSON.stringify(published, null, 2) + "\n",
  );
  return destination;
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  const triple = process.env.NMATH_TARGET;
  const target = targetFor(triple);
  const output = join(root, "packaging/nmath/dist");
  // Use the target toolchain explicitly; a successful cross-build is not execution evidence.
  execFileSync(
    "cargo",
    [
      "build",
      "--locked",
      "--release",
      "--target",
      target.rust,
      "-p",
      "nmath-node",
      "-p",
      "nmath-compiler",
    ],
    { cwd: root, stdio: "inherit" },
  );
  const destination = stagePlatform({
    triple,
    artifacts: join(root, "target", target.rust, "release"),
    output,
    licenseBundle: process.env.NMATH_LICENSE_BUNDLE,
  });
  const publicPackage = stagePublic(output, process.env.NMATH_LICENSE_BUNDLE);
  if (!process.env.npm_execpath)
    throw new Error("Run package:native through the pinned pnpm task.");
  for (const directory of [destination, publicPackage]) {
    execFileSync(
      process.execPath,
      [
        process.env.npm_execpath,
        "--dir",
        directory,
        "pack",
        "--pack-destination",
        output,
      ],
      { stdio: "inherit" },
    );
  }
  console.log(
    JSON.stringify({
      event: "distribution_staged",
      triple,
      destination,
      acceptance: "not-performed",
    }),
  );
}
