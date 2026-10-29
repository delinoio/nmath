// SPDX-License-Identifier: Apache-2.0
// Preparation reads pinned source archives. Rendering uses only staged, hashed texts.
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  realpathSync,
  writeFileSync,
} from "node:fs";
import pathApi, { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const json = (file) => JSON.parse(readFileSync(file, "utf8"));
const hash = (bytes, algorithm = "sha256", encoding = "hex") =>
  createHash(algorithm).update(bytes).digest(encoding);
const run = (command, args, cwd = root) =>
  execFileSync(command, args, {
    cwd,
    encoding: "utf8",
    maxBuffer: 128 * 1024 * 1024,
  });
const sorted = (values) => [...values].sort();
const policy = () => json(join(root, "packaging/nmath/license-policy.json"));
const noticeName =
  /(?:^|[/_.-])(licen[cs]es?|copying|copyright|notices?|authors|patents)(?:$|[/_.-])/i;
const termName = /licen[cs]e|copying|copyright/i;
const utf8 = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });

// Classify canonical filesystem paths, then normalize only archive member names.
export function bundleRelativePath(directory, file, paths = pathApi) {
  const relative = paths.relative(directory, file);
  if (
    !relative ||
    relative === ".." ||
    relative.startsWith(`..${paths.sep}`) ||
    paths.isAbsolute(relative)
  )
    return null;
  return relative.split(paths.sep).join("/");
}

export function checkDigest(bytes, expected) {
  if (hash(bytes) !== expected)
    throw new Error("License input digest mismatch");
}
export function archiveNotices(archive) {
  const paths = sorted(
    run("tar", ["-tf", archive]).split("\n").filter(Boolean),
  );
  const files = [];
  for (const path of paths) {
    if (path.startsWith("/") || path.split("/").includes(".."))
      throw new Error("Unsafe archive path");
    if (path.endsWith("/") || !noticeName.test(path)) continue;
    const bytes = execFileSync("tar", ["-xOf", archive, path], {
      maxBuffer: 32 * 1024 * 1024,
    });
    if (!bytes.length) continue;
    // Preserve bytes; invalid UTF-8 cannot silently become replacement characters.
    utf8.decode(bytes);
    files.push({ path, sha256: hash(bytes), bytes });
  }
  if (!files.some((f) => termName.test(basename(f.path))))
    throw new Error(`Missing license terms in ${basename(archive)}`);
  return files;
}
function stageComponent(stage, component, files) {
  const notices = files.map(({ path, sha256, bytes }) => {
    writeFileSync(join(stage, "texts", sha256), bytes);
    return { path, sha256 };
  });
  return { ...component, notices };
}
function lockPackages(lock) {
  return lock
    .split("[[package]]")
    .slice(1)
    .map((part) =>
      Object.fromEntries(
        [
          ...part.matchAll(/^(name|version|source|checksum) = "([^"]+)"$/gm),
        ].map((m) => [m[1], m[2]]),
      ),
    );
}
function npmIntegrity(lock, name, version) {
  const key = `${name}@${version}`;
  const blocks = lock.split(/\n(?=  \S)/);
  const block = blocks.find(
    (b) =>
      b.split("\n")[0].replace(/^  ['"]?|['"]?:$/g, "") === key &&
      b.includes("resolution: {integrity:"),
  );
  const integrity = block?.match(/integrity: (sha512-[A-Za-z0-9+/=]+)/)?.[1];
  if (!integrity) throw new Error(`Missing locked npm integrity: ${key}`);
  return integrity;
}

export async function prepare(stage, cache, fetchSources = false) {
  mkdirSync(join(stage, "texts"), { recursive: true });
  mkdirSync(cache, { recursive: true });
  const cargoLock = readFileSync(join(root, "Cargo.lock"), "utf8");
  const npmLock = readFileSync(join(root, "pnpm-lock.yaml"), "utf8");
  const locked = lockPackages(cargoLock);
  const metadata = JSON.parse(
    run("cargo", [
      "metadata",
      "--locked",
      "--offline",
      "--format-version",
      "1",
    ]),
  );
  const targets = json(join(root, "packaging/nmath/targets.json"));
  const graphs = {};
  const workspaceIds = new Map(
    metadata.packages
      .filter((p) => !p.source)
      .map((p) => [p.id, `workspace:${p.name}@${p.version}`]),
  );
  const normalizedId = (id) => workspaceIds.get(id) ?? id;
  for (const target of targets) {
    const m = JSON.parse(
      run("cargo", [
        "metadata",
        "--locked",
        "--offline",
        "--format-version",
        "1",
        "--filter-platform",
        target.rust,
      ]),
    );
    // Include build and dev edges conservatively. These are source inventories,
    // not claims that every crate contributes bytes to the final executable.
    graphs[target.triple] = m.resolve.nodes
      .map((n) => ({
        id: normalizedId(n.id),
        features: sorted(n.features),
        dependencies: sorted(n.dependencies.map(normalizedId)),
      }))
      .sort((a, b) => a.id.localeCompare(b.id));
  }
  const selected = new Set(
    Object.values(graphs).flatMap((nodes) => nodes.map((n) => n.id)),
  );
  const components = [];
  function sourceNotices(archive, p, provenance) {
    try {
      return archiveNotices(archive);
    } catch (error) {
      if (!error.message.startsWith("Missing license terms")) throw error;
      // Some published crates omit workspace-level terms. Recover only from the
      // exact VCS commit recorded inside the checksum-verified crate archive.
      const prefix = `${p.name}-${p.version}`;
      const vcs = JSON.parse(
        run("tar", ["-xOf", archive, `${prefix}/.cargo_vcs_info.json`]),
      );
      const revision = vcs.git?.sha1;
      const repository = p.repository?.match(
        /^https:\/\/github\.com\/([A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+)\/?$/,
      )?.[1];
      if (!repository || !/^[a-f0-9]{40}$/.test(revision))
        throw new Error(`Missing pinned supplemental source for ${p.id}`);
      const upstream = join(cache, `${revision}.upstream.tar.gz`);
      if (!existsSync(upstream) && fetchSources) {
        const bytes = execFileSync(
          "gh",
          ["api", `repos/${repository}/tarball/${revision}`],
          { maxBuffer: 128 * 1024 * 1024 },
        );
        writeFileSync(upstream, bytes);
      }
      if (!existsSync(upstream))
        throw new Error(
          `Stage pinned supplemental source ${repository}@${revision} with --fetch-sources`,
        );
      provenance.supplemental = {
        repository,
        revision,
        sha256: hash(readFileSync(upstream)),
        reason: "Published crate omits license terms",
      };
      return archiveNotices(upstream);
    }
  }
  const gitArchives = new Map();
  for (const p of metadata.packages
    .filter((p) => p.source && selected.has(p.id))
    .sort((a, b) => a.id.localeCompare(b.id))) {
    if (!p.license && !p.license_file)
      throw new Error(`Missing declared license: ${p.id}`);
    let archive;
    let provenance;
    if (p.source.startsWith("registry+")) {
      const entry = locked.find(
        (l) =>
          l.name === p.name && l.version === p.version && l.source === p.source,
      );
      if (!entry?.checksum) throw new Error(`Missing Cargo checksum: ${p.id}`);
      const cargoSource = dirname(p.manifest_path);
      archive = join(
        dirname(dirname(dirname(cargoSource))),
        "cache",
        basename(dirname(cargoSource)),
        `${p.name}-${p.version}.crate`,
      );
      if (!existsSync(archive))
        throw new Error(`Stage Cargo archive first: ${p.name}@${p.version}`);
      checkDigest(readFileSync(archive), entry.checksum);
      const originalManifest = execFileSync("tar", [
        "-xOf",
        archive,
        `${p.name}-${p.version}/Cargo.toml`,
      ]);
      checkDigest(readFileSync(p.manifest_path), hash(originalManifest));
      provenance = { source: p.source, sha256: entry.checksum };
    } else if (p.source.startsWith("git+")) {
      const revision = p.source.split("#")[1];
      if (!/^[a-f0-9]{40}$/.test(revision))
        throw new Error("Unpinned Git dependency");
      const checkout = run(
        "git",
        ["rev-parse", "--show-toplevel"],
        dirname(p.manifest_path),
      ).trim();
      if (
        run("git", ["rev-parse", "HEAD"], checkout).trim() !== revision ||
        run(
          "git",
          ["status", "--porcelain", "--untracked-files=no"],
          checkout,
        ).trim()
      )
        throw new Error("Git dependency checkout differs from pinned source");
      archive = join(cache, `${revision}.tar`);
      if (!gitArchives.has(revision)) {
        run(
          "git",
          ["archive", "--format=tar", `--output=${archive}`, revision],
          checkout,
        );
        gitArchives.set(revision, archiveNotices(archive));
      }
      provenance = {
        source: p.source,
        revision,
        sha256: hash(readFileSync(archive)),
      };
    } else throw new Error(`Unsupported source: ${p.source}`);
    components.push(
      stageComponent(
        stage,
        {
          id: p.id,
          kind: "cargo",
          name: p.name,
          version: p.version,
          declaredLicense: p.license,
          provenance,
        },
        gitArchives.get(provenance.revision) ??
          sourceNotices(archive, p, provenance),
      ),
    );
  }
  // Walk installed production dependencies only. Every package is then checked
  // against the lockfile and its original registry archive, not copied from node_modules.
  const seen = new Set();
  const npmArchives = new Map();
  function visit(manifestPath) {
    const manifest = json(manifestPath);
    const req = createRequire(manifestPath);
    for (const name of sorted(
      Object.keys({
        ...manifest.dependencies,
        ...manifest.optionalDependencies,
      }),
    )) {
      let path;
      try {
        path = req.resolve
          .paths(name)
          .map((base) => join(base, name, "package.json"))
          .find(existsSync);
        if (!path) throw new Error("missing");
      } catch {
        throw new Error(
          `Cannot inventory runtime dependency ${name}; stage all optional packages too`,
        );
      }
      path = realpathSync(path);
      if (seen.has(path)) continue;
      seen.add(path);
      visit(path);
    }
  }
  for (const file of [
    "packages/nmath/package.json",
    "apps/nmath-vscode/package.json",
  ])
    visit(join(root, file));
  for (const path of sorted(seen)) {
    const p = json(path);
    const integrity = npmIntegrity(npmLock, p.name, p.version);
    const archive = join(
      cache,
      `${p.name.replaceAll("/", "-")}-${p.version}.tgz`,
    );
    if (!existsSync(archive) && fetchSources) {
      const url = `https://registry.npmjs.org/${p.name}/-/${p.name.split("/").at(-1)}-${p.version}.tgz`;
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Source download failed: ${p.name}`);
      writeFileSync(archive, Buffer.from(await response.arrayBuffer()));
    }
    if (!existsSync(archive))
      throw new Error(
        `Stage npm source archive first: ${basename(archive)} (or prepare with --fetch-sources)`,
      );
    if (
      `sha512-${hash(readFileSync(archive), "sha512", "base64")}` !== integrity
    )
      throw new Error(`npm source integrity mismatch: ${p.name}`);
    const originalManifest = execFileSync("tar", [
      "-xOf",
      archive,
      "package/package.json",
    ]);
    if (hash(originalManifest) !== hash(readFileSync(path)))
      throw new Error(
        `Installed npm manifest differs from pinned archive: ${p.name}`,
      );
    if (!p.license) throw new Error(`Missing npm license: ${p.name}`);
    npmArchives.set(dirname(path), archive);
    components.push(
      stageComponent(
        stage,
        {
          id: `npm:${p.name}@${p.version}`,
          kind: "npm-runtime",
          name: p.name,
          version: p.version,
          declaredLicense: p.license,
          provenance: { integrity, sha256: hash(readFileSync(archive)) },
        },
        archiveNotices(archive),
      ),
    );
  }
  const extensionRoot = join(root, "apps/nmath-vscode");
  const esbuild = createRequire(join(extensionRoot, "package.json"))("esbuild");
  const bundled = esbuild.buildSync({
    entryPoints: [join(extensionRoot, "src/extension.ts")],
    bundle: true,
    platform: "node",
    format: "cjs",
    target: "node20",
    external: ["vscode"],
    sourcemap: false,
    write: false,
    metafile: true,
    absWorkingDir: root,
  });
  const bundleInputs = [];
  for (const input of sorted(Object.keys(bundled.metafile.inputs))) {
    const path = realpathSync(resolve(root, input));
    const owner = [...npmArchives.keys()].find(
      (dir) => bundleRelativePath(dir, path) !== null,
    );
    if (!owner) {
      if (
        bundleRelativePath(realpathSync(join(extensionRoot, "src")), path) ===
        null
      )
        throw new Error(`Uninventoried VSIX input: ${input}`);
      bundleInputs.push({
        path: input,
        owner: "repository",
        sha256: hash(readFileSync(path)),
      });
    } else {
      const relative = bundleRelativePath(owner, path);
      const original = execFileSync("tar", [
        "-xOf",
        npmArchives.get(owner),
        `package/${relative}`,
      ]);
      checkDigest(readFileSync(path), hash(original));
      const manifest = json(join(owner, "package.json"));
      bundleInputs.push({
        path: `${manifest.name}/${relative}`,
        owner: `npm:${manifest.name}@${manifest.version}`,
        sha256: hash(original),
      });
    }
  }
  const toolchain = run("rustc", ["-Vv"]).trim();
  const sysroot = run("rustc", ["--print", "sysroot"]).trim();
  const rustCopyright = readFileSync(
    join(sysroot, "share/doc/rust/COPYRIGHT-library.html"),
  );
  utf8.decode(rustCopyright);
  components.push(
    stageComponent(
      stage,
      {
        id: "rust-standard-library",
        kind: "toolchain",
        name: "Rust standard library",
        version: toolchain,
        declaredLicense: "See upstream COPYRIGHT-library.html; review pending",
        provenance: { compiler: toolchain, sha256: hash(rustCopyright) },
      },
      [
        {
          path: "COPYRIGHT-library.html",
          sha256: hash(rustCopyright),
          bytes: rustCopyright,
        },
      ],
    ),
  );
  const importedRoot = "crates/nmath-core/src/random";
  const importedNotices = ["NUMPY-LICENSE.txt", "NUMPY-RANDOM-LICENSE.txt"].map(
    (name) => {
      const bytes = readFileSync(join(root, importedRoot, name));
      return { path: `${importedRoot}/${name}`, sha256: hash(bytes), bytes };
    },
  );
  components.push(
    stageComponent(
      stage,
      {
        id: "imported:numpy-random@2.5.4",
        kind: "imported-source",
        name: "NumPy random sources",
        version: "2.5.4",
        declaredLicense: "BSD-3-Clause",
        provenance: {
          sourceRecordSha256: hash(
            readFileSync(join(root, importedRoot, "SOURCES.txt")),
          ),
        },
      },
      importedNotices,
    ),
  );
  const inputPaths = sorted(
    new Set([
      "Cargo.lock",
      "pnpm-lock.yaml",
      "Cargo.toml",
      "rust-toolchain",
      "packaging/nmath/targets.json",
      "packaging/nmath/license-policy.json",
      "packages/nmath/package.json",
      "apps/nmath-vscode/package.json",
      ...run("git", [
        "ls-files",
        "crates",
        "packages/nmath/src",
        "packages/nmath/bin",
        "apps/nmath-vscode/src",
      ])
        .trim()
        .split("\n"),
    ]),
  );
  const inputs = Object.fromEntries(
    inputPaths.map((file) => [file, hash(readFileSync(join(root, file)))]),
  );
  const inventory = {
    schemaVersion: 1,
    kind: "actual-source-inventory",
    revision: run("git", ["rev-parse", "HEAD"]).trim(),
    locks: inputs,
    toolchain,
    bundleInputs,
    graphs,
    excludedTargetPackages: metadata.packages
      .filter((p) => p.source && !selected.has(p.id))
      .map((p) => p.id)
      .sort(),
    components: components.sort((a, b) => a.id.localeCompare(b.id)),
    boundaries: {
      npm: "Production npm dependency closure from both shipping package manifests. Native/compiler dependencies are inventoried separately.",
      vsix: "Production dependency closure, conservatively including tree-shaken files. VS Code and Node.js are external host runtimes. VSCE/esbuild/TypeScript are build tools, not bundled by --no-dependencies.",
      native:
        "Cargo inventory includes all resolved targets and build/dev dependencies. Linker runtimes, SDKs, Rust standard library and embedded non-Cargo material require supplemental review.",
    },
    pendingReview: [
      "Review complete license choices, source-header notices and embedded/imported material for each component",
      "Supply target-specific Rust standard-library, linker/runtime and SDK redistribution evidence",
      "Confirm esbuild inputs match the runtime closure and no build dependency is bundled",
      "Re-run after Arrow or other dependency/source changes",
    ],
  };
  writeFileSync(
    join(stage, "inventory.json"),
    JSON.stringify(inventory, null, 2) + "\n",
  );
  const review = {
    schemaVersion: 1,
    inventorySha256: hash(readFileSync(join(stage, "inventory.json"))),
    status: "pending",
    reviewer: "",
    components: inventory.components.map((c) => ({
      id: c.id,
      status: "pending",
      notes: "",
    })),
    boundaries: policy().reviewBoundaries.map((id) => ({
      id,
      status: "pending",
      notes: "",
    })),
    nativeTargets: targets.map((t) => ({
      target: t.triple,
      status: "pending",
      notes: "",
    })),
    supplemental: [],
    supplementTemplate: policy().supplementTemplate,
    pending: inventory.pendingReview,
  };
  writeFileSync(
    join(stage, "review.template.json"),
    JSON.stringify(review, null, 2) + "\n",
  );
  return inventory;
}

export function render(stage, output, reviewFile) {
  const inventoryBytes = readFileSync(join(stage, "inventory.json"));
  const inventory = JSON.parse(inventoryBytes);
  if (
    inventory.schemaVersion !== 1 ||
    inventory.kind !== "actual-source-inventory" ||
    !inventory.components.length
  )
    throw new Error("Invalid actual-source inventory");
  for (const [file, digest] of Object.entries(inventory.locks))
    checkDigest(readFileSync(join(root, file)), digest);
  const review = reviewFile ? json(reviewFile) : undefined;
  if (review) {
    checkDigest(inventoryBytes, review.inventorySha256);
    if (
      review.status !== "reviewed" ||
      !review.reviewer?.trim() ||
      review.pending?.length !== 0 ||
      review.components?.length !== inventory.components.length ||
      inventory.components.some(
        (c) =>
          !review.components.some(
            (r) => r.id === c.id && r.status === "reviewed" && r.notes?.trim(),
          ),
      )
    )
      throw new Error("License review incomplete");
    for (const id of policy().reviewBoundaries) {
      if (
        !review.boundaries?.some(
          (b) => b.id === id && b.status === "reviewed" && b.notes?.trim(),
        )
      )
        throw new Error(`Missing boundary review: ${id}`);
    }
    for (const target of Object.keys(inventory.graphs)) {
      if (
        !review.nativeTargets?.some(
          (t) =>
            t.target === target && t.status === "reviewed" && t.notes?.trim(),
        )
      )
        throw new Error(`Missing native target review: ${target}`);
    }
    if (!review.supplemental?.length)
      throw new Error("Missing toolchain/imported-source supplemental review");
  }
  let text = `NMath third-party license source bundle\nSource revision: ${inventory.revision}\nInventory SHA-256: ${hash(inventoryBytes)}\nReview: ${review ? "reviewed" : "PENDING — local inspection only; not release acceptance"}\n\n`;
  for (const component of inventory.components) {
    if (!component.notices?.some((n) => termName.test(basename(n.path))))
      throw new Error(`Missing license: ${component.id}`);
    text += `\n========== ${component.id} ==========\nDeclared license: ${component.declaredLicense ?? "see license file"}\n`;
    for (const file of component.notices) {
      if (!/^[a-f0-9]{64}$/.test(file.sha256))
        throw new Error("Invalid text digest");
      const bytes = readFileSync(join(stage, "texts", file.sha256));
      checkDigest(bytes, file.sha256);
      text += `\n--- ${file.path} (SHA-256 ${file.sha256}) ---\n${utf8.decode(bytes)}\n`;
    }
  }
  for (const supplement of review?.supplemental ?? []) {
    if (
      !supplement.name?.trim() ||
      !supplement.source?.trim() ||
      !supplement.notes?.trim() ||
      !/^[a-f0-9]{64}$/.test(supplement.sha256)
    )
      throw new Error("Invalid supplemental notice");
    const bytes = readFileSync(join(stage, "texts", supplement.sha256));
    checkDigest(bytes, supplement.sha256);
    text += `\n========== ${supplement.name} ==========\nSource: ${supplement.source}\n${utf8.decode(bytes)}\n`;
  }
  writeFileSync(output, text);
  const receipt = {
    schemaVersion: 1,
    kind: "actual-source-bundle",
    inventorySha256: hash(inventoryBytes),
    bundleSha256: hash(text),
    revision: inventory.revision,
    reviewStatus: review ? "reviewed" : "pending",
    reviewSha256: reviewFile ? hash(readFileSync(reviewFile)) : null,
    components: inventory.components.length,
    locks: inventory.locks,
  };
  writeFileSync(`${output}.json`, JSON.stringify(receipt, null, 2) + "\n");
  return receipt;
}

export function verifyBundle(bundle, { allowPending = false } = {}) {
  if (!bundle)
    throw new Error(
      "NMATH_LICENSE_BUNDLE must name a generated license bundle",
    );
  const receipt = json(`${bundle}.json`);
  if (
    receipt.schemaVersion !== 1 ||
    receipt.kind !== "actual-source-bundle" ||
    !receipt.components ||
    !receipt.locks ||
    !["pending", "reviewed"].includes(receipt.reviewStatus) ||
    (!allowPending && receipt.reviewStatus !== "reviewed")
  )
    throw new Error("A reviewed actual-source license bundle is required");
  checkDigest(readFileSync(bundle), receipt.bundleSha256);
  for (const [file, digest] of Object.entries(receipt.locks))
    checkDigest(readFileSync(join(root, file)), digest);
  return receipt;
}

if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  const [command, stage, destination, option] = process.argv.slice(2);
  if (command === "prepare" && stage && destination) {
    const result = await prepare(
      resolve(stage),
      resolve(destination),
      option === "--fetch-sources",
    );
    console.log(
      JSON.stringify({
        event: "license_sources_prepared",
        components: result.components.length,
        review: "pending",
      }),
    );
  } else if (command === "render" && stage && destination)
    console.log(
      JSON.stringify(render(resolve(stage), resolve(destination), option)),
    );
  else if (command === "verify" && stage)
    console.log(JSON.stringify(verifyBundle(resolve(stage))));
  else
    throw new Error(
      "Usage: license-bundle.mjs prepare <stage> <source-cache> [--fetch-sources] | render <stage> <bundle.txt> [review.json] | verify <bundle.txt>",
    );
}
