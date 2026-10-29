// SPDX-License-Identifier: Apache-2.0
/** The napi-rs platform suffix for this process, for example `linux-x64-gnu`. */
export function platformTriple(): string {
  const { platform, arch } = process;
  if (platform === "darwin") {
    return `darwin-${arch}`;
  }
  if (platform === "win32") {
    return `win32-${arch}-msvc`;
  }
  if (platform === "linux") {
    const report = process.report?.getReport() as
      { header?: { glibcVersionRuntime?: string } } | undefined;
    const libc = report?.header?.glibcVersionRuntime ? "gnu" : "musl";
    return `linux-${arch}-${libc}`;
  }
  return `${platform}-${arch}`;
}
