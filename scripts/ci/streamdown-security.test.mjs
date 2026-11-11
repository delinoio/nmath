// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../", import.meta.url));
const sources = [
  [".agents/skills/streamdown/assets/examples/full-featured.tsx", 1],
  [".agents/skills/streamdown/assets/examples/custom-security.tsx", 1],
  [".agents/skills/streamdown/references/security.md", 2],
];

for (const [path, expectedCallbacks] of sources) {
  test(`${path} trusts only exact hosts and their subdomains`, async () => {
    const source = readFileSync(join(root, path), "utf8");
    const callbacks = [
      ...source.matchAll(
        /onLinkCheck:\s*(?:async\s*)?\(url\)\s*=>\s*\{([\s\S]*?)\n\s*\},/g,
      ),
    ];
    assert.equal(callbacks.length, expectedCallbacks);

    for (const [, body] of callbacks) {
      const hosts = body.match(/const trusted = \[([^\]]+)\]/)?.[1];
      assert.ok(hosts);
      const domains = [...hosts.matchAll(/["']([^"']+)["']/g)].map(
        (match) => match[1],
      );
      assert.ok(domains.length > 0);
      const onLinkCheck = new Function("url", body);

      for (const domain of domains) {
        assert.equal(await onLinkCheck(`https://${domain}/x`), true, domain);
        assert.equal(
          await onLinkCheck(`https://docs.${domain}/x`),
          true,
          `docs.${domain}`,
        );
        const lookalike = `evil${domain}`;
        const trustedAsAnotherSubdomain = domains.some(
          (other) => lookalike === other || lookalike.endsWith("." + other),
        );
        assert.equal(
          await onLinkCheck(`https://${lookalike}/x`),
          trustedAsAnotherSubdomain,
          lookalike,
        );
        assert.equal(
          await onLinkCheck(`https://${domain}.evil.test/x`),
          false,
          `${domain}.evil.test`,
        );
      }
    }
  });
}
