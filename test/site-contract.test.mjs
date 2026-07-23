import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import test from "node:test";

const root = new URL("..", import.meta.url).pathname;

test("technical migration contains the expected number of source posts", async () => {
  const posts = (await readdir(join(root, "src/content/posts"))).filter((file) => file.endsWith(".md"));
  const report = JSON.parse(await readFile(join(root, "migration-report.json"), "utf8"));

  assert.equal(posts.length, report.technicalImported);
  assert.equal(report.totalExported, 79);
  assert.equal(report.technicalImported, 65);
  assert.equal(report.excludedForZhetengxia, 14);
});

test("site is configured as a static Cloudflare Pages project", async () => {
  const astroConfig = await readFile(join(root, "astro.config.mjs"), "utf8");
  const siteConfig = await readFile(join(root, "src/site.config.ts"), "utf8");
  const wrangler = await readFile(join(root, "wrangler.toml"), "utf8");

  assert.match(astroConfig, /site:\s*["']https:\/\/notlate\.cn["']/);
  assert.match(astroConfig, /output:\s*["']static["']/);
  assert.match(siteConfig, /domain:\s*["']notlate\.cn["']/);
  assert.match(siteConfig, /https:\/\/zhetengxia\.com/);
  assert.match(wrangler, /pages_build_output_dir\s*=\s*["']dist["']/);
});

test("each migrated article has stable SEO metadata", async () => {
  const posts = (await readdir(join(root, "src/content/posts"))).filter((file) => file.endsWith(".md"));

  for (const file of posts) {
    const source = await readFile(join(root, "src/content/posts", file), "utf8");
    assert.match(source, /^title:\s+"/m, file);
    assert.match(source, /^description:\s+"/m, file);
    assert.match(source, /^slug:\s+"[a-z0-9-]+"/m, file);
    assert.match(source, /^legacyId:\s+\d+/m, file);
    assert.match(source, /^pubDate:\s+\d{4}-\d{2}-\d{2}/m, file);
  }
});
