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

test("homepage uses publication order and does not expose a standalone diary tab", async () => {
  const homepage = await readFile(join(root, "src/pages/index.astro"), "utf8");
  const siteConfig = await readFile(join(root, "src/site.config.ts"), "utf8");

  assert.match(homepage, /b\.data\.pubDate\.valueOf\(\)\s*-\s*a\.data\.pubDate\.valueOf\(\)/);
  assert.match(homepage, /posts\.slice\(0,\s*7\)/);
  assert.doesNotMatch(siteConfig, /label:\s*["']随笔["']/);
  assert.match(siteConfig, /label:\s*["']折腾实践 ↗["']/);
  assert.match(siteConfig, /href:\s*["']https:\/\/zhetengxia\.com\//);
});

test("recommender and advertising posts use separate categories", async () => {
  const files = (await readdir(join(root, "src/content/posts"))).filter((file) => file.endsWith(".md"));
  const counts = {};
  for (const file of files) {
    const source = await readFile(join(root, "src/content/posts", file), "utf8");
    const category = source.match(/^category:\s*"([^"]+)"/m)?.[1];
    counts[category] = (counts[category] ?? 0) + 1;
  }

  assert.equal(counts["推荐系统"], 18);
  assert.equal(counts["计算广告"], 12);
  assert.equal(counts["折腾实践"], 1);
  assert.equal(counts["推荐系统与机器学习"], undefined);
  assert.equal(counts["工程与算法"], undefined);
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
