#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import process from "node:process";

const dbPath = resolve(process.argv[2] ?? "/tmp/notlate-blog.db");
const outputDir = resolve(process.argv[3] ?? "src/content/posts");
const reportPath = join(dirname(outputDir), "..", "..", "migration-report.json");

const excludedIds = new Set([
  19452715, // AI Code Agent
  19560365, // Code Reader Skills
  19613288, // CLIProxyAPI
  20002343, // Claude 订阅
  20067908, // Codex 移动端
  20125939, // Oracle VPS
  20494127, // 鸿蒙公众号助手
  20688466, // 镜像通知
  20979634, // Claude 水印
  21345748, // 网易 UU
  21367746, // Superpower Skills
  21439874, // Codex TRACE
  21752709, // AI Coding 实践旧版本
  21776554, // AI Coding 实践新版本
]);

const titleReplacements = [
  ["小红书", "xiaohongshu"],
  ["推荐系统", "recommender-system"],
  ["计算广告", "computational-advertising"],
  ["机器学习", "machine-learning"],
  ["概率统计", "probability-statistics"],
  ["移动广告", "mobile-advertising"],
  ["竞价形势", "bid-landscape"],
  ["先验校准", "prior-calibration"],
  ["工作小结", "work-summary"],
  ["工作总结", "work-summary"],
  ["无需服务器个性化域名重定向到其他网站", "custom-domain-redirect"],
  ["推荐系统笔记", "recommender-notes"],
  ["计算广告笔记", "advertising-notes"],
  ["烂笔头系列", "notes"],
  ["方言", "dialect"],
  ["深入研究", "deep-dive"],
  ["学习", "study"],
  ["分析", "analysis"],
  ["总结", "summary"],
  ["如何", "how-to"],
  ["实现", "implement"],
  ["高效", "efficient"],
  ["核心", "core"],
  ["知识点", "knowledge"],
];

const foldSlugs = {
  19452715: null,
  19560365: null,
  19613288: null,
  20002343: "claude-subscription-anti-ban-guide",
  20067908: "codex-mobile-desktop-workflow",
  20125939: null,
  20494127: null,
  20688466: null,
  20979634: "claude-code-watermark-account-risk-2026",
  21345748: null,
  21367746: null,
  21439874: "codex-sqlite-logs-ssd-ramdisk",
  21752709: "ai-coding-practice-experience",
  21776554: "ai-coding-practice-experience",
};

const missingRelativeImages = new Set(["picture7.png", "媒体实战.jpeg", "数据提供方实战.jpeg"]);

function queryRows() {
  const query = "SELECT Id, Title, DateAdded, DateUpdated, SourceUrl, Body, Description FROM blog_Content WHERE IsActive=1 ORDER BY DateAdded, Id";
  const output = execFileSync("sqlite3", ["-json", dbPath, query], {
    encoding: "utf8",
    maxBuffer: 32 * 1024 * 1024,
  });
  return JSON.parse(output);
}

function decodeEntities(value) {
  return value
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

function cleanTitle(value) {
  return decodeEntities(String(value ?? "")).replace(/\s+/g, " ").trim();
}

function makeSlug(title, id, usedSlugs) {
  let value = cleanTitle(title).replace(/[【】\[\]]/g, "");
  for (const [from, to] of titleReplacements) value = value.replaceAll(from, ` ${to} `);
  value = value
    .replace(/ONNX[-\s]?MLIR/gi, "onnx-mlir")
    .replace(/AI编译器/gi, "ai-compiler")
    .replace(/MLIR/gi, "mlir")
    .replace(/IREE/gi, "iree")
    .replace(/Triton/gi, "triton")
    .replace(/TVM/gi, "tvm")
    .replace(/Linalg/gi, "linalg")
    .replace(/MemRef/gi, "memref")
    .replace(/SSA/gi, "ssa")
    .replace(/GPU/gi, "gpu")
    .replace(/NPU/gi, "npu")
    .replace(/QKV/gi, "qkv")
    .replace(/Embedding/gi, "embedding")
    .replace(/ADX/gi, "adx")
    .replace(/DSP/gi, "dsp")
    .replace(/SSP/gi, "ssp")
    .replace(/Bid Landscape/gi, "bid-landscape")
    .replace(/\bvs\b/gi, "vs")
    .replace(/[^a-zA-Z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase();

  if (!value || value.length < 3) value = `post-${id}`;
  const base = value;
  let suffix = 2;
  while (usedSlugs.has(value)) value = `${base}-${suffix++}`;
  usedSlugs.add(value);
  return value;
}

function categoryFor(title) {
  if (/MLIR|ONNX|IREE|inductor|Triton|TVM|Ascend|AI编译器|SSA|Fuse|linalg|Tile|MemRef|Affine|Transform|ShapeHelper|IndexExpr|DialectBuilder|Flow方言|AutoFuse|Pass机制/i.test(title)) {
    return "AI 编译器";
  }
  if (/推荐|小红书|Embedding|机器学习|概率|广告|CTR|点击率|ADX|DSP|SSP|竞价|特征|冷启动|重排|召回|排序|模型评估/i.test(title)) {
    return "推荐系统与机器学习";
  }
  return "工程与算法";
}

function tagsFor(title, category) {
  const tags = new Set([category]);
  const rules = [
    [/MLIR/i, "MLIR"], [/ONNX/i, "ONNX-MLIR"], [/IREE/i, "IREE"], [/Triton/i, "Triton"], [/TVM/i, "TVM"], [/Ascend/i, "Ascend"],
    [/推荐|小红书/i, "推荐系统"], [/Embedding/i, "Embedding"], [/机器学习/i, "机器学习"], [/广告|ADX|DSP|SSP|竞价/i, "计算广告"],
    [/Attention|QKV/i, "Attention"], [/Linalg|linalg/i, "Linalg"], [/Affine/i, "Affine"], [/MemRef/i, "MemRef"], [/Pass/i, "编译器 Pass"],
  ];
  for (const [pattern, tag] of rules) if (pattern.test(title)) tags.add(tag);
  return [...tags].slice(0, 6);
}

function excerptFromBody(body) {
  const text = body
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/[#>*`_[\]~-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return text.length > 150 ? `${text.slice(0, 147)}…` : text;
}

function withoutExportTitle(body) {
  const lines = String(body ?? "").replace(/^\uFEFF/, "").trim().split(/\r?\n/);
  while (lines[0] === "") lines.shift();
  if (lines[0]?.match(/^#\s+/)) lines.shift();
  while (lines[0] === "") lines.shift();
  return lines.join("\n").trim();
}

function rewriteInternalLinks(body, slugById) {
  return body.replace(/https:\/\/www\.cnblogs\.com\/notlate-cn\/p\/(\d+)/g, (full, idText) => {
    const id = Number(idText);
    if (slugById.has(id)) return `https://notlate.cn/blog/${slugById.get(id)}`;
    if (Object.hasOwn(foldSlugs, id)) {
      return foldSlugs[id] ? `https://zhetengxia.com/blog/${foldSlugs[id]}` : "https://zhetengxia.com/";
    }
    return full;
  });
}

function remoteImageUrl(source) {
  const name = source.split(/[\\/]/).pop()?.trim();
  return name ? `https://cdn.jsdelivr.net/gh/notlate-cn/imgs/blogs/${encodeURI(name)}` : null;
}

function rewriteRelativeImages(body, missingImages) {
  let rewritten = body.replace(/!\[([^\]]*)\]\((?!https?:\/\/)([^)]+)\)/g, (full, alt, source) => {
    const cleanSource = source.trim().replace(/^<|>$/g, "");
    const name = cleanSource.split(/[\\/]/).pop() ?? cleanSource;
    if (missingRelativeImages.has(name)) {
      missingImages.add(name);
      return `> 图片资源未包含在博客园数据库备份中：${alt || name}`;
    }
    const remote = remoteImageUrl(cleanSource);
    return remote ? `![${alt}](${remote})` : full;
  });
  rewritten = rewritten.replace(/<img([^>]+)src=["'](?!https?:\/\/)([^"']+)["']([^>]*)>/gi, (full, before, source, after) => {
    const name = source.split(/[\\/]/).pop()?.trim() ?? source;
    if (missingRelativeImages.has(name)) {
      missingImages.add(name);
      return `\n> 图片资源未包含在博客园数据库备份中：${name}\n`;
    }
    const remote = remoteImageUrl(source);
    return remote ? `<img${before}src="${remote}"${after}>` : full;
  });
  return rewritten;
}

function normalizeFenceLanguages(body) {
  return body.replace(/^(\s*>\s*)?```(mlir|MLIR|tablegen|cuda|C\+\+|c\+\+|C|Plain)\s*$/gm, (_, prefix = "", language) => {
    if (["C", "C++", "c++", "cuda"].includes(language)) return `${prefix}\`\`\`cpp`;
    return `${prefix}\`\`\`text`;
  });
}

function addHtmlImageHints(body) {
  return body.replace(/<img\b([^>]*?)(\/?)>/gi, (full, attributes, slash) => {
    let next = attributes;
    if (!/\bloading\s*=/.test(next)) next += ' loading="lazy"';
    if (!/\bdecoding\s*=/.test(next)) next += ' decoding="async"';
    return `<img${next}${slash}>`;
  });
}

function frontmatter(row, slug, body) {
  const title = cleanTitle(row.Title);
  const pubDate = String(row.DateAdded).slice(0, 10);
  const updatedDate = String(row.DateUpdated).slice(0, 10);
  const description = String(row.Description ?? "").trim() || excerptFromBody(body);
  const category = categoryFor(title);
  const tags = tagsFor(title, category);
  const lines = [
    "---",
    `title: ${JSON.stringify(title)}`,
    `description: ${JSON.stringify(description)}`,
    `slug: ${JSON.stringify(slug)}`,
    `legacyId: ${row.Id}`,
    `sourceUrl: ${JSON.stringify(row.SourceUrl || `https://www.cnblogs.com/notlate-cn/p/${row.Id}`)}`,
    `pubDate: ${pubDate}`,
    ...(updatedDate !== pubDate ? [`updatedDate: ${updatedDate}`] : []),
    `category: ${JSON.stringify(category)}`,
    `tags: ${JSON.stringify(tags)}`,
    `featured: ${row.Id >= 19419609 && row.Id <= 19760130}`,
    "---",
    "",
    body,
    "",
  ];
  return lines.join("\n");
}

const rows = queryRows();
const technicalRows = rows.filter((row) => !excludedIds.has(row.Id));
const usedSlugs = new Set();
const slugById = new Map();
for (const row of technicalRows) slugById.set(row.Id, makeSlug(row.Title, row.Id, usedSlugs));

await mkdir(outputDir, { recursive: true });
const remoteImages = new Set();
const missingImages = new Set();
for (const row of technicalRows) {
  const slug = slugById.get(row.Id);
  let body = withoutExportTitle(row.Body);
  body = rewriteInternalLinks(body, slugById);
  body = rewriteRelativeImages(body, missingImages);
  body = normalizeFenceLanguages(body);
  body = addHtmlImageHints(body);
  for (const match of body.matchAll(/https?:\/\/[^\s)"'<>]+/g)) {
    if (/\.(?:png|jpe?g|gif|webp|svg)(?:\?[^\s)"'<>]+)?$/i.test(match[0])) remoteImages.add(match[0]);
  }
  await writeFile(join(outputDir, `${slug}.md`), frontmatter(row, slug, body), "utf8");
}

const excluded = rows.filter((row) => excludedIds.has(row.Id)).map((row) => ({ id: row.Id, title: cleanTitle(row.Title), suggestedSlug: foldSlugs[row.Id] ?? null }));
const report = {
  generatedAt: new Date().toISOString(),
  source: basename(dbPath),
  totalExported: rows.length,
  technicalImported: technicalRows.length,
  excludedForZhetengxia: excluded.length,
  excluded,
  remoteImageCount: remoteImages.size,
  remoteImages: [...remoteImages].sort(),
  missingRelativeImages: [...missingImages].sort(),
};
await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(`Imported ${technicalRows.length}/${rows.length} posts into ${outputDir}`);
console.log(`Excluded ${excluded.length} AI-tool/personal posts for zhetengxia.com`);
console.log(`Found ${remoteImages.size} remote image URLs; see ${reportPath}`);
