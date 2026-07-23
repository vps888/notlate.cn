# NotLate 技术博客

`notlate.cn` 的独立 Astro 静态站点，内容聚焦 MLIR、AI 编译器、推荐系统与机器学习工程实践。

## 本地运行

```bash
npm install
npm run dev
```

## Cloudflare Pages

- Framework preset：`Astro`
- Build command：`npm run build`
- Output directory：`dist`
- Node.js：建议使用 20 或更高版本
- 自定义域名：`notlate.cn`

站点不依赖服务端运行时，Cloudflare Pages 直接托管构建后的静态文件即可。`wrangler.toml` 仅用于需要从 Wrangler 部署时的默认配置。

## 重新导入博客园备份

导出 ZIP 内是 SQLite 数据库。如果以后继续导出，可以先解压数据库，再运行本项目根目录的导入脚本：

```bash
unzip -p ~/Downloads/cnblogs_blog_notlate-cn.*.zip > /tmp/notlate-blog.db
node ./import-cnblogs-export.mjs /tmp/notlate-blog.db src/content/posts
```

脚本会把 AI 编译器、推荐系统和机器学习文章导入本站，并把 AI 工具/个人折腾文章列入 `migration-report.json`，不会自动覆盖 `zhetengxia.com` 的现有文章。

## 内容边界与互链

技术文章放在本站；AI 工具、AI Coding、网络和个人折腾内容继续放在 `zhetengxia.com`。两个站点在导航、页脚和文章提示区互相链接，避免复制同一篇文章造成重复内容。
