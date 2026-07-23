import sitemap from "@astrojs/sitemap";
import { defineConfig } from "astro/config";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";

function rehypeLazyImages() {
  return (tree) => {
    const visit = (node) => {
      if (node.type === "element" && node.tagName === "img") {
        node.properties ??= {};
        node.properties.loading ??= "lazy";
        node.properties.decoding ??= "async";
      }
      for (const child of node.children ?? []) visit(child);
    };
    visit(tree);
  };
}

export default defineConfig({
  site: "https://notlate.cn",
  output: "static",
  build: {
    inlineStylesheets: "always",
  },
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [[rehypeKatex, { strict: false }], rehypeLazyImages],
  },
  integrations: [sitemap()],
});
