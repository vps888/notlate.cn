import sitemap from "@astrojs/sitemap";
import { defineConfig } from "astro/config";

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
    rehypePlugins: [rehypeLazyImages],
  },
  integrations: [sitemap()],
});
