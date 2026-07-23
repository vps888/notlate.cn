import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import { siteConfig } from "../site.config";

export async function GET(context) {
  const posts = (await getCollection("posts")).sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());
  return rss({
    title: siteConfig.name,
    description: siteConfig.description,
    site: context.site,
    items: posts.map((post) => ({
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.pubDate,
      link: `/blog/${post.data.slug}`,
    })),
  });
}
