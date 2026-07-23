import { glob } from "astro/loaders";
import { defineCollection, z } from "astro:content";

const posts = defineCollection({
  loader: glob({
    pattern: "**/*.md",
    base: "./src/content/posts",
  }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    slug: z.string(),
    legacyId: z.number().int(),
    sourceUrl: z.string().url().optional(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    category: z.enum(["AI 编译器", "推荐系统与机器学习", "工程与算法"]),
    tags: z.array(z.string()),
    featured: z.boolean().default(false),
  }),
});

export const collections = { posts };
