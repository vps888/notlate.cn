import type { APIRoute } from "astro";

export const GET: APIRoute = () => new Response(null, {
  status: 301,
  headers: {
    Location: "/sitemap-index.xml",
    "Cache-Control": "public, max-age=3600",
  },
});
