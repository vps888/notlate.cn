import type { APIRoute } from "astro";

// Keep the legacy URL valid for older Search Console submissions.
export const GET: APIRoute = () => new Response(null, {
  status: 301,
  headers: {
    Location: "/sitemap-index.xml",
    "Cache-Control": "public, max-age=3600",
  },
});
