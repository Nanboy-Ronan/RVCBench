# RVCBench homepage

`index.html` is the built homepage. It's a complete, standalone HTML5
document — full `<head>` (meta description, Open Graph/Twitter cards,
`citation_*` tags for Google Scholar, canonical URL, JSON-LD structured
data) plus `robots.txt`, `sitemap.xml`, and `llms.txt` alongside it.

**All content is server-rendered at build time** — the leaderboard,
robustness chart, cross-dataset heatmap, model/protection/dataset lists,
and FAQ are real HTML in the document, not populated by JavaScript after
load. JS only *enhances* what's already there (column sorting, richer
hover tooltips). This matters for both classic search crawlers and AI
answer-engine crawlers (GPTBot, ClaudeBot, PerplexityBot, …), most of
which don't execute JavaScript — verified by loading the built page with
JS disabled and checking the real numbers are still in the text.

## Enable on GitHub Pages

1. Push this `docs/` folder to the `main` branch.
2. In the GitHub repo: **Settings → Pages → Build and deployment → Source:
   "Deploy from a branch"**, then set **Branch: `main` / folder: `/docs`**.
3. GitHub publishes it at `https://nanboy-ronan.github.io/RVCBench/`.
4. Once live, submit `sitemap.xml` in Google Search Console (optional but
   speeds up indexing).

## Editing the page

Source lives in `site-src/`:

- `site-src/data.py` — every number/string on the page (leaderboard,
  robustness table, cross-dataset matrix, model list, protection methods,
  datasets, FAQ, citation). **Single source of truth** — update this when
  new results land, sourced from the root `README.md`.
- `site-src/render.py` — turns `data.py` into the static HTML fragments
  (table rows, the dumbbell SVG, the heatmap, JSON-LD, `llms.txt`).
- `site-src/template.html` — page markup, CSS, and the (small,
  enhancement-only) client-side JS.
- `site-src/build.py` — assembles everything into `docs/index.html` (+
  `docs/llms.txt`, `docs/assets/`) and a separate
  `site-src/artifact.html` (fonts/logo inlined as base64, for publishing
  to a Claude Artifact preview — not committed, rebuilt on demand).
- `site-src/assets/` — self-hosted `.woff2` fonts and the logo.

Rebuild after any edit:

```bash
python3 docs/site-src/build.py
```

The social-share card (`assets/og-image.png`, referenced by the Open
Graph/Twitter meta tags) is a one-off Playwright screenshot, not part of
the regular build — regenerate it by hand if the brand visuals change.
