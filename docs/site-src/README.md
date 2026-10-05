# RVCBench homepage

`index.html` is the built homepage. It's a complete, standalone HTML5
document — full `<head>` (meta description, Open Graph/Twitter cards,
`citation_*` tags for Google Scholar, canonical URL, JSON-LD structured
data) plus `robots.txt` and `llms.txt` alongside it. The combined website build produces
the complete `sitemap.xml` and `llms-full.txt` in its deployment artifact.

**All content is server-rendered at build time** — the leaderboard,
robustness chart, cross-dataset heatmap, model/protection/dataset lists,
and FAQ are real HTML in the document, not populated by JavaScript after
load. JS only *enhances* what's already there (column sorting, richer
hover tooltips). This matters for both classic search crawlers and AI
answer-engine crawlers (GPTBot, ClaudeBot, PerplexityBot, …), most of
which don't execute JavaScript — verified by loading the built page with
JS disabled and checking the real numbers are still in the text.

## Build and publish the documentation website

The homepage stays at `https://nanboy-ronan.github.io/RVCBench/`. The searchable user documentation
is at `https://nanboy-ronan.github.io/RVCBench/docs/`, built with Material for MkDocs from the existing
Markdown guides in `docs/`. `mkdocs.yml` defines the navigation, theme, search and strict link checks.

```bash
python -m pip install -r docs/requirements.txt
python scripts/build_website.py
python scripts/check_website.py
```

The generated `site/` directory contains both the homepage and the documentation. It is ignored by Git.
For a live documentation preview, run `python -m mkdocs serve` and open the URL it prints.

`.github/workflows/docs.yml` checks every PR and deploys `main` through GitHub Pages. The repository's
**Settings → Pages → Build and deployment → Source** must be **GitHub Actions**. Deployment uploads
only the built `site/` artifact. It does not install RVCBench, download model weights or publish PyPI.
The existing CPU CI still checks the homepage's committed generated HTML.

Edit the Markdown guide once to update its GitHub view and its website page. Add new top-level guides
to `mkdocs.yml`; advanced linked guides are still built and searchable even when omitted from the sidebar.
Keep links between guides relative (`metrics.md`); MkDocs turns them into website URLs.
`scripts/docs_hooks.py` keeps links to source code and notebooks pointing to GitHub.

## Editing the page

Source lives in `docs/site-src/` (paths below are relative to `docs/`):

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
  `site-src/artifact.html` (fonts/logo inlined as base64, for a
  single-file preview — not committed, rebuilt on demand).
- `site-src/assets/` — self-hosted `.woff2` fonts and the logo.

Rebuild after any edit:

```bash
python3 docs/site-src/build.py
```

The social-share card (`assets/og-image.png`, referenced by the Open Graph/Twitter meta tags)
is generated from `social-card.html`. Run `python scripts/build_social_card.py` after editing it
(requires Playwright and Chromium). The PNG is committed, so regular builds need no browser.

See [search and AI discoverability](discoverability.md) for the product-claim mapping, metadata checks,
GitHub Pages robots scope and the authenticated indexing steps that must be tracked separately.
