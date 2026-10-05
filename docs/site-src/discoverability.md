# Search and AI discoverability

RVCBench has two public product claims: automatic speech metrics for user-provided audio, and
dataset-backed evaluation of voice cloning models. Homepage, documentation, package metadata and
repository metadata should describe the same capabilities. The paper title remains its published title.

## Search intent and evidence

| User question | Canonical answer | Evidence |
| --- | --- | --- |
| How do I evaluate voice cloning in Python? | [Metrics guide](https://nanboy-ronan.github.io/RVCBench/docs/metrics/) | Seven public metrics, input requirements and batch example |
| How do I benchmark a new voice cloning model? | [Model evaluation](https://nanboy-ronan.github.io/RVCBench/docs/adding_a_model/) | Prompt export, output layout, scoring, resume and comparison |
| Which voice cloning datasets should I use? | [Suites](https://nanboy-ronan.github.io/RVCBench/docs/core_suite/) | Fixed selections, counts, coverage and preview status |
| Does my model need an adapter? | [FAQ](https://nanboy-ronan.github.io/RVCBench/docs/faq/) | WAV output contract and separate model environments |
| How do I install the scoring package? | [Installation](https://nanboy-ronan.github.io/RVCBench/docs/installation/) | PyPI, CPU/GPU setup, model assets and troubleshooting |

Use natural descriptions of actual capabilities. Do not claim every possible metric is implemented,
automatic inference for arbitrary models, guaranteed model superiority, or guaranteed search ranking.
Do not add fabricated reviews, ratings, citations or performance numbers to structured data.

## Build checks

```bash
python scripts/build_website.py
python scripts/check_website.py
```

The checker verifies unique titles and summaries, canonical URLs, social metadata, JSON-LD, all internal
links and anchors, package-version consistency, and complete sitemap coverage. These checks run in the
documentation deployment workflow. Core product descriptions are rendered in HTML without JavaScript.

- Homepage JSON-LD separates the software application, source code, dataset, paper and website.
- Documentation pages include per-page summaries, social previews, article metadata and breadcrumbs.
- `sitemap.xml` covers the homepage and all published guides. Rebuilds do not fabricate modification dates.
- `llms.txt` links to authoritative task guides; `llms-full.txt` exports their actual text with source URLs.
  These files are optional convenience formats, not a requirement or guarantee of inclusion in AI search.
- `python scripts/build_social_card.py` regenerates the share image from `social-card.html`.

## GitHub Pages robots scope

This is a project site under `/RVCBench/`. Standards-compliant crawlers consult
`https://nanboy-ronan.github.io/robots.txt`, not `/RVCBench/robots.txt`. The latter is only a reference
policy. The origin-root file returned HTTP 404 during this audit: no robots prohibition was found.
If a root-domain robots policy is added later, recheck access for Googlebot, Bingbot and search crawlers.
Do not claim that editing the project-local file changes origin-wide crawl policy.

The complete sitemap is linked from the HTML head and included in the project reference policy.
For a custom domain, install the robots policy at that domain's root and update every canonical URL.

## Indexing and measurement

Public HTTP and rendering checks establish technical accessibility, not search-engine indexing or ranking.
Use the owner's authenticated dashboards to complete and monitor the following:

1. Verify the URL-prefix property `https://nanboy-ronan.github.io/RVCBench/` in Google Search Console
   and Bing Webmaster Tools; submit `https://nanboy-ronan.github.io/RVCBench/sitemap.xml`.
2. Inspect homepage, metrics, model evaluation and FAQ URLs for indexing and canonical selection.
3. Track branded and task-based queries, impressions, clicks and landing pages; review Bing AI citations
   where available. Record dates and sources instead of treating one-off search probes as a ranking audit.
4. Refresh the PyPI description and Documentation URL through a package release; published distribution
   metadata cannot be changed merely by editing the repository.

No authenticated Search Console or Bing indexing submission was performed by the website build.

## Official references

- [Google: AI features and your website](https://developers.google.com/search/docs/appearance/ai-features)
- [Google: robots.txt scope](https://developers.google.com/search/docs/crawling-indexing/robots/create-robots-txt)
- [Bing: AI Performance](https://www.bing.com/webmasters/help/ai-performance-9f8e7d6c)
