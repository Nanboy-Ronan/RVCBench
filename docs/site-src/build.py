#!/usr/bin/env python3
"""Build the RVCBench homepage.

Produces two outputs from the same template.html + data.py:

  docs/index.html
      A complete, standalone HTML5 document for GitHub Pages: full <head>
      with meta/Open Graph/Twitter/canonical/citation_* tags and JSON-LD
      structured data, fonts and logo referenced as external files under
      docs/assets/ (cacheable, no huge inline payload).

  docs/site-src/artifact.html
      A content-only fragment (no <!DOCTYPE>/<html>/<head>/<body>) with
      fonts and logo inlined as base64 data URIs, for publishing to a
      Claude Artifact (which requires a single self-contained file and
      injects its own document shell).

Usage:
    python3 docs/site-src/build.py
"""
import base64
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).parent
DOCS = SRC.parent
sys.path.insert(0, str(SRC))

import data as D          # noqa: E402
import render as R        # noqa: E402

# template.html already declares each @font-face block's own font-weight/style;
# these tokens only need to resolve to a src: value (data URI vs. relative file).
FONT_FILES = {
    "FRAUNCES_HERO": "fraunces_hero_900.woff2",
    "FRAUNCES_H2": "fraunces_h2_600.woff2",
    "FRAUNCES_ITALIC": "fraunces_italic_500.woff2",
    "PLEXSANS": "plexsans_var.woff2",
    "PLEXMONO_400": "plexmono_400.woff2",
    "PLEXMONO_500": "plexmono_500.woff2",
    "PLEXMONO_600": "plexmono_600.woff2",
}


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def render_body(template: str) -> str:
    """Fill in every server-rendered content placeholder. Shared by both builds."""
    return (
        template
        .replace("__STATS_GRID__", R.render_stats())
        .replace("__DIMENSIONS_GRID__", R.render_dimensions())
        .replace("__WHY_TABLE__", R.render_why_table())
        .replace("__LEADERBOARD_ROWS__", R.render_leaderboard_rows())
        .replace("__ROBUSTNESS_ROWS__", R.render_robustness_rows())
        .replace("__DUMBBELL_SVG__", R.render_dumbbell_svg())
        .replace("__HEATMAP_TABLE__", R.render_heatmap_table())
        .replace("__MODELS_GRID__", R.render_models_grid())
        .replace("__PROTECT_GRID__", R.render_protect_grid())
        .replace("__DATASETS_ROWS__", R.render_datasets_rows())
        .replace("__FAQ_ITEMS__", R.render_faq())
    )


def build_head(canonical: bool = True) -> str:
    title = "RVCBench — Voice Cloning Robustness Benchmark"
    desc = (
        "RVCBench benchmarks voice-cloning robustness, speaker privacy, and audio-protection methods "
        "across 27 TTS/VC models, 5 protection methods, and 10 dataset conditions, with a public "
        "leaderboard, dataset, and reproducible evaluation pipeline."
    )
    url = D.SITE["url"]
    og_image = url + "assets/og-image.png"
    citation_authors = "\n    ".join(
        f'<meta name="citation_author" content="{a}">' for a in D.CITATION["authors"]
    )
    jsonld = R.render_jsonld()
    canonical_tag = f'<link rel="canonical" href="{url}">' if canonical else ""
    return f"""<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<meta name="robots" content="index, follow">
<meta name="keywords" content="voice cloning benchmark, TTS robustness, speaker privacy, audio deepfake, adversarial audio, speaker verification, voice cloning protection, RVCBench">
{canonical_tag}
<link rel="icon" href="assets/logo.png" type="image/png">
<link rel="apple-touch-icon" href="assets/logo.png">

<meta property="og:type" content="website">
<meta property="og:site_name" content="RVCBench">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}">
<meta property="og:url" content="{url}">
<meta property="og:image" content="{og_image}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title}">
<meta name="twitter:description" content="{desc}">
<meta name="twitter:image" content="{og_image}">

<meta name="citation_title" content="{D.CITATION['title']}">
{citation_authors}
<meta name="citation_publication_date" content="{D.CITATION['year']}">
<meta name="citation_arxiv_id" content="{D.CITATION['arxiv_id']}">
<meta name="citation_online_date" content="{D.CITATION['year']}">

<script type="application/ld+json">
{jsonld}
</script>
"""


def build_pages():
    template = (SRC / "template.html").read_text(encoding="utf-8")

    # External, cacheable asset references (no base64 inflation on GH Pages).
    body = template
    for token, fname in FONT_FILES.items():
        body = body.replace(f"__FONT_{token}_SRC__", f"url(assets/fonts/{fname}) format('woff2')")
    body = body.replace("__LOGO_SRC__", "assets/logo.png")
    body = render_body(body)

    head = build_head(canonical=True)
    doc = (
        "<!doctype html>\n"
        f'<html lang="en">\n<head>\n{head}</head>\n<body>\n{body}\n</body>\n</html>\n'
    )

    out = DOCS / "index.html"
    out.write_text(doc, encoding="utf-8")

    # Copy raw asset files alongside index.html so the relative paths above resolve.
    assets_out = DOCS / "assets"
    (assets_out / "fonts").mkdir(parents=True, exist_ok=True)
    for fname in FONT_FILES.values():
        shutil.copyfile(SRC / "assets/fonts" / fname, assets_out / "fonts" / fname)
    shutil.copyfile(SRC / "assets/logo.png", assets_out / "logo.png")

    print(f"wrote {out} ({len(doc.encode('utf-8')) / 1024:.0f} KB) + docs/assets/")

    llms_out = DOCS / "llms.txt"
    llms_out.write_text(R.render_llms_txt(), encoding="utf-8")
    print(f"wrote {llms_out}")


def build_artifact():
    template = (SRC / "template.html").read_text(encoding="utf-8")
    body = template
    for token, fname in FONT_FILES.items():
        body = body.replace(f"__FONT_{token}_SRC__", f"url(data:font/woff2;base64,{b64(SRC / 'assets/fonts' / fname)}) format('woff2')")
    body = body.replace("__LOGO_SRC__", f"data:image/png;base64,{b64(SRC / 'assets/logo.png')}")
    body = render_body(body)

    out = SRC / "artifact.html"
    out.write_text(body, encoding="utf-8")
    print(f"wrote {out} ({len(body.encode('utf-8')) / 1024:.0f} KB)")


if __name__ == "__main__":
    build_pages()
    build_artifact()
