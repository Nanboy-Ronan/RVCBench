"""Server-side rendering: turns data.py into static HTML fragments.

Every fragment here is plain, crawlable HTML — no client-side rendering
step is required for the content to exist in the document. `build.py`
splices these into template.html at build time; the page's own JS then
only *enhances* what's already in the DOM (re-sorting existing rows,
richer hover tooltips) instead of creating it from scratch.
"""
import html
import json

import data as D


def esc(s):
    return html.escape(str(s), quote=True)


def fmt(v, d=3):
    return "—" if v is None else f"{v:.{d}f}"


# ---------------------------------------------------------------- stats ----
def render_stats():
    return "".join(
        f'<div class="stat"><div class="n tnum">{esc(s["n"])}</div><div class="l">{esc(s["l"])}</div></div>'
        for s in D.STATS
    )


# ----------------------------------------------------------- leaderboard ---
def render_leaderboard_rows():
    rows = sorted(D.LEADERBOARD, key=lambda d: d["sim"], reverse=True)
    max_sim = max(d["sim"] for d in rows)
    out = []
    for i, d in enumerate(rows):
        medal = ' medal' if i < 3 else ''
        pct = round(d["sim"] / max_sim * 100, 1)
        out.append(
            '<tr data-model="{m}" data-sim="{sim}" data-wer="{wer}" data-mos="{mos}" '
            'data-mcd="{mcd}" data-rtf="{rtf}" data-sva="{sva}" data-emo="{emo}">'
            '<td class="num"><span class="rank{medal}">{rank}</span></td>'
            '<td><span class="model-cell">{mname}</span></td>'
            '<td class="num"><span class="sim-bar-wrap"><span class="tnum">{simf}</span>'
            '<span class="sim-bar"><i style="width:{pct}%"></i></span></span></td>'
            '<td class="num tnum">{werf}</td>'
            '<td class="num tnum">{mosf}</td>'
            '<td class="num tnum">{mcdf}</td>'
            '<td class="num tnum">{rtff}</td>'
            '<td class="num tnum">{svaf}</td>'
            '<td class="num tnum">{emof}</td>'
            '</tr>'.format(
                m=esc(d["m"]), sim=d["sim"], wer=d["wer"], mos=d["mos"], mcd=d["mcd"],
                rtf=d["rtf"] if d["rtf"] is not None else "", sva=d["sva"], emo=d["emo"],
                medal=medal, rank=i + 1, mname=esc(d["m"]),
                simf=fmt(d["sim"]), werf=fmt(d["wer"]), mosf=fmt(d["mos"], 2),
                mcdf=fmt(d["mcd"], 2), rtff=fmt(d["rtf"], 2), svaf=fmt(d["sva"]), emof=fmt(d["emo"]),
                pct=pct,
            )
        )
    return "".join(out)


# ------------------------------------------------------------ robustness ---
def render_robustness_rows():
    out = []
    for d in D.ROBUSTNESS:
        out.append(
            f'<tr><td>{esc(d["m"])}</td><td class="num tnum">{fmt(d["clean"])}</td>'
            f'<td class="num tnum">{fmt(d["ss"])}</td><td class="num tnum">{fmt(d["ek"])}</td>'
            f'<td class="num tnum">{fmt(d["sp"])}</td><td class="num tnum">{fmt(d["gr"])}</td>'
            f'<td class="num tnum">{fmt(d["em"])}</td></tr>'
        )
    return "".join(out)


def render_dumbbell_svg():
    W, row_h, top, left, right = 1000, 32, 10, 190, 40
    plot_w = W - left - right
    n = len(D.ROBUSTNESS)
    H = top + n * row_h + 30
    max_v = 0.65

    def x(v):
        return left + (v / max_v) * plot_w

    parts = [f'<svg id="dumbbell" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" '
              f'role="img" aria-label="Speaker similarity: clean prompt vs. best-case protection, per model">']
    for g in (0, .1, .2, .3, .4, .5, .6):
        gx = x(g)
        parts.append(f'<line x1="{gx}" x2="{gx}" y1="{top}" y2="{top + n * row_h}" class="db-gridline"/>')
        parts.append(f'<text x="{gx}" y="{top + n * row_h + 18}" class="db-tick" text-anchor="middle">{g:.1f}</text>')

    for i, d in enumerate(D.ROBUSTNESS):
        y = top + i * row_h + row_h / 2
        methods = {k: d[k] for k in ("ss", "ek", "sp", "gr", "em")}
        present = {k: v for k, v in methods.items() if v is not None}
        min_key = min(present, key=present.get)
        min_val = present[min_key]
        min_name = D.ROBUSTNESS_METHOD_NAME[min_key]
        cx_clean, cx_min = x(d["clean"]), x(min_val)
        parts.append(f'<text x="{left - 14}" y="{y + 4}" class="db-row-label" text-anchor="end">{esc(d["m"])}</text>')
        parts.append(f'<line x1="{cx_min}" x2="{cx_clean}" y1="{y}" y2="{y}" class="db-line"/>')
        parts.append(
            f'<circle cx="{cx_min}" cy="{y}" r="5" class="db-min">'
            f'<title>{esc(d["m"])} · {esc(min_name)}: {min_val:.3f}</title></circle>'
        )
        parts.append(
            f'<circle cx="{cx_clean}" cy="{y}" r="5" class="db-clean">'
            f'<title>{esc(d["m"])} · Clean: {d["clean"]:.3f}</title></circle>'
        )
        parts.append(f'<text x="{cx_min}" y="{y - 10}" class="db-method-label" text-anchor="middle">{esc(min_name)}</text>')
    parts.append('</svg>')
    return "".join(parts)


# --------------------------------------------------------------- heatmap ---
# Column order + dimension grouping (see D.CROSS_DATASET_GROUPS): reorders the
# *presentation* only, so cells still carry the same underlying values as the
# source README table.
_GROUP_ORDER = [c for g in D.CROSS_DATASET_GROUPS for c in g["cols"]]
_COL_INDEX = {c: i for i, c in enumerate(D.CROSS_DATASET_COLUMNS)}
_DIM_TOKEN = {"input": "judge", "generation": "signal", None: None}


def render_heatmap_table():
    max_v = 0.78
    order = [_COL_INDEX[c] for c in _GROUP_ORDER]

    group_cells = ['<th class="rowh" rowspan="2" scope="col">Model</th>']
    for g in D.CROSS_DATASET_GROUPS:
        token = _DIM_TOKEN[g["dim"]]
        cls = f' class="dim-{token}"' if token else ""
        group_cells.append(f'<th colspan="{len(g["cols"])}" scope="colgroup"{cls}>{esc(g["label"])}</th>')
    group_row = "<tr>" + "".join(group_cells) + "</tr>"
    col_row = "<tr>" + "".join(f'<th scope="col">{esc(_GROUP_ORDER[i])}</th>' for i in range(len(_GROUP_ORDER))) + "</tr>"
    head = f"<thead>{group_row}{col_row}</thead>"

    rows = []
    for row in D.CROSS_DATASET:
        cells = [f'<th class="rowh" scope="row">{esc(row["m"])}</th>']
        for ci in order:
            v = row["v"][ci]
            col = D.CROSS_DATASET_COLUMNS[ci]
            if v is None:
                cells.append('<td class="empty">—</td>')
                continue
            t = max(0.0, min(1.0, v / max_v))
            bg = f"color-mix(in oklab, var(--surface-3), var(--signal) {round(t * 100)}%)"
            fg = "var(--text-on-accent)" if t > 0.52 else "var(--text-primary)"
            title = f"{row['m']} · {col}: {v:.3f}"
            cells.append(
                f'<td style="background:{bg};color:{fg}" title="{esc(title)}" '
                f'data-m="{esc(row["m"])}" data-c="{esc(col)}" data-v="{v:.3f}">{v:.2f}</td>'
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="heatmap" id="hmTable">{head}<tbody>{"".join(rows)}</tbody></table>'


# ------------------------------------------------------------- dimensions --
def render_dimensions():
    out = []
    for d in D.DIMENSIONS:
        subtests = "".join(f"<li>{esc(s)}</li>" for s in d["subtests"])
        if d["demo_anchor"]:
            demo = f'<a class="dim-link" href="#{d["demo_anchor"]}">{esc(d["demo_label"])} →</a>'
        else:
            demo = f'<span class="dim-link dim-pending">{esc(d["demo_label"])}</span>'
        out.append(
            f'<div class="dim-card dim-{d["token"]}">'
            f'<h3>{esc(d["name"])}</h3>'
            f'<p class="dim-q">{esc(d["question"])}</p>'
            f'<ul class="dim-list">{subtests}</ul>'
            f'{demo}'
            f'</div>'
        )
    return "".join(out)


# ---------------------------------------------------------------- models ---
def render_models_grid():
    out = []
    for m in D.MODELS:
        cls = "chip" if m["b"] else "chip other"
        out.append(
            f'<div class="{cls}"><span class="name"><span class="badge"></span>{esc(m["n"])}</span>'
            f'<span class="key mono">{esc(m["key"])}</span></div>'
        )
    return "".join(out)


def render_protect_grid():
    return "".join(
        f'<div class="protect-card"><h3>{esc(p["n"])}</h3><p>{esc(p["desc"])}</p></div>'
        for p in D.PROTECTIONS
    )


def render_datasets_rows():
    return "".join(
        f'<tr><td class="mono">{esc(d["k"])}</td><td><span class="lang-tag">{esc(d["lang"])}</span></td>'
        f'<td>{esc(d["desc"])}</td></tr>'
        for d in D.DATASETS
    )


def render_faq():
    out = []
    for item in D.FAQ:
        out.append(
            f'<div class="faq-item" itemscope itemprop="mainEntity" itemtype="https://schema.org/Question">'
            f'<h3 itemprop="name">{esc(item["q"])}</h3>'
            f'<div itemscope itemprop="acceptedAnswer" itemtype="https://schema.org/Answer">'
            f'<p itemprop="text">{esc(item["a"])}</p></div></div>'
        )
    return "".join(out)


# -------------------------------------------------------------- JSON-LD ----
def render_jsonld():
    authors = [{"@type": "Person", "name": n} for n in D.CITATION["authors"]]
    graph = [
        {
            "@type": "Dataset",
            "@id": D.SITE["url"] + "#dataset",
            "name": "RVCBench",
            "alternateName": "RVCBench: Voice Cloning Robustness Benchmark",
            "description": (
                "A benchmark for voice-cloning robustness, speaker privacy, and audio-protection methods, "
                "covering 27 TTS/VC models, 10 dataset configurations, and 5 audio-protection methods."
            ),
            "url": D.SITE["url"],
            "sameAs": [D.SITE["repo"], D.SITE["dataset"], D.SITE["paper"]],
            "license": "https://creativecommons.org/publicdomain/zero/1.0/",
            "creator": authors,
            "keywords": [
                "voice cloning", "text-to-speech", "speaker privacy", "audio deepfake",
                "adversarial audio", "speaker verification", "audio protection", "TTS benchmark",
            ],
            "distribution": {"@type": "DataDownload", "contentUrl": D.SITE["dataset"], "encodingFormat": "application/x-parquet"},
        },
        {
            "@type": "ScholarlyArticle",
            "@id": D.SITE["url"] + "#paper",
            "headline": D.CITATION["title"],
            "author": authors,
            "datePublished": D.CITATION["year"],
            "url": D.SITE["paper"],
            "identifier": f"arXiv:{D.CITATION['arxiv_id']}",
            "about": {"@id": D.SITE["url"] + "#dataset"},
        },
        {
            "@type": "SoftwareSourceCode",
            "@id": D.SITE["url"] + "#code",
            "name": "RVCBench",
            "codeRepository": D.SITE["repo"],
            "programmingLanguage": "Python",
            "license": "https://creativecommons.org/publicdomain/zero/1.0/",
        },
        {
            "@type": "FAQPage",
            "@id": D.SITE["url"] + "#faq",
            "mainEntity": [
                {
                    "@type": "Question",
                    "name": item["q"],
                    "acceptedAnswer": {"@type": "Answer", "text": item["a"]},
                }
                for item in D.FAQ
            ],
        },
    ]
    doc = {"@context": "https://schema.org", "@graph": graph}
    return json.dumps(doc, ensure_ascii=False, indent=2)


# ------------------------------------------------------------- llms.txt ----
def render_llms_txt():
    """https://llmstxt.org convention: a plain-text summary for LLM crawlers."""
    lines = [
        "# RVCBench",
        "",
        "> A benchmark for voice-cloning robustness, speaker privacy, and audio-protection "
        "methods, covering 27 TTS/VC models, 5 protection methods, and 10 dataset configurations.",
        "",
        "RVCBench applies audio-protection perturbations to source speech, runs zero-shot and "
        "fine-tuning voice-cloning models against clean and protected prompts, optionally denoises "
        "protected audio, and scores every run on speaker similarity (SIM), word error rate (WER), "
        "SpeechMOS (MOS), mel-cepstral distortion (MCD), real-time factor (RTF), speaker-verification "
        "accuracy (SVA), and emotion match rate.",
        "",
        "## Key facts",
        "",
    ]
    for item in D.FAQ:
        lines.append(f"- {item['a']}")
    lines += [
        "",
        "## Links",
        "",
        f"- [Paper (arXiv:{D.CITATION['arxiv_id']})]({D.SITE['paper']})",
        f"- [Dataset (Hugging Face)]({D.SITE['dataset']})",
        f"- [Interactive demo]({D.SITE['demo']})",
        f"- [Code repository]({D.SITE['repo']})",
        "",
        "## Citation",
        "",
        "```",
        "@article{jin2026rvcbench,",
        f"  title   = {{{D.CITATION['title']}}},",
        f"  author  = {{{' and '.join(D.CITATION['authors'])}}},",
        f"  journal = {{arXiv preprint arXiv:{D.CITATION['arxiv_id']}}},",
        f"  year    = {{{D.CITATION['year']}}}",
        "}",
        "```",
    ]
    return "\n".join(lines) + "\n"
