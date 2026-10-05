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


# --------------------------------------------------------------- why-table -
def render_why_table():
    rows = "".join(
        f'<tr><th scope="row">{esc(r["dim"])}</th>'
        f'<td>{esc(r["typical"])}</td>'
        f'<td class="why-win">{esc(r["rvcbench"])}</td></tr>'
        for r in D.WHY_COMPARISON
    )
    return (
        '<table class="why-table">'
        '<thead><tr><th scope="col"></th>'
        '<th scope="col">You provide</th>'
        '<th scope="col">RVCBench</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
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
            'data-mcd="{mcd}" data-sva="{sva}" data-emo="{emo}">'
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
                simf=fmt(d["sim"], 2), werf=fmt(d["wer"], 2), mosf=fmt(d["mos"], 2),
                mcdf=fmt(d["mcd"], 2), rtff=fmt(d["rtf"], 2), svaf=fmt(d["sva"], 2), emof=fmt(d["emo"], 2),
                pct=pct,
            )
        )
    return "".join(out)


# ------------------------------------------------------------ robustness ---
def render_robustness_rows():
    out = []
    for d in D.ROBUSTNESS:
        out.append(
            f'<tr><td>{esc(d["m"])}</td><td class="num tnum">{fmt(d["clean"], 2)}</td>'
            f'<td class="num tnum">{fmt(d["ss"], 2)}</td><td class="num tnum">{fmt(d["ek"], 2)}</td>'
            f'<td class="num tnum">{fmt(d["sp"], 2)}</td><td class="num tnum">{fmt(d["gr"], 2)}</td>'
            f'<td class="num tnum">{fmt(d["em"], 2)}</td></tr>'
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
            title = f"{row['m']} · {col}: {v:.2f}"
            cells.append(
                f'<td style="background:{bg};color:{fg}" title="{esc(title)}" '
                f'data-m="{esc(row["m"])}" data-c="{esc(col)}" data-v="{v:.2f}">{v:.2f}</td>'
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
    base = D.SITE["url"]
    description = ("Comprehensive voice cloning evaluation with seven automatic speech metrics and "
                   "ready-to-use benchmark datasets. Score your own audio or evaluate model outputs with provided data.")
    graph = [
        {
            "@type": "WebSite", "@id": base + "#website", "url": base,
            "name": "RVCBench", "description": description, "inLanguage": "en",
        },
        {
            "@type": "WebPage", "@id": base + "#webpage", "url": base,
            "name": "RVCBench | Comprehensive Voice Cloning Evaluation",
            "description": description, "isPartOf": {"@id": base + "#website"},
            "mainEntity": {"@id": base + "#software"},
        },
        {
            "@type": "SoftwareApplication", "@id": base + "#software", "name": "RVCBench",
            "description": description, "url": base, "softwareVersion": D.VERSION,
            "applicationCategory": "DeveloperApplication", "operatingSystem": "Linux",
            "softwareRequirements": "Python 3.10 or newer; FFmpeg for speech scoring",
            "installUrl": D.SITE["pypi"], "softwareHelp": D.SITE["docs"],
            "license": "https://creativecommons.org/publicdomain/zero/1.0/", "author": authors,
            "isAccessibleForFree": True,
            "featureList": [
                "Automatic speech metrics: SIM, SVA, WER, SpeechMOS (UTMOS), MCD, STOI and emotion consistency",
                "Dataset-backed evaluation: export prompts, generate with your model, score and compare",
                "CPU or GPU scoring, cached scorer models and resumable dataset evaluations",
            ],
            "sameAs": [D.SITE["repo"], D.SITE["pypi"]],
        },
        {
            "@type": "Dataset", "@id": base + "#dataset", "name": "RVCBench evaluation datasets",
            "description": ("Reference audio, transcripts and evaluation data for voice cloning across languages, "
                            "speakers and recording conditions. Packaged suites provide fixed input selections."),
            "url": D.SITE["dataset"], "license": "https://creativecommons.org/publicdomain/zero/1.0/",
            "creator": authors, "keywords": ["voice cloning evaluation", "speech evaluation", "benchmark datasets"],
            "includedInDataCatalog": {"@type": "DataCatalog", "name": "Hugging Face Datasets",
                                      "url": "https://huggingface.co/datasets"},
        },
        {
            "@type": "ScholarlyArticle", "@id": base + "#paper", "headline": D.CITATION["title"],
            "author": authors, "datePublished": D.CITATION["year"], "url": D.SITE["paper"],
            "identifier": f"arXiv:{D.CITATION['arxiv_id']}",
            "isPartOf": {"@type": "PublicationVolume", "name": D.CITATION["venue"]},
            "description": "Accepted to NeurIPS 2026.", "about": {"@id": base + "#dataset"},
        },
        {
            "@type": "SoftwareSourceCode", "@id": base + "#code", "name": "RVCBench",
            "codeRepository": D.SITE["repo"], "programmingLanguage": "Python", "softwareVersion": D.VERSION,
            "targetProduct": {"@id": base + "#software"},
            "license": "https://creativecommons.org/publicdomain/zero/1.0/",
        },
        {
            "@type": "FAQPage", "@id": base + "#faq",
            "mainEntity": [{"@type": "Question", "name": item["q"],
                            "acceptedAnswer": {"@type": "Answer", "text": item["a"]}} for item in D.FAQ],
        },
    ]
    return json.dumps({"@context": "https://schema.org", "@graph": graph}, ensure_ascii=False, indent=2)


# ------------------------------------------------------------- llms.txt ----
def render_llms_txt():
    """https://llmstxt.org convention: a plain-text summary for LLM crawlers."""
    lines = [
        "# RVCBench",
        "",
        "> A comprehensive voice cloning evaluation package with automatic speech metrics and datasets. "
        "Supports scoring your own audio and evaluating models with versioned benchmark suites.",
        "",
        "Install with `pip install \"rvcbench[eval]\"` and run `rvcbench setup-scorers`. "
        "Use `rvcbench.metrics.Evaluator` with your own audio, or `rvcbench prompts` and `rvcbench score` "
        "with onboarding-v1 (52 utterances), core-v1 (480), or full-v1 (12,724). "
        "Metrics include SIM, SVA, WER, MOS, MCD, STOI and emotion consistency. "
        "Dataset scoring supports --resume; comparison validates scoring fingerprints.",
        "",
        "## Key facts",
        "",
    ]
    for item in D.FAQ:
        lines.extend([f"### {item['q']}", "", item["a"], ""])
    lines += [
        "",
        "## Links",
        "",
        f"- [Paper (arXiv:{D.CITATION['arxiv_id']})]({D.SITE['paper']})",
        f"- [Dataset (Hugging Face)]({D.SITE['dataset']})",
        f"- [Interactive demo]({D.SITE['demo']})",
        f"- [Code repository]({D.SITE['repo']})",
        f"- [Documentation]({D.SITE['url']}docs/)",
        f"- [Quickstart]({D.SITE['url']}docs/quickstart/)",
        f"- [Automatic metrics for your audio]({D.SITE['url']}docs/metrics/)",
        f"- [Evaluate models with benchmark data]({D.SITE['url']}docs/adding_a_model/)",
        f"- [Evaluation FAQ and coverage]({D.SITE['url']}docs/faq/)",
        f"- [Python API reference]({D.SITE['url']}docs/api/)",
        f"- [PyPI package]({D.SITE['pypi']})",
        f"- [Full documentation text]({D.SITE['url']}llms-full.txt)",
        "",
        "## Citation",
        "",
        "```",
        "@inproceedings{jin2026rvcbench,",
        f"  title   = {{{D.CITATION['title']}}},",
        f"  author  = {{{' and '.join(D.CITATION['authors'])}}},",
        f"  booktitle = {{{D.CITATION['venue']}}},",
        f"  url     = {{{D.SITE['paper']}}},",
        f"  year    = {{{D.CITATION['year']}}}",
        "}",
        "```",
    ]
    return "\n".join(lines) + "\n"


def render_validated_runs(report_dir=None):
    """Render only coverage-checked reports; never substitute for historical rows."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'src'))
    from rvcbench.benchmark.artifacts import validate_report, metric_means
    reports = []
    directory = Path(report_dir) if report_dir is not None else root / 'docs' / 'validated_runs'
    for path in sorted(directory.glob('*.json')):
        payload = json.loads(path.read_text())
        run = payload['run']
        validate_report(run, verify_files=False)
        if run['config']['vc']['model'] == 'smoke':
            raise ValueError('Smoke fixtures must never appear on the public leaderboard')
        means = metric_means(run['samples'], run['coverage']['required_metrics'])
        if means != payload['means']:
            raise ValueError('Report means disagree with sample metrics: ' + str(path))
        reports.append('<tr><td>' + esc(str(run['config']['vc']['model'])) + '</td><td>'
                       + str(run['coverage']['requested']) + '</td><td>'
                       + esc(', '.join(f'{k}: {v:.4f}' for k, v in means.items()))
                       + '</td><td><a href="validated_runs/' + esc(path.name)
                       + '">Manifest and provenance</a></td></tr>')
    if not reports:
        return '<p>No reports have been published under the new manifest protocol yet.</p>'
    return '<table><thead><tr><th>Model</th><th>Samples</th><th>Metrics</th><th>Evidence</th></tr></thead><tbody>' + ''.join(reports) + '</tbody></table>'
