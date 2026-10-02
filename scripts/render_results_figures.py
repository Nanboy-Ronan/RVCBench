#!/usr/bin/env python3
"""Generate benchmark result figures for the RVCBench README.

Usage:
    python scripts/render_results_figures.py \
        --results-dir /path/to/results \
        --out-dir figs
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

TIMESTAMP_RE = re.compile(r"^\d{8}-\d{6}$")

MODEL_NAME_PATTERNS = [
    ("zipvoice_distill", "ZipVoice Distill"),
    ("bark_voice_clone", "Bark Voice Clone"),
    ("fishspeech", "FishSpeech"),
    ("ozspeech", "OZSpeech"),
    ("styletts2", "StyleTTS 2"),
    ("styletts", "StyleTTS 2"),
    ("sparkaudio", "SparkTTS"),
    ("sparktts", "SparkTTS"),
    ("moss_ttsd", "MOSS-TTSD"),
    ("moss", "MOSS-TTSD"),
    ("higgs_audio", "Higgs Audio"),
    ("higgs", "Higgs Audio"),
    ("cosyvoice2", "CosyVoice 2"),
    ("cozyvoice2", "CosyVoice 2"),
    ("cosyvoice", "CosyVoice 2"),
    ("cozyvoice", "CosyVoice 2"),
    ("glowtts", "GlowTTS"),
    ("glm_tts", "GLM-TTS"),
    ("glmtts", "GLM-TTS"),
    ("vibevoice", "VibeVoice"),
    ("mgm_omni", "MGM-Omni"),
    ("qwen3_omni", "Qwen3-Omni"),
    ("qwen3_tts", "Qwen3-TTS"),
    ("qwen_tts", "Qwen3-TTS"),
    ("kimi_audio", "Kimi Audio"),
    ("f5_tts", "F5-TTS"),
    ("f5tts", "F5-TTS"),
    ("maskgct", "MaskGCT"),
    ("openvoice", "OpenVoice V2"),
    ("coqui_xtts", "XTTS-v2"),
    ("xtts_v2", "XTTS-v2"),
    ("xtts", "XTTS-v2"),
    ("index_tts", "IndexTTS"),
    ("indextts", "IndexTTS"),
    ("zipvoice", "ZipVoice"),
    ("vall_e", "VALL-E"),
    ("bertvits2", "BertVITS2"),
    ("bert_ots", "BertVITS2"),
    ("playdiffusion", "PlayDiffusion"),
]

PROTECTION_PATTERNS = [
    ("_with_safespeech", "SafeSpeech"),
    ("_with_enkidu", "Enkidu"),
    ("_with_spec", "Spectral"),
    ("_with_gaussian_noise", "GR-Noise"),
    ("_with_em", "EM"),
    ("_with_pop", "EM"),  # backward compatibility for historically mislabelled EM runs
]

PROT_ORDER = ["SafeSpeech", "Enkidu", "Spectral", "GR-Noise", "EM"]

MODEL_ORDER = [
    "Qwen3-TTS",
    "CosyVoice 2",
    "IndexTTS",
    "GLM-TTS",
    "ZipVoice",
    "MaskGCT",
    "F5-TTS",
    "Higgs Audio",
    "MGM-Omni",
    "PlayDiffusion",
    "FishSpeech",
    "VibeVoice",
    "MOSS-TTSD",
    "OZSpeech",
    "XTTS-v2",
    "SparkTTS",
    "OpenVoice V2",
    "StyleTTS 2",
    "ZipVoice Distill",
    "VALL-E",
    "BertVITS2",
    "GlowTTS",
]


def _pretty_model(config: dict, run_dir: str) -> str:
    variant = (config.get("adversary") or {}).get("variant", "") or ""
    run_name = config.get("run_name", "") or ""
    text = " ".join([variant.lower(), run_name.lower(), run_dir.lower()])
    for key, pretty in MODEL_NAME_PATTERNS:
        if key in text:
            return pretty
    return run_dir


def _find_latest_metrics(run_root: str) -> str | None:
    candidates = [
        d for d in os.listdir(run_root)
        if TIMESTAMP_RE.match(d)
        and os.path.isfile(os.path.join(run_root, d, "metrics.json"))
    ]
    if candidates:
        return os.path.join(run_root, sorted(candidates)[-1], "metrics.json")
    root_p = os.path.join(run_root, "metrics.json")
    return root_p if os.path.isfile(root_p) else None


def _load_clean_libritts(results_root: str) -> dict[str, dict]:
    """Return {model_name: {metric: value}} for clean LibriTTS runs."""
    out: dict[str, dict] = {}
    for run_dir in os.listdir(results_root):
        if "_with_" in run_dir:
            continue
        if "libritts" not in run_dir and "libratts" not in run_dir:
            continue
        if "denoiser" in run_dir or "dns64" in run_dir:
            continue
        run_root = os.path.join(results_root, run_dir)
        if not os.path.isdir(run_root):
            continue
        p = _find_latest_metrics(run_root)
        if p is None:
            continue
        try:
            d = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue
        ge = d.get("generation_evaluation") or {}
        if not isinstance(ge, dict):
            continue
        model = _pretty_model(d.get("config") or {}, run_dir)
        if model not in out:
            out[model] = {
                "SIM":     ge.get("avg_sim"),
                "WER":     ge.get("avg_wer"),
                "MOS":     ge.get("avg_speechmos_mos"),
                "MCD":     ge.get("avg_mcd"),
                "RTF":     ge.get("rtf"),
                "SVA":     ge.get("avg_sva"),
                "Emotion": ge.get("emotion_match_rate"),
            }
    return out


def _load_protection_sim(results_root: str) -> dict[tuple[str, str], float]:
    """Return {(model, protection): SIM} for LibriTTS protected runs."""
    out: dict[tuple[str, str], float] = {}
    for run_dir in os.listdir(results_root):
        if "_with_" not in run_dir:
            continue
        if "libritts" not in run_dir and "libratts" not in run_dir:
            continue
        if "denoiser" in run_dir or "dns64" in run_dir:
            continue
        prot = None
        for suffix, name in PROTECTION_PATTERNS:
            if suffix in run_dir:
                prot = name
                break
        if prot is None:
            continue
        run_root = os.path.join(results_root, run_dir)
        if not os.path.isdir(run_root):
            continue
        p = _find_latest_metrics(run_root)
        if p is None:
            continue
        try:
            d = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue
        ge = d.get("generation_evaluation") or {}
        sim = ge.get("avg_sim")
        if sim is None:
            continue
        model = _pretty_model(d.get("config") or {}, run_dir)
        key = (model, prot)
        if key not in out:
            out[key] = sim
    return out


def _canonical_ds(config: dict, run_dir: str) -> str:
    raw = (config.get("dataset") or {}).get("name", "") or ""
    raw_l = raw.lower()
    dir_l = run_dir.lower()
    if "multispeaker_libri" in raw_l or "multispeaker_libri" in dir_l:
        return "Multispeaker LibriSpeech"
    if "long_librispeech" in raw_l or "long_librispeech" in dir_l:
        return "Long LibriSpeech"
    if "bilingual_uedin" in raw_l or "bilingual_uedin" in dir_l:
        return "Bilingual_uedin"
    if "french" in raw_l or "french" in dir_l:
        return "French"
    if "aishell" in raw_l or "aishell" in dir_l:
        return "AISHELL"
    if "background_clean" in raw_l or "background_clean" in dir_l:
        return "Background-VCTK (clean)"
    if "background_noise" in raw_l or "background_noise" in dir_l:
        return "Background-VCTK (noise)"
    if "vctk_text_robust" in raw_l or "vctk_text_robust" in dir_l:
        return "Hallucination-VCTK"
    if "robotcall" in raw_l or "robotcall" in dir_l:
        return "Robotcall"
    if "libritts" in raw_l or "libritts" in dir_l or "libratts" in dir_l:
        return "LibriTTS"
    if "vctk" in raw_l or "vctk" in dir_l:
        return "VCTK"
    return raw or "Unknown"


def _load_all_datasets(results_root: str) -> dict[str, dict[str, dict]]:
    """Return {dataset: {model: metrics}} for all clean, non-protected runs."""
    data: dict[str, dict[str, dict]] = defaultdict(dict)
    sources: dict[str, dict[str, bool]] = defaultdict(dict)

    for run_dir in os.listdir(results_root):
        run_root = os.path.join(results_root, run_dir)
        if not os.path.isdir(run_root):
            continue
        p = _find_latest_metrics(run_root)
        if p is None:
            continue
        try:
            d = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue
        ge = d.get("generation_evaluation") or {}
        if not isinstance(ge, dict):
            continue
        config = d.get("config") or {}
        run_name = config.get("run_name", "") or ""
        is_protected = "_with_" in run_dir or "_with_" in run_name

        ds = _canonical_ds(config, run_dir)
        model = _pretty_model(config, run_dir)

        existing = sources[ds].get(model)
        if existing is not None:
            if existing is False and is_protected:
                continue
            if existing is True and is_protected:
                continue

        data[ds][model] = {
            "SIM": ge.get("avg_sim"),
            "MOS": ge.get("avg_speechmos_mos"),
            "WER": ge.get("avg_wer"),
        }
        sources[ds][model] = is_protected

    return dict(data)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _cell_cmap(vmin: float, vmax: float, higher_is_better: bool):
    if higher_is_better:
        colors = ["#f7d4d4", "#ffe8b2", "#c8e6c9"]
    else:
        colors = ["#c8e6c9", "#ffe8b2", "#f7d4d4"]
    cmap = LinearSegmentedColormap.from_list("metric", colors, N=256)
    return cmap, Normalize(vmin=vmin, vmax=vmax)


# ---------------------------------------------------------------------------
# Figure 1: Clean LibriTTS Leaderboard
# ---------------------------------------------------------------------------

METRIC_META = {
    "SIM":     ("SIM ↑",     True,  "{:.3f}"),
    "WER":     ("WER ↓",     False, "{:.3f}"),
    "MOS":     ("MOS ↑",     True,  "{:.2f}"),
    "MCD":     ("MCD ↓",     False, "{:.2f}"),
    "RTF":     ("RTF ↓",     False, "{:.2f}"),
    "SVA":     ("SVA ↑",     True,  "{:.3f}"),
    "Emotion": ("Emo ↑",     True,  "{:.3f}"),
}

DISPLAY_METRICS = ["SIM", "WER", "MOS", "MCD", "RTF", "SVA", "Emotion"]


def render_leaderboard(clean_data: dict[str, dict], out_path: str) -> None:
    ordered = [m for m in MODEL_ORDER if m in clean_data]
    ordered.sort(key=lambda m: clean_data[m].get("SIM") or 0, reverse=True)

    n_models = len(ordered)
    n_metrics = len(DISPLAY_METRICS)

    fig_h = 0.42 * n_models + 1.4
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.set_xlim(0, n_metrics)
    ax.set_ylim(-0.5, n_models - 0.5)
    ax.invert_yaxis()
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")
    ax.axis("off")

    col_labels = [METRIC_META[m][0] for m in DISPLAY_METRICS]
    for j, label in enumerate(col_labels):
        ax.text(
            j + 0.5, -0.65, label,
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="#1a1a2e",
        )

    col_vals: dict[str, list[float]] = {k: [] for k in DISPLAY_METRICS}
    for model in ordered:
        for k in DISPLAY_METRICS:
            v = clean_data[model].get(k)
            if v is not None:
                col_vals[k].append(v)

    col_norm: dict[str, tuple] = {}
    for k in DISPLAY_METRICS:
        vals = col_vals[k]
        if vals:
            col_norm[k] = _cell_cmap(min(vals), max(vals), METRIC_META[k][1])

    rank_colors = ["#FFD700", "#C0C0C0", "#CD7F32"]

    for i, model in enumerate(ordered):
        row_bg = "#ffffff" if i % 2 == 0 else "#f0f4ff"
        ax.add_patch(mpatches.FancyBboxPatch(
            (-3.2, i - 0.45), n_metrics + 3.5, 0.9,
            boxstyle="round,pad=0.02", linewidth=0,
            facecolor=row_bg, zorder=0,
        ))

        rank = i + 1
        badge_color = rank_colors[i] if i < 3 else "#e0e0e0"
        badge_text_color = "#5d4037" if i < 3 else "#616161"
        ax.add_patch(mpatches.FancyBboxPatch(
            (-3.1, i - 0.28), 0.55, 0.56,
            boxstyle="round,pad=0.04", linewidth=0,
            facecolor=badge_color, zorder=1,
        ))
        ax.text(-2.83, i, f"#{rank}", ha="center", va="center",
                fontsize=8, fontweight="bold", color=badge_text_color, zorder=2)

        ax.text(-2.4, i, model, ha="left", va="center",
                fontsize=9.5, color="#1a1a2e", zorder=2)

        for j, metric in enumerate(DISPLAY_METRICS):
            v = clean_data[model].get(metric)
            if v is not None and metric in col_norm:
                cmap, norm = col_norm[metric]
                cell_color = cmap(norm(v))
                ax.add_patch(mpatches.FancyBboxPatch(
                    (j + 0.06, i - 0.38), 0.88, 0.76,
                    boxstyle="round,pad=0.02", linewidth=0,
                    facecolor=cell_color, zorder=1,
                ))
                fmt = METRIC_META[metric][2]
                ax.text(j + 0.5, i, fmt.format(v),
                        ha="center", va="center",
                        fontsize=9, color="#1a1a2e", fontweight="500", zorder=2)
            else:
                ax.text(j + 0.5, i, "—",
                        ha="center", va="center",
                        fontsize=9, color="#9e9e9e", zorder=2)

    ax.axhline(-0.5, color="#dee2e6", linewidth=1.2, zorder=3)

    for j in range(1, n_metrics):
        ax.axvline(j, ymin=0, ymax=1, color="#dee2e6", linewidth=0.5,
                   linestyle="--", alpha=0.6, zorder=0)

    ax.set_xlim(-3.2, n_metrics + 0.3)

    fig.suptitle(
        "RVCBench Leaderboard · LibriTTS (clean)",
        fontsize=13, fontweight="bold", color="#1a1a2e", y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Protection Robustness Heatmap
# ---------------------------------------------------------------------------

def render_protection_heatmap(
    clean_data: dict[str, dict],
    prot_data: dict[tuple[str, str], float],
    out_path: str,
) -> None:
    models_with_prot = {m for m, _ in prot_data}
    ordered = [m for m in MODEL_ORDER if m in clean_data and m in models_with_prot]
    ordered.sort(key=lambda m: clean_data[m].get("SIM") or 0, reverse=True)

    all_prots = ["Clean"] + PROT_ORDER
    n_models = len(ordered)
    n_prots = len(all_prots)

    mat = np.full((n_models, n_prots), np.nan)
    for i, model in enumerate(ordered):
        sim_clean = clean_data[model].get("SIM")
        if sim_clean is not None:
            mat[i, 0] = sim_clean
        for j, prot in enumerate(PROT_ORDER, start=1):
            v = prot_data.get((model, prot))
            if v is not None:
                mat[i, j] = v

    vmin = np.nanmin(mat)
    vmax = np.nanmax(mat)
    colors = ["#d32f2f", "#ef9a9a", "#fff9c4", "#a5d6a7", "#2e7d32"]
    cmap = LinearSegmentedColormap.from_list("sim_heat", colors, N=256)
    norm = Normalize(vmin=vmin, vmax=vmax)

    cell_w = 1.3
    cell_h = 0.52
    fig_w = n_prots * cell_w + 3.2
    fig_h = n_models * cell_h + 1.6

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")
    ax.set_xlim(-3.2, n_prots * cell_w)
    ax.set_ylim(-0.5, n_models - 0.5)
    ax.invert_yaxis()
    ax.axis("off")

    col_colors = {
        "Clean":      "#1565c0",
        "SafeSpeech": "#6a1b9a",
        "Enkidu":     "#1b5e20",
        "Spectral":   "#e65100",
        "GR-Noise":   "#37474f",
        "EM":         "#880e4f",
    }
    for j, prot in enumerate(all_prots):
        color = col_colors.get(prot, "#333")
        ax.text(
            j * cell_w + cell_w / 2, -0.72,
            prot, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=color,
        )

    rank_colors_3 = ["#FFD700", "#C0C0C0", "#CD7F32"]
    for i, model in enumerate(ordered):
        row_bg = "#ffffff" if i % 2 == 0 else "#f0f4ff"
        ax.add_patch(mpatches.FancyBboxPatch(
            (-3.2, i - 0.45), n_prots * cell_w + 3.3, 0.9,
            boxstyle="round,pad=0.02", linewidth=0,
            facecolor=row_bg, zorder=0,
        ))

        rank = i + 1
        badge_color = rank_colors_3[i] if i < 3 else "#e0e0e0"
        badge_text_color = "#5d4037" if i < 3 else "#616161"
        ax.add_patch(mpatches.FancyBboxPatch(
            (-3.1, i - 0.28), 0.55, 0.56,
            boxstyle="round,pad=0.04", linewidth=0,
            facecolor=badge_color, zorder=1,
        ))
        ax.text(-2.83, i, f"#{rank}", ha="center", va="center",
                fontsize=8, fontweight="bold", color=badge_text_color, zorder=2)

        ax.text(-2.4, i, model, ha="left", va="center",
                fontsize=9, color="#1a1a2e", zorder=2)

        for j in range(n_prots):
            v = mat[i, j]
            x0 = j * cell_w + 0.06
            if not np.isnan(v):
                face = cmap(norm(v))
                lum = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2]
                txt_color = "#1a1a2e" if lum > 0.45 else "white"
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x0, i - 0.38), cell_w - 0.12, 0.76,
                    boxstyle="round,pad=0.02", linewidth=0,
                    facecolor=face, zorder=1,
                ))
                fw = "bold" if j == 0 else "normal"
                ax.text(j * cell_w + cell_w / 2, i,
                        f"{v:.3f}",
                        ha="center", va="center",
                        fontsize=8.5, color=txt_color, fontweight=fw, zorder=2)
            else:
                ax.text(j * cell_w + cell_w / 2, i, "—",
                        ha="center", va="center",
                        fontsize=8.5, color="#9e9e9e", zorder=2)

    ax.axvline(cell_w, ymin=0, ymax=1, color="#90a4ae",
               linewidth=1.5, linestyle="-", alpha=0.7, zorder=3)
    ax.axhline(-0.5, color="#dee2e6", linewidth=1.2, zorder=3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.025, pad=0.01, aspect=40)
    cbar.set_label("Speaker Similarity (SIM)", fontsize=9, labelpad=4)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Protection Robustness · Speaker Similarity (SIM) on LibriTTS",
        fontsize=13, fontweight="bold", color="#1a1a2e", y=1.002,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1.0])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Multi-dataset SIM heatmap
# ---------------------------------------------------------------------------

DATASET_ORDER = [
    "LibriTTS",
    "VCTK",
    "Multispeaker LibriSpeech",
    "Long LibriSpeech",
    "AISHELL",
    "French",
    "Bilingual_uedin",
    "Background-VCTK (clean)",
    "Background-VCTK (noise)",
    "Hallucination-VCTK",
]

DATASET_SHORT = {
    "LibriTTS":               "LibriTTS",
    "VCTK":                   "VCTK",
    "Multispeaker LibriSpeech": "Multi-spk",
    "Long LibriSpeech":       "Long",
    "AISHELL":                "AISHELL",
    "French":                 "French",
    "Bilingual_uedin":        "Bilingual",
    "Background-VCTK (clean)": "BG-clean",
    "Background-VCTK (noise)": "BG-noise",
    "Hallucination-VCTK":     "Hallucin.",
}


def render_multidataset_heatmap(
    all_data: dict[str, dict[str, dict]],
    out_path: str,
) -> None:
    libritts_sim = {
        m: (v.get("SIM") or 0)
        for m, v in all_data.get("LibriTTS", {}).items()
    }
    top_models = sorted(
        [m for m in MODEL_ORDER if m in libritts_sim],
        key=lambda m: libritts_sim.get(m, 0),
        reverse=True,
    )[:16]

    datasets = [ds for ds in DATASET_ORDER if ds in all_data]

    n_models = len(top_models)
    n_ds = len(datasets)

    mat = np.full((n_models, n_ds), np.nan)
    for i, model in enumerate(top_models):
        for j, ds in enumerate(datasets):
            v = (all_data.get(ds, {}).get(model) or {}).get("SIM")
            if v is not None:
                mat[i, j] = v

    vmin = np.nanmin(mat)
    vmax = np.nanmax(mat)
    colors = ["#b71c1c", "#ef9a9a", "#fff9c4", "#a5d6a7", "#1b5e20"]
    cmap = LinearSegmentedColormap.from_list("multi_heat", colors, N=256)
    norm = Normalize(vmin=vmin, vmax=vmax)

    cell_w = 1.1
    cell_h = 0.52
    fig_w = n_ds * cell_w + 3.2
    fig_h = n_models * cell_h + 1.8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")
    ax.set_xlim(-3.2, n_ds * cell_w)
    ax.set_ylim(-0.5, n_models - 0.5)
    ax.invert_yaxis()
    ax.axis("off")

    for j, ds in enumerate(datasets):
        ax.text(
            j * cell_w + cell_w / 2, -0.82,
            DATASET_SHORT.get(ds, ds),
            ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#1a1a2e", rotation=30,
        )

    rank_colors_3 = ["#FFD700", "#C0C0C0", "#CD7F32"]
    for i, model in enumerate(top_models):
        row_bg = "#ffffff" if i % 2 == 0 else "#f0f4ff"
        ax.add_patch(mpatches.FancyBboxPatch(
            (-3.2, i - 0.45), n_ds * cell_w + 3.3, 0.9,
            boxstyle="round,pad=0.02", linewidth=0,
            facecolor=row_bg, zorder=0,
        ))

        rank = i + 1
        badge_color = rank_colors_3[i] if i < 3 else "#e0e0e0"
        badge_text_color = "#5d4037" if i < 3 else "#616161"
        ax.add_patch(mpatches.FancyBboxPatch(
            (-3.1, i - 0.28), 0.55, 0.56,
            boxstyle="round,pad=0.04", linewidth=0,
            facecolor=badge_color, zorder=1,
        ))
        ax.text(-2.83, i, f"#{rank}", ha="center", va="center",
                fontsize=8, fontweight="bold", color=badge_text_color, zorder=2)

        ax.text(-2.4, i, model, ha="left", va="center",
                fontsize=9, color="#1a1a2e", zorder=2)

        for j in range(n_ds):
            v = mat[i, j]
            if not np.isnan(v):
                face = cmap(norm(v))
                lum = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2]
                txt_color = "#1a1a2e" if lum > 0.45 else "white"
                ax.add_patch(mpatches.FancyBboxPatch(
                    (j * cell_w + 0.06, i - 0.38), cell_w - 0.12, 0.76,
                    boxstyle="round,pad=0.02", linewidth=0,
                    facecolor=face, zorder=1,
                ))
                ax.text(j * cell_w + cell_w / 2, i,
                        f"{v:.3f}",
                        ha="center", va="center",
                        fontsize=7.5, color=txt_color, zorder=2)
            else:
                ax.text(j * cell_w + cell_w / 2, i, "—",
                        ha="center", va="center",
                        fontsize=8, color="#bdbdbd", zorder=2)

    ax.axhline(-0.5, color="#dee2e6", linewidth=1.2, zorder=3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.025, pad=0.01, aspect=40)
    cbar.set_label("Speaker Similarity (SIM)", fontsize=9, labelpad=4)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Speaker Similarity (SIM) Across All Benchmark Datasets",
        fontsize=13, fontweight="bold", color="#1a1a2e", y=1.02,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1.0])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RVCBench result figures.")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(REPO_ROOT, "results"),
        help="Root directory containing benchmark result folders.",
    )
    parser.add_argument(
        "--out-dir",
        default="figs",
        help="Directory to write output PNG files.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir

    print("Loading clean LibriTTS results …")
    clean = _load_clean_libritts(results_dir)
    print(f"  {len(clean)} models found.")

    print("Loading protection robustness results …")
    prot = _load_protection_sim(results_dir)
    print(f"  {len(prot)} (model, protection) pairs found.")

    print("Loading multi-dataset results …")
    all_ds = _load_all_datasets(results_dir)
    print(f"  {len(all_ds)} datasets found.")

    render_leaderboard(clean, os.path.join(out_dir, "results_leaderboard.png"))
    render_protection_heatmap(clean, prot, os.path.join(out_dir, "results_protection_heatmap.png"))
    render_multidataset_heatmap(all_ds, os.path.join(out_dir, "results_multidataset_heatmap.png"))
    print("Done.")


if __name__ == "__main__":
    main()
