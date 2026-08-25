"""
Report figures.

Every figure is generated from the JSON and NPZ artefacts written by
``src.train``, so nothing in the write-up is transcribed by hand and a rerun
regenerates the whole figure set from the measurements.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"

PALETTE = {
    "primary": "#1f4e79",
    "accent": "#c0392b",
    "muted": "#7f8c8d",
    "good": "#27ae60",
    "warn": "#e67e22",
    "grid": "#dfe4e8",
}


def _style(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def _save(fig, name: str) -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path.relative_to(ROOT)}")
    return path


def _load(name: str):
    p = REPORTS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------
def fig_leakage_audit():
    audit = _load("audit")
    if not audit:
        return None
    keys = ["A_reference_pipeline", "B_identifiers_only", "C_no_identifiers",
            "D_honest_labels", "E_label_permutation", "F_novel_variant_split",
            "G_oversample_before_split"]
    labels = ["Reference\npipeline", "Identifier\ncolumns only",
              "Biology only\n(no identifiers)", "Honest labels\n(unknowns dropped)",
              "Labels\nshuffled", "Novel-variant\nsplit",
              "Over-sample\nbefore split"]
    keys = [k for k in keys if k in audit]
    labels = [l for k, l in zip(
        ["A_reference_pipeline", "B_identifiers_only", "C_no_identifiers",
         "D_honest_labels", "E_label_permutation", "F_novel_variant_split",
         "G_oversample_before_split"], labels) if k in audit]

    train = [audit[k]["train_auc"] or 0 for k in keys]
    test = [audit[k]["test_auc"] or 0 for k in keys]

    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(keys))
    ax.bar(x - 0.2, train, 0.4, label="Training AUC", color=PALETTE["muted"])
    ax.bar(x + 0.2, test, 0.4, label="Held-out test AUC", color=PALETTE["primary"])
    ax.axhline(0.5, color=PALETTE["accent"], linestyle="--", linewidth=1.2,
               label="Chance")
    for xi, t in zip(x, test):
        ax.text(xi + 0.2, t + 0.015, f"{t:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 1.08)
    _style(ax, "Where the reference pipeline's score comes from", "", "AUC-ROC")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    return _save(fig, "01_leakage_audit")
