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
