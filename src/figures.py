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


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
def fig_model_comparison():
    cv = _load("cv")
    if not cv:
        return None
    blocked = _load("blocked_cv") or {"models": {}}
    names = cv["ranking"]
    means = [cv["models"][n]["cv_auc_mean"] for n in names]
    stds = [cv["models"][n]["cv_auc_std"] for n in names]
    bl = [blocked["models"].get(n, {}).get("blocked_auc_mean") for n in names]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(names) + 2.2))
    y = np.arange(len(names))
    ax.barh(y, means, xerr=stds, color=PALETTE["primary"], height=0.55,
            error_kw=dict(ecolor=PALETTE["muted"], lw=1, capsize=3),
            label="Repeated stratified CV")
    have = [(yi, b) for yi, b in zip(y, bl) if b is not None]
    if have:
        ax.scatter([b for _, b in have], [yi for yi, _ in have], s=34,
                   color=PALETTE["accent"], zorder=5, marker="D",
                   label="Position-blocked CV")
    for yi, m in zip(y, means):
        ax.text(m + 0.008, yi, f"{m:.3f}", va="center", fontsize=7.5)
    ax.axvline(0.5, color=PALETTE["muted"], linestyle=":", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.45, max(means) + 0.08)
    _style(ax, "Model comparison on leakage-free features", "AUC-ROC", "")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    return _save(fig, "02_model_comparison")


# ---------------------------------------------------------------------------
# ROC, precision-recall, calibration, decision curve
# ---------------------------------------------------------------------------
def fig_performance_panel():
    npz = REPORTS / "test_predictions.npz"
    if not npz.exists():
        return None
    from .evaluate import calibration_curve_points, decision_curve

    d = np.load(npz)
    y, p_cal, p_raw = d["y"], d["p_cal"], d["p_raw"]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))

    fpr, tpr, _ = roc_curve(y, p_cal)
    from sklearn.metrics import auc as _auc
    axes[0, 0].plot(fpr, tpr, color=PALETTE["primary"], lw=2,
                    label=f"Model (AUC {_auc(fpr, tpr):.3f})")
    axes[0, 0].plot([0, 1], [0, 1], "--", color=PALETTE["muted"], lw=1,
                    label="Chance")
    _style(axes[0, 0], "ROC curve, held-out test set",
           "1 - specificity", "Sensitivity")
    axes[0, 0].legend(fontsize=8, frameon=False, loc="lower right")

    prec, rec, _ = precision_recall_curve(y, p_cal)
    axes[0, 1].plot(rec, prec, color=PALETTE["primary"], lw=2, label="Model")
    axes[0, 1].axhline(y.mean(), color=PALETTE["accent"], linestyle="--", lw=1.2,
                       label=f"Prevalence ({y.mean():.3f})")
    _style(axes[0, 1], "Precision-recall curve", "Recall (sensitivity)",
           "Precision (PPV)")
    axes[0, 1].legend(fontsize=8, frameon=False)

    for probs, lab, col in [(p_raw, "Uncalibrated", PALETTE["warn"]),
                            (p_cal, "Isotonic-calibrated", PALETTE["primary"])]:
        xs, ys, ns = calibration_curve_points(y, probs, n_bins=8)
        if len(xs):
            axes[1, 0].plot(xs, ys, "o-", color=col, lw=1.8, ms=5, label=lab)
    axes[1, 0].plot([0, 1], [0, 1], "--", color=PALETTE["muted"], lw=1,
                    label="Perfect calibration")
    _style(axes[1, 0], "Calibration", "Predicted risk", "Observed frequency")
    axes[1, 0].legend(fontsize=8, frameon=False)

    thr, nb, all_nb = decision_curve(y, p_cal)
    axes[1, 1].plot(thr, nb, color=PALETTE["primary"], lw=2, label="Model")
    axes[1, 1].plot(thr, all_nb, "--", color=PALETTE["warn"], lw=1.4,
                    label="Test everyone")
    axes[1, 1].axhline(0, color=PALETTE["muted"], lw=1, linestyle=":",
                       label="Test no one")
    axes[1, 1].set_ylim(min(-0.02, nb.min() - 0.01), max(nb.max(), 0.05) * 1.35)
    _style(axes[1, 1], "Decision curve (clinical net benefit)",
           "Threshold probability", "Net benefit")
    axes[1, 1].legend(fontsize=8, frameon=False)

    fig.tight_layout()
    return _save(fig, "03_performance_panel")


# ---------------------------------------------------------------------------
# Biology of the cohort
# ---------------------------------------------------------------------------
def fig_biology():
    from .datasets import load_champ, split_by_label
    from .features import normalise_variant_type, normalise_severity

    lab, _ = split_by_label(load_champ())
    y = (lab["inhibitor"] == 1).astype(int).values
    vt = np.array([normalise_variant_type(v) for v in lab["Variant Type"]])
    sv = np.array([normalise_severity(v) for v in lab["Reported Clinical Severity"]])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    order = ["large_structural", "nonsense", "frameshift", "splice",
             "small_structural", "missense", "synonymous", "regulatory"]
    order = [o for o in order if (vt == o).sum() >= 5]
    rates = [y[vt == o].mean() * 100 for o in order]
    ns = [(vt == o).sum() for o in order]
    colors = [PALETTE["accent"] if r >= 25 else
              PALETTE["warn"] if r >= 15 else PALETTE["primary"] for r in rates]
    axes[0].barh(range(len(order)), rates, color=colors, height=0.6)
    for i, (r, nn) in enumerate(zip(rates, ns)):
        axes[0].text(r + 0.7, i, f"{r:.1f}%  (n={nn})", va="center", fontsize=7.5)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([o.replace("_", " ") for o in order], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(rates) * 1.4)
    _style(axes[0], "Inhibitor rate by molecular consequence", "% inhibitor-positive", "")

    sorder = [s for s in ["severe", "moderate", "mild", "mixed", "unknown"]
              if (sv == s).sum() >= 5]
    srates = [y[sv == s].mean() * 100 for s in sorder]
    sns_ = [(sv == s).sum() for s in sorder]
    axes[1].bar(range(len(sorder)), srates, color=PALETTE["primary"], width=0.55)
    for i, (r, nn) in enumerate(zip(srates, sns_)):
        axes[1].text(i, r + 0.6, f"{r:.1f}%\nn={nn}", ha="center", fontsize=7.5)
    axes[1].set_xticks(range(len(sorder)))
    axes[1].set_xticklabels(sorder, fontsize=8)
    axes[1].set_ylim(0, max(srates) * 1.35)
    _style(axes[1], "Inhibitor rate by clinical severity", "", "% inhibitor-positive")

    fig.tight_layout()
    return _save(fig, "04_cohort_biology")


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
def fig_explainability(importance_df=None, block_df=None, attention_df=None):
    parts = [d for d in (importance_df, block_df, attention_df) if d is not None]
    if not parts:
        return None
    ncol = len(parts)
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 5.4))
    if ncol == 1:
        axes = [axes]
    i = 0
    if importance_df is not None:
        d = importance_df.head(15).iloc[::-1]
        axes[i].barh(range(len(d)), d["mean_abs_shap"], color=PALETTE["primary"],
                     height=0.66)
        axes[i].set_yticks(range(len(d)))
        axes[i].set_yticklabels(d["feature"], fontsize=7.5)
        _style(axes[i], "SHAP global importance", "mean |SHAP|", "")
        i += 1
    if block_df is not None:
        d = block_df.iloc[::-1]
        axes[i].barh(range(len(d)), d["share"] * 100, color=PALETTE["good"],
                     height=0.6)
        axes[i].set_yticks(range(len(d)))
        axes[i].set_yticklabels(d["block"], fontsize=8)
        for j, s in enumerate(d["share"] * 100):
            axes[i].text(s + 0.4, j, f"{s:.1f}%", va="center", fontsize=7.5)
        _style(axes[i], "SHAP attribution by biological block", "% of total |SHAP|", "")
        i += 1
    if attention_df is not None:
        d = attention_df.iloc[::-1]
        axes[i].barh(range(len(d)), d["mean_attention"], xerr=d["std_attention"],
                     color=PALETTE["warn"], height=0.6,
                     error_kw=dict(ecolor=PALETTE["muted"], lw=1, capsize=3))
        axes[i].set_yticks(range(len(d)))
        axes[i].set_yticklabels(d["block"], fontsize=8)
        _style(axes[i], "Intrinsic block attention", "mean attention weight", "")
    fig.tight_layout()
    return _save(fig, "05_explainability")


# ---------------------------------------------------------------------------
# External validation
# ---------------------------------------------------------------------------
def fig_external():
    ext = _load("external")
    fin = _load("final")
    if not (ext and fin):
        return None
    labels = ["CHAMP (F8)\ninternal test", "CHBMP (F9)\nzero-shot transfer"]
    vals = [fin["test_calibrated_youden"]["auc_roc"], ext["metrics"]["auc_roc"]]
    los = [fin["auc_ci"]["lo"], ext["auc_ci"]["lo"]]
    his = [fin["auc_ci"]["hi"], ext["auc_ci"]["hi"]]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(2)
    err = np.array([[v - lo for v, lo in zip(vals, los)],
                    [hi - v for v, hi in zip(vals, his)]])
    ax.bar(x, vals, 0.45, yerr=err, color=[PALETTE["primary"], PALETTE["good"]],
           error_kw=dict(ecolor=PALETTE["muted"], lw=1.2, capsize=6))
    ax.axhline(0.5, color=PALETTE["accent"], linestyle="--", lw=1.2,
               label="Chance")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.0)
    _style(ax, "Internal and cross-gene external validation", "", "AUC-ROC (95% CI)")
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, "06_external_validation")


def build_all() -> list:
    made = []
    for fn in (fig_leakage_audit, fig_model_comparison, fig_performance_panel,
               fig_biology, fig_external):
        try:
            p = fn()
            if p:
                made.append(p)
        except Exception as exc:                      # a missing stage is fine
            print(f"  !! {fn.__name__}: {type(exc).__name__}: {exc}")
    return made


if __name__ == "__main__":
    build_all()
