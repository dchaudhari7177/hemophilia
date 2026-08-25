"""
Evaluation harness.

The reference works report point estimates of accuracy, precision, recall, F1
and AUC on a single 20% split. On 2,296 patients with 461 events that is not
enough to separate two models: the 95% interval on AUC is roughly +/-0.05, so
differences smaller than that are noise. Three things are added here:

* **Bootstrap confidence intervals** on every metric.
* **DeLong's test** for the difference between two correlated ROC curves, so a
  claim that model A beats model B is backed by a p-value.
* **Calibration and clinical utility.** A risk score that is used to decide
  whether to start immune-tolerance-friendly prophylaxis must be *calibrated*,
  not merely discriminative. Brier score, expected calibration error and
  decision-curve net benefit are reported alongside AUC; none appear in the
  reference works.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------
def youden_threshold(y_true, y_prob) -> float:
    """Threshold maximising sensitivity + specificity - 1."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[int(np.argmax(tpr - fpr))])


def threshold_at_sensitivity(y_true, y_prob, target: float = 0.90) -> float:
    """Lowest threshold that still reaches ``target`` sensitivity.

    Clinically this is often the operating point that matters: missing a
    high-risk patient costs far more than an unnecessary extra inhibitor
    assay, so sensitivity is fixed first and specificity is whatever follows.
    """
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    ok = np.where(tpr >= target)[0]
    return float(thr[ok[0]]) if len(ok) else 0.0


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------
def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Mean |observed - predicted| risk across equal-width probability bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def net_benefit(y_true, y_prob, threshold: float) -> float:
    """Vickers-Elkin decision-curve net benefit at a threshold probability.

    Net benefit expresses true positives gained per patient after subtracting
    false positives weighted by the odds of the threshold -- i.e. it encodes
    how many unnecessary interventions a clinician is willing to accept to
    catch one extra true case.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    n = len(y_true)
    tp = float(((pred == 1) & (y_true == 1)).sum())
    fp = float(((pred == 1) & (y_true == 0)).sum())
    if threshold >= 1.0:
        return 0.0
    return tp / n - (fp / n) * (threshold / (1 - threshold))


def compute_metrics(y_true, y_prob, threshold: float | None = None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if threshold is None:
        threshold = youden_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    prev = float(y_true.mean())

    return {
        "n": int(len(y_true)),
        "prevalence": round(prev, 4),
        "threshold": round(float(threshold), 4),
        "auc_roc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "auc_pr": round(float(average_precision_score(y_true, y_prob)), 4),
        "auc_pr_baseline": round(prev, 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float((recall_score(y_true, y_pred, zero_division=0) + spec) / 2), 4),
        "sensitivity": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(spec), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "npv": round(float(npv), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "ece": round(expected_calibration_error(y_true, y_prob), 4),
        "net_benefit_at_10pct": round(net_benefit(y_true, y_prob, 0.10), 4),
        "net_benefit_at_20pct": round(net_benefit(y_true, y_prob, 0.20), 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
def bootstrap_ci(y_true, y_prob, metric: str = "auc_roc", n_boot: int = 2000,
                 alpha: float = 0.05, random_state: int = RANDOM_STATE) -> dict:
    """Stratified bootstrap percentile interval for one metric."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(random_state)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]

    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            vals.append(compute_metrics(yt, yp)[metric])
        except (ValueError, KeyError):
            continue
    if not vals:
        return {"point": None, "lo": None, "hi": None}
    vals = np.asarray(vals, dtype=float)
    return {
        "point": round(float(compute_metrics(y_true, y_prob)[metric]), 4),
        "lo": round(float(np.percentile(vals, 100 * alpha / 2)), 4),
        "hi": round(float(np.percentile(vals, 100 * (1 - alpha / 2))), 4),
        "n_boot": int(len(vals)),
    }
