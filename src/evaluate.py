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
