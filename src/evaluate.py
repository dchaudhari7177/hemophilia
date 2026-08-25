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
