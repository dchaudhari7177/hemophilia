"""
Model zoo: the reference architectures, strong tabular baselines, and the
biologically-blocked attention network introduced by this project.

Everything here exposes the scikit-learn estimator API (``fit`` / ``predict_proba``)
so that a single cross-validation and calibration harness can drive all of them.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def _seed_everything(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Shared preprocessing
# ---------------------------------------------------------------------------
def make_preprocessor() -> Pipeline:
    """Median imputation + z-scoring, fitted inside each CV fold.

    Fitting the imputer and the scaler on the whole dataset before splitting is
    a subtle leak that the reference notebook commits; keeping them inside a
    Pipeline means the harness refits them per fold automatically.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017), as used by the reference notebook.

    ``alpha`` up-weights the positive class and ``gamma`` down-weights examples
    the model already classifies confidently, concentrating gradient on the
    hard inhibitor-positive patients.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        ce = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
        p_t = target * p + (1 - target) * (1 - p)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        return (alpha_t * (1 - p_t) ** self.gamma * ce).mean()
