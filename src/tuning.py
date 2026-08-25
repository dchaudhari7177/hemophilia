"""
Hyperparameter search and clinically-constrained gradient boosting.

Two ideas beyond a plain grid search.

**Nested search.** The reference works tune with ``GridSearchCV`` and then
report the best cross-validated score. That score is optimistically biased:
the same folds chose the hyperparameters and graded them. Here the search runs
in an inner loop and is scored on an outer loop it never touched, so the
reported number is what a new cohort should expect.

**Monotone constraints.** Some directions are not in question. A null variant
cannot lower inhibitor risk relative to a missense; a severe phenotype cannot
lower it relative to a mild one. Gradient boosting is free to learn the
opposite from noise in a 369-event training set, and on small data it often
does. Pinning the sign of these features costs a little training fit and buys
a model that cannot contradict established immunology -- which also makes it
defensible in front of a clinician.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict)

from .models import RANDOM_STATE, build_pipeline

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Directional priors from haemophilia immunology
# ---------------------------------------------------------------------------
# +1: feature can only increase predicted inhibitor risk
# -1: feature can only decrease it
MONOTONE_PRIORS: dict[str, int] = {
    "is_null_mutation": +1,        # no endogenous FVIII -> never tolerised
    "is_truncating": +1,
    "vtype_large_structural": +1,  # large deletions carry the highest risk
    "vtype_nonsense": +1,
    "vtype_frameshift": +1,
    "vtype_missense": -1,          # missense makes protein -> lowest risk
    "vtype_synonymous": -1,
    "severity_severe": +1,
    "severity_mild": -1,
    "severity_ordinal": +1,
    "null_and_severe": +1,
    "fraction_protein_lost": +1,
    "n_domains_lost": +1,
}


def monotone_vector(feature_names: list[str]) -> list[int]:
    """Constraint vector aligned to the design matrix, 0 = unconstrained."""
    return [MONOTONE_PRIORS.get(n, 0) for n in feature_names]
