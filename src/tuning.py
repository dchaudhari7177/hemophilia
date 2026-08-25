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


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------
def search_spaces(random_state: int = RANDOM_STATE) -> dict:
    import lightgbm as lgb
    import xgboost as xgb

    return {
        "ExtraTrees": (
            ExtraTreesClassifier(class_weight="balanced", n_jobs=-1,
                                 random_state=random_state),
            {"clf__n_estimators": randint(300, 1200),
             "clf__min_samples_leaf": randint(1, 20),
             "clf__max_features": uniform(0.05, 0.6),
             "clf__max_depth": [None, 8, 12, 20],
             "clf__criterion": ["gini", "entropy"]},
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1,
                                   random_state=random_state),
            {"clf__n_estimators": randint(300, 1200),
             "clf__min_samples_leaf": randint(1, 20),
             "clf__max_features": uniform(0.05, 0.6),
             "clf__max_depth": [None, 8, 12, 20]},
        ),
        "LightGBM": (
            lgb.LGBMClassifier(class_weight="balanced", n_jobs=-1, verbose=-1,
                               random_state=random_state),
            {"clf__n_estimators": randint(150, 900),
             "clf__learning_rate": loguniform(5e-3, 1e-1),
             "clf__num_leaves": randint(4, 48),
             "clf__min_child_samples": randint(5, 60),
             "clf__subsample": uniform(0.5, 0.5),
             "clf__subsample_freq": [1],
             "clf__colsample_bytree": uniform(0.3, 0.7),
             "clf__reg_lambda": loguniform(1e-2, 50),
             "clf__reg_alpha": loguniform(1e-3, 10)},
        ),
        "XGBoost": (
            xgb.XGBClassifier(eval_metric="logloss", n_jobs=-1,
                              random_state=random_state),
            {"clf__n_estimators": randint(150, 900),
             "clf__learning_rate": loguniform(5e-3, 1e-1),
             "clf__max_depth": randint(2, 8),
             "clf__min_child_weight": randint(1, 20),
             "clf__subsample": uniform(0.5, 0.5),
             "clf__colsample_bytree": uniform(0.3, 0.7),
             "clf__reg_lambda": loguniform(1e-2, 50),
             "clf__scale_pos_weight": uniform(1, 5)},
        ),
        "LogisticRegression": (
            LogisticRegression(penalty="elasticnet", solver="saga",
                               max_iter=8000, class_weight="balanced",
                               random_state=random_state),
            {"clf__C": loguniform(1e-3, 10),
             "clf__l1_ratio": uniform(0, 1)},
        ),
    }
