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
            ExtraTreesClassifier(class_weight="balanced", n_jobs=1,
                                 random_state=random_state),
            {"clf__n_estimators": randint(300, 1200),
             "clf__min_samples_leaf": randint(1, 20),
             "clf__max_features": uniform(0.05, 0.6),
             "clf__max_depth": [None, 8, 12, 20],
             "clf__criterion": ["gini", "entropy"]},
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced_subsample", n_jobs=1,
                                   random_state=random_state),
            {"clf__n_estimators": randint(300, 1200),
             "clf__min_samples_leaf": randint(1, 20),
             "clf__max_features": uniform(0.05, 0.6),
             "clf__max_depth": [None, 8, 12, 20]},
        ),
        "LightGBM": (
            lgb.LGBMClassifier(class_weight="balanced", n_jobs=1, verbose=-1,
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
            xgb.XGBClassifier(eval_metric="logloss", n_jobs=1,
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
            LogisticRegression(solver="liblinear", max_iter=3000,
                               class_weight="balanced",
                               random_state=random_state),
            {"clf__C": loguniform(1e-3, 10),
             "clf__penalty": ["l1", "l2"]},
        ),
    }


# ---------------------------------------------------------------------------
# Nested search
# ---------------------------------------------------------------------------
def nested_search(X, y, name: str, estimator, space, n_iter: int = 40,
                  inner_splits: int = 4, outer_splits: int = 5,
                  random_state: int = RANDOM_STATE) -> dict:
    """Tune on inner folds, score on untouched outer folds."""
    from sklearn.metrics import roc_auc_score

    # The search fans out over folds and candidates; the estimators inside it
    # are therefore configured single-threaded, or the two layers oversubscribe
    # every core and the search runs slower than a serial one.
    outer = StratifiedKFold(outer_splits, shuffle=True, random_state=random_state)
    inner = StratifiedKFold(inner_splits, shuffle=True, random_state=random_state)

    outer_scores, chosen = [], []
    for tr, te in outer.split(X, y):
        search = RandomizedSearchCV(
            build_pipeline(estimator), space, n_iter=n_iter, scoring="roc_auc",
            cv=inner, n_jobs=-1, random_state=random_state, refit=True,
            error_score=0.0)
        search.fit(X[tr], y[tr])
        p = search.best_estimator_.predict_proba(X[te])[:, 1]
        outer_scores.append(float(roc_auc_score(y[te], p)))
        chosen.append({k.replace("clf__", ""): (round(v, 5) if isinstance(v, float) else v)
                       for k, v in search.best_params_.items()})

    # one final search on everything, to produce the deployable estimator
    final = RandomizedSearchCV(
        build_pipeline(estimator), space, n_iter=n_iter, scoring="roc_auc",
        cv=inner, n_jobs=-1, random_state=random_state, refit=True,
        error_score=0.0)
    final.fit(X, y)

    return {
        "model": name,
        "nested_auc_mean": round(float(np.mean(outer_scores)), 4),
        "nested_auc_std": round(float(np.std(outer_scores)), 4),
        "nested_auc_folds": [round(s, 4) for s in outer_scores],
        "inner_best_auc": round(float(final.best_score_), 4),
        "optimism_from_tuning": round(float(final.best_score_ - np.mean(outer_scores)), 4),
        "best_params": {k.replace("clf__", ""): (round(v, 5) if isinstance(v, float) else v)
                        for k, v in final.best_params_.items()},
        "params_per_outer_fold": chosen,
        "_estimator": final.best_estimator_,
    }


# ---------------------------------------------------------------------------
# Clinically-constrained booster
# ---------------------------------------------------------------------------
def constrained_lightgbm(feature_names: list[str], params: dict | None = None,
                         random_state: int = RANDOM_STATE):
    """LightGBM whose response to the established risk factors cannot invert."""
    import lightgbm as lgb

    base = dict(n_estimators=500, learning_rate=0.02, num_leaves=15,
                min_child_samples=25, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.6, reg_lambda=5.0, class_weight="balanced",
                n_jobs=-1, verbose=-1, random_state=random_state)
    base.update(params or {})
    base["monotone_constraints"] = monotone_vector(feature_names)
    base["monotone_constraints_method"] = "advanced"
    return lgb.LGBMClassifier(**base)


def evaluate(estimator, X, y, cv=None, random_state: int = RANDOM_STATE) -> float:
    from sklearn.metrics import roc_auc_score
    cv = cv or StratifiedKFold(5, shuffle=True, random_state=random_state)
    p = cross_val_predict(build_pipeline(estimator), X, y, cv=cv,
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))
