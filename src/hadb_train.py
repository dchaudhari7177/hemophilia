"""Model selection and evaluation on the HADB patient-level cohort.

Protocol
--------
The unit of analysis is one allele report (a patient). 2,643 variants generate
4,966 labelled records, and a handful of recurrent variants generate up to 104
records each. If those records were split at random, a model could memorise a
variant in training and be scored on the same variant at test time -- the
patient-level version of the identifier leak that inflated the earlier CHAMP
results. **Every split here is grouped by ``mut_id``**, so a variant appears on
exactly one side of any boundary.

A second, harsher split groups by *study*. Reporting centres differ in how they
screen for inhibitors and in which patients they publish, so study-grouped
scores estimate how the model behaves in a centre it has never seen. That is
the number to quote for deployment, and it is expected to be lower.

Nothing is resampled. Imbalance is handled with class weights only, because
over-sampling before the split is precisely what produced the 97-99% figures
this project exists to correct.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .hadb import build_features, load_hadb

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


# ---------------------------------------------------------------------------
# Cohort assembly
# ---------------------------------------------------------------------------
@dataclass
class Cohort:
    """A labelled design matrix plus the grouping keys its splits must respect."""

    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray          # mut_id -- the variant a record belongs to
    studies: np.ndarray         # reporting publication
    blocks: dict[str, list[str]] = field(default_factory=dict)
    frame: pd.DataFrame | None = None

    def __len__(self) -> int:
        return len(self.y)

    @property
    def prevalence(self) -> float:
        return float(self.y.mean())

    @property
    def majority_baseline(self) -> float:
        """Accuracy of always predicting the majority class."""
        return float(max(self.y.mean(), 1 - self.y.mean()))


def build_cohort(include_clinical: bool = True,
                 include_context: bool = True,
                 df: pd.DataFrame | None = None) -> Cohort:
    df = load_hadb() if df is None else df
    X_all, blocks = build_features(df, include_clinical=include_clinical,
                                   include_context=include_context)
    mask = df["y"].notna().to_numpy()
    X = X_all.loc[mask].reset_index(drop=True)
    sub = df.loc[mask].reset_index(drop=True)
    return Cohort(
        X=X,
        y=sub["y"].to_numpy(dtype=float),
        groups=sub["mut_id"].to_numpy(),
        studies=sub["study"].to_numpy(),
        blocks={k: [c for c in v if c in X.columns] for k, v in blocks.items()},
        frame=sub,
    )


def unlabelled_frame(df: pd.DataFrame | None = None,
                     include_clinical: bool = True,
                     include_context: bool = True):
    """The records whose outcome was never recorded, kept for semi-supervision."""
    df = load_hadb() if df is None else df
    X_all, _ = build_features(df, include_clinical=include_clinical,
                              include_context=include_context)
    mask = df["y"].isna().to_numpy()
    return X_all.loc[mask].reset_index(drop=True), df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def holdout_split(cohort: Cohort, test_size: float = 0.2,
                  random_state: int = RANDOM_STATE):
    """Variant-grouped train/test split. Returns boolean masks."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=random_state)
    train_idx, test_idx = next(splitter.split(cohort.X, cohort.y, cohort.groups))
    train = np.zeros(len(cohort), dtype=bool)
    test = np.zeros(len(cohort), dtype=bool)
    train[train_idx] = True
    test[test_idx] = True
    return train, test


def grouped_folds(y, groups, n_splits: int = 5, random_state: int = RANDOM_STATE):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=random_state)
    return list(cv.split(np.zeros(len(y)), y, groups))


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------
def _numeric_prep() -> Pipeline:
    """Median imputation then scaling.

    Missingness is already encoded explicitly by the ``*_measured`` indicator
    features, so imputing the value itself does not silently invent a
    measurement the model cannot tell apart from a real one.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


def model_zoo(random_state: int = RANDOM_STATE, pos_weight: float = 5.0) -> dict:
    """Every candidate is imbalance-aware; none of them resample."""
    import lightgbm as lgb
    import xgboost as xgb

    def wrap(est, scale: bool = True):
        steps = [("impute", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scale", StandardScaler()))
        steps.append(("clf", est))
        return Pipeline(steps)

    zoo = {
        "logistic_l2": wrap(LogisticRegression(
            max_iter=5000, class_weight="balanced", C=1.0,
            random_state=random_state)),
        "logistic_elasticnet": wrap(LogisticRegression(
            max_iter=5000, class_weight="balanced", penalty="elasticnet",
            solver="saga", l1_ratio=0.5, C=0.5, random_state=random_state)),
        "random_forest": wrap(RandomForestClassifier(
            n_estimators=600, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=random_state), scale=False),
        "extra_trees": wrap(ExtraTreesClassifier(
            n_estimators=600, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=random_state), scale=False),
        "gradient_boosting": wrap(GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=random_state), scale=False),
        "hist_gradient_boosting": wrap(HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, class_weight="balanced",
            random_state=random_state), scale=False),
        "xgboost": wrap(xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_lambda=2.0, scale_pos_weight=pos_weight, n_jobs=-1,
            eval_metric="logloss", tree_method="hist",
            random_state=random_state), scale=False),
        "lightgbm": wrap(lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=2.0, class_weight="balanced", n_jobs=-1, verbose=-1,
            random_state=random_state), scale=False),
        "svm_rbf": wrap(SVC(
            C=2.0, gamma="scale", class_weight="balanced", probability=True,
            random_state=random_state)),
        "knn": wrap(KNeighborsClassifier(n_neighbors=25, weights="distance")),
        "mlp": wrap(MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=800,
            early_stopping=True, n_iter_no_change=25,
            random_state=random_state)),
    }
    return zoo


def stacking_ensemble(random_state: int = RANDOM_STATE,
                      pos_weight: float = 5.0) -> Pipeline:
    """Stack the three strongest families under a logistic meta-learner.

    The meta-learner sees out-of-fold probabilities only, and its internal CV
    is supplied by the caller as grouped folds so the stack does not leak a
    variant across its own levels.
    """
    import lightgbm as lgb
    import xgboost as xgb

    base = [
        ("xgb", xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3, reg_lambda=2.0,
            scale_pos_weight=pos_weight, n_jobs=-1, eval_metric="logloss",
            tree_method="hist", random_state=random_state)),
        ("lgb", lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            class_weight="balanced", n_jobs=-1, verbose=-1,
            random_state=random_state)),
        ("rf", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=random_state)),
    ]
    stack = StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=2000,
                                           class_weight="balanced"),
        stack_method="predict_proba",
        n_jobs=1,
        passthrough=False,
    )
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", stack)])


def pos_weight_for(y) -> float:
    n_pos = float((y == 1).sum())
    return float((len(y) - n_pos) / max(n_pos, 1.0))
