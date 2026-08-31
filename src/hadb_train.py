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
from sklearn.isotonic import IsotonicRegression
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


class BoundedIsotonic:
    """Isotonic calibration that is not allowed to claim certainty.

    Plain isotonic regression is a step function, so its top step takes the
    value of whatever the highest-scoring training bin happened to contain. If
    that bin is small and entirely positive, the calibrator returns exactly
    1.0 -- and the app then tells a clinician that a patient will *certainly*
    develop an inhibitor. Nothing in the data supports that: the worst
    observed stratum, severe patients with a large deletion, runs at 53%, and
    the top calibration bin at 60%.

    The extremes are therefore bounded by what was actually observed. Training
    scores are split into deciles, each decile's positive rate is
    Laplace-smoothed (``(k+1)/(n+2)``, so a bin that happens to be all-positive
    still lands short of 1), and predictions are clipped to the range those
    smoothed rates span. Ranking is untouched, so AUC is unchanged; only the
    numbers quoted to a human are made honest.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def fit(self, scores, y):
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y, dtype=float)
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0,
                                       y_max=1.0).fit(scores, y)

        edges = np.unique(np.quantile(scores, np.linspace(0, 1, self.n_bins + 1)))
        rates = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (scores >= lo) & (scores <= hi)
            if m.sum() >= 5:
                rates.append((y[m].sum() + 1.0) / (m.sum() + 2.0))
        if not rates:                       # degenerate: fall back to prevalence
            rates = [float(y.mean())]
        self.lo_ = float(min(rates))
        self.hi_ = float(max(rates))
        return self

    def predict(self, scores):
        return np.clip(self.iso_.predict(np.asarray(scores, dtype=float)),
                       self.lo_, self.hi_)


class RankEnsemble:
    """Average the rank transforms of several fitted members.

    Rank averaging is used rather than probability averaging because the
    members are calibrated on different scales -- a forest's 0.6 and a boosted
    tree's 0.6 are not the same claim. Ranking removes the scale, and a single
    isotonic layer afterwards puts the blend back onto a probability axis.

    This lives in ``src`` rather than in the training script because the fitted
    object is pickled into the shipped artefact. A class defined in a ``python
    script.py`` entry point pickles as ``__main__.RankEnsemble`` and cannot be
    loaded by the app or the notebook.
    """

    def __init__(self, members: dict):
        self.members = members

    def fit(self, X, y):
        for m in self.members.values():
            m.fit(X, y)
        # Each member's rank transform is defined against the *training*
        # distribution, not against whatever rows happen to be scored together.
        # Ranking within the input batch would make a single patient's score
        # depend on who else was in the request -- and scoring one patient
        # alone would rank them 1/1 and return the same number for everybody.
        self.reference_ = {
            name: np.sort(m.predict_proba(X)[:, 1])
            for name, m in self.members.items()
        }
        return self

    def decision_scores(self, X) -> np.ndarray:
        cols = []
        for name, m in self.members.items():
            p = m.predict_proba(X)[:, 1]
            ref = getattr(self, "reference_", {}).get(name)
            if ref is None or len(ref) == 0:
                from scipy.stats import rankdata
                cols.append(rankdata(p) / max(len(p), 1))
            else:
                cols.append(np.searchsorted(ref, p, side="right") / len(ref))
        return np.column_stack(cols).mean(1)

    def predict_proba(self, X) -> np.ndarray:
        s = self.decision_scores(X)
        return np.column_stack([1 - s, s])
