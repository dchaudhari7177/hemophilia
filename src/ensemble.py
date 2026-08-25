"""
Stacked and averaged ensembles built on out-of-fold predictions.

Two things worth being careful about, both of which the reference notebook gets
wrong.

**The meta-learner must never see in-fold predictions.** If base models are
fitted on the whole training set and their predictions on that same set are fed
to a meta-learner, the meta-learner is trained on scores that are already
overfitted, and it learns to trust whichever base model memorises hardest.
Out-of-fold predictions are the fix, and they have to come from folds the base
model did not see.

**A stack is not automatically better.** With 369 events the meta-learner has
very little to work with, and the honest answer is often that a single tuned
model wins. This module reports the comparison rather than assuming the answer.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .models import RANDOM_STATE, build_pipeline


def out_of_fold_matrix(models: dict, X, y, n_splits: int = 5,
                       random_state: int = RANDOM_STATE):
    """Column per base model of predictions made on unseen folds."""
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    names = list(models)
    oof = np.zeros((len(y), len(names)))
    for j, name in enumerate(names):
        for tr, te in cv.split(X, y):
            pipe = build_pipeline(models[name])
            pipe.fit(X[tr], y[tr])
            oof[te, j] = pipe.predict_proba(X[te])[:, 1]
    return oof, names


class StackedEnsemble(BaseEstimator, ClassifierMixin):
    """Out-of-fold stacking with a regularised logistic meta-learner.

    The meta-learner is deliberately simple. With this few events a flexible
    meta-model overfits the base predictions immediately; a penalised linear
    blend is what the data can support.
    """

    def __init__(self, base_models: dict | None = None, n_splits: int = 5,
                 C: float = 1.0, random_state: int = RANDOM_STATE):
        self.base_models = base_models
        self.n_splits = n_splits
        self.C = C
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])

        oof, self.names_ = out_of_fold_matrix(
            self.base_models, X, y, self.n_splits, self.random_state)

        self.meta_ = LogisticRegression(
            C=self.C, max_iter=5000, class_weight="balanced",
            random_state=self.random_state)
        self.meta_.fit(oof, y)

        # refit each base learner on everything for inference
        self.fitted_ = {n: build_pipeline(self.base_models[n]).fit(X, y)
                        for n in self.names_}
        self.meta_weights_ = dict(zip(self.names_,
                                      np.round(self.meta_.coef_[0], 4)))
        return self

    def _base_matrix(self, X):
        return np.column_stack([self.fitted_[n].predict_proba(X)[:, 1]
                                for n in self.names_])

    def predict_proba(self, X):
        return self.meta_.predict_proba(self._base_matrix(np.asarray(X, dtype=float)))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class WeightedAverageEnsemble(BaseEstimator, ClassifierMixin):
    """Average of base probabilities, weighted by out-of-fold AUC.

    Simpler than stacking and, on small event counts, frequently better: there
    are no meta-parameters to overfit. Weights are computed from out-of-fold
    scores so a model cannot earn weight by memorising.
    """

    def __init__(self, base_models: dict | None = None, n_splits: int = 5,
                 power: float = 4.0, random_state: int = RANDOM_STATE):
        self.base_models = base_models
        self.n_splits = n_splits
        self.power = power
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.metrics import roc_auc_score

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])

        oof, self.names_ = out_of_fold_matrix(
            self.base_models, X, y, self.n_splits, self.random_state)
        aucs = np.array([roc_auc_score(y, oof[:, j])
                         for j in range(oof.shape[1])])
        # sharpen: weight by (AUC - 0.5)^power so a near-chance model
        # contributes essentially nothing
        raw = np.clip(aucs - 0.5, 1e-6, None) ** self.power
        self.weights_ = raw / raw.sum()
        self.oof_auc_ = dict(zip(self.names_, np.round(aucs, 4)))
        self.weight_map_ = dict(zip(self.names_, np.round(self.weights_, 4)))

        self.fitted_ = {n: build_pipeline(self.base_models[n]).fit(X, y)
                        for n in self.names_}
        return self

    def predict_proba(self, X):
        M = np.column_stack([self.fitted_[n].predict_proba(np.asarray(X, dtype=float))[:, 1]
                             for n in self.names_])
        p = M @ self.weights_
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_ensembles(zoo: dict, members: list[str] | None = None,
                    random_state: int = RANDOM_STATE) -> dict:
    """Assemble both ensembles from a subset of the model zoo.

    Members default to a deliberately *diverse* set -- a linear model, a bagged
    tree ensemble and a boosted one -- because averaging correlated models buys
    nothing.
    """
    members = members or ["LogisticRegression", "ExtraTrees", "RandomForest",
                          "LightGBM", "XGBoost"]
    base = {m: clone(zoo[m]) for m in members if m in zoo}
    return {
        "StackedEnsemble": StackedEnsemble(base_models=base,
                                           random_state=random_state),
        "WeightedAverageEnsemble": WeightedAverageEnsemble(
            base_models=base, random_state=random_state),
    }
