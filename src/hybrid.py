"""
Hybrid tree + neural ensembling.

The screen in ``.devtools/exp_stack.py`` found something worth acting on. The
project's original stack used five classical learners, and they turn out to be
nearly the same model: ExtraTrees and RandomForest agree at r = 0.965 on
out-of-fold predictions. Averaging models that make the same mistakes buys
nothing, which is why that stack (0.7395) did not beat ExtraTrees alone
(0.7412).

ExtraTrees and DeepMLP agree at only r = 0.853. A tree ensemble partitions the
feature space into axis-aligned boxes; a self-normalising MLP fits a smooth
function over it. On the same 369 events they are wrong about different
patients, and combining them is worth roughly +0.004 AUC.

Two mechanisms are provided.

``RankAverageEnsemble``
    Averages the *ranks* of each member's predictions rather than the
    probabilities. Rank averaging is scale-free, so a member that is poorly
    calibrated but well-ordered contributes its full ordering information --
    which matters here because the neural members are trained with focal loss
    and are badly calibrated before isotonic regression is applied.

``SeedBaggedTorch``
    A neural network fitted from one random initialisation on 1,836 rows is a
    high-variance estimator. Averaging several seeds reduces that variance
    without touching bias, and the ensemble is then a stabler stacking member
    than any single fit.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .models import RANDOM_STATE, build_pipeline


# ---------------------------------------------------------------------------
# Variance reduction on the neural member
# ---------------------------------------------------------------------------
class SeedBaggedTorch(ClassifierMixin, BaseEstimator):
    """Average a torch classifier over several random initialisations.

    Each fit sees the same data; only the initialisation and the batch order
    change. The spread between those fits is pure estimator variance, and
    averaging it away costs nothing but training time.
    """

    def __init__(self, base_estimator=None, n_seeds: int = 5,
                 random_state: int = RANDOM_STATE):
        self.base_estimator = base_estimator
        self.n_seeds = n_seeds
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.members_ = []
        for s in range(self.n_seeds):
            m = clone(self.base_estimator)
            m.set_params(random_state=self.random_state + 1000 * s)
            self.members_.append(m.fit(X, y))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.members_], axis=0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Rank averaging
# ---------------------------------------------------------------------------
class RankAverageEnsemble(ClassifierMixin, BaseEstimator):
    """Average member ranks, then map back to a [0, 1] score.

    Rank averaging ignores each member's probability scale entirely, so a
    well-ordered but poorly-calibrated member contributes fully. The output is
    a normalised rank, not a probability -- calibration is applied afterwards
    by the training pipeline, which is where it belongs.

    Weights are optional and are computed from out-of-fold AUC so that a member
    cannot earn influence by memorising.
    """

    def __init__(self, base_models: dict | None = None, weighted: bool = True,
                 n_splits: int = 5, random_state: int = RANDOM_STATE):
        self.base_models = base_models
        self.weighted = weighted
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.metrics import roc_auc_score

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.names_ = list(self.base_models)

        if self.weighted:
            cv = StratifiedKFold(self.n_splits, shuffle=True,
                                 random_state=self.random_state)
            aucs = []
            for name in self.names_:
                oof = np.zeros(len(y))
                for tr, te in cv.split(X, y):
                    pipe = build_pipeline(self.base_models[name]).fit(X[tr], y[tr])
                    oof[te] = pipe.predict_proba(X[te])[:, 1]
                aucs.append(roc_auc_score(y, oof))
            aucs = np.asarray(aucs)
            raw = np.clip(aucs - 0.5, 1e-6, None) ** 2
            self.weights_ = raw / raw.sum()
            self.oof_auc_ = dict(zip(self.names_, np.round(aucs, 4)))
        else:
            self.weights_ = np.full(len(self.names_), 1.0 / len(self.names_))
            self.oof_auc_ = {}

        self.weight_map_ = dict(zip(self.names_, np.round(self.weights_, 4)))
        self.fitted_ = {n: build_pipeline(self.base_models[n]).fit(X, y)
                        for n in self.names_}
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        if n == 1:
            # a single row has no rank information; fall back to the weighted
            # probability average so single-patient scoring still works
            p = float(np.dot(
                [self.fitted_[m].predict_proba(X)[0, 1] for m in self.names_],
                self.weights_))
            return np.array([[1 - p, p]])
        R = np.zeros(n)
        for w, name in zip(self.weights_, self.names_):
            pr = self.fitted_[name].predict_proba(X)[:, 1]
            R += w * (rankdata(pr) / n)
        R = (R - R.min()) / max(R.max() - R.min(), 1e-12)
        return np.column_stack([1 - R, R])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_hybrids(blocks: dict[str, list[int]],
                  random_state: int = RANDOM_STATE) -> dict:
    """The hybrid candidates, built from the diversity the screen identified."""
    from .ensemble import StackedEnsemble
    from .models import classical_models, neural_models

    classical = classical_models(random_state)
    neural = neural_models(blocks, random_state)

    et = clone(classical["ExtraTrees"])
    mlp = clone(neural["DeepMLP"])
    bagged_mlp = SeedBaggedTorch(base_estimator=clone(neural["DeepMLP"]),
                                 n_seeds=5, random_state=random_state)

    return {
        "Hybrid_RankAvg": RankAverageEnsemble(
            base_models={"ExtraTrees": et, "DeepMLP": mlp},
            random_state=random_state),
        "Hybrid_RankAvg_Bagged": RankAverageEnsemble(
            base_models={"ExtraTrees": clone(et), "DeepMLP_bagged": bagged_mlp},
            random_state=random_state),
        "Hybrid_Stack": StackedEnsemble(
            base_models={"ExtraTrees": clone(et), "DeepMLP": clone(mlp)},
            random_state=random_state),
        "Hybrid_Stack_Wide": StackedEnsemble(
            base_models={"ExtraTrees": clone(et),
                         "RandomForest": clone(classical["RandomForest"]),
                         "DeepMLP": clone(mlp),
                         "BioBlockAttention": clone(neural["BioBlockAttention"])},
            random_state=random_state),
    }
