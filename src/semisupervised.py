"""
Making use of the 1,744 CHAMP rows whose inhibitor outcome was never recorded.

Those rows are 43% of the database. The reference pipeline relabels every one
of them as inhibitor-negative, which is unsupported (a publication that did not
state the outcome is not a publication that reported a negative outcome) and
self-serving (it pads the majority class, so accuracy rises without the model
improving). The corrected baseline drops them instead, which is honest but
throws away 43% of the feature information.

Note on framing
---------------
CHAMP gives us labelled *positives* (461), labelled *negatives* (1,835) and
unlabelled rows (1,744). That is a semi-supervised problem, not a
positive-unlabelled one: classical PU corrections such as Elkan & Noto (2008)
assume the labelled set contains positives only, and applying their 1/c
rescaling here would estimate "was this variant's outcome written down",
not "did this patient develop an inhibitor". This module therefore uses
semi-supervised methods, and keeps the labelled-vs-unlabelled classifier only
in its proper role -- as a diagnostic for whether the missingness is
informative.

Three components
----------------
``ReportingBiasProbe``
    Asks whether a variant's *features* predict whether its outcome was
    reported at all. If they do not (AUC near 0.5) the outcomes are missing at
    random and simply dropping them is unbiased. If they do, the missingness is
    informative and any analysis that ignores it -- including the corrected
    baseline -- inherits a selection bias that has to be stated.

``SelfTrainingSSL``
    Iteratively adopts the unlabelled rows the current model is most confident
    about, at a capped rate, with the adopted rows down-weighted so that
    pseudo-labels can never outvote real observations.

``estimate_unlabelled_prevalence``
    Scores the unlabelled pool to quantify how many false negatives the
    reference pipeline's relabelling actually injects.

Every number is still measured against genuinely labelled held-out patients, so
the unlabelled rows cannot flatter the reported performance.
"""

from __future__ import annotations

import inspect

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Is the missingness informative?
# ---------------------------------------------------------------------------
class ReportingBiasProbe:
    """Diagnostic: predict *whether* a variant's inhibitor status was recorded.

    This is a missing-data test, not a risk model. Under missing-at-random the
    probe should be near chance. Anything appreciably above chance means the
    labelled subset is not a random sample of CHAMP, which constrains how far
    any model trained on it -- ours included -- can be generalised.
    """

    def __init__(self, base_estimator=None, n_splits: int = 5,
                 random_state: int = RANDOM_STATE):
        self.base_estimator = base_estimator
        self.n_splits = n_splits
        self.random_state = random_state

    def run(self, X_labelled, X_unlabelled) -> dict:
        X = np.vstack([np.asarray(X_labelled, dtype=float),
                       np.asarray(X_unlabelled, dtype=float)])
        s = np.concatenate([np.ones(len(X_labelled)), np.zeros(len(X_unlabelled))])
        cv = StratifiedKFold(self.n_splits, shuffle=True,
                             random_state=self.random_state)
        oof = cross_val_predict(clone(self.base_estimator), X, s, cv=cv,
                                method="predict_proba")[:, 1]
        auc = float(roc_auc_score(s, oof))
        return {
            "n_labelled": int(len(X_labelled)),
            "n_unlabelled": int(len(X_unlabelled)),
            "reporting_auc": round(auc, 4),
            "interpretation": (
                "missing at random -- dropping unlabelled rows is unbiased"
                if auc < 0.60 else
                "informative missingness -- the labelled subset is not a random "
                "sample of CHAMP and this limits external generalisation"),
        }


# ---------------------------------------------------------------------------
# Semi-supervised self-training
# ---------------------------------------------------------------------------
class SelfTrainingSSL(BaseEstimator, ClassifierMixin):
    """Confidence-capped self-training over the unlabelled pool.

    Each round adopts at most ``max_adopt_per_round`` rows, and only those the
    current model scores above ``hi`` or below ``lo``. The positive/negative
    split of each adopted batch is held at the labelled prevalence so that
    self-training cannot drift the class prior, and adopted rows carry weight
    ``pseudo_weight`` < 1 so a single early mistake cannot cascade.
    """

    def __init__(self, base_estimator=None, n_rounds: int = 5, hi: float = 0.85,
                 lo: float = 0.05, max_adopt_per_round: int = 200,
                 pseudo_weight: float = 0.5, random_state: int = RANDOM_STATE):
        self.base_estimator = base_estimator
        self.n_rounds = n_rounds
        self.hi = hi
        self.lo = lo
        self.max_adopt_per_round = max_adopt_per_round
        self.pseudo_weight = pseudo_weight
        self.random_state = random_state

    @staticmethod
    def _weight_kwarg(model, w):
        """Pass sample weights only to estimators that accept them."""
        try:
            if "sample_weight" in inspect.signature(model.fit).parameters:
                return {"sample_weight": w}
        except (TypeError, ValueError):
            pass
        return {}

    def fit_ssl(self, X_labelled, y_labelled, X_unlabelled):
        X_l = np.asarray(X_labelled, dtype=float)
        y_l = np.asarray(y_labelled).astype(int)
        X_u = np.asarray(X_unlabelled, dtype=float)
        self.classes_ = np.array([0, 1])
        prevalence = float(y_l.mean())

        X_cur, y_cur = X_l.copy(), y_l.copy()
        w_cur = np.ones(len(y_l))
        remaining = np.ones(len(X_u), dtype=bool)
        self.history_: list[dict] = []

        model = clone(self.base_estimator)
        model.fit(X_cur, y_cur, **self._weight_kwarg(model, w_cur))

        for rnd in range(self.n_rounds):
            if not remaining.any():
                break
            score = model.predict_proba(X_u)[:, 1]
            idx_rem = np.where(remaining)[0]
            cand_pos = idx_rem[score[idx_rem] >= self.hi]
            cand_neg = idx_rem[score[idx_rem] <= self.lo]

            cap_pos = min(len(cand_pos),
                          max(1, int(self.max_adopt_per_round * prevalence)))
            cap_neg = min(len(cand_neg), self.max_adopt_per_round - cap_pos)
            take_pos = cand_pos[np.argsort(-score[cand_pos])][:cap_pos]
            take_neg = cand_neg[np.argsort(score[cand_neg])][:cap_neg]
            adopted = np.concatenate([take_pos, take_neg]).astype(int)
            if len(adopted) == 0:
                break

            X_cur = np.vstack([X_cur, X_u[adopted]])
            y_cur = np.concatenate([
                y_cur,
                np.concatenate([np.ones(len(take_pos)), np.zeros(len(take_neg))]),
            ]).astype(int)
            w_cur = np.concatenate([w_cur,
                                    np.full(len(adopted), self.pseudo_weight)])
            remaining[adopted] = False

            model = clone(self.base_estimator)
            model.fit(X_cur, y_cur, **self._weight_kwarg(model, w_cur))
            self.history_.append({
                "round": rnd + 1,
                "adopted_positive": int(len(take_pos)),
                "adopted_negative": int(len(take_neg)),
                "pool_remaining": int(remaining.sum()),
            })

        self.model_ = model
        return self

    def fit(self, X, y):
        """sklearn entry point; trains without an unlabelled pool."""
        return self.fit_ssl(X, y, np.empty((0, np.asarray(X).shape[1])))

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# What does relabelling the unknowns as negative actually cost?
# ---------------------------------------------------------------------------
def estimate_unlabelled_prevalence(model, X_unlabelled) -> dict:
    """Score the unlabelled pool and summarise the implied inhibitor rate.

    If the reference pipeline's assumption held, a well-fitted model would
    score almost every unlabelled row as negative. Any appreciable predicted
    prevalence is a direct estimate of the false negatives that relabelling
    injects into training.
    """
    p = model.predict_proba(np.asarray(X_unlabelled, dtype=float))[:, 1]
    return {
        "n_unlabelled": int(len(p)),
        "mean_predicted_risk": round(float(p.mean()), 4),
        "median_predicted_risk": round(float(np.median(p)), 4),
        "predicted_positive_at_0.5": int((p >= 0.5).sum()),
        "predicted_positive_fraction_at_0.5": round(float((p >= 0.5).mean()), 4),
        "implied_false_negatives_if_relabelled_zero": int((p >= 0.5).sum()),
    }
