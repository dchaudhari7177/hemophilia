"""
Positive-unlabelled learning over the 1,744 CHAMP rows with no recorded outcome.

Those rows are 43% of the database. The reference pipeline relabels them all as
inhibitor-negative, which is both unsupported and self-serving (it inflates
accuracy by padding the majority class). Dropping them, as the corrected
baseline does, is honest but wasteful: their *features* are perfectly good, and
the class-prior information they carry is exactly what a small dataset needs.

Positive-unlabelled learning is the principled middle path. We treat the data
as one labelled-positive set P (461 patients), one labelled-negative set N
(1,835 patients) and one unlabelled set U (1,744 patients), and use U for
semi-supervised regularisation rather than pretending to know its labels.

Two estimators are provided:

``ElkanNotoPU``
    Fits a non-traditional classifier that separates *labelled* from
    *unlabelled*, estimates the label frequency c = P(labelled | positive) on a
    held-out validation split, and rescales its output by 1/c to recover the
    true posterior. This is the standard Elkan & Noto (2008) correction.

``SelfTrainingPU``
    Iteratively pseudo-labels the most confident unlabelled rows and folds them
    back into training, with a confidence floor and a cap on how many rows can
    be adopted per round so that a single early mistake cannot cascade.

Both are evaluated only against genuinely labelled held-out patients, so the
unlabelled rows can never flatter the reported numbers.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42


class ElkanNotoPU(BaseEstimator, ClassifierMixin):
    """Elkan & Noto (2008) positive-unlabelled correction.

    ``fit`` expects ``y`` in {0, 1} for labelled rows; the unlabelled feature
    matrix is passed separately to ``fit_pu``.
    """

    def __init__(self, base_estimator=None, n_splits: int = 5,
                 random_state: int = RANDOM_STATE):
        self.base_estimator = base_estimator
        self.n_splits = n_splits
        self.random_state = random_state

    def fit_pu(self, X_labelled, y_labelled, X_unlabelled):
        X_labelled = np.asarray(X_labelled, dtype=float)
        y_labelled = np.asarray(y_labelled).astype(int)
        X_unlabelled = np.asarray(X_unlabelled, dtype=float)
        self.classes_ = np.array([0, 1])

        # Stage 1 -- a classifier that separates "this row has a recorded
        # outcome" from "this row does not". Rows whose outcome was recorded as
        # positive are the P set.
        X_all = np.vstack([X_labelled, X_unlabelled])
        s = np.concatenate([np.ones(len(X_labelled)), np.zeros(len(X_unlabelled))])

        # Stage 2 -- estimate c = P(labelled | positive) out-of-fold, so the
        # constant is not fitted on the same rows that produced it.
        cv = StratifiedKFold(self.n_splits, shuffle=True,
                             random_state=self.random_state)
        oof = np.zeros(len(s))
        for tr, te in cv.split(X_all, s):
            m = clone(self.base_estimator)
            m.fit(X_all[tr], s[tr])
            oof[te] = m.predict_proba(X_all[te])[:, 1]

        pos_mask = np.zeros(len(s), dtype=bool)
        pos_mask[:len(y_labelled)] = y_labelled == 1
        self.c_ = float(np.mean(oof[pos_mask])) if pos_mask.any() else 1.0
        self.c_ = float(np.clip(self.c_, 1e-3, 1.0))

        self.model_ = clone(self.base_estimator)
        self.model_.fit(X_all, s)
        return self

    def fit(self, X, y):                       # sklearn compatibility
        return self.fit_pu(X, y, np.empty((0, np.asarray(X).shape[1])))

    def predict_proba(self, X):
        p = self.model_.predict_proba(np.asarray(X, dtype=float))[:, 1] / self.c_
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SelfTrainingPU(BaseEstimator, ClassifierMixin):
    """Confidence-capped self-training over the unlabelled pool.

    Each round adopts at most ``max_adopt_per_round`` unlabelled rows, and only
    those the current model scores beyond ``hi``/below ``lo``. Adopted rows
    enter the next round with a reduced sample weight so that pseudo-labels
    never outvote real observations.
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

    def fit_pu(self, X_labelled, y_labelled, X_unlabelled):
        X_l = np.asarray(X_labelled, dtype=float)
        y_l = np.asarray(y_labelled).astype(int)
        X_u = np.asarray(X_unlabelled, dtype=float)
        self.classes_ = np.array([0, 1])

        X_cur, y_cur = X_l.copy(), y_l.copy()
        w_cur = np.ones(len(y_l))
        remaining = np.ones(len(X_u), dtype=bool)
        self.history_: list[dict] = []

        model = clone(self.base_estimator)
        model.fit(X_cur, y_cur, **self._weight_kwarg(model, w_cur))

        for rnd in range(self.n_rounds):
            if not remaining.any():
                break
            p = model.predict_proba(X_u[remaining])[:, 1]
            idx_rem = np.where(remaining)[0]

            take_pos = idx_rem[p >= self.hi]
            take_neg = idx_rem[p <= self.lo]
            # keep the pseudo-labelled batch in the same ratio as the real data
            # so self-training cannot drift the prior
            cap_pos = min(len(take_pos),
                          int(self.max_adopt_per_round * y_l.mean()) or 1)
            cap_neg = min(len(take_neg),
                          self.max_adopt_per_round - cap_pos)
            score = model.predict_proba(X_u)[:, 1]
            take_pos = take_pos[np.argsort(-score[take_pos])][:cap_pos]
            take_neg = take_neg[np.argsort(score[take_neg])][:cap_neg]
            adopted = np.concatenate([take_pos, take_neg])
            if len(adopted) == 0:
                break

            X_cur = np.vstack([X_cur, X_u[adopted]])
            y_cur = np.concatenate([y_cur,
                                    np.concatenate([np.ones(len(take_pos)),
                                                    np.zeros(len(take_neg))])]).astype(int)
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

    @staticmethod
    def _weight_kwarg(model, w):
        """Pass sample weights only to estimators that accept them."""
        import inspect
        try:
            sig = inspect.signature(model.fit)
            if "sample_weight" in sig.parameters:
                return {"sample_weight": w}
        except (TypeError, ValueError):
            pass
        return {}

    def fit(self, X, y):
        return self.fit_pu(X, y, np.empty((0, np.asarray(X).shape[1])))

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def estimate_unlabelled_prevalence(model, X_unlabelled) -> dict:
    """Score the unlabelled pool and summarise the implied inhibitor rate.

    This is a substantive result in its own right: if the reference pipeline's
    assumption were correct, the model would score almost every unlabelled row
    as negative. Any appreciable predicted prevalence is direct evidence that
    relabelling them 0 injects false negatives into training.
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
