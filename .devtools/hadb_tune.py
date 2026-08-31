"""Tune the leading families under variant-grouped CV, then combine them.

Two stages:

**Tuning** -- randomised search over each family, scored by AUC-ROC on
precomputed variant-grouped folds. The folds are passed in as a list so the
search cannot silently fall back to a random split.

**Combination** -- the sklearn ``StackingClassifier`` cannot be used here: it
generates its own internal folds at random, which puts the same variant on
both sides of the meta-learner's training boundary and inverted the screen's
score. This builds the out-of-fold matrix by hand from grouped folds, and
compares three ways of combining the members:

  * ``mean``  -- average of calibrated probabilities
  * ``rank``  -- average of within-fold rank transforms, which is robust to
    members being calibrated differently
  * ``stack`` -- logistic meta-learner on grouped out-of-fold probabilities

Everything is selected on cross-validated scores. The held-out set is not
touched here.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import compute_metrics  # noqa: E402
from src.hadb_train import (  # noqa: E402
    RANDOM_STATE,
    REPORTS,
    build_cohort,
    grouped_folds,
    holdout_split,
    model_zoo,
    pos_weight_for,
)

warnings.filterwarnings("ignore")

SEARCH_SPACES = {
    "random_forest": {
        "clf__n_estimators": [400, 600, 900, 1200],
        "clf__max_depth": [None, 8, 12, 16, 24],
        "clf__min_samples_leaf": [1, 2, 3, 5, 8],
        "clf__max_features": ["sqrt", "log2", 0.3, 0.5],
        "clf__min_samples_split": [2, 5, 10],
    },
    "extra_trees": {
        "clf__n_estimators": [400, 600, 900, 1200],
        "clf__max_depth": [None, 8, 12, 16, 24],
        "clf__min_samples_leaf": [1, 2, 3, 5, 8],
        "clf__max_features": ["sqrt", "log2", 0.3, 0.5],
    },
    "xgboost": {
        "clf__n_estimators": [300, 500, 800, 1200],
        "clf__learning_rate": [0.01, 0.02, 0.05, 0.08],
        "clf__max_depth": [3, 4, 5, 6, 8],
        "clf__subsample": [0.6, 0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.5, 0.7, 0.8, 1.0],
        "clf__min_child_weight": [1, 3, 5, 10],
        "clf__reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
        "clf__gamma": [0, 0.1, 0.5, 1.0],
    },
    "lightgbm": {
        "clf__n_estimators": [300, 500, 800, 1200],
        "clf__learning_rate": [0.01, 0.02, 0.05, 0.08],
        "clf__num_leaves": [15, 31, 63, 127],
        "clf__min_child_samples": [10, 20, 40, 60],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.5, 0.7, 0.9],
        "clf__reg_lambda": [0.5, 1.0, 5.0, 10.0],
    },
    "hist_gradient_boosting": {
        "clf__max_iter": [300, 500, 800],
        "clf__learning_rate": [0.02, 0.05, 0.08],
        "clf__max_leaf_nodes": [15, 31, 63],
        "clf__min_samples_leaf": [10, 20, 40],
        "clf__l2_regularization": [0.5, 1.0, 5.0],
    },
}


def oof_probs(model, X, y, folds) -> np.ndarray:
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        m = clone(model)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


def main() -> None:
    t0 = time.time()
    cohort = build_cohort()
    train_mask, _ = holdout_split(cohort)
    X = cohort.X.loc[train_mask].reset_index(drop=True)
    y = cohort.y[train_mask]
    groups = cohort.groups[train_mask]
    folds = grouped_folds(y, groups, n_splits=5)
    pw = pos_weight_for(y)
    zoo = model_zoo(pos_weight=pw)

    out: dict = {"n_train": int(len(y)),
                 "prevalence": round(float(y.mean()), 4),
                 "protocol": "randomised search, 5-fold grouped by mut_id"}

    # -- stage 1: tune each family -----------------------------------------
    tuned, tuned_params = {}, {}
    print("tuning (40 draws each, grouped folds):")
    for name, space in SEARCH_SPACES.items():
        t = time.time()
        search = RandomizedSearchCV(
            zoo[name], space, n_iter=40, scoring="roc_auc", cv=folds,
            random_state=RANDOM_STATE, n_jobs=-1, refit=True, error_score=0.0)
        search.fit(X, y)
        tuned[name] = search.best_estimator_
        tuned_params[name] = {k: (v if not isinstance(v, np.generic) else v.item())
                              for k, v in search.best_params_.items()}
        baseline = None
        out.setdefault("tuning", {})[name] = {
            "best_cv_auc": round(float(search.best_score_), 4),
            "best_params": tuned_params[name],
            "seconds": round(time.time() - t, 1),
        }
        print(f"  {name:24s} {search.best_score_:.4f}  ({time.time()-t:.0f}s)")

    # -- stage 2: combine ---------------------------------------------------
    print("\nbuilding grouped out-of-fold matrix for the ensemble:")
    oof = {}
    for name, model in tuned.items():
        oof[name] = oof_probs(model, X, y, folds)
        auc = compute_metrics(y, oof[name])["auc_roc"]
        print(f"  {name:24s} oof AUC {auc:.4f}")
        out["tuning"][name]["oof_auc"] = round(float(auc), 4)

    names = list(oof)
    P = np.column_stack([oof[n] for n in names])
    R = np.column_stack([rankdata(oof[n]) / len(y) for n in names])

    combos = {}
    combos["mean"] = {"auc_roc": float(compute_metrics(y, P.mean(1))["auc_roc"])}
    combos["rank"] = {"auc_roc": float(compute_metrics(y, R.mean(1))["auc_roc"])}

    # The meta-learner is itself cross-validated on the same grouped folds, so
    # its score is not read off the data it was fitted on.
    meta_oof = np.full(len(y), np.nan)
    for tr, te in folds:
        meta = LogisticRegression(max_iter=2000, class_weight="balanced")
        meta.fit(P[tr], y[tr])
        meta_oof[te] = meta.predict_proba(P[te])[:, 1]
    combos["stack"] = {"auc_roc": float(compute_metrics(y, meta_oof)["auc_roc"])}

    # Greedy forward selection with replacement (Caruana): add whichever member
    # most improves the running average, which routinely beats a plain mean
    # when members are correlated but unequal.
    selected: list[int] = []
    best_auc, best_hist = -1.0, []
    for _ in range(12):
        cand_auc, cand_i = -1.0, None
        for i in range(len(names)):
            blend = R[:, selected + [i]].mean(1)
            a = float(compute_metrics(y, blend)["auc_roc"])
            if a > cand_auc:
                cand_auc, cand_i = a, i
        if cand_auc <= best_auc + 1e-6:
            break
        selected.append(cand_i)
        best_auc = cand_auc
        best_hist.append({"added": names[cand_i], "auc_roc": round(cand_auc, 4)})
    combos["greedy_rank"] = {
        "auc_roc": best_auc,
        "members": [names[i] for i in selected],
        "path": best_hist,
    }

    for k, v in combos.items():
        v["auc_roc"] = round(float(v["auc_roc"]), 4)
        print(f"  ensemble {k:14s} {v['auc_roc']:.4f}")
    out["ensembles"] = combos
    out["member_names"] = names

    best_single = max(out["tuning"].items(), key=lambda kv: kv[1]["oof_auc"])
    best_ens = max(combos.items(), key=lambda kv: kv[1]["auc_roc"])
    out["selection"] = {
        "best_single_model": best_single[0],
        "best_single_auc": best_single[1]["oof_auc"],
        "best_ensemble": best_ens[0],
        "best_ensemble_auc": best_ens[1]["auc_roc"],
    }
    out["elapsed_seconds"] = round(time.time() - t0, 1)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "hadb_tuning.json").write_text(json.dumps(out, indent=2))
    np.savez(REPORTS / "hadb_oof.npz", y=y, groups=groups,
             **{f"oof_{n}": oof[n] for n in names})
    print(f"\nbest single {best_single[0]} {best_single[1]['oof_auc']:.4f} | "
          f"best ensemble {best_ens[0]} {best_ens[1]['auc_roc']:.4f}")
    print(f"wrote {REPORTS / 'hadb_tuning.json'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
