"""
Ablations: what is each part of the feature set actually worth?

Three questions, each answered by measurement rather than assertion.

1. **Signal decomposition.** How far do the two obvious variables (variant type
   and clinical severity) get on their own, and how much do the other 119
   engineered features add on top? If the answer is "nothing", the engineering
   was decoration and should be said to be.

2. **Leave-one-block-out.** Drop each biological block in turn. A block whose
   removal costs nothing is not contributing, however good the story behind it
   sounds.

3. **Feature-count sweep.** With 369 events, 135 features may simply be too
   many. Ranking by importance and sweeping the cut-off shows where the
   variance-bias trade-off actually sits.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .models import RANDOM_STATE, build_pipeline


def _default_model(random_state: int = RANDOM_STATE):
    return ExtraTreesClassifier(
        n_estimators=600, min_samples_leaf=4, max_features="sqrt",
        class_weight="balanced", n_jobs=-1, random_state=random_state)


def _auc(X, y, cols=None, model=None, n_splits: int = 5,
         random_state: int = RANDOM_STATE) -> float:
    Xi = X if cols is None else X[:, cols]
    if Xi.shape[1] == 0:
        return 0.5
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    p = cross_val_predict(build_pipeline(model or _default_model()), Xi, y,
                          cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def signal_decomposition(X, y, feature_names: list[str],
                         blocks: dict[str, list[int]]) -> dict:
    """How much does each source of information contribute on its own?"""
    idx = {n: i for i, n in enumerate(feature_names)}
    vt = [idx[n] for n in feature_names if n.startswith("vtype_")]
    sev = [idx[n] for n in feature_names if n.startswith("severity_")]
    null = [idx["is_null_mutation"]] if "is_null_mutation" in idx else []

    sets = {
        "null-mutation flag alone": null,
        "variant type only": vt,
        "clinical severity only": sev,
        "variant type + severity": vt + sev,
        "all features": list(range(X.shape[1])),
    }
    for name, cols in blocks.items():
        sets[f"block: {name}"] = cols

    out = {}
    for name, cols in sets.items():
        out[name] = {"n_features": len(cols),
                     "auc": round(_auc(X, y, cols), 4)}
    baseline = out["variant type + severity"]["auc"]
    out["_lift_over_variant_type_and_severity"] = round(
        out["all features"]["auc"] - baseline, 4)
    return out


def leave_one_block_out(X, y, blocks: dict[str, list[int]]) -> dict:
    """Cost of removing each biological block from the full feature set."""
    full = _auc(X, y)
    out = {"full_auc": round(full, 4), "blocks": {}}
    all_cols = set(range(X.shape[1]))
    for name, cols in blocks.items():
        keep = sorted(all_cols - set(cols))
        auc = _auc(X, y, keep)
        out["blocks"][name] = {
            "n_removed": len(cols),
            "auc_without": round(auc, 4),
            "cost_of_removal": round(full - auc, 4),
        }
    out["blocks"] = dict(sorted(out["blocks"].items(),
                                key=lambda kv: -kv[1]["cost_of_removal"]))
    return out


def feature_count_sweep(X, y, feature_names: list[str],
                        counts=(10, 20, 30, 50, 80, 110, None),
                        random_state: int = RANDOM_STATE) -> dict:
    """Rank features once, then sweep how many of the top-k to keep.

    The ranking is computed on the training data being swept, so the reported
    curve is mildly optimistic about *where* the peak sits. It is used to choose
    a feature count, not to report performance -- the held-out test set does
    that.
    """
    ranker = _default_model(random_state)
    ranker.fit(np.nan_to_num(X, nan=0.0), y)
    order = np.argsort(-ranker.feature_importances_)

    out = {"ranking_top_20": [feature_names[i] for i in order[:20]], "sweep": {}}
    for k in counts:
        cols = list(order) if k is None else list(order[:k])
        label = "all" if k is None else str(k)
        out["sweep"][label] = round(_auc(X, y, cols), 4)
    best = max(out["sweep"].items(), key=lambda kv: kv[1])
    out["best_k"] = best[0]
    out["best_auc"] = best[1]
    return out


def run(X, y, feature_names: list[str], blocks: dict[str, list[int]]) -> dict:
    return {
        "protocol": "StratifiedKFold(5) on the training split, ExtraTrees",
        "signal_decomposition": signal_decomposition(X, y, feature_names, blocks),
        "leave_one_block_out": leave_one_block_out(X, y, blocks),
        "feature_count_sweep": feature_count_sweep(X, y, feature_names),
    }
