"""Screen every candidate model on the HADB cohort under variant-grouped CV.

Writes reports/hadb_screen.json. Nothing here touches the held-out test set;
selection happens on cross-validated scores only.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import compute_metrics  # noqa: E402
from src.hadb_train import (  # noqa: E402
    REPORTS,
    Cohort,
    build_cohort,
    grouped_folds,
    holdout_split,
    model_zoo,
    pos_weight_for,
    stacking_ensemble,
)

warnings.filterwarnings("ignore")


def cross_val_probs(model, cohort: Cohort, mask, folds):
    """Out-of-fold probabilities over the rows selected by ``mask``."""
    X = cohort.X.loc[mask].reset_index(drop=True)
    y = cohort.y[mask]
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        from sklearn.base import clone
        m = clone(model)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return y, oof


def summarise(y, prob, baseline) -> dict:
    m = compute_metrics(y, prob)
    m["majority_baseline_accuracy"] = round(baseline, 4)
    m["accuracy_over_baseline"] = round(m.get("accuracy", 0) - baseline, 4)
    return {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating))
                else v) for k, v in m.items()}


def main() -> None:
    t0 = time.time()
    cohort = build_cohort()
    train_mask, test_mask = holdout_split(cohort)
    print(f"cohort: {len(cohort)} records, prevalence {cohort.prevalence:.4f}, "
          f"{len(np.unique(cohort.groups))} variants")
    print(f"train {train_mask.sum()} / test {test_mask.sum()} "
          f"(grouped by variant, no mut_id straddles the split)")

    y_tr = cohort.y[train_mask]
    g_tr = cohort.groups[train_mask]
    folds = grouped_folds(y_tr, g_tr, n_splits=5)
    pw = pos_weight_for(y_tr)
    baseline = float(max(y_tr.mean(), 1 - y_tr.mean()))

    results = {}
    zoo = model_zoo(pos_weight=pw)
    zoo["stacking"] = stacking_ensemble(pos_weight=pw)

    for name, model in zoo.items():
        t = time.time()
        try:
            y, oof = cross_val_probs(model, cohort, train_mask, folds)
            results[name] = summarise(y, oof, baseline)
            results[name]["fit_seconds"] = round(time.time() - t, 1)
            print(f"  {name:24s} AUC {results[name]['auc_roc']:.4f}  "
                  f"PR {results[name].get('auc_pr', float('nan')):.4f}  "
                  f"({results[name]['fit_seconds']}s)")
        except Exception as exc:  # keep the screen going, record the failure
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  {name:24s} FAILED {exc}")

    payload = {
        "protocol": "5-fold StratifiedGroupKFold grouped by mut_id, "
                    "training partition only, no resampling",
        "n_train": int(train_mask.sum()),
        "n_test_heldout": int(test_mask.sum()),
        "prevalence_train": round(float(y_tr.mean()), 4),
        "majority_baseline_accuracy": round(baseline, 4),
        "n_features": int(cohort.X.shape[1]),
        "feature_blocks": {k: len(v) for k, v in cohort.blocks.items()},
        "models": results,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "hadb_screen.json").write_text(json.dumps(payload, indent=2))

    ranked = sorted(((v.get("auc_roc", 0), k) for k, v in results.items()),
                    reverse=True)
    print("\nranked by cross-validated AUC-ROC:")
    for auc, name in ranked:
        print(f"  {auc:.4f}  {name}")
    print(f"\nwrote {REPORTS / 'hadb_screen.json'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
