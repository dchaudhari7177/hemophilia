"""Fit the selected model, score it once on the held-out set, and ship it.

Order of operations matters here and is deliberate:

1. The model and its operating thresholds are chosen on **training** folds.
2. Probabilities are calibrated on training out-of-fold predictions, because a
   risk that is quoted to a clinician has to mean what it says -- a stated 25%
   should come true about a quarter of the time.
3. Only then is the held-out set scored, once, and reported with bootstrap
   confidence intervals and a DeLong comparison against the genomic-only model.

Writes reports/hadb_final.json and models/hadb_model.joblib.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import (  # noqa: E402
    accuracy_threshold,
    bootstrap_ci,
    calibration_curve_points,
    compute_metrics,
    decision_curve,
    delong_test,
    threshold_at_sensitivity,
    youden_threshold,
)
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
MODELS = ROOT / "models"


def oof_probs(model, X, y, folds):
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        m = clone(model)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


class RankEnsemble:
    """Average the rank transforms of several fitted members.

    Rank averaging is used rather than probability averaging because the
    members are calibrated on different scales -- a forest's 0.6 and a boosted
    tree's 0.6 are not the same claim. Ranking removes the scale, and a single
    isotonic layer afterwards puts the blend back onto a probability axis.
    """

    def __init__(self, members: dict):
        self.members = members

    def fit(self, X, y):
        for m in self.members.values():
            m.fit(X, y)
        return self

    def decision_scores(self, X) -> np.ndarray:
        cols = [rankdata(m.predict_proba(X)[:, 1]) / len(X)
                for m in self.members.values()]
        return np.column_stack(cols).mean(1)

    def predict_proba(self, X) -> np.ndarray:
        s = self.decision_scores(X)
        return np.column_stack([1 - s, s])


def main() -> None:
    t0 = time.time()
    tuning_path = REPORTS / "hadb_tuning.json"
    tuning = json.loads(tuning_path.read_text()) if tuning_path.exists() else {}

    cohort = build_cohort()
    train_mask, test_mask = holdout_split(cohort)
    Xtr = cohort.X.loc[train_mask].reset_index(drop=True)
    ytr = cohort.y[train_mask]
    gtr = cohort.groups[train_mask]
    Xte = cohort.X.loc[test_mask].reset_index(drop=True)
    yte = cohort.y[test_mask]
    folds = grouped_folds(ytr, gtr, n_splits=5)
    pw = pos_weight_for(ytr)

    # -- rebuild the tuned members ----------------------------------------
    zoo = model_zoo(pos_weight=pw)
    members = {}
    for name, info in tuning.get("tuning", {}).items():
        est = clone(zoo[name])
        est.set_params(**info["best_params"])
        members[name] = est
    if not members:                       # tuning not run: fall back to defaults
        members = {k: clone(zoo[k]) for k in
                   ["random_forest", "extra_trees", "lightgbm", "xgboost"]}

    chosen = tuning.get("ensembles", {}).get("greedy_rank", {}).get("members")
    if chosen:
        # Greedy selection picks with replacement; duplicates are weights.
        weights = {n: chosen.count(n) for n in set(chosen)}
        members = {n: members[n] for n in weights if n in members}
    print(f"ensemble members: {list(members)}")

    ens = RankEnsemble(members)

    # -- training out-of-fold scores, thresholds and calibration ----------
    oof = np.full(len(ytr), np.nan)
    for tr, te in folds:
        fold_ens = RankEnsemble({n: clone(m) for n, m in members.items()})
        fold_ens.fit(Xtr.iloc[tr], ytr[tr])
        oof[te] = fold_ens.decision_scores(Xtr.iloc[te])
    train_oof_metrics = compute_metrics(ytr, oof)
    print(f"train out-of-fold AUC {train_oof_metrics['auc_roc']:.4f}")

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    calibrator.fit(oof, ytr)
    oof_cal = calibrator.predict(oof)

    thresholds = {
        "youden": float(youden_threshold(ytr, oof_cal)),
        "accuracy_max": float(accuracy_threshold(ytr, oof_cal)),
        "sensitivity_90": float(threshold_at_sensitivity(ytr, oof_cal, 0.90)),
        "sensitivity_80": float(threshold_at_sensitivity(ytr, oof_cal, 0.80)),
    }
    print("thresholds (chosen on training folds):",
          {k: round(v, 3) for k, v in thresholds.items()})

    # -- fit on all training data, then score the held-out set once -------
    ens.fit(Xtr, ytr)
    raw_te = ens.decision_scores(Xte)
    cal_te = calibrator.predict(raw_te)

    baseline_acc = float(max(yte.mean(), 1 - yte.mean()))
    results = {}
    for tname, thr in thresholds.items():
        m = compute_metrics(yte, cal_te, threshold=thr)
        m["majority_baseline_accuracy"] = round(baseline_acc, 4)
        m["accuracy_over_baseline"] = round(m["accuracy"] - baseline_acc, 4)
        results[f"test_{tname}"] = m
    print(f"held-out AUC {results['test_youden']['auc_roc']:.4f} "
          f"(n={len(yte)}, prevalence {yte.mean():.3f})")

    auc_ci = bootstrap_ci(yte, cal_te, "auc_roc", n_boot=2000)
    pr_ci = bootstrap_ci(yte, cal_te, "auc_pr", n_boot=2000)
    ci = {
        "auc_roc_ci95": [auc_ci["lo"], auc_ci["hi"]],
        "auc_pr_ci95": [pr_ci["lo"], pr_ci["hi"]],
    }
    print(f"  95% CI [{auc_ci['lo']:.4f}, {auc_ci['hi']:.4f}]")

    # A single 20% split holds 974 records and about 159 positives, so its AUC
    # carries roughly +/-0.04 of sampling noise -- too wide to be the headline.
    # The primary generalisation estimate is therefore repeated variant-grouped
    # CV over the whole labelled cohort, and the held-out figure above is the
    # independent confirmation that nothing was tuned into that estimate.
    print("\nrepeated variant-grouped CV over the full labelled cohort:")
    repeated = []
    for seed in (42, 202, 7):
        seed_folds = grouped_folds(cohort.y, cohort.groups, random_state=seed)
        seed_oof = np.full(len(cohort.y), np.nan)
        for tr, te in seed_folds:
            fe = RankEnsemble({n: clone(m) for n, m in members.items()})
            fe.fit(cohort.X.iloc[tr], cohort.y[tr])
            seed_oof[te] = fe.decision_scores(cohort.X.iloc[te])
        a = float(compute_metrics(cohort.y, seed_oof)["auc_roc"])
        repeated.append(a)
        print(f"  seed {seed}: {a:.4f}")
    full_cv = {
        "auc_roc_mean": round(float(np.mean(repeated)), 4),
        "auc_roc_std": round(float(np.std(repeated)), 4),
        "auc_roc_values": [round(v, 4) for v in repeated],
        "n": int(len(cohort.y)),
        "note": ("5-fold StratifiedGroupKFold by mut_id, 3 seeds, whole "
                 "labelled cohort. This is the headline estimate; the "
                 "single held-out split is the independent check."),
    }
    print(f"  mean {full_cv['auc_roc_mean']:.4f} +/- {full_cv['auc_roc_std']:.4f}")

    # -- what the clinical layer is worth on the held-out set -------------
    genomic = build_cohort(include_clinical=False, include_context=False)
    Xg_tr = genomic.X.loc[train_mask].reset_index(drop=True)
    Xg_te = genomic.X.loc[test_mask].reset_index(drop=True)
    g_model = clone(zoo["random_forest"]).fit(Xg_tr, ytr)
    g_prob = g_model.predict_proba(Xg_te)[:, 1]
    delong = delong_test(yte, cal_te, g_prob)
    print(f"genomic-only held-out AUC {compute_metrics(yte, g_prob)['auc_roc']:.4f} "
          f"| DeLong p = {delong.get('p_value')}")

    payload = {
        "protocol": (
            "Members and thresholds selected on variant-grouped training "
            "folds; isotonic calibration fitted on training out-of-fold "
            "scores; held-out set scored once."),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "test_prevalence": round(float(yte.mean()), 4),
        "majority_baseline_accuracy": round(baseline_acc, 4),
        "ensemble_members": list(members),
        "thresholds": {k: round(v, 4) for k, v in thresholds.items()},
        "train_oof": train_oof_metrics,
        "repeated_full_cohort_cv": full_cv,
        **results,
        "confidence_intervals": ci,
        "genomic_only_test": compute_metrics(yte, g_prob),
        "delong_full_vs_genomic": delong,
        "calibration_curve": calibration_curve_points(yte, cal_te),
        "decision_curve": decision_curve(yte, cal_te),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    def jsonable(o):
        """numpy scalars and arrays are not JSON types; curves are arrays."""
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(f"not serialisable: {type(o).__name__}")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "hadb_final.json").write_text(json.dumps(payload, indent=2,
                                                        default=jsonable))
    np.savez(REPORTS / "hadb_test_predictions.npz",
             y_true=yte, prob_cal=cal_te, prob_raw=raw_te, prob_genomic=g_prob)

    MODELS.mkdir(exist_ok=True)
    joblib.dump({
        "ensemble": ens,
        "calibrator": calibrator,
        "thresholds": thresholds,
        "feature_names": list(cohort.X.columns),
        "feature_blocks": cohort.blocks,
        "train_reference": Xtr.median().to_dict(),
        "metrics": results["test_youden"],
        "provenance": "EAHAD/HADB VTH-2024-000215, patient-level, "
                      "variant-grouped split",
    }, MODELS / "hadb_model.joblib", compress=3)

    print(f"\nwrote reports/hadb_final.json and models/hadb_model.joblib "
          f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
