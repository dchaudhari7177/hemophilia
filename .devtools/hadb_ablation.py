"""What each layer of the HADB feature set is actually worth, plus the
controls that prove the score is not an artefact.

Three questions, one script:

1. **Ablation ladder** -- genotype alone, + domain/position, + chemistry,
   + patient clinical measurements, + reporting region. The jump at the
   clinical rung is the entire argument for adding this dataset, so it gets
   measured rather than asserted.

2. **Shuffled-label control** -- refit on permuted outcomes. Any AUC
   meaningfully above 0.5 means the design matrix carries row identity, which
   is how the earlier CHAMP work reached 1.000 train AUC.

3. **Split-strictness ladder** -- ungrouped, variant-grouped, study-grouped.
   The gap between ungrouped and variant-grouped is the size of the leak this
   protocol closes; the study-grouped number is what to quote for a new
   centre.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import compute_metrics  # noqa: E402
from src.hadb_train import (  # noqa: E402
    REPORTS,
    build_cohort,
    grouped_folds,
    model_zoo,
    pos_weight_for,
)

warnings.filterwarnings("ignore")
SEEDS = (42, 202, 7)


def oof_auc(model, X, y, folds) -> tuple[float, float]:
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        m = clone(model)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    met = compute_metrics(y, oof)
    return float(met["auc_roc"]), float(met.get("auc_pr", np.nan))


def repeated_auc(model, X, y, groups, seeds=SEEDS) -> dict:
    aucs, prs = [], []
    for s in seeds:
        a, p = oof_auc(model, X, y, grouped_folds(y, groups, random_state=s))
        aucs.append(a)
        prs.append(p)
    return {
        "auc_roc_mean": round(float(np.mean(aucs)), 4),
        "auc_roc_std": round(float(np.std(aucs)), 4),
        "auc_pr_mean": round(float(np.mean(prs)), 4),
        "n_seeds": len(seeds),
    }


def main() -> None:
    t0 = time.time()
    out: dict = {}

    full = build_cohort()
    pw = pos_weight_for(full.y)
    model = model_zoo(pos_weight=pw)["random_forest"]
    print(f"cohort {len(full)} records / {len(np.unique(full.groups))} variants "
          f"/ prevalence {full.prevalence:.4f}")

    # -- 1. ablation ladder -------------------------------------------------
    genomic = build_cohort(include_clinical=False, include_context=False)
    rungs = {
        "genotype_only": genomic.blocks["genotype"],
        "plus_domain": genomic.blocks["genotype"] + genomic.blocks["domain"],
        "plus_position": (genomic.blocks["genotype"] + genomic.blocks["domain"]
                          + genomic.blocks["position"]),
        "genomic_all": list(genomic.X.columns),
    }
    ladder = {}
    for name, cols in rungs.items():
        ladder[name] = repeated_auc(model, genomic.X[cols], genomic.y,
                                    genomic.groups)
        ladder[name]["n_features"] = len(cols)
        print(f"  {name:22s} {ladder[name]['auc_roc_mean']:.4f} "
              f"+/- {ladder[name]['auc_roc_std']:.4f}  ({len(cols)} features)")

    clin = build_cohort(include_clinical=True, include_context=False)
    ladder["plus_clinical"] = repeated_auc(model, clin.X, clin.y, clin.groups)
    ladder["plus_clinical"]["n_features"] = clin.X.shape[1]
    print(f"  {'plus_clinical':22s} {ladder['plus_clinical']['auc_roc_mean']:.4f} "
          f"+/- {ladder['plus_clinical']['auc_roc_std']:.4f}  "
          f"({clin.X.shape[1]} features)")

    ladder["plus_region"] = repeated_auc(model, full.X, full.y, full.groups)
    ladder["plus_region"]["n_features"] = full.X.shape[1]
    print(f"  {'plus_region':22s} {ladder['plus_region']['auc_roc_mean']:.4f} "
          f"+/- {ladder['plus_region']['auc_roc_std']:.4f}  "
          f"({full.X.shape[1]} features)")

    # Clinical measurements could in principle be a consequence of the outcome
    # rather than a predictor of it: an inhibitor suppresses measured factor
    # activity. Reporting the genomic-only rung alongside the full model is
    # what lets a reader discount that concern instead of taking it on trust.
    ladder["clinical_contribution"] = round(
        ladder["plus_clinical"]["auc_roc_mean"]
        - ladder["genomic_all"]["auc_roc_mean"], 4)
    out["ablation"] = ladder

    # -- 2. shuffled-label control -----------------------------------------
    print("\nshuffled-label control (expect ~0.50):")
    shuffled = []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        y_perm = full.y.copy()
        rng.shuffle(y_perm)
        a, _ = oof_auc(model, full.X, y_perm,
                       grouped_folds(y_perm, full.groups, random_state=s))
        shuffled.append(a)
        print(f"  seed {s}: {a:.4f}")
    out["shuffled_label_control"] = {
        "auc_roc_mean": round(float(np.mean(shuffled)), 4),
        "auc_roc_values": [round(v, 4) for v in shuffled],
        "verdict": ("clean -- the matrix carries no row identity"
                    if abs(np.mean(shuffled) - 0.5) < 0.05
                    else "SUSPECT -- investigate identifier columns"),
    }

    # -- 3. split-strictness ladder ----------------------------------------
    print("\nsplit strictness:")
    strict = {}
    ungrouped = [
        (tr, te) for tr, te in
        StratifiedKFold(5, shuffle=True, random_state=42).split(full.X, full.y)
    ]
    a, p = oof_auc(model, full.X, full.y, ungrouped)
    strict["ungrouped_random"] = {"auc_roc": round(a, 4), "auc_pr": round(p, 4)}
    strict["grouped_by_variant"] = repeated_auc(model, full.X, full.y,
                                                full.groups)
    strict["grouped_by_study"] = repeated_auc(model, full.X, full.y,
                                              full.studies)
    strict["leak_from_ungrouping"] = round(
        strict["ungrouped_random"]["auc_roc"]
        - strict["grouped_by_variant"]["auc_roc_mean"], 4)
    for k in ["ungrouped_random", "grouped_by_variant", "grouped_by_study"]:
        v = strict[k]
        print(f"  {k:22s} {v.get('auc_roc', v.get('auc_roc_mean')):.4f}")
    print(f"  leak closed by grouping: {strict['leak_from_ungrouping']:+.4f} AUC")
    out["split_strictness"] = strict

    out["elapsed_seconds"] = round(time.time() - t0, 1)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "hadb_ablation.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {REPORTS / 'hadb_ablation.json'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
