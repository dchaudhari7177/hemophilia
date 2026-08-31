"""Audit the two derived CSVs that shipped alongside the HADB supplement.

``HemophiliaA_Merged_MMC2_MMC3.csv`` and ``HemophiliaA_ML_Ready_Inhibitor.csv``
are a reasonable first pass at making the supplement modellable, and they
reproduce -- independently, on a new dataset -- the two failure modes this
project was rebuilt to correct. Both are demonstrated numerically rather than
argued, because "this would leak" is a claim and "this scores 1.000" is
evidence.

Experiment A -- unrecorded outcomes relabelled as negatives.
Experiment B -- the outcome, aggregated per variant, left in as a feature.
Experiment C -- variant-level aggregation discards the patient layer.

Writes reports/hadb_audit.json.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import compute_metrics  # noqa: E402
from src.hadb import FORBIDDEN, load_hadb  # noqa: E402
from src.hadb_train import (  # noqa: E402
    REPORTS,
    build_cohort,
    grouped_folds,
    model_zoo,
    pos_weight_for,
)

warnings.filterwarnings("ignore")
HADB = ROOT / "data" / "raw" / "hadb"


def oof_auc(model, X, y, folds) -> float:
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        m = clone(model)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return float(compute_metrics(y, oof)["auc_roc"])


def main() -> None:
    t0 = time.time()
    out: dict = {}
    df = load_hadb()

    # -- A. relabelling unrecorded outcomes --------------------------------
    per_variant = df.groupby("mut_id")["y"].agg(known="count", pos="sum")
    ml = pd.read_csv(HADB / "HemophiliaA_ML_Ready_Inhibitor.csv",
                     low_memory=False)
    n_with_outcome = int((per_variant["known"] > 0).sum())
    true_prev = float((per_variant.loc[per_variant["known"] > 0, "pos"] > 0).mean())

    out["A_unrecorded_relabelled_as_negative"] = {
        "variants_in_registry": int(len(per_variant)),
        "variants_with_a_recorded_outcome": n_with_outcome,
        "rows_in_ml_ready_file": int(len(ml)),
        "rows_labelled_negative_without_an_outcome":
            int(len(ml) - n_with_outcome),
        "prevalence_in_ml_ready_file": round(float(ml["inhibitor_target"].mean()), 4),
        "prevalence_among_variants_actually_followed_up": round(true_prev, 4),
        "prevalence_among_patient_records": round(float(df["y"].mean()), 4),
        "finding": (
            "1,063 variants carry inhibitor_target = 0 although no inhibitor "
            "outcome was ever recorded for them. Absence of a report is not a "
            "negative result. The substitution drops apparent prevalence from "
            "23.5% to 13.3% and pads the majority class, which raises accuracy "
            "without the model predicting anything better."),
    }

    # -- B. the outcome left in as a feature -------------------------------
    merged = pd.read_csv(HADB / "HemophiliaA_Merged_MMC2_MMC3.csv",
                         low_memory=False)
    leak_cols = [c for c in merged.columns if c in FORBIDDEN]
    cohort = build_cohort()
    folds = grouped_folds(cohort.y, cohort.groups)
    model = model_zoo(pos_weight=pos_weight_for(cohort.y))["random_forest"]

    honest = oof_auc(model, cohort.X, cohort.y, folds)

    # Rebuild the cohort with the variant-level positive rate bolted on, the
    # way it appears in the merged CSV.
    rate = df["mut_id"].map(
        merged.set_index("mut_id")["inhibitor_positive_rate"])
    X_leaky = cohort.X.copy()
    X_leaky["inhibitor_positive_rate"] = rate.loc[
        df["y"].notna().to_numpy()].reset_index(drop=True).astype(float)
    leaky = oof_auc(model, X_leaky, cohort.y, folds)

    out["B_outcome_aggregate_used_as_feature"] = {
        "forbidden_columns_present_in_merged_csv": sorted(leak_cols),
        "auc_without_them": round(honest, 4),
        "auc_with_inhibitor_positive_rate": round(leaky, 4),
        "inflation": round(leaky - honest, 4),
        "finding": (
            "inhibitor_positive_rate is the target averaged over the very "
            "records being predicted. Adding that one column moves AUC by the "
            "amount reported above with no new biology involved. uinhibitor, "
            "useverity, uclotting, uratio, uantigen and utype are the same "
            "quantity in other clothing and are all in hadb.FORBIDDEN."),
    }

    # -- C. what variant-level aggregation costs ---------------------------
    # Collapse each variant to a majority-vote label and a single feature row,
    # then score it the same way, to price the aggregation itself.
    lab = df[df["y"].notna()].copy()
    from src.hadb import build_features
    Xall, _ = build_features(df)
    Xlab = Xall.loc[df["y"].notna().to_numpy()].reset_index(drop=True)
    lab = lab.reset_index(drop=True)
    Xlab["_mut"] = lab["mut_id"].to_numpy()
    Xlab["_y"] = lab["y"].to_numpy()
    agg = Xlab.groupby("_mut").mean(numeric_only=True)
    y_v = (agg.pop("_y") > 0.5).astype(float).to_numpy()
    g_v = agg.index.to_numpy()
    variant_auc = oof_auc(model, agg.reset_index(drop=True), y_v,
                          grouped_folds(y_v, g_v))

    out["C_variant_level_aggregation"] = {
        "n_rows_patient_level": int(len(cohort)),
        "n_rows_variant_level": int(len(agg)),
        "auc_patient_level": round(honest, 4),
        "auc_variant_level_majority_vote": round(variant_auc, 4),
        "finding": (
            "Collapsing to one row per variant throws away 2,323 patient "
            "observations and the per-patient factor level that goes with "
            "them. Two patients carrying the same variant genuinely differ in "
            "outcome, and a majority vote deletes that variation rather than "
            "modelling it."),
    }

    out["elapsed_seconds"] = round(time.time() - t0, 1)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "hadb_audit.json").write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2)[:2600])
    print(f"\nwrote {REPORTS / 'hadb_audit.json'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
