"""
Forensic audit of the reference pipeline's headline result.

The reference deep-learning notebook for this dataset reports 99.63% accuracy
and AUC-ROC 0.9999, and the classical-ML paper it builds on reports 97.37%
accuracy. Both numbers are far above what the underlying biology can support:
inhibitor development is driven substantially by treatment intensity, product
type, exposure timing and HLA haplotype, none of which appear in CHAMP.

This module does not argue the point -- it measures it. Four experiments run
the reference preprocessing verbatim and then vary one thing at a time:

  A. REFERENCE          reference preprocessing, reference labels
  B. IDENTIFIERS_ONLY   *only* the near-unique columns (HGVS cDNA, hg19
                        coordinate, protein notation, codon). If these alone
                        reproduce the headline score, the score is identity
                        lookup, not biology.
  C. NO_IDENTIFIERS     reference preprocessing minus those columns
  D. HONEST_LABELS      reference preprocessing, but "Not reported" excluded
                        instead of relabelled 0

plus two stress tests:

  E. LABEL_PERMUTATION  labels shuffled. A model that has learned biology
                        collapses to AUC 0.5; a model that is memorising row
                        identity still fits the training set perfectly, which
                        exposes the capacity being exploited.
  F. NOVEL_VARIANT      test variants drawn from stretches of the gene that do
                        not appear in training, i.e. the real clinical task of
                        scoring a newly discovered mutation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .datasets import LABEL_UNKNOWN, protein_region_blocks

RANDOM_STATE = 42

# The columns the reference pipeline label-encodes that are effectively row ids.
IDENTIFIER_LIKE = [
    "HGVS cDNA", "hg19 Coordinates", "HGVS Protein", "Mature Protein", "Codon",
]


def reference_preprocess(df: pd.DataFrame, honest_labels: bool = False):
    """Reproduce the reference notebook's preprocessing.

    Faithful to the original: drop comment/reference/year/unnamed columns, map
    ``History of Inhibitor`` with ``{"yes": 1, everything else: 0}``, then
    ``LabelEncoder`` every remaining object column -- including the near-unique
    identifier columns.
    """
    df = df.copy()
    target = "History of Inhibitor"

    drop_pats = ["comment", "reference", "year", "newly added", "unnamed",
                 "inhibitor", "gene"]

    if honest_labels:
        df = df[df["inhibitor"] != LABEL_UNKNOWN]
        y = (df["inhibitor"] == 1).astype(int)
    else:
        t = df[target].astype(str).str.strip().str.lower()
        y = t.map({"yes": 1, "no": 0, "not reported": 0,
                   "nan": 0, "unknown": 0}).fillna(0).astype(int)

    X = df.drop(columns=[c for c in df.columns
                         if any(p in str(c).lower() for p in drop_pats)],
                errors="ignore")
    X = X.fillna("Unknown")
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.apply(pd.to_numeric, errors="coerce").fillna(-1)
    return X.reset_index(drop=True), y.reset_index(drop=True)


def _fit_score(X, y, train_idx, test_idx) -> dict:
    """A Random Forest matching the reference paper's best classical model."""
    clf = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1,
        class_weight="balanced_subsample")
    clf.fit(X.iloc[train_idx], y.iloc[train_idx])
    p_te = clf.predict_proba(X.iloc[test_idx])[:, 1]
    p_tr = clf.predict_proba(X.iloc[train_idx])[:, 1]
    yte, ytr = y.iloc[test_idx], y.iloc[train_idx]
    return {
        "test_accuracy": round(float(accuracy_score(yte, (p_te >= 0.5).astype(int))), 4),
        "test_auc": round(float(roc_auc_score(yte, p_te)), 4) if yte.nunique() > 1 else None,
        "train_accuracy": round(float(accuracy_score(ytr, (p_tr >= 0.5).astype(int))), 4),
        "train_auc": round(float(roc_auc_score(ytr, p_tr)), 4) if ytr.nunique() > 1 else None,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_prevalence": round(float(yte.mean()), 4),
        "majority_class_accuracy": round(float(max(yte.mean(), 1 - yte.mean())), 4),
    }


def _stratified_idx(y, test_size=0.2, seed=RANDOM_STATE):
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)


def _ros_then_split(X: pd.DataFrame, y: pd.Series, seed: int = RANDOM_STATE) -> dict:
    """Random Over-Sample the minority class, *then* split. Deliberately wrong."""
    rng = np.random.default_rng(seed)
    pos = np.where(y.values == 1)[0]
    neg = np.where(y.values == 0)[0]
    extra = rng.choice(pos, size=max(len(neg) - len(pos), 0), replace=True)
    idx = np.concatenate([np.arange(len(y)), extra])
    Xr = X.iloc[idx].reset_index(drop=True)
    yr = y.iloc[idx].reset_index(drop=True)
    tr, te = _stratified_idx(yr, seed=seed)
    out = _fit_score(Xr, yr, tr, te)
    # how many test rows are verbatim copies of a training row?
    tr_keys = set(map(tuple, Xr.iloc[tr].to_numpy()))
    dup = sum(1 for r in Xr.iloc[te].to_numpy() if tuple(r) in tr_keys)
    out["test_rows_also_in_train"] = int(dup)
    out["fraction_test_rows_duplicated_from_train"] = round(dup / len(te), 4)
    return out


def run_audit(champ: pd.DataFrame) -> dict:
    results: dict[str, dict] = {}

    # ---- A. the reference pipeline, verbatim ---------------------------
    Xa, ya = reference_preprocess(champ, honest_labels=False)
    tr, te = _stratified_idx(ya)
    results["A_reference_pipeline"] = {
        "description": "Reference preprocessing and labels ('Not reported' -> 0).",
        **_fit_score(Xa, ya, tr, te),
    }

    # ---- B. identifier columns alone -----------------------------------
    id_cols = [c for c in Xa.columns if c in IDENTIFIER_LIKE]
    results["B_identifiers_only"] = {
        "description": ("Only the near-unique identifier columns "
                        f"({', '.join(id_cols)}). Carries no biology."),
        "columns_used": id_cols,
        **_fit_score(Xa[id_cols], ya, tr, te),
    }

    # ---- C. everything except the identifiers --------------------------
    bio_cols = [c for c in Xa.columns if c not in IDENTIFIER_LIKE]
    results["C_no_identifiers"] = {
        "description": "Reference pipeline with identifier columns removed.",
        "columns_used": bio_cols,
        **_fit_score(Xa[bio_cols], ya, tr, te),
    }

    # ---- D. honest labels ----------------------------------------------
    Xd, yd = reference_preprocess(champ, honest_labels=True)
    trd, ted = _stratified_idx(yd)
    results["D_honest_labels"] = {
        "description": ("Reference features, but rows with an unrecorded "
                        "outcome are excluded rather than called negative."),
        **_fit_score(Xd, yd, trd, ted),
    }

    # ---- E. label permutation ------------------------------------------
    rng = np.random.default_rng(RANDOM_STATE)
    y_perm = pd.Series(rng.permutation(ya.values))
    results["E_label_permutation"] = {
        "description": ("Labels shuffled. Test AUC must fall to ~0.50; a "
                        "train AUC that stays near 1.0 shows the model is "
                        "fitting row identity, not signal."),
        **_fit_score(Xa, y_perm, tr, te),
    }

    # ---- F. novel-variant (position-blocked) split ----------------------
    blocks = protein_region_blocks(champ, n_blocks=5)
    if len(blocks) == len(Xa):
        holdout = blocks == 4
        trf = np.where(~holdout)[0]
        tef = np.where(holdout)[0]
        results["F_novel_variant_split"] = {
            "description": ("Test set is a contiguous stretch of the gene that "
                            "never appears in training -- the real task of "
                            "scoring a newly discovered variant."),
            **_fit_score(Xa, ya, trf, tef),
        }

    # ---- G. oversampling applied before the split ----------------------
    # The classical-ML reference applies Random Over-Sampling to the whole
    # dataset and *then* runs stratified k-fold. Because ROS duplicates
    # minority rows verbatim, the same patient can land in both the training
    # and the evaluation fold, and the classifier is graded on rows it has
    # already memorised. This experiment reproduces that protocol so the
    # inflation can be measured rather than argued about.
    results["G_oversample_before_split"] = {
        "description": ("Random Over-Sampling applied before the train/test "
                        "split, as in the classical-ML reference. Duplicated "
                        "minority rows appear on both sides of the split."),
        **_ros_then_split(Xa, ya),
    }

    # ---- headline comparison -------------------------------------------
    a = results["A_reference_pipeline"]
    b = results["B_identifiers_only"]
    c = results["C_no_identifiers"]
    results["_interpretation"] = {
        "reference_test_auc": a["test_auc"],
        "identifiers_alone_test_auc": b["test_auc"],
        "biology_alone_test_auc": c["test_auc"],
        "share_of_auc_lift_attributable_to_identifiers": (
            round((b["test_auc"] - 0.5) / (a["test_auc"] - 0.5), 3)
            if a["test_auc"] and a["test_auc"] > 0.5 else None),
        "accuracy_inflation_from_relabelling_unknowns": (
            round(a["test_accuracy"] - results["D_honest_labels"]["test_accuracy"], 4)),
    }
    return results


def save_audit(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
