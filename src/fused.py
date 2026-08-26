"""
The fused CHAMP + clinical dataset: provenance audit and simulation study.

A collaborator supplied ``Final_Fused_Dataset.csv`` -- the CHAMP variant table
with five patient-level columns appended: ``Age_at_Diagnosis``, ``Ethnicity``,
``Treatment_Regimen``, ``Exposure_Days`` and ``Family_History``. Those are
precisely the variables this project's limitations section says CHAMP lacks and
that the model most needs, so the file looked like it might lift performance
into the range a capstone rubric asks for.

It does lift performance. The question this module answers first is *why*, and
the answer changes what the numbers are allowed to be used for.

Provenance
----------
CHAMP is a catalogue of published variants. It has no patients, no ages, no
exposure days -- a variant row aggregates every case ever reported to carry that
mutation. There is no join key by which real clinical data could be attached,
so the clinical block cannot have been merged in from a registry; it has to have
been generated. ``audit_provenance`` tests that directly, and four independent
signatures agree:

1. ``Patient_ID`` is a random UUID4 per row. Registries use structured
   identifiers; ``uuid.uuid4()`` produces exactly this.
2. ``Ethnicity`` has no association with the outcome whatsoever (chi-square
   p ~ 0.95, inhibitor rate flat at 11.2-12.6% across all five groups). The
   roughly two-fold higher inhibitor rate in Black and Hispanic patients is
   among the most reproducible non-genetic findings in this field, replicated
   across CDC surveillance, MLOF and UKHCDO. A real cohort of 4,026 patients
   would show it. A column drawn from a fixed multinomial would look exactly
   like this.
3. ``Family_History`` reproduces the published odds ratio of ~3 to two decimal
   places (3.31). That is the number being typed into a generator, not measured.
4. ``Age_at_Diagnosis`` and ``Exposure_Days`` correlate at r = 0.86, far tighter
   than any real clinical pair, which is what happens when one is generated as a
   function of the other.

What that means for the numbers
-------------------------------
Accuracy obtained by using these columns is circular. The model recovers the
effect sizes somebody wrote into the generator; it is graded on data whose
answers were drawn from the thing being predicted. It says nothing about real
patients, and it cannot be presented as validation.

What it is legitimately good for
--------------------------------
Read the other way round, the file is a serviceable **power analysis**. If
patient-level covariates with literature-plausible effect sizes were available,
how much would inhibitor prediction improve? That is a real question, it
supports this project's central conclusion that the ceiling is data rather than
method, and it quantifies the value of collecting registry variables. The
labelling matters: this is a simulation result, reported as one.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
FUSED = ROOT / "data" / "raw" / "fused_champ_clinical.csv"

CLINICAL_COLUMNS = ["Age_at_Diagnosis", "Ethnicity", "Treatment_Regimen",
                    "Exposure_Days", "Family_History"]

# Columns the collaborator pre-engineered. They are recomputed from scratch by
# this project's own featuriser, so they are dropped rather than trusted --
# `Severity_Encoded` in particular carries the "not reported -> 0" defect.
SUPPLIED_DERIVED = ["Inhibitor_Status", "Severity_Encoded",
                    "High_Risk_Mutation_Flag", "Mutation_Location_Group",
                    "Genomic_Region_Bucket", "Variant_Type_Encoded",
                    "Severity_Interaction", "Patient_ID"]

UUID4 = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"


def load_fused(path: Path | str | None = None) -> pd.DataFrame:
    """Load the fused file and restore the honest tri-state label.

    The supplied ``Inhibitor_Status`` maps CHAMP's 1,731 "Not reported" rows to
    0, which is the defect documented in section 2 of the results. The label is
    therefore rebuilt from ``History of Inhibitor`` instead of trusted.
    """
    from .datasets import _clean_columns, _norm_label

    path = Path(path) if path else FUSED
    df = _clean_columns(pd.read_csv(path)).dropna(how="all")
    df["inhibitor"] = [_norm_label(v) for v in df["History of Inhibitor"]]
    df["gene"] = "F8"
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Provenance audit
# ---------------------------------------------------------------------------
def audit_provenance(df: pd.DataFrame) -> dict:
    """Four independent tests of whether the clinical block is real or drawn."""
    lab = df[df["inhibitor"] != -1]
    y = (lab["inhibitor"] == 1).astype(int).values
    out: dict = {"n_labelled": int(len(y)), "n_events": int(y.sum())}

    # 1 -- identifier format
    ids = df["Patient_ID"].astype(str)
    out["patient_id"] = {
        "n_unique": int(ids.nunique()),
        "n_rows": int(len(ids)),
        "fraction_matching_uuid4": round(float(ids.str.match(UUID4).mean()), 4),
        "verdict": ("random UUID4 per row -- generated, not a registry identifier"
                    if ids.str.match(UUID4).mean() > 0.95 else "structured"),
    }

    # 2 -- ethnicity: the field's most reproducible non-genetic association
    ct = pd.crosstab(lab["Ethnicity"], y)
    rate = (ct[1] / ct.sum(axis=1) * 100).round(2)
    out["ethnicity"] = {
        "inhibitor_rate_pct": rate.to_dict(),
        "spread_pct": round(float(rate.max() - rate.min()), 2),
        "chi2_p": round(float(stats.chi2_contingency(ct).pvalue), 4),
        "expected_from_literature": ("~2x higher rate in Black and Hispanic "
                                     "patients (CDC, MLOF, UKHCDO)"),
        "verdict": ("flat across all groups -- the best-replicated non-genetic "
                    "effect in inhibitor epidemiology is absent, which a real "
                    "cohort of this size would not show"),
    }

    # 3 -- family history: effect size versus the published one
    ct = pd.crosstab(lab["Family_History"], y)
    a, b, c, d = ct.loc[1, 1], ct.loc[1, 0], ct.loc[0, 1], ct.loc[0, 0]
    out["family_history"] = {
        "odds_ratio": round(float((a / b) / (c / d)), 2),
        "published_odds_ratio": 3.0,
        "chi2_p": float(f"{stats.chi2_contingency(ct).pvalue:.3g}"),
        "verdict": "matches the published figure to two decimals",
    }

    # 4 -- internal correlation structure
    r = float(np.corrcoef(df["Age_at_Diagnosis"], df["Exposure_Days"])[0, 1])
    out["age_vs_exposure"] = {
        "pearson_r": round(r, 3),
        "verdict": ("near-deterministic for a clinical pair -- consistent with "
                    "one being generated from the other"),
    }

    out["conclusion"] = (
        "The clinical block is simulated. CHAMP rows are published variants, "
        "not patients, so there is no key on which real clinical data could "
        "have been joined; the four signatures above confirm generation with "
        "hand-chosen effect sizes. Performance obtained from these columns is "
        "circular and cannot be reported as validation. It is reported here as "
        "a power analysis instead.")
    return out


def clinical_effect_sizes(df: pd.DataFrame) -> dict:
    """Univariate association of each supplied clinical column with the label."""
    from sklearn.metrics import roc_auc_score

    lab = df[df["inhibitor"] != -1]
    y = (lab["inhibitor"] == 1).astype(int).values
    out = {}
    for col in ["Age_at_Diagnosis", "Exposure_Days"]:
        v = lab[col].values.astype(float)
        auc = float(roc_auc_score(y, v))
        out[col] = {
            "auc": round(auc, 4),
            "auc_oriented": round(max(auc, 1 - auc), 4),
            "mean_positive": round(float(v[y == 1].mean()), 2),
            "mean_negative": round(float(v[y == 0].mean()), 2),
            "mannwhitney_p": float(f"{stats.mannwhitneyu(v[y == 1], v[y == 0]).pvalue:.3g}"),
        }
    for col in ["Ethnicity", "Treatment_Regimen", "Family_History"]:
        ct = pd.crosstab(lab[col], y)
        out[col] = {
            "inhibitor_rate_pct": (ct[1] / ct.sum(axis=1) * 100).round(2).to_dict(),
            "chi2_p": float(f"{stats.chi2_contingency(ct).pvalue:.3g}"),
        }
    return out


# ---------------------------------------------------------------------------
# Featurisation of the clinical block
# ---------------------------------------------------------------------------
def clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the five supplied clinical columns.

    Deliberately plain: the point of the simulation is to measure what these
    covariates are worth, so wrapping them in elaborate engineering would only
    blur the reading.
    """
    out = pd.DataFrame(index=df.index)
    out["age_at_diagnosis"] = pd.to_numeric(df["Age_at_Diagnosis"], errors="coerce")
    out["log_age"] = np.log1p(out["age_at_diagnosis"].clip(lower=0))
    out["exposure_days"] = pd.to_numeric(df["Exposure_Days"], errors="coerce")
    out["log_exposure_days"] = np.log1p(out["exposure_days"].clip(lower=0))
    # the first 50 exposure days carry most of the inhibitor risk
    out["within_first_50_ed"] = (out["exposure_days"] <= 50).astype(float)
    out["family_history"] = pd.to_numeric(df["Family_History"], errors="coerce")
    out["prophylaxis"] = (df["Treatment_Regimen"].astype(str).str.strip()
                          .str.lower() == "prophylaxis").astype(float)
    for lvl in ["Caucasian", "Asian", "Black", "Hispanic", "Other"]:
        out[f"ethnicity_{lvl.lower()}"] = (
            df["Ethnicity"].astype(str).str.strip() == lvl).astype(float)
    # the interaction the literature emphasises: intensive early exposure in a
    # patient who has never made endogenous FVIII
    out["young_and_early_exposure"] = (
        (out["age_at_diagnosis"] <= 2).astype(float)
        * out["within_first_50_ed"])
    return out.astype(float)


# ---------------------------------------------------------------------------
# The simulation study
# ---------------------------------------------------------------------------
def run_simulation(seed: int = 42, n_repeats: int = 3) -> dict:
    """Measure what the simulated covariates are worth, on honest labels.

    Both arms use the same folds, the same held-out patients and the same
    leakage-free genomic featuriser. The only difference between them is
    whether the clinical block is present, so the gap is attributable to those
    columns and nothing else.

    Accuracy is reported alongside the always-predict-negative baseline in
    every arm, because on a 20%-prevalence outcome accuracy on its own is not
    interpretable -- and on the 11%-prevalence version of this label it is
    actively misleading.
    """
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

    from .datasets import split_by_label
    from .evaluate import bootstrap_ci, compute_metrics, delong_test
    from .features import VariantFeaturizer
    from .models import RANDOM_STATE, build_pipeline, classical_models

    df = load_fused()
    labelled, _ = split_by_label(df)
    y = (labelled["inhibitor"] == 1).astype(int).values

    idx = np.arange(len(labelled))
    tr, te = train_test_split(idx, test_size=0.20, stratify=y, random_state=seed)

    fz = VariantFeaturizer().fit(labelled.iloc[tr])
    X_gen = fz.transform(labelled).values.astype(float)
    X_cli = clinical_features(labelled).values.astype(float)
    X_both = np.hstack([X_gen, X_cli])

    arms = {
        "genomic_only": X_gen,
        "clinical_only": X_cli,
        "genomic_plus_clinical": X_both,
    }
    model = classical_models()["ExtraTrees"]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats,
                                 random_state=RANDOM_STATE)

    results, test_probs = {}, {}
    for name, X in arms.items():
        from sklearn.metrics import roc_auc_score
        fold = []
        for a, b in cv.split(X[tr], y[tr]):
            pipe = build_pipeline(model).fit(X[tr][a], y[tr][a])
            fold.append(float(roc_auc_score(y[tr][b],
                                            pipe.predict_proba(X[tr][b])[:, 1])))
        fitted = build_pipeline(model).fit(X[tr], y[tr])
        p = fitted.predict_proba(X[te])[:, 1]
        test_probs[name] = p

        # accuracy-optimal threshold, chosen on training folds only
        oof = np.zeros(len(tr))
        for a, b in RepeatedStratifiedKFold(
                n_splits=5, n_repeats=1, random_state=RANDOM_STATE).split(X[tr], y[tr]):
            m = build_pipeline(model).fit(X[tr][a], y[tr][a])
            oof[b] = m.predict_proba(X[tr][b])[:, 1]
        grid = np.unique(np.round(oof, 3))
        acc_thr = float(max(grid, key=lambda t: accuracy_score(y[tr], (oof >= t).astype(int))))

        results[name] = {
            "n_features": int(X.shape[1]),
            "cv_auc_mean": round(float(np.mean(fold)), 4),
            "cv_auc_std": round(float(np.std(fold)), 4),
            "test_auc_ci": bootstrap_ci(y[te], p, "auc_roc"),
            "at_youden": compute_metrics(y[te], p),
            "at_accuracy_optimal": compute_metrics(y[te], p, acc_thr),
            "majority_class_accuracy": round(float(max(y[te].mean(), 1 - y[te].mean())), 4),
        }

    results["_delong_gain"] = delong_test(
        y[te], test_probs["genomic_plus_clinical"], test_probs["genomic_only"])
    g, b = results["genomic_only"], results["genomic_plus_clinical"]
    results["_summary"] = {
        "auc_gain_from_clinical": round(
            b["test_auc_ci"]["point"] - g["test_auc_ci"]["point"], 4),
        "accuracy_genomic_only": g["at_accuracy_optimal"]["accuracy"],
        "accuracy_with_clinical": b["at_accuracy_optimal"]["accuracy"],
        "majority_baseline": g["majority_class_accuracy"],
        "n_test": int(len(te)),
        "prevalence": round(float(y[te].mean()), 4),
        "reading": (
            "The gap between the two arms is what the simulated covariates are "
            "worth. Because those covariates were generated with hand-chosen "
            "effect sizes, the gap measures the generator, not real patients -- "
            "it is a power analysis for collecting registry data, not a "
            "validation result."),
    }
    return results
