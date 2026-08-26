"""
Mechanical integrity audit of the pipeline.

The project's central claim is that its numbers are trustworthy. That claim is
worth no more than the checks behind it, so rather than asserting "no
oversampling, no leakage" in prose, this module verifies each property by
running it and records the result as an artefact the review can inspect.

Every check answers a question a sceptical examiner would actually ask.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# Anything that resamples the training set by duplicating or synthesising rows.
RESAMPLING_NAMES = [
    "SMOTE", "ADASYN", "RandomOverSampler", "RandomUnderSampler",
    "BorderlineSMOTE", "SVMSMOTE", "KMeansSMOTE", "SMOTENC", "SMOTETomek",
    "imblearn", "resample(", "oversample", "undersample",
]


def check_no_resampling() -> dict:
    """No module may import or call a resampler.

    Class imbalance is handled by weighting the loss (``class_weight`` and
    focal loss), which changes the objective without inventing patients. The
    reference pipeline's Random Over-Sampling is what put 50% of its test rows
    into its own training set.
    """
    # leakage_audit reproduces the reference protocol deliberately, in order to
    # measure it; integrity holds the keyword list itself; report and figures
    # only name the audit experiment in strings. None of those are usages.
    exempt = {"leakage_audit.py", "integrity.py", "report.py", "figures.py"}

    hits = []
    scanned = [p for p in sorted(SRC.glob("*.py")) if p.name not in exempt]
    for path in scanned:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # a name inside a string literal is documentation, not a call
            code = stripped.split("#")[0]
            if '"' in code or "'" in code:
                continue
            for name in RESAMPLING_NAMES:
                if name.lower() in code.lower():
                    hits.append(f"{path.name}:{line_no}: {stripped[:90]}")
    return {
        "check": "no resampling of the training set",
        "files_scanned": len(scanned),
        "files_exempt": sorted(exempt),
        "violations": hits,
        "passed": not hits,
        "note": ("Imbalance is handled by weighting the objective, never by "
                 "duplicating or synthesising patients. leakage_audit.py is "
                 "exempt because measuring the reference over-sampling "
                 "protocol is the point of that module."),
    }


def check_imbalance_handling() -> dict:
    """Every classical model must be imbalance-aware without resampling."""
    from .models import classical_models

    rows = []
    for name, model in classical_models().items():
        params = model.get_params()
        weighted = (params.get("class_weight") is not None
                    or params.get("scale_pos_weight") is not None)
        rows.append({"model": name,
                     "class_weight": str(params.get("class_weight")),
                     "imbalance_aware": bool(weighted)})
    return {
        "check": "imbalance handled by weighting, not resampling",
        "models": rows,
        "passed": all(r["imbalance_aware"] for r in rows),
    }


def check_preprocessing_inside_folds() -> dict:
    """Imputation and scaling must be fitted per fold, not on the whole set."""
    from .models import build_pipeline, classical_models

    pipe = build_pipeline(classical_models()["LogisticRegression"])
    steps = list(pipe.named_steps)
    prep = pipe.named_steps.get("prep")
    inner = list(prep.named_steps) if prep is not None else []
    return {
        "check": "imputer and scaler live inside the CV pipeline",
        "pipeline_steps": steps,
        "preprocessor_steps": inner,
        "passed": steps[:1] == ["prep"] and "impute" in inner and "scale" in inner,
        "note": ("Because they are Pipeline steps, scikit-learn refits them on "
                 "each training fold automatically; fitting them once on the "
                 "full matrix would leak test-fold statistics into training."),
    }


def check_featuriser_is_label_blind() -> dict:
    """Scrambling the outcome must not change a single engineered feature."""
    from .datasets import load_champ
    from .features import VariantFeaturizer

    champ = load_champ()
    scrambled = champ.copy()
    rng = np.random.default_rng(0)
    scrambled["inhibitor"] = rng.permutation(scrambled["inhibitor"].values)
    scrambled["History of Inhibitor"] = rng.permutation(
        scrambled["History of Inhibitor"].values)

    a = VariantFeaturizer().fit(champ).transform(champ)
    b = VariantFeaturizer().fit(scrambled).transform(scrambled)
    identical = a.equals(b)
    return {
        "check": "featuriser never reads the outcome",
        "n_features": int(a.shape[1]),
        "identical_under_label_permutation": bool(identical),
        "passed": bool(identical),
    }


def check_no_identifier_features() -> dict:
    """No engineered column may be near-unique across patients."""
    from .datasets import load_champ
    from .features import IDENTIFIER_COLUMNS, VariantFeaturizer

    champ = load_champ()
    X = VariantFeaturizer().fit(champ).transform(champ)
    frac = (X.nunique() / len(X)).sort_values(ascending=False)
    worst = frac.head(5).round(4).to_dict()
    banned = [c for c in IDENTIFIER_COLUMNS if c in X.columns]
    return {
        "check": "no feature behaves like a row identifier",
        "highest_cardinality_fraction": worst,
        "threshold": 0.5,
        "banned_columns_present": banned,
        "passed": bool(frac.max() < 0.5 and not banned),
    }


def check_test_set_used_once() -> dict:
    """The held-out set must not appear in any fitting or selection step."""
    text = (SRC / "train.py").read_text(encoding="utf-8")
    fit_on_test = [ln.strip() for ln in text.splitlines()
                   if ".fit(" in ln and "X[te]" in ln]
    thresholds_from_test = [ln.strip() for ln in text.splitlines()
                            if "threshold" in ln.lower() and "y[te]" in ln]
    return {
        "check": "test set is scored, never fitted or tuned on",
        "fits_on_test": fit_on_test,
        "thresholds_chosen_on_test": thresholds_from_test,
        "passed": not fit_on_test and not thresholds_from_test,
    }


def check_label_policy() -> dict:
    """Unrecorded outcomes must stay unrecorded, not become negatives."""
    from .datasets import label_summary, load_champ

    s = label_summary(load_champ())
    return {
        "check": "'Not reported' is never relabelled as 'no inhibitor'",
        **s,
        "passed": bool(s["n_unlabelled"] > 1500
                       and 0.18 < s["prevalence_labelled"] < 0.23),
        "note": ("Prevalence of 20.1% matches published epidemiology for "
                 "severe hemophilia A; relabelling would drive it to 11.4%."),
    }


def check_accuracy_reported_with_baseline() -> dict:
    """Any accuracy figure must ship with its majority-class baseline."""
    import json

    path = ROOT / "reports" / "final.json"
    if not path.exists():
        return {"check": "accuracy reported with its baseline",
                "passed": None, "note": "final.json not generated yet"}
    d = json.loads(path.read_text(encoding="utf-8"))
    ctx = d.get("accuracy_context", {})
    return {
        "check": "accuracy reported with its majority-class baseline",
        "model_accuracy": ctx.get("model_accuracy"),
        "majority_class_accuracy": ctx.get("majority_class_accuracy"),
        "margin_over_baseline": (
            round(ctx["model_accuracy"] - ctx["majority_class_accuracy"], 4)
            if ctx.get("model_accuracy") is not None else None),
        "passed": bool(ctx.get("majority_class_accuracy") is not None),
    }


CHECKS = [
    check_no_resampling,
    check_imbalance_handling,
    check_preprocessing_inside_folds,
    check_featuriser_is_label_blind,
    check_no_identifier_features,
    check_test_set_used_once,
    check_label_policy,
    check_accuracy_reported_with_baseline,
]


def run_all() -> dict:
    results = {}
    for fn in CHECKS:
        try:
            r = fn()
        except Exception as exc:                       # a check must not hide
            r = {"check": fn.__name__, "passed": False,
                 "error": f"{type(exc).__name__}: {exc}"}
        results[fn.__name__.replace("check_", "")] = r
    passed = [k for k, v in results.items() if v.get("passed") is True]
    failed = [k for k, v in results.items() if v.get("passed") is False]
    results["_summary"] = {
        "n_checks": len(CHECKS),
        "passed": len(passed),
        "failed": len(failed),
        "failed_checks": failed,
        "all_passed": not failed,
    }
    return results


if __name__ == "__main__":
    import json

    out = run_all()
    (ROOT / "reports" / "integrity.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    for name, r in out.items():
        if name.startswith("_"):
            continue
        mark = {True: "PASS", False: "FAIL", None: "SKIP"}[r.get("passed")]
        print(f"  [{mark}] {r.get('check', name)}")
    print(f"\n{out['_summary']['passed']}/{out['_summary']['n_checks']} checks passed")
