"""
Training driver.

Stages (run with ``python -m src.train --stage <name>``):

  audit     reproduce the reference pipeline and measure where its score comes from
  cv        repeated stratified cross-validation over the whole model zoo
  blocked   position-blocked cross-validation (generalising to unseen regions)
  final     fit the selected model, calibrate it, evaluate on the held-out test set
  subgroups per-stratum performance (severe, null, missense, ...)
  ssl       missing-data probe and semi-supervised use of the unlabelled pool
  external  transfer the F8 model onto the CHBMP F9 cohort
  all       every stage, in order

Every stage writes a JSON file into ``reports/`` so the write-up is generated
from measurements rather than transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     train_test_split)

from .datasets import (LABEL_UNKNOWN, load_champ, load_chbmp, label_summary,
                       protein_region_blocks, split_by_label)
from .evaluate import (bootstrap_ci, compute_metrics, delong_test,
                       threshold_at_sensitivity, youden_threshold)
from .features import VariantFeaturizer, block_index
from .models import (RANDOM_STATE, build_pipeline, classical_models,
                     neural_models)
from .semisupervised import (ReportingBiasProbe, SelfTrainingSSL,
                             estimate_unlabelled_prevalence)

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
TEST_SIZE = 0.20
N_SPLITS = 5
N_REPEATS = 3


def _save(name: str, obj) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    print(f"  -> {path.relative_to(ROOT)}")
    return path


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Shared data preparation
# ---------------------------------------------------------------------------
def prepare(seed: int = RANDOM_STATE):
    """Load CHAMP, featurise, and carve off a held-out test set.

    The featuriser is fitted on the training rows only. It never sees a label,
    so this is belt-and-braces rather than a strict requirement, but it keeps
    the pipeline defensible under the strictest reading.
    """
    champ = load_champ()
    labelled, unlabelled = split_by_label(champ)
    y = (labelled["inhibitor"] == 1).astype(int).values

    idx = np.arange(len(labelled))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE, stratify=y,
                                      random_state=seed)

    fz = VariantFeaturizer().fit(labelled.iloc[tr_idx])
    X = fz.transform(labelled).values.astype(float)
    X_unl = fz.transform(unlabelled).values.astype(float)

    return {
        "champ": champ,
        "labelled": labelled,
        "unlabelled": unlabelled,
        "featurizer": fz,
        "X": X,
        "y": y,
        "X_unlabelled": X_unl,
        "train_idx": tr_idx,
        "test_idx": te_idx,
        "blocks": block_index(fz.blocks_, fz.columns_),
        "feature_names": fz.columns_,
    }


def all_models(blocks, include_neural: bool = True,
               include_ensembles: bool = True) -> dict:
    zoo = dict(classical_models())
    if include_neural:
        zoo.update(neural_models(blocks))
    if include_ensembles:
        from .ensemble import build_ensembles
        zoo.update(build_ensembles(zoo))
    return zoo


# ---------------------------------------------------------------------------
# Stage: leakage audit
# ---------------------------------------------------------------------------
def stage_audit() -> dict:
    from .leakage_audit import run_audit
    _log("Stage AUDIT -- reproducing the reference pipeline")
    champ = load_champ()
    res = run_audit(champ)
    res["_label_summary"] = label_summary(champ)
    _save("audit", res)
    return res


# ---------------------------------------------------------------------------
# Stage: repeated stratified cross-validation
# ---------------------------------------------------------------------------
def _cv_scores(model, X, y, cv, name: str) -> dict:
    """Fold AUCs plus the repeat-averaged out-of-fold prediction vector.

    With ``RepeatedStratifiedKFold`` each row is held out once per repeat, so
    the per-row predictions are averaged across repeats before the pooled
    metrics are computed. The fold AUCs stay separate -- their spread is the
    honest measure of how much of a model's lead is noise.
    """
    from sklearn.metrics import roc_auc_score

    fold_aucs: list[float] = []
    seen = np.zeros(len(y), dtype=int)
    total = np.zeros(len(y), dtype=float)

    for tr, te in cv.split(X, y):
        pipe = build_pipeline(model)
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[te])[:, 1]
        total[te] += p
        seen[te] += 1
        fold_aucs.append(float(roc_auc_score(y[te], p)))

    oof = total / np.maximum(seen, 1)
    m = compute_metrics(y, oof)
    return {
        "model": name,
        "cv_auc_mean": round(float(np.mean(fold_aucs)), 4),
        "cv_auc_std": round(float(np.std(fold_aucs)), 4),
        "cv_auc_folds": [round(a, 4) for a in fold_aucs],
        "oof_metrics": m,
        "_oof": oof,
    }


def stage_cv(data=None, include_neural: bool = True) -> dict:
    data = data or prepare()
    X, y = data["X"], data["y"]
    tr = data["train_idx"]
    Xtr, ytr = X[tr], y[tr]

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=RANDOM_STATE)
    zoo = all_models(data["blocks"], include_neural)

    _log(f"Stage CV -- {len(zoo)} models, {N_SPLITS}x{N_REPEATS} CV "
         f"on {len(ytr)} patients ({ytr.sum()} events)")

    results, oofs = {}, {}
    for name, model in zoo.items():
        t0 = time.time()
        r = _cv_scores(model, Xtr, ytr, cv, name)
        oofs[name] = r.pop("_oof")
        r["fit_seconds"] = round(time.time() - t0, 1)
        results[name] = r
        _log(f"  {name:20s} AUC {r['cv_auc_mean']:.4f} "
             f"+/- {r['cv_auc_std']:.4f}   ({r['fit_seconds']}s)")

    np.savez(REPORTS / "cv_oof.npz", y=ytr, **oofs)
    ranked = sorted(results.values(), key=lambda r: -r["cv_auc_mean"])
    out = {"n_train": int(len(ytr)), "n_events": int(ytr.sum()),
           "n_features": int(X.shape[1]),
           "protocol": f"RepeatedStratifiedKFold({N_SPLITS}x{N_REPEATS})",
           "ranking": [r["model"] for r in ranked],
           "models": results}
    _save("cv", out)
    return out


# ---------------------------------------------------------------------------
# Stage: position-blocked cross-validation
# ---------------------------------------------------------------------------
def stage_blocked(data=None, include_neural: bool = True,
                  include_ensembles: bool = False) -> dict:
    """Hold out contiguous stretches of the gene.

    Standard k-fold can place a variant at residue 490 in training and residue
    491 in test. Those are neighbouring residues in the same epitope, so the
    model is interpolating rather than predicting. Blocking by genomic region
    measures whether the model transfers to a stretch of F8 it has never seen,
    which is what happens when a novel mutation is discovered.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    data = data or prepare()
    X, y = data["X"], data["y"]
    groups = protein_region_blocks(data["labelled"], n_blocks=10)
    zoo = all_models(data["blocks"], include_neural, include_ensembles)

    _log(f"Stage BLOCKED -- GroupKFold over 10 genomic regions")
    gkf = GroupKFold(n_splits=5)
    results = {}
    for name, model in zoo.items():
        aucs = []
        for tr, te in gkf.split(X, y, groups):
            if len(np.unique(y[te])) < 2:
                continue
            pipe = build_pipeline(model)
            pipe.fit(X[tr], y[tr])
            aucs.append(float(roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1])))
        results[name] = {
            "blocked_auc_mean": round(float(np.mean(aucs)), 4),
            "blocked_auc_std": round(float(np.std(aucs)), 4),
            "folds": [round(a, 4) for a in aucs],
        }
        _log(f"  {name:20s} blocked AUC {results[name]['blocked_auc_mean']:.4f}")

    out = {"protocol": "GroupKFold(5) over 10 contiguous cDNA regions",
           "models": results}
    _save("blocked_cv", out)
    return out


# ---------------------------------------------------------------------------
# Stage: final model, calibration, held-out evaluation
# ---------------------------------------------------------------------------
def stage_final(data=None, best_name: str | None = None) -> dict:
    data = data or prepare()
    X, y = data["X"], data["y"]
    tr, te = data["train_idx"], data["test_idx"]

    if best_name is None:
        cv_path = REPORTS / "cv.json"
        best_name = (json.loads(cv_path.read_text())["ranking"][0]
                     if cv_path.exists() else "LightGBM")
    _log(f"Stage FINAL -- selected model: {best_name}")

    zoo = all_models(data["blocks"])
    base = build_pipeline(zoo[best_name])

    # Isotonic calibration on internal CV folds of the training set only.
    cal = CalibratedClassifierCV(base, method="isotonic",
                                 cv=StratifiedKFold(5, shuffle=True,
                                                    random_state=RANDOM_STATE))
    cal.fit(X[tr], y[tr])

    uncal = build_pipeline(zoo[best_name]).fit(X[tr], y[tr])

    p_test_cal = cal.predict_proba(X[te])[:, 1]
    p_test_raw = uncal.predict_proba(X[te])[:, 1]

    # Threshold chosen on training-set OOF predictions, never on the test set.
    oof = np.zeros(len(tr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE).split(X[tr], y[tr]):
        m = build_pipeline(zoo[best_name]).fit(X[tr][a], y[tr][a])
        oof[b] = m.predict_proba(X[tr][b])[:, 1]
    thr_youden = youden_threshold(y[tr], oof)
    thr_sens90 = threshold_at_sensitivity(y[tr], oof, 0.90)

    out = {
        "selected_model": best_name,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "test_events": int(y[te].sum()),
        "thresholds": {"youden_on_train_oof": round(thr_youden, 4),
                       "sensitivity90_on_train_oof": round(thr_sens90, 4)},
        "test_calibrated_youden": compute_metrics(y[te], p_test_cal, thr_youden),
        "test_calibrated_sens90": compute_metrics(y[te], p_test_cal, thr_sens90),
        "test_uncalibrated_youden": compute_metrics(y[te], p_test_raw, thr_youden),
        "auc_ci": bootstrap_ci(y[te], p_test_cal, "auc_roc"),
        "auc_pr_ci": bootstrap_ci(y[te], p_test_cal, "auc_pr"),
        "sensitivity_ci": bootstrap_ci(y[te], p_test_cal, "sensitivity"),
        "calibration_effect": {
            "brier_uncalibrated": compute_metrics(y[te], p_test_raw)["brier"],
            "brier_calibrated": compute_metrics(y[te], p_test_cal)["brier"],
            "ece_uncalibrated": compute_metrics(y[te], p_test_raw)["ece"],
            "ece_calibrated": compute_metrics(y[te], p_test_cal)["ece"],
        },
    }

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": cal, "featurizer": data["featurizer"],
                 "thresholds": out["thresholds"],
                 "feature_names": data["feature_names"]},
                MODELS / "final_model.joblib")
    np.savez(REPORTS / "test_predictions.npz",
             y=y[te], p_cal=p_test_cal, p_raw=p_test_raw)
    _save("final", out)
    return out


# ---------------------------------------------------------------------------
# Stage: performance inside each clinical stratum
# ---------------------------------------------------------------------------
def stage_subgroups(data=None) -> dict:
    """Does the model work for the patients it would actually be used on?

    Prophylaxis decisions are made almost entirely in severe hemophilia A. An
    overall AUC that looks respectable while the severe stratum sits at chance
    would be useless in clinic, and the overall number would never show it.
    """
    from .subgroups import subgroup_report, to_records

    data = data or prepare()
    npz = REPORTS / "test_predictions.npz"
    if not npz.exists():
        stage_final(data)
    d = np.load(npz)

    te = data["test_idx"]
    rows = data["labelled"].iloc[te].reset_index(drop=True)
    final = json.loads((REPORTS / "final.json").read_text())
    thr = final["thresholds"]["youden_on_train_oof"]

    _log("Stage SUBGROUPS -- per-stratum performance on the held-out test set")
    table = subgroup_report(rows, d["y"], d["p_cal"], thr)
    for _, r in table.iterrows():
        _log(f"  {r['subgroup']:22s} n={r['n']:4d} events={r['events']:3d} "
             f"AUC={r['auc_roc'] if r['auc_roc'] else '--'}")

    out = {"threshold": thr,
           "note": ("Strata with fewer than 10 events in either class are "
                    "listed with their counts but no AUC, because an estimate "
                    "from that few events would not be stable."),
           "subgroups": to_records(table)}
    _save("subgroups", out)
    return out


# ---------------------------------------------------------------------------
# Stage: missing-data probe and semi-supervised learning
# ---------------------------------------------------------------------------
def stage_ssl(data=None) -> dict:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    data = data or prepare()
    X, y, Xu = data["X"], data["y"], data["X_unlabelled"]
    tr, te = data["train_idx"], data["test_idx"]

    _log("Stage SSL -- missingness probe and self-training")
    probe_base = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                        n_jobs=-1, random_state=RANDOM_STATE)
    probe = ReportingBiasProbe(base_estimator=probe_base).run(
        np.nan_to_num(X, nan=0.0), np.nan_to_num(Xu, nan=0.0))

    def mk():
        return build_pipeline(lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=15,
            min_child_samples=25, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=1.0, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1))

    supervised = mk().fit(X[tr], y[tr])
    p_sup = supervised.predict_proba(X[te])[:, 1]

    ssl = SelfTrainingSSL(base_estimator=mk(), n_rounds=5,
                          max_adopt_per_round=250)
    ssl.fit_ssl(X[tr], y[tr], Xu)
    p_ssl = ssl.predict_proba(X[te])[:, 1]

    out = {
        "reporting_bias_probe": probe,
        "unlabelled_risk_profile": estimate_unlabelled_prevalence(supervised, Xu),
        "self_training_history": ssl.history_,
        "supervised_test_auc": round(float(roc_auc_score(y[te], p_sup)), 4),
        "semisupervised_test_auc": round(float(roc_auc_score(y[te], p_ssl)), 4),
        "delong_ssl_vs_supervised": delong_test(y[te], p_ssl, p_sup),
    }
    _save("ssl", out)
    return out


# ---------------------------------------------------------------------------
# Stage: external validation on the CHBMP F9 cohort
# ---------------------------------------------------------------------------
def stage_external(data=None) -> dict:
    """Score hemophilia B patients with the hemophilia A model.

    F8 and F9 are different genes coding different proteins, so nothing about
    F8-specific position or chemistry transfers directly. What can transfer is
    the mutation-class immunology: null variants abolish the protein, so the
    patient was never tolerised to the factor they are later infused with.
    A model that survives this transfer has learned that mechanism; a model
    that has memorised F8 collapses to chance.
    """
    data = data or prepare()
    _log("Stage EXTERNAL -- transferring the F8 model onto CHBMP (F9)")

    chbmp = load_chbmp()
    lab_b, _ = split_by_label(chbmp)
    yb = (lab_b["inhibitor"] == 1).astype(int).values

    fz = data["featurizer"]
    Xb = fz.transform(lab_b).values.astype(float)

    artefact = MODELS / "final_model.joblib"
    if not artefact.exists():
        stage_final(data)
    bundle = joblib.load(artefact)
    p = bundle["model"].predict_proba(Xb)[:, 1]

    out = {
        "cohort": "CHBMP (CDC Hemophilia B Mutation Project, F9), 2022 release",
        "label_summary": label_summary(chbmp),
        "n_scored": int(len(yb)),
        "n_events": int(yb.sum()),
        "prevalence": round(float(yb.mean()), 4),
        "metrics": compute_metrics(yb, p),
        "auc_ci": bootstrap_ci(yb, p, "auc_roc"),
        "note": ("Zero-shot transfer: no F9 patient was used at any point in "
                 "training, feature fitting or threshold selection."),
        "caveat": (
            "The positional features are computed with FVIII coordinates. "
            "Factor IX is a 415-residue mature protein against FVIII's 2,332, "
            "so every F9 variant lands in the low-numbered bins and the domain, "
            "epitope-distance and truncation-extent features carry no real "
            "information here -- they are near-constant across the cohort and "
            "therefore contribute almost nothing to the ranking. What is being "
            "tested is whether the *consequence-class* signal (null vs missense, "
            "truncating vs in-frame, splice disruption) plus clinical severity "
            "transfers across genes. That is the intended test, and it is the "
            "part of the model that should be gene-agnostic; but the result "
            "should not be read as evidence that the FVIII structural features "
            "generalise."),
    }
    _save("external", out)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    choices=["audit", "cv", "blocked", "final", "ssl",
                             "external", "subgroups", "all"])
    ap.add_argument("--no-neural", action="store_true",
                    help="skip the torch models (much faster)")
    args = ap.parse_args()

    include_neural = not args.no_neural
    t0 = time.time()

    if args.stage == "audit":
        stage_audit()
        return

    data = prepare()
    _log(f"Data ready: X={data['X'].shape}, "
         f"{int(data['y'].sum())} events / {len(data['y'])} labelled patients, "
         f"{len(data['unlabelled'])} unlabelled")

    if args.stage in ("cv", "all"):
        stage_cv(data, include_neural)
    if args.stage in ("blocked", "all"):
        stage_blocked(data, include_neural)
    if args.stage in ("final", "all"):
        stage_final(data)
    if args.stage in ("subgroups", "all"):
        stage_subgroups(data)
    if args.stage in ("ssl", "all"):
        stage_ssl(data)
    if args.stage in ("external", "all"):
        stage_external(data)
    if args.stage == "all":
        stage_audit()

    _log(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
