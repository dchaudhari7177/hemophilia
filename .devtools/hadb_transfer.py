"""Cross-registry transfer between HADB and CHAMP.

Train on one consortium's patients, score the other's, with no refitting and
no access to the target registry's labels. Both directions are run, because a
model that only transfers one way is usually exploiting a coverage difference
rather than biology.

Two controls sit alongside the transfer numbers:

* **within-registry** grouped CV on each side, so the drop caused by moving
  between registries can be separated from the difficulty of the task;
* **CHAMP with its unrecorded outcomes relabelled as negatives**, which is the
  protocol the reference works used, to show what that choice does to the
  apparent score on a fixed model.

Writes reports/hadb_transfer.json.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import load_champ  # noqa: E402
from src.evaluate import bootstrap_ci, compute_metrics  # noqa: E402
from src.hadb import load_hadb  # noqa: E402
from src.hadb_train import (  # noqa: E402
    REPORTS,
    grouped_folds,
    model_zoo,
    pos_weight_for,
)
from src.harmonise import (  # noqa: E402
    champ_variant_keys,
    hadb_variant_keys,
    harmonise_champ,
    harmonise_hadb,
)

warnings.filterwarnings("ignore")


def oof_auc(model, X, y, folds) -> dict:
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        m = clone(model)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return compute_metrics(y, oof)


def main() -> None:
    t0 = time.time()
    out: dict = {"feature_space": "harmonised intersection of CHAMP and HADB"}

    # -- assemble both cohorts in the shared space -------------------------
    h = load_hadb()
    Xh_all = harmonise_hadb(h)
    hm = h["y"].notna().to_numpy()
    Xh = Xh_all.loc[hm].reset_index(drop=True)
    yh = h.loc[hm, "y"].to_numpy(dtype=float)
    gh = h.loc[hm, "mut_id"].to_numpy()

    c = load_champ()
    Xc_all = harmonise_champ(c)
    cm = (c["inhibitor"] != -1).to_numpy()
    Xc = Xc_all.loc[cm].reset_index(drop=True)
    yc = c.loc[cm, "inhibitor"].to_numpy(dtype=float)

    out["cohorts"] = {
        "hadb": {"n": int(len(yh)), "prevalence": round(float(yh.mean()), 4),
                 "unit": "allele report (patient)"},
        "champ": {"n": int(len(yc)), "prevalence": round(float(yc.mean()), 4),
                  "unit": "variant"},
        "n_shared_features": int(Xh.shape[1]),
    }
    print(f"HADB {len(yh)} @ {yh.mean():.3f} | CHAMP {len(yc)} @ {yc.mean():.3f} "
          f"| {Xh.shape[1]} shared features")

    zoo_h = model_zoo(pos_weight=pos_weight_for(yh))
    zoo_c = model_zoo(pos_weight=pos_weight_for(yc))
    mh, mc = zoo_h["random_forest"], zoo_c["random_forest"]

    # -- within-registry references ----------------------------------------
    print("\nwithin-registry (grouped CV, harmonised features):")
    out["within_hadb"] = oof_auc(mh, Xh, yh, grouped_folds(yh, gh))
    print(f"  HADB  {out['within_hadb']['auc_roc']:.4f}")
    # CHAMP is one row per variant, so a stratified split is already grouped.
    from sklearn.model_selection import StratifiedKFold
    champ_folds = list(StratifiedKFold(5, shuffle=True, random_state=42)
                       .split(Xc, yc))
    out["within_champ"] = oof_auc(mc, Xc, yc, champ_folds)
    print(f"  CHAMP {out['within_champ']['auc_roc']:.4f}")

    # -- transfer ----------------------------------------------------------
    print("\ntransfer (train on one registry, score the other untouched):")
    fitted_h = clone(mh).fit(Xh, yh)
    p_hc = fitted_h.predict_proba(Xc)[:, 1]
    m_hc = compute_metrics(yc, p_hc)
    ci = bootstrap_ci(yc, p_hc, "auc_roc", n_boot=2000)
    m_hc["auc_roc_ci95"] = [ci["lo"], ci["hi"]]
    out["hadb_to_champ"] = m_hc
    print(f"  HADB -> CHAMP {m_hc['auc_roc']:.4f} "
          f"[{m_hc['auc_roc_ci95'][0]:.4f}, {m_hc['auc_roc_ci95'][1]:.4f}]")

    fitted_c = clone(mc).fit(Xc, yc)
    p_ch = fitted_c.predict_proba(Xh)[:, 1]
    m_ch = compute_metrics(yh, p_ch)
    ci = bootstrap_ci(yh, p_ch, "auc_roc", n_boot=2000)
    m_ch["auc_roc_ci95"] = [ci["lo"], ci["hi"]]
    out["champ_to_hadb"] = m_ch
    print(f"  CHAMP -> HADB {m_ch['auc_roc']:.4f} "
          f"[{m_ch['auc_roc_ci95'][0]:.4f}, {m_ch['auc_roc_ci95'][1]:.4f}]")

    out["transfer_penalty"] = {
        "hadb_to_champ": round(out["within_champ"]["auc_roc"]
                               - m_hc["auc_roc"], 4),
        "champ_to_hadb": round(out["within_hadb"]["auc_roc"]
                               - m_ch["auc_roc"], 4),
    }

    # -- the transfer numbers above are contaminated, and here is why -------
    # Transfer scoring *higher* than within-registry CV is not a success; it
    # is a symptom. CHAMP and EAHAD are compiled from overlapping published
    # literature, so a large share of CHAMP's variants are already in HADB and
    # the "unseen" cohort is substantially memorised. The honest external
    # number is the one measured on the disjoint remainder.
    train_keys = hadb_variant_keys(h[h["y"].notna()])
    champ_keys = champ_variant_keys(c.loc[cm])
    shared_all = np.array([k in train_keys for k in champ_keys])

    # The key needs a reference residue and a mature position to identify a
    # variant. Frameshifts and large structural changes usually record neither,
    # so they can never match and would be mislabelled "disjoint" -- which
    # would compare a 77%-frameshift subset against a 60%-missense one and
    # measure composition rather than novelty. Overlap is therefore assessed
    # only on the substitution-like stratum where the key actually resolves.
    resolvable = np.array([k[1] is not None and k[2] != "" for k in champ_keys])
    shared = shared_all & resolvable
    disjoint = resolvable & ~shared_all

    out["registry_overlap"] = {
        "champ_labelled_rows": int(len(champ_keys)),
        "rows_with_a_resolvable_residue_key": int(resolvable.sum()),
        "resolvable_and_present_in_hadb": int(shared.sum()),
        "overlap_fraction_within_resolvable": round(
            float(shared.sum() / max(resolvable.sum(), 1)), 4),
        "resolvable_and_novel": int(disjoint.sum()),
        "note": (
            "Matched on (mutation class, mature residue, reference residue, "
            "substituted residue). The two consortia curate the same "
            "published literature, so overlap is expected -- and it means the "
            "headline transfer figure is not an external result. Restricted "
            "to substitution-like variants because the key cannot resolve a "
            "frameshift; comparing across that boundary would measure "
            "mutation-class composition instead of novelty."),
    }
    print(f"\nregistry overlap (substitution-like stratum): "
          f"{shared.sum()}/{resolvable.sum()} "
          f"({shared.sum() / max(resolvable.sum(), 1):.1%}) of CHAMP's "
          f"resolvable variants are also in HADB")

    if disjoint.sum() > 30 and 0 < yc[disjoint].mean() < 1:
        m_sh = compute_metrics(yc[shared], p_hc[shared])
        m_dis = compute_metrics(yc[disjoint], p_hc[disjoint])
        ci_d = bootstrap_ci(yc[disjoint], p_hc[disjoint], "auc_roc", n_boot=2000)
        m_dis["auc_roc_ci95"] = [ci_d["lo"], ci_d["hi"]]
        out["hadb_to_champ_shared_only"] = m_sh
        out["hadb_to_champ_novel_only"] = m_dis
        out["contamination_effect"] = {
            "auc_on_variants_hadb_had_seen": m_sh["auc_roc"],
            "auc_on_variants_novel_to_hadb": m_dis["auc_roc"],
            "difference": round(m_sh["auc_roc"] - m_dis["auc_roc"], 4),
            "prevalence_shared": round(float(yc[shared].mean()), 4),
            "prevalence_novel": round(float(yc[disjoint].mean()), 4),
            "reading": (
                "Both strata are substitution-like, so composition is held "
                "roughly fixed and the gap is attributable to familiarity "
                "rather than to mutation class. The novel-variant figure is "
                "the external result this project claims."),
        }
        print(f"  on variants HADB had seen  {m_sh['auc_roc']:.4f} "
              f"(prevalence {yc[shared].mean():.3f})")
        print(f"  on variants novel to HADB  {m_dis['auc_roc']:.4f} "
              f"[{ci_d['lo']:.4f}, {ci_d['hi']:.4f}] "
              f"(prevalence {yc[disjoint].mean():.3f})  <- external result")

        # The novel stratum has a different prevalence and composition from
        # CHAMP as a whole, so comparing it against whole-cohort CHAMP CV would
        # be comparing two different questions. Scoring CHAMP's own
        # cross-validated predictions on exactly these rows makes it a like-for
        # -like contest: same patients, same labels, two training sources.
        champ_oof = np.full(len(yc), np.nan)
        for tr, te in champ_folds:
            mm = clone(mc)
            mm.fit(Xc.iloc[tr], yc[tr])
            champ_oof[te] = mm.predict_proba(Xc.iloc[te])[:, 1]
        m_own = compute_metrics(yc[disjoint], champ_oof[disjoint])
        out["champ_own_cv_on_the_same_novel_rows"] = m_own
        out["contamination_effect"]["champ_own_cv_on_novel_rows"] = \
            m_own["auc_roc"]
        out["contamination_effect"]["hadb_advantage_on_novel_rows"] = round(
            m_dis["auc_roc"] - m_own["auc_roc"], 4)
        print(f"  CHAMP's own CV on those same rows  {m_own['auc_roc']:.4f} "
              f"-> HADB training is worth "
              f"{m_dis['auc_roc'] - m_own['auc_roc']:+.4f} on identical rows")

    # -- what relabelling unrecorded outcomes does -------------------------
    # Same fitted model, same variants; only the label convention changes.
    y_relabelled = (c["inhibitor"] == 1).to_numpy(dtype=float)
    p_all = fitted_h.predict_proba(Xc_all)[:, 1]
    m_re = compute_metrics(y_relabelled, p_all)
    out["champ_with_unrecorded_as_negative"] = {
        **m_re,
        "n": int(len(y_relabelled)),
        "prevalence": round(float(y_relabelled.mean()), 4),
        "note": (
            "Identical model, identical variants. Only the treatment of the "
            "1,744 rows with no recorded outcome changes. Prevalence falls "
            f"from {yc.mean():.3f} to {y_relabelled.mean():.3f} and the "
            "majority-class baseline rises accordingly, so accuracy climbs "
            "while the ranking quality does not improve."),
    }
    print(f"\nCHAMP relabelled: accuracy {m_re['accuracy']:.4f} at prevalence "
          f"{y_relabelled.mean():.4f} (AUC {m_re['auc_roc']:.4f})")

    out["elapsed_seconds"] = round(time.time() - t0, 1)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "hadb_transfer.json").write_text(json.dumps(out, indent=2,
                                                            default=float))
    print(f"\nwrote reports/hadb_transfer.json in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
