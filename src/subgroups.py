"""
Subgroup performance.

An overall AUC hides the question a clinician actually has: *does this work for
the patients I would use it on?* Inhibitor prophylaxis decisions are made almost
entirely in **severe** hemophilia A — that is where the 25-40% incidence sits and
where a risk score would change management. A model with a respectable overall
AUC that is at chance inside the severe stratum would be useless in clinic and
the overall number would never reveal it.

Neither reference work reports subgroup performance. This module reports it for
every stratum with enough events to support an estimate, and says so explicitly
when a stratum does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import bootstrap_ci, compute_metrics
from .features import (normalise_chain, normalise_severity,
                       normalise_variant_type, NULL_VARIANT_TYPES)

MIN_EVENTS = 10          # below this an AUC estimate is not worth reporting


def _stratify(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    vt = np.array([normalise_variant_type(v) for v in rows["Variant Type"]])
    sev = np.array([normalise_severity(v)
                    for v in rows["Reported Clinical Severity"]])
    chain = np.array([normalise_chain(v) for v in rows["Subtype"]])
    null = np.array([v in NULL_VARIANT_TYPES for v in vt])

    groups = {
        "All patients": np.ones(len(rows), dtype=bool),
        "Severe phenotype": sev == "severe",
        "Moderate phenotype": sev == "moderate",
        "Mild phenotype": sev == "mild",
        "Null variants": null,
        "Non-null variants": ~null,
        "Missense only": vt == "missense",
        "Truncating only": np.isin(vt, ["nonsense", "frameshift"]),
        "Large structural": vt == "large_structural",
        "Light chain": chain == "light",
        "Heavy chain": chain == "heavy",
    }
    return groups


def subgroup_report(rows: pd.DataFrame, y: np.ndarray, p: np.ndarray,
                    threshold: float | None = None) -> pd.DataFrame:
    """Metrics per clinical stratum, with the small ones flagged rather than hidden."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    out = []
    for name, mask in _stratify(rows).items():
        n, events = int(mask.sum()), int(y[mask].sum())
        row = {"subgroup": name, "n": n, "events": events,
               "prevalence": round(float(y[mask].mean()), 4) if n else None}
        if events < MIN_EVENTS or n - events < MIN_EVENTS:
            row.update({"auc_roc": None, "auc_ci": None, "sensitivity": None,
                        "specificity": None,
                        "note": f"too few events ({events}) for a stable estimate"})
        else:
            m = compute_metrics(y[mask], p[mask], threshold)
            ci = bootstrap_ci(y[mask], p[mask], "auc_roc", n_boot=800)
            row.update({
                "auc_roc": m["auc_roc"],
                "auc_ci": f"{ci['lo']:.3f}-{ci['hi']:.3f}" if ci["lo"] else None,
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "note": "",
            })
        out.append(row)
    return pd.DataFrame(out)


def to_records(df: pd.DataFrame) -> list[dict]:
    return df.where(pd.notna(df), None).to_dict(orient="records")
