"""Inference for the patient-level inhibitor risk model.

The inputs are what a haemophilia treatment centre holds after genetic testing
and a baseline factor assay: the class of the F8 variant, where it sits in the
protein, and the patient's own FVIII activity and clinical severity. Nothing
here requires information that only becomes available after treatment starts.

Output is a calibrated probability, a risk band, a call at a stated operating
threshold, and the features that drove the number.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .hadb import build_features, normalise_crm, normalise_effect, normalise_severity

ROOT = Path(__file__).resolve().parents[1]
ARTEFACT = ROOT / "models" / "hadb_model.joblib"

# Bands are anchored to epidemiology rather than to round numbers: the
# background inhibitor rate in severe hemophilia A is 20-40%, so "high" begins
# where an individual's estimated risk exceeds the top of that range.
RISK_BANDS = [
    (0.00, 0.10, "Low"),
    (0.10, 0.25, "Moderate"),
    (0.25, 0.40, "High"),
    (0.40, 1.01, "Very high"),
]

#: The raw columns ``build_features`` reads. A prediction row is assembled in
#: exactly this shape so training and inference share one code path.
RAW_COLUMNS = [
    "effect", "mut_type", "location", "ntchange", "n_bp", "d_id", "aa_numb",
    "nuc_numb", "e_i_numb", "locnumb", "aa_first", "aa_last",
    "fviii_activity", "fviii_antigen", "act_ant_ratio", "severity",
    "crm_type", "region",
]

INPUT_FIELDS = [
    "effect", "domain", "exon", "aa_position", "aa_first", "aa_last",
    "mut_type", "location", "ntchange", "n_bp", "severity", "fviii_activity",
    "fviii_antigen", "crm_type", "region",
]


def frame_from_inputs(records: list[dict]) -> pd.DataFrame:
    """Assemble the raw frame ``build_features`` expects from user input.

    Missing fields stay missing. The model was trained with median imputation
    and explicit ``*_measured`` indicators, so an unknown antigen level is
    handled the same way at inference as it was in training rather than being
    silently invented.
    """
    rows = []
    for r in records:
        aa_pos = r.get("aa_position")
        rows.append({
            "effect": normalise_effect(r.get("effect")),
            "mut_type": r.get("mut_type") or "Point",
            "location": r.get("location") or "Exon",
            "ntchange": r.get("ntchange"),
            "n_bp": r.get("n_bp", 1),
            "d_id": r.get("domain"),
            # build_features converts precursor -> mature by subtracting the
            # 19-residue signal peptide, so a mature position supplied by the
            # caller is shifted back into precursor numbering first.
            "aa_numb": (float(aa_pos) + 19) if aa_pos not in (None, "") else np.nan,
            "nuc_numb": r.get("nuc_numb", np.nan),
            "e_i_numb": r.get("exon"),
            "locnumb": r.get("locnumb", np.nan),
            "aa_first": r.get("aa_first"),
            "aa_last": r.get("aa_last"),
            "fviii_activity": _num(r.get("fviii_activity")),
            "fviii_antigen": _num(r.get("fviii_antigen")),
            "act_ant_ratio": _num(r.get("act_ant_ratio")),
            "severity": normalise_severity(r.get("severity")),
            "crm_type": normalise_crm(r.get("crm_type")),
            "region": r.get("region") or "Unknown",
        })
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def _num(v):
    if v in (None, "", "nan"):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


class HADBRiskModel:
    """The fitted rank ensemble plus its isotonic calibrator."""

    def __init__(self, artefact: Path | str = ARTEFACT):
        bundle = joblib.load(artefact)
        self.ensemble = bundle["ensemble"]
        self.calibrator = bundle["calibrator"]
        self.thresholds = bundle["thresholds"]
        self.feature_names = bundle["feature_names"]
        self.feature_blocks = bundle["feature_blocks"]
        self.metrics = bundle.get("metrics", {})
        self.provenance = bundle.get("provenance", "")

    @staticmethod
    def _band(p: float) -> str:
        for lo, hi, name in RISK_BANDS:
            if lo <= p < hi:
                return name
        return "Very high"

    def design_matrix(self, records: list[dict]) -> pd.DataFrame:
        raw = frame_from_inputs(records)
        X, _ = build_features(raw)
        # Align to the training column order; anything the caller could not
        # supply is left NaN for the pipeline's imputer to handle.
        return X.reindex(columns=self.feature_names)

    def predict(self, records, threshold_name: str = "youden") -> list[dict]:
        if isinstance(records, dict):
            records = [records]
        if threshold_name not in self.thresholds:
            raise ValueError(
                f"unknown threshold {threshold_name!r}; "
                f"available: {sorted(self.thresholds)}")
        thr = float(self.thresholds[threshold_name])

        X = self.design_matrix(records)
        raw_scores = self.ensemble.decision_scores(X)
        probs = self.calibrator.predict(raw_scores)

        out = []
        for rec, p in zip(records, probs):
            p = float(np.clip(p, 0.0, 1.0))
            out.append({
                "label": rec.get("label"),
                "risk": round(p, 4),
                "risk_percent": f"{p:.1%}",
                "band": self._band(p),
                "call": "flag for enhanced monitoring" if p >= thr else "routine",
                "threshold": round(thr, 4),
                "threshold_rule": threshold_name,
            })
        return out

    def explain(self, record: dict, top: int = 8) -> pd.DataFrame:
        """Which features moved this patient away from the cohort baseline.

        A leave-one-out contribution: each feature in turn is reset to its
        training median and the change in score is reported. It is not a Shapley
        value, but it is exact for the model as evaluated and needs no
        background sampling, which keeps single-patient scoring fast.
        """
        X = self.design_matrix([record])
        base = float(self.ensemble.decision_scores(X)[0])
        median = pd.Series(
            joblib.load(ARTEFACT).get("train_reference", {}))
        rows = []
        for col in self.feature_names:
            if col not in median.index or pd.isna(X.iloc[0][col]):
                continue
            Xp = X.copy()
            Xp.loc[Xp.index[0], col] = median[col]
            delta = base - float(self.ensemble.decision_scores(Xp)[0])
            if abs(delta) > 1e-9:
                rows.append({"feature": col, "value": X.iloc[0][col],
                             "contribution": round(delta, 5),
                             "direction": "increases risk" if delta > 0
                                          else "lowers risk"})
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return (df.reindex(df["contribution"].abs().sort_values(ascending=False)
                           .index).head(top).reset_index(drop=True))
