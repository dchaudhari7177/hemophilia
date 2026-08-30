"""
Inference: score a single patient's F8 variant.

The point of the whole exercise is a tool a haemophilia treatment centre could
actually use, so the input here is what a centre holds after genetic testing --
the HGVS description of the variant plus the measured FVIII activity stratum --
and the output is a calibrated risk, a band, a call at a stated threshold, and
the reasons behind it.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARTEFACT = ROOT / "models" / "final_model.joblib"

# Bands are anchored to the epidemiology rather than to round numbers: the
# background inhibitor rate in severe hemophilia A is 20-40%, so "high" starts
# where a patient's estimated risk exceeds the upper end of that range.
RISK_BANDS = [
    (0.00, 0.10, "Low"),
    (0.10, 0.25, "Moderate"),
    (0.25, 0.40, "High"),
    (0.40, 1.01, "Very high"),
]

INPUT_FIELDS = [
    "HGVS cDNA", "HGVS Protein", "Mature Protein", "Variant Type", "Mechanism",
    "Exon", "Domain", "Subtype", "In Poly A", "Reported Clinical Severity",
]


class InhibitorRiskModel:
    """Loaded artefact plus the featuriser it was trained with."""

    def __init__(self, artefact: Path | str = ARTEFACT):
        bundle = joblib.load(artefact)
        self.model = bundle["model"]
        self.featurizer = bundle["featurizer"]
        self.thresholds = bundle["thresholds"]
        self.feature_names = bundle["feature_names"]
        self.shap_background = bundle.get("shap_background")

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _band(p: float) -> str:
        for lo, hi, name in RISK_BANDS:
            if lo <= p < hi:
                return name
        return "Very high"

    def _frame(self, records: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        for col in INPUT_FIELDS:
            if col not in df.columns:
                df[col] = np.nan
        return df

    # -- public API ------------------------------------------------------
    def predict(self, records: list[dict] | dict,
                threshold_name: str = "youden_on_train_oof") -> list[dict]:
        if isinstance(records, dict):
            records = [records]
        df = self._frame(records)
        X = self.featurizer.transform(df).values.astype(float)
        p = self.model.predict_proba(X)[:, 1]
        thr = self.thresholds[threshold_name]
        return [
            {
                "probability": round(float(pi), 4),
                "risk_band": self._band(float(pi)),
                "prediction": "inhibitor-risk positive" if pi >= thr
                              else "inhibitor-risk negative",
                "threshold": round(float(thr), 4),
                "threshold_rule": threshold_name,
            }
            for pi in p
        ]

    def explain(self, record: dict, top: int = 8) -> pd.DataFrame:
        """Per-patient SHAP attribution over the engineered features."""
        from .explain import explain_patient, shap_values

        df = self._frame([record])
        X = self.featurizer.transform(df).values.astype(float)
        vals, Xt = shap_values(self._base_estimator(), X, self.feature_names,
                               max_rows=1, background=self.shap_background)
        return explain_patient(vals, Xt, self.feature_names, 0, top=top)

    def explain_lime(self, record: dict, top: int = 8) -> pd.DataFrame:
        """LIME attribution for one patient.

        Reported alongside SHAP because the two answer different questions and
        their disagreement is itself informative -- see ``src/explain.py``.
        """
        from .explain import lime_explain_patient, lime_explainer

        if self.shap_background is None:
            raise ValueError(
                "LIME needs the background sample stored in the artefact; "
                "retrain with `python -m src.train --stage final`.")
        df = self._frame([record])
        X = self.featurizer.transform(df).values.astype(float)
        est = self._base_estimator()
        expl = lime_explainer(est, self.shap_background, self.feature_names)
        return lime_explain_patient(est, expl, X[0], self.feature_names, top=top)

    def _base_estimator(self):
        """The uncalibrated estimator underneath the calibration wrapper.

        SHAP explains the model that produces the score, so it is pointed at
        the base learner rather than the isotonic wrapper; calibration is a
        monotone transform and does not change the ranking of contributions.
        """
        cal = getattr(self.model, "calibrated_classifiers_", None)
        if cal:
            return getattr(cal[0], "estimator", self.model)
        return self.model


def score_dataframe(df: pd.DataFrame, artefact: Path | str = ARTEFACT) -> pd.DataFrame:
    """Score a whole CHAMP-shaped table and append the risk columns."""
    m = InhibitorRiskModel(artefact)
    X = m.featurizer.transform(df).values.astype(float)
    p = m.model.predict_proba(X)[:, 1]
    out = df.copy()
    out["inhibitor_risk"] = np.round(p, 4)
    out["risk_band"] = [InhibitorRiskModel._band(float(x)) for x in p]
    return out


if __name__ == "__main__":
    demo = {
        "HGVS cDNA": "c.6496C>T",
        "HGVS Protein": "p.(Arg2166*)",
        "Mature Protein": "Arg2147*",
        "Variant Type": "Nonsense",
        "Mechanism": "Substitution",
        "Exon": "23",
        "Domain": "C1",
        "Subtype": "Light chain",
        "In Poly A": "N",
        "Reported Clinical Severity": "Severe",
    }
    model = InhibitorRiskModel()
    print(model.predict(demo)[0])
    print(model.explain(demo).to_string(index=False))
