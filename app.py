"""
Clinical decision-support front end.

A haemophilia treatment centre enters a patient's F8 variant and FVIII activity
stratum; the app returns a calibrated inhibitor risk, a band, a call at the
chosen operating threshold, and the biological reasons behind the number.

Run with ``python app.py`` and open http://127.0.0.1:5000.
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.predict import ARTEFACT, InhibitorRiskModel

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)

# Vocabularies offered in the form, taken from the CHAMP field definitions.
OPTIONS = {
    "variant_type": [
        "Missense", "Nonsense", "Frameshift", "Splice site change",
        "Large structural change (>50 bp)",
        "Small structural change (in-frame, <50 bp)", "Synonymous",
        "Promoter", "5'UTR", "3'UTR",
    ],
    "mechanism": [
        "Substitution", "Deletion", "Duplication", "Insertion", "Inversion",
        "Deletion/Insertion",
    ],
    "domain": ["A1", "A2", "A3", "B", "C1", "C2", "Signal", "a1", "a2", "a3"],
    "subtype": ["Heavy chain", "Light chain", "Single domain"],
    "severity": ["Severe", "Moderate", "Mild", "Not reported"],
    "in_poly_a": ["N", "Y"],
    "threshold_rule": ["youden_on_train_oof", "sensitivity90_on_train_oof"],
}

_model: InhibitorRiskModel | None = None


def get_model() -> InhibitorRiskModel:
    global _model
    if _model is None:
        if not ARTEFACT.exists():
            raise FileNotFoundError(
                f"No trained artefact at {ARTEFACT}. "
                "Run `python -m src.train --stage final` first.")
        _model = InhibitorRiskModel()
    return _model


def _record(payload: dict) -> dict:
    return {
        "HGVS cDNA": payload.get("hgvs_cdna") or None,
        "HGVS Protein": payload.get("hgvs_protein") or None,
        "Mature Protein": payload.get("mature_protein") or None,
        "Variant Type": payload.get("variant_type") or None,
        "Mechanism": payload.get("mechanism") or None,
        "Exon": payload.get("exon") or None,
        "Domain": payload.get("domain") or None,
        "Subtype": payload.get("subtype") or None,
        "In Poly A": payload.get("in_poly_a") or None,
        "Reported Clinical Severity": payload.get("severity") or None,
    }


@app.get("/")
def index():
    metrics = {}
    final = ROOT / "reports" / "final.json"
    if final.exists():
        d = json.loads(final.read_text(encoding="utf-8"))
        t = d["test_calibrated_youden"]
        metrics = {
            "model": d["selected_model"],
            "auc": t["auc_roc"],
            "sensitivity": t["sensitivity"],
            "specificity": t["specificity"],
            "n_test": d["n_test"],
        }
    return render_template("index.html", options=OPTIONS, metrics=metrics)


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(force=True, silent=True) or {}
    rule = payload.get("threshold_rule", "youden_on_train_oof")
    if rule not in OPTIONS["threshold_rule"]:
        return jsonify({"error": f"unknown threshold_rule {rule!r}"}), 400
    try:
        model = get_model()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    record = _record(payload)
    if not any(record.values()):
        return jsonify({"error": "no variant details supplied"}), 400

    result = model.predict(record, threshold_name=rule)[0]
    try:
        expl = model.explain(record, top=8)
        result["explanation"] = expl.to_dict(orient="records")
    except Exception as exc:                    # explanation is best-effort
        result["explanation"] = []
        result["explanation_error"] = f"{type(exc).__name__}: {exc}"
    result["disclaimer"] = (
        "Research tool. Not a validated medical device. Does not replace "
        "clinical judgement or laboratory inhibitor testing.")
    return jsonify(result)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_available": ARTEFACT.exists()})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
