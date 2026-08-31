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

from src.hadb_predict import ARTEFACT as HADB_ARTEFACT, HADBRiskModel
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


# ---------------------------------------------------------------------------
# v2: the patient-level model trained on the EAHAD/HADB cohort
# ---------------------------------------------------------------------------
# The v1 form above asks only about the variant, because CHAMP holds nothing
# else. This one also asks what the treatment centre measured at diagnosis,
# which is where the extra discrimination comes from.
HADB_OPTIONS = {
    "effect": ["Missense", "Nonsense", "Frameshift", "Splice", "In-frame",
               "Silent", "Large Deletion", "Large Duplication"],
    "domain": ["A1", "A2", "A3", "B", "C1", "C2", "a1", "a2", "a3", "Signal",
               "Splice Site", "Multiple Domains", "5UTR", "3UTR"],
    "severity": ["Severe", "Moderate", "Mild", "Unknown"],
    "crm_type": ["Unknown", "I", "II", "NU"],
    "region": ["Unknown", "europe_other", "north_america", "east_asia",
               "south_asia", "middle_east", "latin_america", "africa",
               "oceania"],
    "residues": ["Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His",
                 "Ile", "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp",
                 "Tyr", "Val"],
    "threshold_rule": ["youden", "sensitivity_80", "sensitivity_90",
                       "accuracy_max"],
}

_hadb_model: HADBRiskModel | None = None


def get_hadb_model() -> HADBRiskModel:
    global _hadb_model
    if _hadb_model is None:
        if not HADB_ARTEFACT.exists():
            raise FileNotFoundError(
                f"No trained artefact at {HADB_ARTEFACT}. "
                "Run `python .devtools/hadb_final.py` first.")
        _hadb_model = HADBRiskModel()
    return _hadb_model


@app.get("/hadb")
def hadb_index():
    metrics = {}
    final = ROOT / "reports" / "hadb_final.json"
    if final.exists():
        d = json.loads(final.read_text(encoding="utf-8"))
        t = d["test_youden"]
        metrics = {
            "auc": t["auc_roc"],
            "sensitivity": t["sensitivity"],
            "specificity": t["specificity"],
            "n_test": d["n_test"],
            "prevalence": d["test_prevalence"],
        }
    return render_template("hadb.html", options=HADB_OPTIONS, metrics=metrics)


@app.post("/api/hadb/predict")
def api_hadb_predict():
    payload = request.get_json(force=True, silent=True) or {}
    rule = payload.get("threshold_rule", "youden")
    if rule not in HADB_OPTIONS["threshold_rule"]:
        return jsonify({"error": f"unknown threshold_rule {rule!r}"}), 400
    try:
        model = get_hadb_model()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    record = {k: (v if v not in ("", None) else None) for k, v in payload.items()
              if k != "threshold_rule"}
    if not any(record.values()):
        return jsonify({"error": "no patient details supplied"}), 400

    result = model.predict(record, threshold_name=rule)[0]
    try:
        expl = model.explain(record, top=8)
        result["explanation"] = expl.to_dict(orient="records")
    except Exception as exc:                    # explanation is best-effort
        result["explanation"] = []
        result["explanation_error"] = f"{type(exc).__name__}: {exc}"
    result["disclaimer"] = (
        "Research tool. Not a validated medical device. Does not replace "
        "clinical judgement or laboratory inhibitor testing. Intended use is "
        "prioritising monitoring intensity during the first fifty exposure "
        "days, not diagnosis.")
    return jsonify(result)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok",
                    "model_available": ARTEFACT.exists(),
                    "hadb_model_available": HADB_ARTEFACT.exists()})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
