"""
End-to-end checks on the trained artefact.

These are sanity checks against known biology rather than accuracy tests. A
model can post a respectable AUC and still rank a large deletion below a
conservative missense substitution -- and if it does, no clinician should use
it, whatever the AUC says.

Skipped when no artefact has been trained yet.
"""

import numpy as np
import pytest

from src.predict import ARTEFACT, InhibitorRiskModel

pytestmark = pytest.mark.skipif(
    not ARTEFACT.exists(),
    reason="no trained artefact; run `python -m src.train --stage final` first")


# A null variant in a severely affected patient: the classic high-risk profile.
HIGH_RISK = {
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

# A conservative missense in a mild patient: FVIII protein is still made, so
# the immune system has been tolerised to it.
LOW_RISK = {
    "HGVS cDNA": "c.103T>C",
    "HGVS Protein": "p.(Tyr35His)",
    "Mature Protein": "Tyr16His",
    "Variant Type": "Missense",
    "Mechanism": "Substitution",
    "Exon": "1",
    "Domain": "A1",
    "Subtype": "Heavy chain",
    "In Poly A": "N",
    "Reported Clinical Severity": "Mild",
}

# A multi-exon deletion: the highest-risk class in the CHAMP counts.
LARGE_DELETION = {
    "HGVS cDNA": "c.-171-?_143+?del",
    "Variant Type": "Large structural change (>50 bp)",
    "Mechanism": "Deletion",
    "Exon": "1-5",
    "Domain": "A1",
    "Subtype": "Single domain",
    "In Poly A": "N",
    "Reported Clinical Severity": "Severe",
}


@pytest.fixture(scope="module")
def model():
    return InhibitorRiskModel()


def test_probabilities_are_valid(model):
    for record in (HIGH_RISK, LOW_RISK, LARGE_DELETION):
        p = model.predict(record)[0]["probability"]
        assert 0.0 <= p <= 1.0


def test_null_variant_outranks_conservative_missense(model):
    """The single most established finding in the inhibitor literature."""
    hi = model.predict(HIGH_RISK)[0]["probability"]
    lo = model.predict(LOW_RISK)[0]["probability"]
    assert hi > lo, (
        f"nonsense/severe scored {hi:.3f} but missense/mild scored {lo:.3f}; "
        "the model has the central biology backwards")


def test_large_deletion_is_high_risk(model):
    """Large structural changes carry the highest observed inhibitor rate."""
    p = model.predict(LARGE_DELETION)[0]["probability"]
    lo = model.predict(LOW_RISK)[0]["probability"]
    assert p > lo


def test_risk_bands_are_monotone_in_probability(model):
    order = ["Low", "Moderate", "High", "Very high"]
    prev = -1
    for p in (0.02, 0.15, 0.30, 0.60):
        band = InhibitorRiskModel._band(p)
        assert order.index(band) >= prev
        prev = order.index(band)


def test_batch_scoring_matches_single_scoring(model):
    """A patient must score the same alone as inside a batch."""
    records = [HIGH_RISK, LOW_RISK, LARGE_DELETION]
    batch = [r["probability"] for r in model.predict(records)]
    single = [model.predict(r)[0]["probability"] for r in records]
    np.testing.assert_allclose(batch, single, rtol=1e-9)


def test_high_sensitivity_threshold_is_not_above_the_balanced_one(model):
    """Trading specificity for sensitivity can only lower the cut-off."""
    a = model.predict(HIGH_RISK, "youden_on_train_oof")[0]["threshold"]
    b = model.predict(HIGH_RISK, "sensitivity90_on_train_oof")[0]["threshold"]
    assert b <= a + 1e-9


def test_missing_fields_do_not_crash(model):
    sparse = {"Variant Type": "Nonsense", "Reported Clinical Severity": "Severe"}
    out = model.predict(sparse)[0]
    assert 0.0 <= out["probability"] <= 1.0


def test_explanation_returns_ranked_drivers(model):
    expl = model.explain(HIGH_RISK, top=5)
    assert len(expl) <= 5
    assert set(expl.columns) >= {"feature", "shap", "direction"}
    # ranked by absolute contribution
    mags = expl["shap"].abs().to_numpy()
    assert (np.diff(mags) <= 1e-9).all()
