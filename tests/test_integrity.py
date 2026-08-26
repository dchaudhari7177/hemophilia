"""
The integrity audit, run as tests.

``src/integrity.py`` writes a report for the review to inspect. The same checks
belong in the test suite too, so that a change which reintroduces resampling,
un-weights a model, or lets the featuriser see the outcome fails immediately
rather than being discovered in a JSON file nobody opened.
"""

import pytest

from src import integrity


def test_no_resampling_anywhere():
    """Imbalance is weighted, never resampled.

    Random Over-Sampling before a split is what put half of the reference
    pipeline's test set into its own training data. Nothing in this codebase
    may duplicate or synthesise a patient.
    """
    r = integrity.check_no_resampling()
    assert r["passed"], f"resampling reintroduced: {r['violations']}"


def test_every_classical_model_is_imbalance_aware():
    """A model trained unweighted cannot be fairly compared with one that is."""
    r = integrity.check_imbalance_handling()
    unweighted = [m["model"] for m in r["models"] if not m["imbalance_aware"]]
    assert not unweighted, f"no imbalance handling on: {unweighted}"


def test_preprocessing_is_fitted_inside_folds():
    r = integrity.check_preprocessing_inside_folds()
    assert r["passed"], r


def test_featuriser_is_label_blind():
    r = integrity.check_featuriser_is_label_blind()
    assert r["passed"], "features changed when the outcome was scrambled"


def test_no_identifier_features():
    r = integrity.check_no_identifier_features()
    assert r["passed"], (
        f"a feature is identifier-like: {r['highest_cardinality_fraction']}")


def test_test_set_is_never_fitted_on():
    r = integrity.check_test_set_used_once()
    assert r["passed"], r


def test_unrecorded_outcomes_stay_unrecorded():
    r = integrity.check_label_policy()
    assert r["passed"], (
        f"label policy broken: prevalence {r['prevalence_labelled']}, "
        f"{r['n_unlabelled']} unlabelled rows")


def test_full_audit_reports_a_summary():
    out = integrity.run_all()
    s = out["_summary"]
    assert s["n_checks"] >= 7
    # accuracy-with-baseline is allowed to skip when final.json is absent
    hard_failures = [c for c in s["failed_checks"]
                     if c != "accuracy_reported_with_baseline"]
    assert not hard_failures, f"integrity failures: {hard_failures}"


@pytest.mark.parametrize("name", ["SMOTE", "RandomOverSampler", "imblearn"])
def test_resampling_names_are_actually_screened(name):
    """Guard the guard: the scanner must still be looking for these."""
    assert name in integrity.RESAMPLING_NAMES
