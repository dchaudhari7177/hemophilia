"""
Tests for the statistics the write-up depends on.

These matter more than usual here: the whole point of the project is that the
reference numbers were not trustworthy, so the machinery used to make the
replacement claims has to be checked rather than assumed.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluate import (bootstrap_ci, compute_metrics, decision_curve,
                          delong_test, expected_calibration_error, net_benefit,
                          threshold_at_sensitivity, youden_threshold)


@pytest.fixture(scope="module")
def scores():
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.2, 900)
    good = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.35, 900), 0, 1)
    weak = np.clip(0.2 + 0.15 * y + rng.normal(0, 0.35, 900), 0, 1)
    return y, good, weak


def test_delong_auc_matches_sklearn(scores):
    y, good, _ = scores
    r = delong_test(y, good, good)
    assert r["auc_a"] == pytest.approx(roc_auc_score(y, good), abs=1e-3)


def test_delong_against_itself_is_never_significant(scores):
    y, good, _ = scores
    r = delong_test(y, good, good)
    assert r["delta"] == pytest.approx(0.0, abs=1e-9)
    assert r["p_value"] == pytest.approx(1.0)


def test_delong_detects_a_real_difference(scores):
    y, good, weak = scores
    r = delong_test(y, good, weak)
    assert r["delta"] > 0
    assert r["p_value"] < 0.01


def test_bootstrap_interval_brackets_the_point_estimate(scores):
    y, good, _ = scores
    ci = bootstrap_ci(y, good, "auc_roc", n_boot=400)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["hi"] - ci["lo"] < 0.25       # sane width at n=900


def test_perfect_and_random_scores_bound_the_auc():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert compute_metrics(y, y.astype(float))["auc_roc"] == 1.0
    assert compute_metrics(y, 1.0 - y)["auc_roc"] == 0.0


def test_youden_threshold_separates_a_clean_split():
    y = np.array([0] * 50 + [1] * 50)
    p = np.array([0.1] * 50 + [0.9] * 50)
    t = youden_threshold(y, p)
    pred = (p >= t).astype(int)
    assert (pred == y).all()


def test_sensitivity_threshold_reaches_its_target(scores):
    y, good, _ = scores
    t = threshold_at_sensitivity(y, good, 0.90)
    achieved = ((good >= t) & (y == 1)).sum() / y.sum()
    assert achieved >= 0.90 - 1e-9


def test_calibration_error_is_zero_for_a_perfectly_calibrated_score():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 40000)
    y = rng.binomial(1, p)
    assert expected_calibration_error(y, p, n_bins=10) < 0.02


def test_calibration_error_catches_a_systematically_shifted_score():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.55, 20000)
    y = rng.binomial(1, p)
    shifted = np.clip(p + 0.3, 0, 1)
    assert expected_calibration_error(y, shifted) > expected_calibration_error(y, p)


def test_net_benefit_of_treating_no_one_is_zero():
    y = np.array([0, 1, 0, 1, 0])
    assert net_benefit(y, np.zeros(5), 0.2) == 0.0


def test_net_benefit_of_treating_everyone_matches_the_closed_form():
    y = np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 0])
    t, prev = 0.2, y.mean()
    expected = prev - (1 - prev) * (t / (1 - t))
    assert net_benefit(y, np.ones(len(y)), t) == pytest.approx(expected)


def test_decision_curve_returns_aligned_arrays(scores):
    y, good, _ = scores
    thr, model, treat_all = decision_curve(y, good)
    assert len(thr) == len(model) == len(treat_all)
    assert np.isfinite(model).all()


def test_metrics_report_the_prevalence_as_the_pr_baseline(scores):
    y, good, _ = scores
    m = compute_metrics(y, good)
    assert m["auc_pr_baseline"] == pytest.approx(y.mean(), abs=1e-4)
    assert m["auc_pr"] > m["auc_pr_baseline"]


def test_confusion_counts_sum_to_n(scores):
    y, good, _ = scores
    m = compute_metrics(y, good)
    assert m["tp"] + m["tn"] + m["fp"] + m["fn"] == m["n"]
