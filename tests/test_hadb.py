"""Checks on the patient-level HADB pipeline.

These guard the properties the results depend on -- the tri-state label, the
forbidden columns, variant grouping -- and pin the two defects found while
smoke-testing inference, so neither can come back silently.
"""

import numpy as np
import pandas as pd
import pytest

from src.hadb import (
    FORBIDDEN,
    build_features,
    load_hadb,
    normalise_inhibitor,
    normalise_severity,
    parse_activity,
)
from src.hadb_train import (
    BoundedIsotonic,
    build_cohort,
    grouped_folds,
    holdout_split,
)


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("Yes", 1.0), ("yes", 1.0), ("No", 0.0), ("no ", 0.0),
])
def test_recorded_outcomes_map_to_a_class(raw, expected):
    assert normalise_inhibitor(raw) == expected


@pytest.mark.parametrize("raw", ["Not reported", "Not", "", None, float("nan"),
                                 "?", "NULL"])
def test_unrecorded_outcomes_never_become_negatives(raw):
    """The single substitution that inflated every reference result."""
    assert np.isnan(normalise_inhibitor(raw))


@pytest.mark.parametrize("raw, expected", [
    ("5", 5.0),
    ("<1", 0.5),            # left-censored: half the bound, so it sorts below 1
    ("<2", 1.0),
    (">5", 5.0),
    ("23 to 40", 31.5),     # a reported range takes its midpoint
    ("9|<1?", 9.0),         # annotated entry: the first reading wins
    ("", float("nan")),
    ("not done", float("nan")),
])
def test_activity_parsing(raw, expected):
    got = parse_activity(raw)
    if expected != expected:
        assert got != got
    else:
        assert got == pytest.approx(expected)


def test_censored_activity_sorts_below_the_observed_bound():
    assert parse_activity("<1") < parse_activity("1")


@pytest.mark.parametrize("raw, expected", [
    ("Severe", "Severe"), ("severe ", "Severe"), ("svere", "Severe"),
    ("Moderate", "Moderate"), ("moderare", "Moderate"),
    ("Severe/moderate", "Severe"),   # mixed reports take the severe end
    ("Mild/Moderate", "Moderate"),
    ("non-severe", "Moderate"),
    ("Not reported", "Unknown"), ("Unclassified", "Unknown"), ("", "Unknown"),
])
def test_severity_normalisation(raw, expected):
    assert normalise_severity(raw) == expected


# ---------------------------------------------------------------------------
# The design matrix
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def hadb():
    return load_hadb()


def test_no_forbidden_column_reaches_the_model(hadb):
    """uinhibitor and the u* summaries are the outcome under another name."""
    X, _ = build_features(hadb)
    assert not (set(X.columns) & FORBIDDEN)


def test_no_feature_correlates_perfectly_with_the_outcome(hadb):
    X, _ = build_features(hadb)
    mask = hadb["y"].notna().to_numpy()
    y = hadb.loc[mask, "y"].to_numpy(dtype=float)
    Xl = X.loc[mask]
    for col in Xl.columns:
        v = Xl[col]
        if v.notna().sum() < 10 or v.nunique(dropna=True) < 2:
            continue
        r = abs(np.corrcoef(v.fillna(v.median()), y)[0, 1])
        assert r < 0.95, f"{col} is almost the label itself (|r| = {r:.3f})"


def test_matrix_is_entirely_numeric_and_finite_or_nan(hadb):
    X, _ = build_features(hadb)
    assert all(pd.api.types.is_float_dtype(X[c]) for c in X.columns)
    assert not np.isinf(X.to_numpy(dtype=float)).any()


def test_ablation_flags_actually_remove_their_blocks(hadb):
    full, blocks = build_features(hadb)
    genomic, _ = build_features(hadb, include_clinical=False,
                                include_context=False)
    assert genomic.shape[1] < full.shape[1]
    for col in blocks["clinical"] + blocks["context"]:
        assert col not in genomic.columns


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cohort():
    return build_cohort()


def test_label_prevalence_matches_published_epidemiology(cohort):
    # 15-30% is the published range for hemophilia A. Landing inside it without
    # adjustment is evidence the tri-state label is being read correctly.
    assert 0.15 <= cohort.prevalence <= 0.30


def test_holdout_split_shares_no_variant(cohort):
    train, test = holdout_split(cohort)
    assert not (set(cohort.groups[train]) & set(cohort.groups[test]))


def test_every_cv_fold_is_grouped_by_variant(cohort):
    for tr, te in grouped_folds(cohort.y, cohort.groups):
        assert not (set(cohort.groups[tr]) & set(cohort.groups[te]))
        assert cohort.y[te].sum() > 0        # each fold must contain positives


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def test_bounded_isotonic_never_claims_certainty():
    """Plain isotonic returns exactly 1.0 here; the app must not quote that."""
    rng = np.random.default_rng(0)
    scores = np.linspace(0, 1, 500)
    y = (rng.random(500) < scores).astype(float)
    y[-40:] = 1.0                            # an all-positive top bin

    cal = BoundedIsotonic().fit(scores, y)
    out = cal.predict(scores)
    assert out.max() < 1.0
    assert out.min() > 0.0
    assert (out.max(), out.min()) == (pytest.approx(cal.hi_),
                                      pytest.approx(cal.lo_))


def test_bounded_isotonic_preserves_ranking():
    """Clipping the extremes must not reorder anyone."""
    rng = np.random.default_rng(1)
    scores = rng.random(400)
    y = (rng.random(400) < scores).astype(float)
    cal = BoundedIsotonic().fit(scores, y)
    p = cal.predict(scores)
    order_in = np.argsort(scores)
    assert np.all(np.diff(p[order_in]) >= -1e-12)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
from src.hadb_predict import ARTEFACT as HADB_ARTEFACT  # noqa: E402

needs_model = pytest.mark.skipif(
    not HADB_ARTEFACT.exists(),
    reason="no HADB artefact; run `python .devtools/hadb_final.py` first")

SEVERE_NULL = dict(effect="Large Deletion", domain="A2", exon=14,
                   severity="Severe", fviii_activity=0.5, crm_type="I")
MILD_MISSENSE = dict(effect="Missense", domain="C1", exon=23, severity="Mild",
                     fviii_activity=18.0, aa_first="Arg", aa_last="Cys",
                     crm_type="II")


@needs_model
def test_scoring_does_not_depend_on_the_batch():
    """A patient's risk must not change with who else is in the request.

    Rank-averaging within the input batch made this fail: scored alone, every
    member ranked the single row 1/1 and returned the same number for
    everybody.
    """
    from src.hadb_predict import HADBRiskModel
    m = HADBRiskModel()
    batch = [r["risk"] for r in m.predict([SEVERE_NULL, MILD_MISSENSE])]
    solo = [m.predict(c)[0]["risk"] for c in (SEVERE_NULL, MILD_MISSENSE)]
    assert batch == pytest.approx(solo)
    assert batch[0] != batch[1]


@needs_model
def test_a_null_variant_outranks_a_conservative_missense():
    from src.hadb_predict import HADBRiskModel
    m = HADBRiskModel()
    high, low = m.predict([SEVERE_NULL, MILD_MISSENSE])
    assert high["risk"] > low["risk"]


@needs_model
def test_quoted_risk_stays_inside_what_the_cohort_supports():
    from src.hadb_predict import HADBRiskModel
    m = HADBRiskModel()
    for case in (SEVERE_NULL, MILD_MISSENSE):
        p = m.predict(case)[0]["risk"]
        assert 0.0 < p < 1.0


@needs_model
def test_explanation_is_non_empty_and_ranked():
    from src.hadb_predict import HADBRiskModel
    m = HADBRiskModel()
    expl = m.explain(SEVERE_NULL, top=6)
    assert not expl.empty
    contribs = expl["contribution"].abs().to_numpy()
    assert np.all(np.diff(contribs) <= 1e-9)
    # The biology has to be legible: a large deletion is a null variant, and
    # that should be among the reasons the risk is high.
    assert any("null" in f or "LargeDeletion" in f or "truncating" in f
               for f in expl["feature"])
