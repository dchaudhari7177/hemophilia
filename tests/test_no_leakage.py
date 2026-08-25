"""
Guard rails against the failure this project exists to correct.

If any of these tests goes red, the pipeline has regained the property that
made the reference results unreproducible.
"""

import numpy as np
import pandas as pd
import pytest

from src.datasets import LABEL_UNKNOWN, load_champ, split_by_label
from src.features import IDENTIFIER_COLUMNS, VariantFeaturizer


@pytest.fixture(scope="module")
def champ():
    return load_champ()


@pytest.fixture(scope="module")
def featurised(champ):
    fz = VariantFeaturizer().fit(champ)
    return fz, fz.transform(champ)


def test_unknown_outcomes_are_not_relabelled_negative(champ):
    """The reference pipeline's central mistake must stay fixed."""
    s = champ["inhibitor"].value_counts()
    assert s.get(LABEL_UNKNOWN, 0) > 1500, "unlabelled rows disappeared"
    labelled, _ = split_by_label(champ)
    prevalence = (labelled["inhibitor"] == 1).mean()
    # 20.1% matches the published epidemiology; calling unknowns negative
    # would drop it to about 11%.
    assert 0.18 < prevalence < 0.23


def test_no_feature_is_a_row_identifier(featurised):
    """No engineered column may be near-unique across patients.

    A column with as many distinct values as it has rows is an index, and a
    model given an index will memorise it.
    """
    _fz, X = featurised
    n = len(X)
    for col in X.columns:
        uniq = X[col].nunique(dropna=True)
        assert uniq < 0.5 * n, (
            f"{col!r} takes {uniq} distinct values over {n} rows -- "
            "that is identifier-like")


def test_identifier_columns_never_reach_the_matrix(featurised):
    _fz, X = featurised
    for banned in IDENTIFIER_COLUMNS:
        assert banned not in X.columns


def test_features_are_finite_or_nan_never_inf(featurised):
    _fz, X = featurised
    assert not np.isinf(X.to_numpy(dtype=float)).any()


def test_featuriser_never_reads_the_outcome(champ):
    """Fitting on data with scrambled labels must give identical features."""
    scrambled = champ.copy()
    rng = np.random.default_rng(0)
    scrambled["inhibitor"] = rng.permutation(scrambled["inhibitor"].values)
    scrambled["History of Inhibitor"] = rng.permutation(
        scrambled["History of Inhibitor"].values)

    a = VariantFeaturizer().fit(champ).transform(champ)
    b = VariantFeaturizer().fit(scrambled).transform(scrambled)
    pd.testing.assert_frame_equal(a, b)


def test_transform_is_row_independent(featurised, champ):
    """Scoring one patient must give the same answer as scoring the batch.

    If a feature were computed from batch statistics of the rows being scored,
    a single-patient prediction would differ from the same patient inside a
    cohort -- which would make the deployed tool disagree with its own
    validation.
    """
    fz, X = featurised
    for i in (0, 17, 500, 2000):
        single = fz.transform(champ.iloc[[i]])
        np.testing.assert_allclose(
            single.to_numpy(dtype=float)[0],
            X.to_numpy(dtype=float)[i],
            rtol=1e-9, equal_nan=True,
            err_msg=f"row {i} scores differently alone than in a batch")


def test_holdout_variants_are_never_seen_in_training(champ):
    """Sanity check on the raw data that motivated the whole audit."""
    labelled, _ = split_by_label(champ)
    assert labelled["HGVS cDNA"].duplicated().sum() == 0, (
        "labelled rows share HGVS strings; the identifier argument would "
        "need revisiting")
