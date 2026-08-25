"""
Checks on the biological reference tables.

These constants drive the chemistry features, and a transcription error in them
would be invisible in the metrics -- the model would simply learn slightly wrong
chemistry and nobody would notice. So they are checked against published values.
"""

import pytest

from src import biology as bio


# Grantham (1974), Table 2. The published table is rounded to integers, so a
# tolerance of 5 is the right check on a recomputation from the formula.
GRANTHAM_PUBLISHED = {
    ("S", "W"): 177, ("I", "V"): 29, ("R", "W"): 101, ("C", "W"): 215,
    ("L", "I"): 5, ("D", "E"): 45, ("G", "W"): 184, ("F", "Y"): 22,
    ("K", "R"): 26, ("A", "G"): 60,
}


@pytest.mark.parametrize("pair,expected", list(GRANTHAM_PUBLISHED.items()))
def test_grantham_matches_the_published_table(pair, expected):
    assert bio.grantham_distance(*pair) == pytest.approx(expected, abs=5)


def test_grantham_is_symmetric_and_zero_on_the_diagonal():
    for a in "ARNDCQEGHILKMFPSTWYV":
        assert bio.grantham_distance(a, a) == pytest.approx(0.0, abs=1e-9)
        for b in "ARNDCQEGHILKMFPSTWYV":
            assert bio.grantham_distance(a, b) == pytest.approx(
                bio.grantham_distance(b, a), abs=1e-9)


@pytest.mark.parametrize("aa,expected", [("W", 11), ("A", 4), ("C", 9),
                                         ("G", 6), ("P", 7)])
def test_blosum62_diagonal(aa, expected):
    assert bio.blosum62(aa, aa) == expected


def test_blosum62_is_symmetric():
    for a in "ARNDCQEGHILKMFPSTWYV":
        for b in "ARNDCQEGHILKMFPSTWYV":
            assert bio.blosum62(a, b) == bio.blosum62(b, a)


def test_domains_tile_the_mature_protein_without_gaps_or_overlaps():
    """Every residue 1..2332 must belong to exactly one domain."""
    spans = sorted((lo, hi) for _n, lo, hi in bio.FVIII_DOMAINS)
    assert spans[0][0] == 1
    assert spans[-1][1] == bio.MATURE_LEN
    for (_lo1, hi1), (lo2, _hi2) in zip(spans, spans[1:]):
        assert lo2 == hi1 + 1, f"gap or overlap between {hi1} and {lo2}"


def test_signal_peptide_arithmetic_is_consistent():
    assert bio.PRECURSOR_LEN - bio.SIGNAL_PEPTIDE_LEN == bio.MATURE_LEN


def test_domain_lookup_agrees_with_the_boundary_table():
    for name, lo, hi in bio.FVIII_DOMAINS:
        assert bio.domain_of(lo) == name
        assert bio.domain_of(hi) == name
        assert bio.domain_of((lo + hi) // 2) == name


def test_domain_lookup_handles_out_of_range_and_missing():
    assert bio.domain_of(-5) == "Signal"
    assert bio.domain_of(None) == "Unknown"
    assert bio.domain_of(float("nan")) == "Unknown"
    assert bio.domain_of(bio.MATURE_LEN + 500) == "Unknown"


def test_heavy_light_boundary_sits_between_the_b_and_a3_domains():
    assert bio.domain_of(bio.HEAVY_LIGHT_BOUNDARY) == "B"
    assert bio.domain_of(bio.HEAVY_LIGHT_BOUNDARY + 1) == "a3"


def test_epitopes_lie_inside_the_domains_they_are_named_for():
    for name, lo, hi in bio.INHIBITOR_EPITOPES:
        expected = name.split("_")[0]
        assert bio.domain_of(lo) == expected, f"{name} starts in {bio.domain_of(lo)}"
        assert bio.domain_of(hi) == expected, f"{name} ends in {bio.domain_of(hi)}"


def test_epitope_distance_is_zero_inside_an_epitope():
    for _name, lo, hi in bio.INHIBITOR_EPITOPES:
        assert bio.nearest_epitope_distance((lo + hi) // 2) == 0.0


def test_epitope_distance_grows_with_separation():
    a2_lo = bio.INHIBITOR_EPITOPES[0][1]
    near = bio.nearest_epitope_distance(a2_lo - 10)
    far = bio.nearest_epitope_distance(a2_lo - 100)
    assert 0 < near < far


def test_amino_acid_tables_cover_all_twenty_residues():
    residues = set("ARNDCQEGHILKMFPSTWYV")
    for table in (bio.HYDROPATHY, bio.VOLUME, bio.CHARGE, bio.POLARITY):
        assert residues <= set(table)


def test_three_letter_codes_round_trip():
    assert bio.THREE_TO_ONE["Arg"] == "R"
    assert bio.THREE_TO_ONE["Ter"] == "*"
    assert len({v for k, v in bio.THREE_TO_ONE.items()
                if v in "ARNDCQEGHILKMFPSTWYV"}) == 20
