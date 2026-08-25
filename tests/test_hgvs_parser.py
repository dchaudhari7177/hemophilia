"""Parser tests, anchored on rows taken verbatim from CHAMP."""

import math

import pytest

from src.hgvs_parser import parse_cdna, parse_protein


@pytest.mark.parametrize("cdna,pos,span,sub", [
    ("c.101A>T", 101, 1, True),
    ("c.106_107del", 106, 2, False),
    ("c.5815G>T", 5815, 1, True),
    ("c.-112G>A", -112, 1, True),
    ("c.96_107del", 96, 12, False),
])
def test_cdna_coordinates(cdna, pos, span, sub):
    c = parse_cdna(cdna)
    assert c.cdna_pos == pos
    assert c.span_nt == span
    assert bool(c.op_sub) is sub


def test_intronic_offset_is_signed():
    """c.1538-10 sits 10 nt inside the preceding intron (acceptor side)."""
    c = parse_cdna("c.1538-10_1546del")
    assert c.is_intronic == 1
    assert c.intron_offset == -10

    donor = parse_cdna("c.388+2delGTG")
    assert donor.is_intronic == 1
    assert donor.intron_offset == 2


def test_unknown_breakpoint_flagged():
    c = parse_cdna("c.-171-?_143+?del")
    assert c.unknown_breakpoint == 1
    assert c.op_del == 1


def test_substitution_alleles():
    c = parse_cdna("c.101A>T")
    assert (c.ref_nt, c.alt_nt) == ("A", "T")


def test_delins_is_not_counted_as_plain_del():
    c = parse_cdna("c.145_156delinsGAAGAATGC")
    assert c.op_delins == 1
    assert c.op_del == 0


@pytest.mark.parametrize("prot,mature,expected", [
    ("p.(Asp34Glu)", "Asp15Glu", 15),
    ("p.(Ala1939Ser)", "Ala1920Ser", 1920),
    ("p.(Cys8*)", "Cys-12*", -12),      # HGVS numbering has no residue 0
    ("p.(Arg15*)", "Arg-5*", -5),
])
def test_mature_numbering_matches_champ(prot, mature, expected):
    assert parse_protein(prot, mature).mature_pos == expected


def test_frameshift_termination_codon():
    p = parse_protein("p.(Met36Alafs*3)", "Met17Alafs*3")
    assert p.is_frameshift == 1
    assert p.fs_ter_offset == 3
    assert p.ptc_mature_pos == 17 + 3 - 1


def test_nonsense_flagged_and_ptc_located():
    p = parse_protein("p.(Cys8*)", "Cys-12*")
    assert p.is_nonsense == 1
    assert p.ptc_mature_pos == -12


def test_synonymous():
    p = parse_protein("p.(=)", "=")
    assert p.is_synonymous == 1
    assert p.is_frameshift == 0


def test_unparsable_input_returns_nan_not_an_exception():
    for junk in (None, float("nan"), "", "???", "not a variant"):
        c = parse_cdna(junk)
        assert math.isnan(c.cdna_pos) or isinstance(c.cdna_pos, float)
        p = parse_protein(junk, junk)
        assert math.isnan(p.mature_pos)
