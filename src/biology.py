"""
FVIII / F8 structural and biochemical reference tables.

Everything in this module is *prior biological knowledge* -- it is derived from
UniProt P00451, RefSeq NM_000132.4 and the published FVIII inhibitor-epitope
literature. None of it is derived from the CHAMP labels, so it can be applied to
train, validation and test rows alike without leaking the outcome.

This is the layer the reference works do not have: instead of label-encoding the
raw HGVS string (which is unique per patient and therefore an identifier), we
turn each variant into a vector of *mechanistic* descriptors that transfer to a
patient the model has never seen.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# 1. FVIII domain architecture
# --------------------------------------------------------------------------
# FVIII is translated as a 2351-residue precursor. The first 19 residues are the
# signal peptide; the circulating mature protein is 2332 residues.
#   mature_position = precursor_position - SIGNAL_PEPTIDE_LEN
# CHAMP stores both numberings ("HGVS Protein" = precursor, "Mature Protein").

SIGNAL_PEPTIDE_LEN = 19
PRECURSOR_LEN = 2351
MATURE_LEN = 2332

# Domain boundaries in MATURE numbering (inclusive), UniProt P00451.
FVIII_DOMAINS = [
    ("A1", 1, 336),
    ("a1", 337, 372),   # acidic region a1
    ("A2", 373, 719),
    ("a2", 720, 740),   # acidic region a2
    ("B", 741, 1648),   # B domain -- dispensable for coagulant activity
    ("a3", 1649, 1689),  # acidic region a3
    ("A3", 1690, 2019),
    ("C1", 2020, 2172),
    ("C2", 2173, 2332),
]

# The heavy chain (A1-a1-A2-a2-B) and light chain (a3-A3-C1-C2) are separated by
# proteolytic processing at residue 1648/1649.
HEAVY_LIGHT_BOUNDARY = 1648

# The B domain is spliced out on activation and is not required for function.
B_DOMAIN_START, B_DOMAIN_END = 741, 1648


# --------------------------------------------------------------------------
# 2. Immunodominant inhibitor epitopes
# --------------------------------------------------------------------------
# Anti-FVIII alloantibodies cluster on a small number of surface patches. These
# spans (mature numbering) are the classical immunodominant regions reported in
# the inhibitor literature. A variant that removes or alters one of these
# regions plausibly changes the epitope repertoire presented to the immune
# system, so proximity to them is a mechanistically motivated feature.
INHIBITOR_EPITOPES = [
    ("A2_epitope", 484, 508),     # classical A2 inhibitor epitope (R484-I508)
    ("A3_epitope", 1804, 1819),   # A3 epitope overlapping the VWF site
    ("C1_epitope", 2091, 2115),   # C1 epitope
    ("C2_epitope_1", 2181, 2243),  # C2 immunodominant epitope
    ("C2_epitope_2", 2248, 2312),  # second C2 epitope
]

# Functionally critical interaction surfaces (mature numbering).
FUNCTIONAL_SITES = [
    ("vwf_binding", 1670, 1684),   # a3 acidic sulfated region, VWF binding
    ("fixa_binding", 1811, 1818),  # FIXa interaction
    ("fx_binding", 337, 372),      # a1 acidic region, FX interaction
    ("phospholipid_c2", 2199, 2332),  # C2 membrane-binding surface
    ("thrombin_r372", 372, 372),   # thrombin cleavage site
    ("thrombin_r740", 740, 740),   # thrombin cleavage site
    ("thrombin_r1689", 1689, 1689),  # thrombin cleavage site
]
