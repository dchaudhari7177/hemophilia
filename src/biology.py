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


# --------------------------------------------------------------------------
# 3. Amino-acid physicochemistry
# --------------------------------------------------------------------------
THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "Sec": "U", "Xaa": "X",
}

# Kyte-Doolittle hydropathy.
HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}

# Side-chain volume (A^3), Zamyatnin.
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
    "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
    "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
    "Y": 193.6, "V": 140.0,
}

# Formal charge at physiological pH.
CHARGE = {
    "D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.5,
    "A": 0.0, "N": 0.0, "C": 0.0, "Q": 0.0, "G": 0.0, "I": 0.0, "L": 0.0,
    "M": 0.0, "F": 0.0, "P": 0.0, "S": 0.0, "T": 0.0, "W": 0.0, "Y": 0.0,
    "V": 0.0,
}

# Grantham polarity.
POLARITY = {
    "A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5, "Q": 10.5,
    "E": 12.3, "G": 9.0, "H": 10.4, "I": 5.2, "L": 4.9, "K": 11.3,
    "M": 5.7, "F": 5.2, "P": 8.0, "S": 9.2, "T": 8.6, "W": 5.4,
    "Y": 6.2, "V": 5.9,
}

AROMATIC = set("FWY")
ALIPHATIC = set("AVLIM")


def grantham_distance(a: str, b: str) -> float:
    """Grantham (1974) chemical dissimilarity between two residues.

    Computed from the published composition/polarity/volume formula rather than
    stored as a 20x20 table, so the values stay auditable.
    """
    if a not in POLARITY or b not in POLARITY:
        return float("nan")
    # Grantham composition: atomic weight ratio of non-carbon atoms in the end
    # group to carbons in the side chain.
    comp = {
        "S": 1.42, "R": 0.65, "L": 0.00, "P": 0.39, "T": 0.71, "A": 0.00,
        "V": 0.00, "G": 0.74, "I": 0.00, "F": 0.00, "Y": 0.20, "C": 2.75,
        "H": 0.58, "Q": 0.89, "N": 1.33, "K": 0.33, "D": 1.38, "E": 0.92,
        "M": 0.00, "W": 0.13,
    }
    alpha, beta, gamma = 1.833, 0.1018, 0.000399
    d = (
        alpha * (comp[a] - comp[b]) ** 2
        + beta * (POLARITY[a] - POLARITY[b]) ** 2
        + gamma * (VOLUME[a] - VOLUME[b]) ** 2
    )
    return 50.723 * (d ** 0.5)


# BLOSUM62 substitution scores, standard 20 residues.
_BLOSUM62_ORDER = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62_ROWS = [
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
]
BLOSUM62 = {
    (_BLOSUM62_ORDER[i], _BLOSUM62_ORDER[j]): _BLOSUM62_ROWS[i][j]
    for i in range(20)
    for j in range(20)
}


def blosum62(a: str, b: str) -> float:
    return float(BLOSUM62.get((a, b), float("nan")))


# --------------------------------------------------------------------------
# 4. Lookup helpers
# --------------------------------------------------------------------------
def domain_of(mature_pos: float | None) -> str:
    """Return the FVIII domain containing a mature-numbering residue."""
    if mature_pos is None or mature_pos != mature_pos:  # NaN check
        return "Unknown"
    if mature_pos < 1:
        return "Signal"
    for name, lo, hi in FVIII_DOMAINS:
        if lo <= mature_pos <= hi:
            return name
    return "Unknown"


def in_span(pos: float | None, lo: int, hi: int) -> int:
    if pos is None or pos != pos:
        return 0
    return int(lo <= pos <= hi)


def distance_to_span(pos: float | None, lo: int, hi: int) -> float:
    """0 inside the span, otherwise residues to the nearest edge."""
    if pos is None or pos != pos:
        return float("nan")
    if pos < lo:
        return lo - pos
    if pos > hi:
        return pos - hi
    return 0.0


def nearest_epitope_distance(mature_pos: float | None) -> float:
    if mature_pos is None or mature_pos != mature_pos:
        return float("nan")
    return min(distance_to_span(mature_pos, lo, hi)
               for _, lo, hi in INHIBITOR_EPITOPES)


# --------------------------------------------------------------------------
# 5. F8 transcript facts (RefSeq NM_000132.4)
# --------------------------------------------------------------------------
N_EXONS = 26
LAST_EXON = 26
# Exon 14 encodes almost the whole B domain and is by far the largest exon
# (3106 bp); exon 26 is the 3'-terminal exon, which matters for NMD escape.
B_DOMAIN_EXON = 14
# A premature termination codon escapes nonsense-mediated decay if it lies in
# the last exon or within ~50-55 nt of the final exon-exon junction.
NMD_ESCAPE_WINDOW_NT = 55
