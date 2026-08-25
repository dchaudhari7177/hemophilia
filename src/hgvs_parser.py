"""
Parse HGVS cDNA and protein notation into structured, non-identifying fields.

The reference pipelines label-encode the raw HGVS string. Because almost every
CHAMP row carries a distinct HGVS string (4038 distinct values across 4050
rows, and *zero* duplicates among the 2296 labelled rows), that encoding is an
row identifier -- the model can memorise it and cannot use it on a new patient.

Here we throw the identity away and keep only what the notation *means*:
position, span, consequence, nucleotide/residue change and splice offsets.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict

from .biology import THREE_TO_ONE, SIGNAL_PEPTIDE_LEN


# c.101A>T   c.-112G>A   c.1538-10_1546del   c.388+2delGTG   c.-171-?_143+?del
_CDNA_POS = re.compile(r"(?P<sign>-|\*)?(?P<base>\d+)(?P<off>[+-](?:\d+|\?))?")
_SUBST = re.compile(r"(?P<ref>[ACGT]+)>(?P<alt>[ACGT]+)\s*$")

_PROT_SIMPLE = re.compile(
    r"p\.?\(?(?P<ref>[A-Z][a-z]{2})(?P<pos>-?\d+)(?P<alt>[A-Z][a-z]{2}|\*|=|del|dup)?"
)
_PROT_FS = re.compile(
    r"(?P<ref>[A-Z][a-z]{2})(?P<pos>-?\d+)(?P<alt>[A-Z][a-z]{2})?fs\*?(?P<ter>\d+|\?)?"
)
_MATURE_SIMPLE = re.compile(
    r"^(?P<ref>[A-Z][a-z]{2})(?P<pos>-?\d+)(?P<alt>[A-Z][a-z]{2}|\*|=|del|dup)?"
)


@dataclass
class CdnaParse:
    """Structured view of an ``HGVS cDNA`` string."""
    cdna_pos: float = math.nan       # first coding position mentioned
    cdna_pos_end: float = math.nan   # last coding position mentioned
    span_nt: float = math.nan        # nucleotides between the two anchors
    intron_offset: float = math.nan  # +/- distance into the intron
    is_intronic: int = 0
    is_utr5: int = 0
    is_utr3: int = 0
    is_promoter: int = 0
    unknown_breakpoint: int = 0      # a '?' breakpoint -> large rearrangement
    ref_nt: str = ""
    alt_nt: str = ""
    ref_len: float = math.nan
    alt_len: float = math.nan
    op_del: int = 0
    op_dup: int = 0
    op_ins: int = 0
    op_inv: int = 0
    op_delins: int = 0
    op_sub: int = 0


@dataclass
class ProteinParse:
    """Structured view of an ``HGVS Protein`` / ``Mature Protein`` string."""
    prot_pos: float = math.nan       # precursor numbering
    mature_pos: float = math.nan     # mature numbering (precursor - 19)
    ref_aa: str = ""
    alt_aa: str = ""
    is_frameshift: int = 0
    is_nonsense: int = 0             # direct stop-gain
    is_synonymous: int = 0
    is_inframe_del: int = 0
    is_inframe_dup: int = 0
    is_delins: int = 0
    fs_ter_offset: float = math.nan  # residues from the frameshift to the PTC
    ptc_mature_pos: float = math.nan  # where translation actually stops


def _to_float(x: str | None) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def parse_cdna(raw) -> CdnaParse:
    """Parse an HGVS cDNA description. Never raises; unknown -> NaN fields."""
    out = CdnaParse()
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return out
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "?"}:
        return out

    body = s[2:] if s.lower().startswith("c.") else s

    if "?" in body:
        out.unknown_breakpoint = 1

    low = body.lower()
    out.op_delins = int("delins" in low)
    out.op_del = int("del" in low and not out.op_delins)
    out.op_dup = int("dup" in low)
    out.op_ins = int("ins" in low and not out.op_delins)
    out.op_inv = int("inv" in low)

    m = _SUBST.search(body)
    if m:
        out.op_sub = 1
        out.ref_nt, out.alt_nt = m.group("ref"), m.group("alt")
        out.ref_len, out.alt_len = float(len(out.ref_nt)), float(len(out.alt_nt))

    # Positional anchors: take the first and last coordinate in the string.
    anchors = list(_CDNA_POS.finditer(body.split("del")[0].split("dup")[0]
                                      .split("ins")[0].split("inv")[0] or body))
    if not anchors:
        anchors = list(_CDNA_POS.finditer(body))
    if anchors:
        first, last = anchors[0], anchors[-1]

        def coord(mm) -> float:
            base = _to_float(mm.group("base"))
            if mm.group("sign") == "-":
                base = -base
            elif mm.group("sign") == "*":
                base = base + 7053  # 3'UTR offset past the stop codon
            return base

        out.cdna_pos = coord(first)
        out.cdna_pos_end = coord(last)
        if not math.isnan(out.cdna_pos) and not math.isnan(out.cdna_pos_end):
            out.span_nt = abs(out.cdna_pos_end - out.cdna_pos) + 1

        # Splice offsets: c.1538-10 is 10 nt into the preceding intron.
        offs = [mm.group("off") for mm in anchors if mm.group("off")]
        numeric = [_to_float(o.lstrip("+-")) * (-1 if o.startswith("-") else 1)
                   for o in offs if o.lstrip("+-").isdigit()]
        if numeric:
            out.is_intronic = 1
            # keep the offset closest to the exon boundary
            out.intron_offset = min(numeric, key=abs)
        elif offs:  # only '?' offsets
            out.is_intronic = 1
            out.unknown_breakpoint = 1

    if not math.isnan(out.cdna_pos):
        out.is_utr5 = int(out.cdna_pos < 0)
        out.is_promoter = int(out.cdna_pos < -100)
        out.is_utr3 = int(out.cdna_pos > 7053)

    return out


def _to_mature(precursor: float) -> float:
    """Precursor -> mature numbering, skipping the non-existent residue 0."""
    if math.isnan(precursor):
        return math.nan
    m = precursor - SIGNAL_PEPTIDE_LEN
    return m if m > 0 else m - 1


def _to_precursor(mature: float) -> float:
    if math.isnan(mature):
        return math.nan
    return mature + SIGNAL_PEPTIDE_LEN if mature > 0 else mature + SIGNAL_PEPTIDE_LEN + 1


def _aa1(three: str | None) -> str:
    if not three:
        return ""
    if three in {"*", "=", "del", "dup"}:
        return three
    return THREE_TO_ONE.get(three, "")


def parse_protein(hgvs_protein, mature_protein=None) -> ProteinParse:
    """Parse the protein-level consequence.

    ``hgvs_protein`` uses precursor numbering, ``mature_protein`` uses mature
    numbering. Either may be missing; whichever is present drives the parse and
    the other numbering is derived via the 19-residue signal peptide offset.
    """
    out = ProteinParse()

    def _clean(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        v = str(v).strip()
        return "" if v.lower() in {"nan", "none"} else v

    p, mp = _clean(hgvs_protein), _clean(mature_protein)
    src = p or mp
    if not src:
        return out

    if src in {"p.(=)", "=", "p.="}:
        out.is_synonymous = 1
        return out

    low = src.lower()
    out.is_delins = int("delins" in low)
    out.is_inframe_del = int("del" in low and "delins" not in low and "fs" not in low)
    out.is_inframe_dup = int("dup" in low and "fs" not in low)

    fs = _PROT_FS.search(src)
    if fs:
        out.is_frameshift = 1
        out.ref_aa = _aa1(fs.group("ref"))
        out.alt_aa = _aa1(fs.group("alt"))
        pos = _to_float(fs.group("pos"))
        out.fs_ter_offset = _to_float(fs.group("ter"))
    else:
        m = _PROT_SIMPLE.search(src) or _MATURE_SIMPLE.search(src)
        if not m:
            return out
        out.ref_aa = _aa1(m.group("ref"))
        alt = m.group("alt")
        out.alt_aa = _aa1(alt) if alt else ""
        pos = _to_float(m.group("pos"))
        if out.alt_aa == "*":
            out.is_nonsense = 1
        elif out.alt_aa == "=":
            out.is_synonymous = 1
            out.alt_aa = out.ref_aa

    # Reconcile the two numbering systems. HGVS numbering has no residue 0, so
    # positions inside the signal peptide count backwards from -1: precursor
    # residue 19 is mature -1, residue 18 is mature -2, and so on.
    if p and src is p:
        out.prot_pos = pos
        out.mature_pos = _to_mature(pos)
    else:
        out.mature_pos = pos
        out.prot_pos = _to_precursor(pos)

    # Where does translation actually terminate?
    if out.is_frameshift and not math.isnan(out.fs_ter_offset):
        out.ptc_mature_pos = out.mature_pos + out.fs_ter_offset - 1
    elif out.is_nonsense:
        out.ptc_mature_pos = out.mature_pos

    return out


def parse_row(cdna, hgvs_protein, mature_protein) -> dict:
    """Convenience wrapper returning one flat dict per variant."""
    c = asdict(parse_cdna(cdna))
    p = asdict(parse_protein(hgvs_protein, mature_protein))
    c.pop("ref_nt", None)
    c.pop("alt_nt", None)
    return {**c, **p}
