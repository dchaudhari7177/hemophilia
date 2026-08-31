"""A feature space both registries can be expressed in.

CHAMP (CDC, United States) and HADB/EAHAD (European consortium) curate F8
variants independently, with different field names, different vocabularies and
different patients. Anything that transfers between them is mutation-class
immunology rather than a quirk of one curation team, which makes cross-registry
transfer the strongest external check available to this project.

The shared space is deliberately the *intersection* of what both hold:

    genotype class, FVIII domain, exon, mature-protein position,
    substituted-residue chemistry, and reported clinical severity

HADB's per-patient factor level, antigen, CRM type and reporting centre have no
CHAMP counterpart, so they are excluded here. The harmonised model is therefore
weaker than the full patient-level model by construction -- that is the price
of being comparable, and both numbers are reported.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from .biology import (
    AROMATIC,
    CHARGE,
    HYDROPATHY,
    LAST_EXON,
    MATURE_LEN,
    POLARITY,
    VOLUME,
    B_DOMAIN_EXON,
    HEAVY_LIGHT_BOUNDARY,
    blosum62,
    domain_of,
    grantham_distance,
    nearest_epitope_distance,
)
from .hadb import (
    NULL_EFFECTS,
    SEVERITY_ORDINAL,
    _clean,
    canon_domain,
    normalise_effect,
    normalise_severity,
)
from .hgvs_parser import parse_protein

#: Columns of the harmonised matrix, in a fixed order so a model trained on one
#: registry can be applied to the other without silent column reordering.
EFFECT_LEVELS = ["Missense", "Nonsense", "Frameshift", "Splice", "InFrame",
                 "Silent", "LargeDeletion", "LargeDuplication"]
DOMAIN_LEVELS = ["A1", "A2", "A3", "B", "C1", "C2", "a1", "a2", "a3",
                 "Signal", "UTR", "SpliceSite", "Multiple", "Unknown"]
SEVERITY_LEVELS = ["Mild", "Moderate", "Severe", "Unknown"]


def _champ_effect(variant_type) -> str:
    """CHAMP's ``Variant Type`` vocabulary onto the shared effect classes."""
    s = _clean(variant_type).lower()
    if not s:
        return "Unknown"
    if "large structural" in s:
        return "LargeDeletion"
    if "small structural" in s or "in-frame" in s:
        return "InFrame"
    if "splice" in s:
        return "Splice"
    if "synonymous" in s:
        return "Silent"
    if "missense" in s:
        return "Missense"
    if "nonsense" in s:
        return "Nonsense"
    if "frameshift" in s:
        return "Frameshift"
    if "promoter" in s or "utr" in s:
        return "Other"
    return "Other"


def _parse_exon(v) -> float:
    s = _clean(v)
    if not s:
        return math.nan
    m = re.match(r"^(\d+)", s)
    return float(m.group(1)) if m else math.nan


def _assemble(effect: pd.Series,
              domain: pd.Series,
              exon: pd.Series,
              mature: pd.Series,
              ref: pd.Series,
              alt: pd.Series,
              severity: pd.Series,
              index) -> pd.DataFrame:
    """Build the fixed harmonised matrix from already-normalised inputs."""
    f = pd.DataFrame(index=index)

    for lvl in EFFECT_LEVELS:
        f[f"effect_{lvl}"] = (effect == lvl).astype(float)
    f["is_null_variant"] = effect.isin(NULL_EFFECTS).astype(float)
    f["is_truncating"] = effect.isin(
        {"Nonsense", "Frameshift", "LargeDeletion", "LargeDuplication"}
    ).astype(float)

    for lvl in DOMAIN_LEVELS:
        f[f"domain_{lvl}"] = (domain == lvl).astype(float)
    f["in_C_domain"] = domain.isin({"C1", "C2"}).astype(float)
    f["in_A_domain"] = domain.isin({"A1", "A2", "A3"}).astype(float)

    f["mature_pos"] = mature
    f["relative_pos"] = mature / MATURE_LEN
    f["is_heavy_chain"] = (mature <= HEAVY_LIGHT_BOUNDARY).astype(float)
    f["epitope_distance"] = mature.map(
        lambda p: nearest_epitope_distance(p) if pd.notna(p) else np.nan)
    f["structural_domain_known"] = mature.map(
        lambda p: 0.0 if pd.isna(p) or domain_of(p) == "Unknown" else 1.0)
    f["exon_number"] = exon
    f["is_exon_14"] = (exon == B_DOMAIN_EXON).astype(float)
    f["is_last_exon"] = (exon == LAST_EXON).astype(float)
    f["nmd_escape"] = ((exon == LAST_EXON)
                       & effect.isin({"Nonsense", "Frameshift"})).astype(float)
    trunc = np.where(effect.isin({"Nonsense", "Frameshift"}),
                     1.0 - (mature / MATURE_LEN).clip(0, 1), 0.0)
    f["truncation_fraction"] = trunc

    scorable = [(a, b) if (a in HYDROPATHY and b in HYDROPATHY) else ("", "")
                for a, b in zip(ref, alt)]
    f["is_substitution"] = ((ref != "") & (alt != "") & (ref != alt)
                            & (alt != "*")).astype(float)
    f["grantham"] = [grantham_distance(a, b) if a else np.nan
                     for a, b in scorable]
    f["blosum62"] = [blosum62(a, b) if a else np.nan for a, b in scorable]
    for tag, table in [("hydropathy", HYDROPATHY), ("volume", VOLUME),
                       ("charge", CHARGE), ("polarity", POLARITY)]:
        r = ref.map(lambda a: float(table[a]) if a in table else np.nan)
        f[f"ref_{tag}"] = r
        f[f"delta_{tag}"] = alt.map(
            lambda a: float(table[a]) if a in table else np.nan) - r
    f["ref_aromatic"] = ref.map(lambda a: float(a in AROMATIC))
    f["alt_aromatic"] = alt.map(lambda a: float(a in AROMATIC))
    f["ref_is_cysteine"] = (ref == "C").astype(float)
    f["alt_is_cysteine"] = (alt == "C").astype(float)
    f["creates_stop"] = (alt == "*").astype(float)

    f["severity_ordinal"] = severity.map(SEVERITY_ORDINAL)
    for lvl in SEVERITY_LEVELS:
        f[f"severity_{lvl}"] = (severity == lvl).astype(float)

    return f.replace([np.inf, -np.inf], np.nan).astype(float)


def harmonise_hadb(df: pd.DataFrame) -> pd.DataFrame:
    """Project HADB allele records into the shared space."""
    from .hadb import SIGNAL_PEPTIDE_LEN, _aa1, parse_exon_number

    mature = pd.to_numeric(df["aa_numb"], errors="coerce") - SIGNAL_PEPTIDE_LEN
    mature = mature.where(mature != 0, np.nan)
    return _assemble(
        effect=df["effect"],
        domain=df["d_id"].map(canon_domain),
        exon=df["e_i_numb"].map(parse_exon_number),
        mature=mature,
        ref=df["aa_first"].map(_aa1),
        alt=df["aa_last"].map(_aa1),
        severity=df["severity"],
        index=df.index,
    )


def harmonise_champ(df: pd.DataFrame) -> pd.DataFrame:
    """Project CHAMP variant rows into the shared space."""
    parsed = [parse_protein(row.get("HGVS Protein"), row.get("Mature Protein"))
              for _, row in df.iterrows()]
    mature = pd.Series([p.mature_pos for p in parsed], index=df.index,
                       dtype=float)
    ref = pd.Series([p.ref_aa or "" for p in parsed], index=df.index)
    alt = pd.Series([p.alt_aa or "" for p in parsed], index=df.index)

    effect = df["Variant Type"].map(_champ_effect)
    # CHAMP records the mechanism separately, so a large structural change can
    # be resolved into deletion vs duplication rather than collapsed.
    mech = df["Mechanism"].map(lambda v: _clean(v).lower())
    effect = effect.where(
        ~((effect == "LargeDeletion") & mech.str.contains("duplication|insertion",
                                                          na=False)),
        "LargeDuplication")

    return _assemble(
        effect=effect,
        domain=df["Domain"].map(canon_domain),
        exon=df["Exon"].map(_parse_exon),
        mature=mature,
        ref=ref,
        alt=alt,
        severity=df["Reported Clinical Severity"].map(normalise_severity),
        index=df.index,
    )
