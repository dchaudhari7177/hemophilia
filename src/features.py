"""
Leakage-free, biology-informed featurisation of F8/F9 variants.

Design rule enforced throughout this module
-------------------------------------------
A feature is admissible only if it could be computed for a patient the model
has never seen, from information a haemophilia treatment centre actually holds
at the time of diagnosis. Concretely that rules out:

  * the raw HGVS cDNA / protein / mature-protein strings (unique per row -> the
    model would memorise row identity),
  * the hg19 genomic coordinate (likewise unique),
  * the literature reference number and year of report (a property of the
    *paper*, not the patient; it also encodes reporting-era confounding),
  * the free-text comment field.

Everything the parser extracts *from* those strings -- position, span,
consequence, residue chemistry -- is admissible, because those quantities
recur across patients and carry mechanism rather than identity.

Feature blocks
--------------
The columns are emitted in named blocks so that the attention network in
``models.py`` can weight whole biological axes, and so that block-level
ablations are easy to run.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from . import biology as bio
from .hgvs_parser import parse_cdna, parse_protein

# Columns that must never reach a model. Kept explicit so the leakage audit can
# reuse the same list.
IDENTIFIER_COLUMNS = [
    "HGVS cDNA", "HGVS cDNA Name",
    "hg19 Coordinates", "hg19 Nucleotide No.", "Yoshitake Nucleotide No.",
    "HGVS Protein", "HGVS Protein Name",
    "Mature Protein", "Mature Protein Change",
    "Codon", "Comments",
    "Reference Number", "Year Reported", "Year",
    "Newly Added in the Current Version",
]

FEATURE_BLOCKS: dict[str, list[str]] = {}


# ---------------------------------------------------------------------------
# Normalisation of the free-ish categorical columns
# ---------------------------------------------------------------------------
_VARIANT_TYPE_MAP = {
    "missense": "missense",
    "nonsense": "nonsense",
    "frameshift": "frameshift",
    "splice site change": "splice",
    "large structural change (>50 bp)": "large_structural",
    "large structure change (>50bp)": "large_structural",
    "small structural change (in-frame, <50 bp)": "small_structural",
    "small structural change (in-frame, <50bp)": "small_structural",
    "synonymous": "synonymous",
    "promoter": "regulatory",
    "5'utr": "regulatory",
    "3'utr": "regulatory",
}

_MECHANISM_MAP = {
    "substitution": "substitution",
    "deletion": "deletion",
    "duplication": "duplication",
    "insertion": "insertion",
    "inversion": "inversion",
    "deletion/insertion": "delins",
    "deletion/duplication": "complex",
    "duplication/insertion": "complex",
    "deletion/inversion": "complex",
    "duplication/inversion": "complex",
    "deletion/duplication/inversion": "complex",
}

# Null (cross-reactive-material-negative) variant classes. This is the single
# strongest established genomic predictor of inhibitor development: patients
# who make no FVIII protein at all have never been immunologically tolerised
# to it, so infused FVIII is seen as foreign.
NULL_VARIANT_TYPES = {"large_structural", "frameshift", "nonsense", "splice"}


def _norm(s) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "unknown"
    t = re.sub(r"\s+", " ", str(s)).strip().lower()
    return t if t and t not in {"nan", "none", "n/a"} else "unknown"


def normalise_variant_type(s) -> str:
    return _VARIANT_TYPE_MAP.get(_norm(s), "other")


def normalise_mechanism(s) -> str:
    return _MECHANISM_MAP.get(_norm(s), "other")


def normalise_severity(s) -> str:
    """Collapse the 13 raw spellings into 4 clinically meaningful strata."""
    t = _norm(s)
    if "not reported" in t or t == "unknown":
        return "unknown"
    has_sev, has_mod, has_mild = "severe" in t, "moderate" in t, "mild" in t
    if has_sev and not (has_mod or has_mild):
        return "severe"
    if has_mod and not has_sev and not has_mild:
        return "moderate"
    if has_mild and not (has_sev or has_mod):
        return "mild"
    return "mixed"


_SEVERITY_ORDINAL = {"mild": 0.0, "mixed": 1.0, "moderate": 1.0,
                     "severe": 2.0, "unknown": np.nan}


def normalise_chain(s) -> str:
    t = _norm(s)
    if "heavy" in t:
        return "heavy"
    if "light" in t:
        return "light"
    if "single" in t:
        return "single_domain"
    return "unknown"


def parse_exon(s) -> float:
    """First exon number mentioned; promoter/UTR rows -> NaN."""
    t = _norm(s)
    if t in {"unknown", "promoter"}:
        return np.nan
    m = re.search(r"\d+", t)
    return float(m.group()) if m else np.nan


def parse_exon_last(s) -> float:
    t = _norm(s)
    nums = re.findall(r"\d+", t)
    return float(nums[-1]) if nums else np.nan


# ---------------------------------------------------------------------------
# Exon map, derived from coordinates only (never from labels)
# ---------------------------------------------------------------------------
def build_exon_map(df: pd.DataFrame, cdna_col: str, exon_col: str) -> pd.DataFrame:
    """Empirical cDNA span of each exon.

    Rather than hard-coding RefSeq coordinates (and risking a transcription
    error), we recover each exon's cDNA interval from the coordinates already
    present in the variant table. This uses only X, never y, so it is safe to
    compute on the full table.
    """
    rows = []
    for cdna, exon in zip(df[cdna_col], df[exon_col]):
        c = parse_cdna(cdna)
        e = parse_exon(exon)
        if math.isnan(e) or math.isnan(c.cdna_pos) or c.is_intronic or c.unknown_breakpoint:
            continue
        if parse_exon_last(exon) != e:      # multi-exon event, no clean anchor
            continue
        rows.append((e, c.cdna_pos))
    if not rows:
        return pd.DataFrame(columns=["exon", "start", "end", "length"])
    t = pd.DataFrame(rows, columns=["exon", "pos"])
    # Robust bounds: a handful of rows carry unconventional notation whose
    # first coordinate lands far outside the exon, so trim to the inner 90%
    # before taking the span.
    g = (t.groupby("exon")["pos"]
           .agg(start=lambda v: v.quantile(0.05),
                end=lambda v: v.quantile(0.95),
                n_obs="count")
           .reset_index())
    g["length"] = (g["end"] - g["start"] + 1).clip(lower=1)
    return g
