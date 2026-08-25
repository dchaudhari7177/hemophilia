"""
Dataset loading and label handling for CHAMP (F8) and CHBMP (F9).

Label policy -- the second correction this project makes to the reference work
--------------------------------------------------------------------------
CHAMP's ``History of Inhibitor`` column takes three states: ``Yes`` (n=461),
``No`` (n=1835) and ``Not reported`` (n=1742). The reference pipeline maps
``Not reported`` to 0, i.e. it asserts that every patient whose inhibitor
status was never recorded did not develop an inhibitor.

That is wrong in two separate ways:

1. It is factually unsupported. "Not reported" means the source publication did
   not state the outcome, not that the outcome was negative.
2. It manufactures an easy majority class. Prevalence falls from 20.1% (the
   epidemiologically correct figure, and the one the reference paper's own
   introduction quotes) to 11.4%, and accuracy on the padded set is inflated
   simply because the padding is all one class.

Here the three states are kept distinct: labelled rows train and evaluate the
model, and unlabelled rows are handed to the positive-unlabelled learner in
``pu_learning.py`` instead of being silently relabelled.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"

LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNKNOWN = 1, 0, -1


def _norm_label(v) -> int:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return LABEL_UNKNOWN
    t = re.sub(r"\s+", " ", str(v)).strip().lower()
    if t in {"yes", "y"}:
        return LABEL_POSITIVE
    if t in {"no", "n"}:
        return LABEL_NEGATIVE
    return LABEL_UNKNOWN          # "not reported", "np", blank


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c).replace("\n", " ")).strip()
                  for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    return df


def load_champ(path: Path | str | None = None) -> pd.DataFrame:
    """CHAMP F8 variant list with a tri-state ``inhibitor`` column added."""
    path = Path(path) if path else DATA_RAW / "champ.csv"
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(
        path, sheet_name="CHAMP Variant List", header=0)
    df = _clean_columns(df)
    df = df.dropna(how="all")
    df = df.drop_duplicates()
    df["inhibitor"] = [_norm_label(v) for v in df["History of Inhibitor"]]
    df["gene"] = "F8"
    return df.reset_index(drop=True)


def load_chbmp(path: Path | str | None = None) -> pd.DataFrame:
    """CHBMP F9 variant list, column names harmonised onto the CHAMP schema.

    Hemophilia B shares the therapeutic problem (inhibitors against infused
    factor) and the CDC curation protocol, but a different gene and protein.
    It is therefore a genuine *external* cohort: a model that transfers to it
    has learned mutation-class immunology rather than F8-specific quirks.
    """
    path = Path(path) if path else DATA_RAW / "CHBMP-Variant-List-2022.xlsx"
    df = pd.read_excel(path, sheet_name="CHBMP Variant List", header=0)
    df = _clean_columns(df)
    df = df.rename(columns={
        "HGVS cDNA Name": "HGVS cDNA",
        "HGVS Protein Name": "HGVS Protein",
        "Mature Protein Change": "Mature Protein",
        "hg19 Nucleotide No.": "hg19 Coordinates",
        "Reported Severity": "Reported Clinical Severity",
    })
    df = df.dropna(how="all").drop_duplicates()
    df["inhibitor"] = [_norm_label(v) for v in df["History of Inhibitor"]]
    df["gene"] = "F9"
    return df.reset_index(drop=True)


def split_by_label(df: pd.DataFrame):
    """Return (labelled, unlabelled) views."""
    lab = df[df["inhibitor"] != LABEL_UNKNOWN].reset_index(drop=True)
    unl = df[df["inhibitor"] == LABEL_UNKNOWN].reset_index(drop=True)
    return lab, unl


def label_summary(df: pd.DataFrame) -> dict:
    v = df["inhibitor"].value_counts()
    pos, neg, unk = (int(v.get(LABEL_POSITIVE, 0)), int(v.get(LABEL_NEGATIVE, 0)),
                     int(v.get(LABEL_UNKNOWN, 0)))
    lab = pos + neg
    return {
        "n_total": int(len(df)),
        "n_positive": pos,
        "n_negative": neg,
        "n_unlabelled": unk,
        "n_labelled": lab,
        "prevalence_labelled": round(pos / lab, 4) if lab else None,
        "prevalence_if_unlabelled_called_negative":
            round(pos / len(df), 4) if len(df) else None,
    }


# ---------------------------------------------------------------------------
# Splitting strategies
# ---------------------------------------------------------------------------
def protein_region_blocks(df: pd.DataFrame, n_blocks: int = 10) -> np.ndarray:
    """Assign each variant to a contiguous block of the FVIII coding sequence.

    Used for *position-blocked* cross-validation. Ordinary stratified k-fold
    lets the model see variants at, say, residue 490 in training and residue
    491 in test -- neighbouring residues in the same epitope, which is closer
    to interpolation than to prediction. Blocking by region forces the model to
    generalise to a stretch of the protein it has never seen, which is the real
    clinical situation when a novel variant is found.
    """
    from .hgvs_parser import parse_cdna

    pos = np.array([parse_cdna(v).cdna_pos for v in df["HGVS cDNA"]], dtype=float)
    filled = np.where(np.isnan(pos), np.nanmedian(pos), pos)
    order = np.argsort(filled, kind="stable")
    blocks = np.empty(len(df), dtype=int)
    edges = np.array_split(order, n_blocks)
    for b, idx in enumerate(edges):
        blocks[idx] = b
    return blocks
