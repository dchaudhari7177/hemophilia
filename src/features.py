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
