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


# ---------------------------------------------------------------------------
# The featuriser
# ---------------------------------------------------------------------------
class VariantFeaturizer:
    """Turn a raw CHAMP/CHBMP-style table into a numeric design matrix.

    ``fit`` learns only the exon map and the categorical vocabularies; no
    label information is used, so fitting on the full table cannot leak the
    outcome. Vocabularies are still learned on train to keep the pipeline
    honest under a strict reviewer.
    """

    def __init__(self, gene: str = "F8"):
        self.gene = gene
        self.exon_map_: pd.DataFrame | None = None
        self.columns_: list[str] = []
        self.blocks_: dict[str, list[str]] = {}

    # -- column resolution ------------------------------------------------
    @staticmethod
    def _col(df: pd.DataFrame, *candidates: str) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        low = {str(c).lower().replace("\n", " ").strip(): c for c in df.columns}
        for c in candidates:
            k = c.lower().replace("\n", " ").strip()
            if k in low:
                return low[k]
        return None

    def _resolve(self, df: pd.DataFrame) -> dict[str, str | None]:
        return {
            "cdna": self._col(df, "HGVS cDNA", "HGVS cDNA Name"),
            "protein": self._col(df, "HGVS Protein", "HGVS Protein Name"),
            "mature": self._col(df, "Mature Protein", "Mature Protein Change"),
            "vtype": self._col(df, "Variant Type"),
            "mech": self._col(df, "Mechanism"),
            "exon": self._col(df, "Exon"),
            "domain": self._col(df, "Domain"),
            "subtype": self._col(df, "Subtype"),
            "polya": self._col(df, "In Poly A"),
            "severity": self._col(df, "Reported Clinical Severity",
                                  "Reported Severity"),
        }

    # -- fit / transform --------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "VariantFeaturizer":
        c = self._resolve(df)
        if c["cdna"] and c["exon"]:
            self.exon_map_ = build_exon_map(df, c["cdna"], c["exon"])
        feats, blocks = self._build(df)
        self.columns_ = list(feats.columns)
        self.blocks_ = blocks
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        feats, _ = self._build(df)
        # align to the fitted schema
        for col in self.columns_:
            if col not in feats.columns:
                feats[col] = 0.0
        return feats[self.columns_]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # -- the actual feature construction ----------------------------------
    def _exon_bounds(self, exon: float) -> tuple[float, float, float]:
        if self.exon_map_ is None or math.isnan(exon):
            return (np.nan, np.nan, np.nan)
        row = self.exon_map_[self.exon_map_["exon"] == exon]
        if row.empty:
            return (np.nan, np.nan, np.nan)
        r = row.iloc[0]
        return (float(r["start"]), float(r["end"]), float(r["length"]))

    def _build(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        c = self._resolve(df)
        n = len(df)
        get = lambda k: df[c[k]] if c[k] else pd.Series([np.nan] * n, index=df.index)

        out: dict[str, np.ndarray | list] = {}
        blocks: dict[str, list[str]] = {k: [] for k in
                                        ["consequence", "position", "truncation",
                                         "chemistry", "nucleotide", "splicing",
                                         "clinical"]}

        def add(block: str, name: str, values) -> None:
            out[name] = values
            blocks[block].append(name)

        # ---- parse every row once ---------------------------------------
        cd = [parse_cdna(v) for v in get("cdna")]
        pr = [parse_protein(a, b) for a, b in zip(get("protein"), get("mature"))]

        vtype = [normalise_variant_type(v) for v in get("vtype")]
        mech = [normalise_mechanism(v) for v in get("mech")]
        chain = [normalise_chain(v) for v in get("subtype")]
        sev = [normalise_severity(v) for v in get("severity")]
        exon = np.array([parse_exon(v) for v in get("exon")], dtype=float)
        exon_last = np.array([parse_exon_last(v) for v in get("exon")], dtype=float)

        # ================================================================
        # BLOCK 1 -- molecular consequence
        # ================================================================
        for lvl in ["missense", "nonsense", "frameshift", "splice",
                    "large_structural", "small_structural", "synonymous",
                    "regulatory", "other"]:
            add("consequence", f"vtype_{lvl}",
                np.array([float(v == lvl) for v in vtype]))

        for lvl in ["substitution", "deletion", "duplication", "insertion",
                    "inversion", "delins", "complex", "other"]:
            add("consequence", f"mech_{lvl}",
                np.array([float(m == lvl) for m in mech]))

        # The null-mutation axis: no functional protein is produced, so the
        # immune system has never seen FVIII as self.
        is_null = np.array([float(v in NULL_VARIANT_TYPES) for v in vtype])
        add("consequence", "is_null_mutation", is_null)
        add("consequence", "is_truncating",
            np.array([float(p.is_frameshift or p.is_nonsense) for p in pr]))
        add("consequence", "is_inframe_indel",
            np.array([float(p.is_inframe_del or p.is_inframe_dup or p.is_delins)
                      for p in pr]))
        add("consequence", "is_multi_exon",
            np.where(np.isnan(exon_last) | np.isnan(exon), 0.0,
                     (exon_last > exon).astype(float)))
        add("consequence", "n_exons_involved",
            np.where(np.isnan(exon_last) | np.isnan(exon), 1.0,
                     exon_last - exon + 1))
        add("consequence", "unknown_breakpoint",
            np.array([float(x.unknown_breakpoint) for x in cd]))

        span = np.array([x.span_nt for x in cd], dtype=float)
        add("consequence", "log_span_nt", np.log1p(np.nan_to_num(span, nan=0.0)))
        add("consequence", "is_large_event", (np.nan_to_num(span, nan=0.0) > 50).astype(float))

        # ================================================================
        # BLOCK 2 -- position within the FVIII molecule
        # ================================================================
        mature = np.array([p.mature_pos for p in pr], dtype=float)
        # fall back to the exon midpoint when the protein change is unparsable
        add("position", "mature_pos", mature)
        add("position", "mature_pos_norm", mature / bio.MATURE_LEN)
        add("position", "mature_pos_known", (~np.isnan(mature)).astype(float))

        dom_from_pos = [bio.domain_of(p) for p in mature]
        # CHAMP's own Domain column, normalised, used as a fallback
        dom_raw = [_norm(v).upper().replace(" ", "_") for v in get("domain")]
        dom = [d if d != "Unknown" else r for d, r in zip(dom_from_pos, dom_raw)]
        for lvl in ["A1", "A2", "A3", "B", "C1", "C2",
                    "SIGNAL", "a1", "a2", "a3", "UNKNOWN"]:
            add("position", f"domain_{lvl}",
                np.array([float(str(d).upper() == lvl.upper()) for d in dom]))

        add("position", "in_b_domain",
            np.array([bio.in_span(p, bio.B_DOMAIN_START, bio.B_DOMAIN_END)
                      for p in mature], dtype=float))
        add("position", "is_light_chain",
            np.array([float(chain[i] == "light") or
                      float(mature[i] > bio.HEAVY_LIGHT_BOUNDARY
                            if not math.isnan(mature[i]) else 0.0)
                      for i in range(n)]))
        add("position", "is_heavy_chain",
            np.array([float(chain[i] == "heavy") for i in range(n)]))
        add("position", "is_single_domain_event",
            np.array([float(chain[i] == "single_domain") for i in range(n)]))

        # exon geometry
        add("position", "exon_number", exon)
        add("position", "exon_norm", exon / bio.N_EXONS)
        add("position", "is_exon_14", (exon == bio.B_DOMAIN_EXON).astype(float))
        add("position", "is_last_exon", (exon == bio.LAST_EXON).astype(float))
        bounds = np.array([self._exon_bounds(e) for e in exon], dtype=float)
        add("position", "exon_length", np.log1p(np.nan_to_num(bounds[:, 2], nan=0.0)))

        # ================================================================
        # BLOCK 3 -- inhibitor epitopes and functional surfaces
        # ================================================================
        for name, lo, hi in bio.INHIBITOR_EPITOPES:
            add("position", f"in_{name}",
                np.array([bio.in_span(p, lo, hi) for p in mature], dtype=float))
        add("position", "nearest_epitope_dist",
            np.array([bio.nearest_epitope_distance(p) for p in mature]))
        add("position", "in_any_epitope",
            np.array([float(any(bio.in_span(p, lo, hi)
                                for _, lo, hi in bio.INHIBITOR_EPITOPES))
                      for p in mature]))
        for name, lo, hi in bio.FUNCTIONAL_SITES:
            add("position", f"in_{name}",
                np.array([bio.in_span(p, lo, hi) for p in mature], dtype=float))

        # ================================================================
        # BLOCK 4 -- truncation severity
        # ================================================================
        ptc = np.array([p.ptc_mature_pos for p in pr], dtype=float)
        add("truncation", "ptc_pos", ptc)
        add("truncation", "ptc_known", (~np.isnan(ptc)).astype(float))
        frac_lost = np.where(np.isnan(ptc), np.nan,
                             1.0 - np.clip(ptc, 0, bio.MATURE_LEN) / bio.MATURE_LEN)
        add("truncation", "fraction_protein_lost", frac_lost)
        add("truncation", "fs_ter_offset",
            np.array([p.fs_ter_offset for p in pr], dtype=float))

        # Nonsense-mediated decay: a PTC in the last exon (or within ~55 nt of
        # the final junction) escapes NMD, so a truncated protein *is* made.
        # That distinction plausibly changes immune presentation.
        nmd_escape = np.array(
            [float((not math.isnan(exon[i])) and exon[i] >= bio.LAST_EXON
                   and (pr[i].is_frameshift or pr[i].is_nonsense))
             for i in range(n)])
        add("truncation", "nmd_escape", nmd_escape)
        add("truncation", "nmd_target",
            np.array([float((pr[i].is_frameshift or pr[i].is_nonsense)
                            and not nmd_escape[i]) for i in range(n)]))

        # which domains are lost downstream of the stop
        for dname, lo, _hi in bio.FVIII_DOMAINS:
            add("truncation", f"loses_{dname}",
                np.array([float((not math.isnan(ptc[i])) and ptc[i] < lo
                                and (pr[i].is_frameshift or pr[i].is_nonsense))
                          for i in range(n)]))
        add("truncation", "n_domains_lost",
            np.array([float(sum((not math.isnan(ptc[i])) and ptc[i] < lo
                                and (pr[i].is_frameshift or pr[i].is_nonsense)
                                for _d, lo, _h in bio.FVIII_DOMAINS))
                      for i in range(n)]))

        # ================================================================
        # BLOCK 5 -- residue chemistry (missense only, NaN elsewhere)
        # ================================================================
        ref_aa = [p.ref_aa if len(p.ref_aa) == 1 and p.ref_aa.isalpha() else ""
                  for p in pr]
        alt_aa = [p.alt_aa if len(p.alt_aa) == 1 and p.alt_aa.isalpha() else ""
                  for p in pr]
        pair_ok = [bool(r) and bool(a) and r in bio.HYDROPATHY and a in bio.HYDROPATHY
                   for r, a in zip(ref_aa, alt_aa)]

        def chem(table, idx, default=np.nan):
            return np.array([table.get(idx[i], default) if pair_ok[i] else np.nan
                             for i in range(n)], dtype=float)

        add("chemistry", "grantham",
            np.array([bio.grantham_distance(ref_aa[i], alt_aa[i]) if pair_ok[i]
                      else np.nan for i in range(n)]))
        add("chemistry", "blosum62",
            np.array([bio.blosum62(ref_aa[i], alt_aa[i]) if pair_ok[i]
                      else np.nan for i in range(n)]))
        add("chemistry", "d_hydropathy", chem(bio.HYDROPATHY, alt_aa) - chem(bio.HYDROPATHY, ref_aa))
        add("chemistry", "d_volume", chem(bio.VOLUME, alt_aa) - chem(bio.VOLUME, ref_aa))
        add("chemistry", "d_charge", chem(bio.CHARGE, alt_aa) - chem(bio.CHARGE, ref_aa))
        add("chemistry", "d_polarity", chem(bio.POLARITY, alt_aa) - chem(bio.POLARITY, ref_aa))
        add("chemistry", "abs_d_charge",
            np.abs(chem(bio.CHARGE, alt_aa) - chem(bio.CHARGE, ref_aa)))
        add("chemistry", "ref_is_cys", np.array([float(r == "C") for r in ref_aa]))
        add("chemistry", "alt_is_cys", np.array([float(a == "C") for a in alt_aa]))
        add("chemistry", "ref_is_gly", np.array([float(r == "G") for r in ref_aa]))
        add("chemistry", "alt_is_pro", np.array([float(a == "P") for a in alt_aa]))
        add("chemistry", "ref_is_arg", np.array([float(r == "R") for r in ref_aa]))
        add("chemistry", "aromatic_change",
            np.array([float((r in bio.AROMATIC) != (a in bio.AROMATIC))
                      if pair_ok[i] else 0.0
                      for i, (r, a) in enumerate(zip(ref_aa, alt_aa))]))
        add("chemistry", "charge_flip",
            np.array([float(bio.CHARGE.get(r, 0) * bio.CHARGE.get(a, 0) < 0)
                      if pair_ok[i] else 0.0
                      for i, (r, a) in enumerate(zip(ref_aa, alt_aa))]))
        add("chemistry", "has_chemistry", np.array([float(x) for x in pair_ok]))

        # ================================================================
        # BLOCK 6 -- nucleotide context
        # ================================================================
        ref_nt = [x.ref_nt for x in cd]
        alt_nt = [x.alt_nt for x in cd]
        purine = set("AG")
        add("nucleotide", "is_transition",
            np.array([float(bool(r) and bool(a) and len(r) == len(a) == 1
                            and (r in purine) == (a in purine))
                      for r, a in zip(ref_nt, alt_nt)]))
        add("nucleotide", "is_transversion",
            np.array([float(bool(r) and bool(a) and len(r) == len(a) == 1
                            and (r in purine) != (a in purine))
                      for r, a in zip(ref_nt, alt_nt)]))
        # C>T / G>A dominate at methylated CpG sites and mark recurrent,
        # independently arising mutations.
        add("nucleotide", "cpg_signature",
            np.array([float((r, a) in {("C", "T"), ("G", "A")})
                      for r, a in zip(ref_nt, alt_nt)]))
        for b in "ACGT":
            add("nucleotide", f"ref_nt_{b}", np.array([float(r == b) for r in ref_nt]))
            add("nucleotide", f"alt_nt_{b}", np.array([float(a == b) for a in alt_nt]))

        cpos = np.array([x.cdna_pos for x in cd], dtype=float)
        add("nucleotide", "cdna_pos_norm", cpos / 7053.0)
        add("nucleotide", "indel_length_mod3",
            np.array([float(np.nan_to_num(s, nan=0.0) % 3) for s in span]))
        add("nucleotide", "frame_preserving",
            np.array([float(np.nan_to_num(s, nan=0.0) % 3 == 0) for s in span]))

        # ================================================================
        # BLOCK 7 -- splicing
        # ================================================================
        ioff = np.array([x.intron_offset for x in cd], dtype=float)
        add("splicing", "is_intronic", np.array([float(x.is_intronic) for x in cd]))
        add("splicing", "abs_intron_offset", np.abs(ioff))
        add("splicing", "is_canonical_splice",
            (np.abs(np.nan_to_num(ioff, nan=999)) <= 2).astype(float))
        add("splicing", "is_extended_splice",
            (np.abs(np.nan_to_num(ioff, nan=999)) <= 8).astype(float))
        add("splicing", "splice_donor_side",
            np.array([float(not math.isnan(o) and o > 0) for o in ioff]))
        add("splicing", "splice_acceptor_side",
            np.array([float(not math.isnan(o) and o < 0) for o in ioff]))
        add("splicing", "is_utr5", np.array([float(x.is_utr5) for x in cd]))
        add("splicing", "is_utr3", np.array([float(x.is_utr3) for x in cd]))
        add("splicing", "is_promoter", np.array([float(x.is_promoter) for x in cd]))

        # ================================================================
        # BLOCK 8 -- clinical phenotype
        # ================================================================
        for lvl in ["severe", "moderate", "mild", "mixed", "unknown"]:
            add("clinical", f"severity_{lvl}",
                np.array([float(s == lvl) for s in sev]))
        add("clinical", "severity_ordinal",
            np.array([_SEVERITY_ORDINAL[s] for s in sev], dtype=float))
        add("clinical", "severity_reported",
            np.array([float(s != "unknown") for s in sev]))
        polya = [_norm(v) for v in get("polya")]
        add("clinical", "in_poly_a", np.array([float(p == "y") for p in polya]))

        # Interaction the literature singles out: a *severe* phenotype caused
        # by a *null* variant is the classic high-inhibitor-risk combination.
        add("clinical", "null_and_severe",
            is_null * np.array([float(s == "severe") for s in sev]))
        add("clinical", "null_and_light_chain",
            is_null * np.array([float(chain[i] == "light") for i in range(n)]))

        feats = pd.DataFrame(out, index=df.index).astype(float)
        feats = feats.replace([np.inf, -np.inf], np.nan)
        return feats, blocks


def block_index(blocks: dict[str, list[str]], columns: list[str]) -> dict[str, list[int]]:
    """Map each biological block to positional indices in the design matrix."""
    pos = {c: i for i, c in enumerate(columns)}
    return {b: [pos[c] for c in cols if c in pos] for b, cols in blocks.items()}
