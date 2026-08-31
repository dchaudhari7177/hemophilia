"""EAHAD / HADB F8 cohort (Blood Adv. 2024, VTH-2024-000215).

Two supplementary tables ship with the paper and they sit at different levels
of analysis, which is the whole reason this dataset is worth adding:

``mmc2`` -- one row per *variant* (6,211 rows). Curated genotype annotation:
    mutation class, protein consequence, domain, exon, codon and nucleotide
    change, CpG status.

``mmc3`` -- one row per *allele report* (10,064 rows), i.e. per patient
    observation, keyed back to ``mut_id``. This is the layer CHAMP never had:
    it carries the individual baseline FVIII activity, clinical severity,
    antigen level, CRM type and reporting centre alongside the inhibitor
    outcome.

Modelling therefore happens at the **patient** level, with variant annotation
joined in -- not at the variant level with the outcome collapsed to a majority
vote, which is what ``HemophiliaA_ML_Ready_Inhibitor.csv`` does.

Leakage discipline carried over from the CHAMP rebuild
------------------------------------------------------
1. ``uinhibitor`` in mmc2 is the variant-level *summary of the outcome*,
   computed from mmc3. It is the target under another name and is dropped.
2. ``useverity`` / ``uclotting`` / ``uratio`` / ``uantigen`` / ``utype`` are
   likewise mmc2 summaries of the mmc3 patient rows. Using them would let a
   record see its own aggregate, so they are dropped in favour of that
   record own measurements.
3. ``comments`` / ``pri_comments`` / ``ucomments`` are free text and five rows
   literally say "inhibitor". All text columns are dropped.
4. ``inhibitor_yes_count`` / ``inhibitor_positive_rate`` in the supplied
   merged CSV are the target aggregated per variant. Never features.
5. Unrecorded outcomes ("Not reported", blank, "Not") stay **unlabelled**.
   They are never relabelled as negatives -- that single choice is what
   inflated the earlier reference results.
6. Records sharing a ``mut_id`` must never straddle a train/test boundary, so
   every split in this project is grouped by variant.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .biology import (
    blosum62,
    domain_of,
    grantham_distance,
    nearest_epitope_distance,
)

ROOT = Path(__file__).resolve().parents[1]
HADB_DIR = ROOT / "data" / "raw" / "hadb"
VARIANTS_CSV = HADB_DIR / "BVTH_VTH-2024-000215-mmc2.csv"
RECORDS_CSV = HADB_DIR / "BVTH_VTH-2024-000215-mmc3.csv"

SIGNAL_PEPTIDE_LEN = 19
MATURE_LEN = 2332

#: Columns that must never reach a model. See the module docstring.
FORBIDDEN = {
    "uinhibitor", "useverity", "uclotting", "uratio", "uantigen", "utype",
    "udiscrep", "ucomments", "comments", "pri_comments",
    "inhibitor_no_count", "inhibitor_yes_count", "inhibitor_total_known",
    "inhibitor_positive_rate", "inhibitor_target", "Inhibitors",
}

_NULLISH = {"", "nan", "none", "null", "na", "n/a", "?", "-"}


# ---------------------------------------------------------------------------
# 1. Value normalisation
# ---------------------------------------------------------------------------
def _clean(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    s = str(v).strip()
    return "" if s.lower() in _NULLISH else s


def normalise_inhibitor(v) -> float:
    """Yes -> 1, No -> 0, anything unrecorded -> NaN (never a fake negative)."""
    s = _clean(v).lower()
    if s in {"yes", "y", "positive"}:
        return 1.0
    if s in {"no", "n", "negative"}:
        return 0.0
    return math.nan


def normalise_severity(v) -> str:
    """Collapse 27 spellings of the clinical phenotype into four buckets."""
    s = _clean(v).lower().replace(" ", "")
    if not s:
        return "Unknown"
    if s in {"unclassified", "notreported", "notaffected"}:
        return "Unknown"
    if s in {"non-severe", "nonsevere"}:
        return "Moderate"
    # Mixed reports ("severe/moderate") take the more severe end, which is how
    # the registry itself reads them.
    if re.search(r"sev|svere", s):
        return "Severe"
    if re.search(r"mod|moderare|mpderate", s):
        return "Moderate"
    if "mild" in s:
        return "Mild"
    return "Unknown"


SEVERITY_ORDINAL = {"Mild": 0.0, "Moderate": 1.0, "Severe": 2.0,
                    "Unknown": math.nan}


def parse_activity(v) -> float:
    """Parse a reported factor level into a percentage.

    The column mixes plain numbers, censored values ("<1"), ranges
    ("23 to 40") and annotated entries ("9|<1?"). Censored values are given
    half the bound, the usual substitution for a left-censored assay reading,
    which also keeps "<1" strictly below any observed 1.
    """
    s = _clean(v).replace("%", "").replace(",", ".")
    if not s:
        return math.nan
    s = s.split("|")[0].strip()
    m = re.fullmatch(r"([<>])=?\s*([\d.]+)\??", s)
    if m:
        val = float(m.group(2))
        return val * 0.5 if m.group(1) == "<" else val
    m = re.fullmatch(r"([\d.]+)\s*(?:to|-|–|and)\s*([\d.]+)\??", s, flags=re.I)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    m = re.match(r"^([\d.]+)", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return math.nan
    return math.nan


def normalise_effect(v) -> str:
    s = _clean(v).lower()
    if not s:
        return "Unknown"
    if "large" in s and "del" in s:
        return "LargeDeletion"
    if "large" in s and ("dup" in s or "ins" in s):
        return "LargeDuplication"
    if "nonsense" in s:
        return "Nonsense"
    if "frameshift" in s:
        return "Frameshift"
    if "missense" in s:
        return "Missense"
    if "splice" in s:
        return "Splice"
    if "silent" in s or "synonym" in s:
        return "Silent"
    if "in-frame" in s or "inframe" in s:
        return "InFrame"
    return "Other"


#: Effects that abolish endogenous FVIII protein. The immunological argument
#: for inhibitor risk is that a patient who never expresses any FVIII has no
#: central tolerance to it, so infused factor is seen as foreign.
NULL_EFFECTS = {"LargeDeletion", "LargeDuplication", "Nonsense", "Frameshift"}


def normalise_crm(v) -> str:
    """CRM type: I (no antigen), II (dysfunctional protein present), NU."""
    s = _clean(v).upper()
    return s if s in {"I", "II", "NU"} else "Unknown"


#: Reporting centre country -> coarse region. Ancestry is a genuine risk
#: modifier for inhibitor development (higher incidence is reported in
#: patients of African ancestry), but the registry records the *reporting
#: laboratory* country, not the patient ancestry, so this is treated as a weak
#: proxy and is ablated separately rather than trusted.
_REGION = {
    "africa": {"south africa", "egypt", "tunisia", "morocco", "nigeria",
               "algeria", "kenya"},
    "east_asia": {"china", "japan", "korea", "south korea", "taiwan",
                  "hong kong", "singapore", "thailand", "malaysia",
                  "vietnam", "indonesia"},
    "south_asia": {"india", "pakistan", "bangladesh", "sri lanka", "nepal",
                   "iran"},
    "middle_east": {"turkey", "israel", "saudi arabia", "lebanon", "jordan",
                    "kuwait", "united arab emirates", "iraq", "syria",
                    "qatar"},
    "latin_america": {"brazil", "argentina", "mexico", "chile", "colombia",
                      "venezuela", "peru", "cuba", "uruguay", "costa rica"},
    "north_america": {"usa", "united states", "canada", "u.s.a.", "us"},
    "oceania": {"australia", "new zealand"},
}


def normalise_region(v) -> str:
    s = _clean(v).lower()
    if not s:
        return "Unknown"
    for region, members in _REGION.items():
        if s in members:
            return region
    return "europe_other"


def parse_exon_number(v) -> float:
    s = _clean(v)
    if not s:
        return math.nan
    m = re.match(r"^(\d+)", s)
    return float(m.group(1)) if m else math.nan


# ---------------------------------------------------------------------------
# 2. Loading
# ---------------------------------------------------------------------------
def load_hadb(variants_path: Path | str | None = None,
              records_path: Path | str | None = None) -> pd.DataFrame:
    """Return one row per allele report, with variant annotation joined in."""
    variants = pd.read_csv(variants_path or VARIANTS_CSV, low_memory=False)
    records = pd.read_csv(records_path or RECORDS_CSV, low_memory=False)

    keep_variant = [
        "mut_id", "mut_type", "mut_effect", "d_id", "location", "e_i_numb",
        "locnumb", "aa_numb_old", "aa_numb", "codon_change", "codon_first",
        "codon_last", "n_bp", "nuc_numb", "ntchange", "aa_first", "aa_last",
        "CpG", "Count_mut_id",
    ]
    keep_variant = [c for c in keep_variant if c in variants.columns]
    df = records.merge(variants[keep_variant], on="mut_id", how="left",
                       suffixes=("", "_v"))

    df["y"] = df["Inhibitors"].map(normalise_inhibitor)
    df["severity"] = df["cli_phe"].map(normalise_severity)
    df["fviii_activity"] = df["clotting"].map(parse_activity)
    df["fviii_antigen"] = df["antigen"].map(parse_activity)
    df["act_ant_ratio"] = pd.to_numeric(df.get("act/ant"), errors="coerce")
    df["crm_type"] = df["type"].map(normalise_crm)
    df["region"] = df["pa_race"].map(normalise_region)
    df["effect"] = df["mut_effect"].map(normalise_effect)
    df["study"] = df["reference"].map(_clean).replace("", "unknown")
    return df


def label_summary(df: pd.DataFrame) -> dict:
    y = df["y"]
    return {
        "n_records": int(len(df)),
        "n_labelled": int(y.notna().sum()),
        "n_unlabelled": int(y.isna().sum()),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "prevalence": round(float(y.mean(skipna=True)), 4),
        "n_variants_labelled": int(df.loc[y.notna(), "mut_id"].nunique()),
        "n_studies_labelled": int(df.loc[y.notna(), "study"].nunique()),
    }


# ---------------------------------------------------------------------------
# 3. Feature construction
# ---------------------------------------------------------------------------
from .biology import (  # noqa: E402  (kept next to its use)
    AROMATIC,
    CHARGE,
    HYDROPATHY,
    POLARITY,
    THREE_TO_ONE,
    VOLUME,
    B_DOMAIN_EXON,
    HEAVY_LIGHT_BOUNDARY,
    LAST_EXON,
)

_DOMAIN_CANON = {
    "5utr": "UTR", "3utr": "UTR", "utr": "UTR",
    "signal peptide": "Signal", "splice site": "SpliceSite",
    "multiple domains": "Multiple", "undefined": "Unknown",
    "a1 linker": "a1", "a2 linker": "a2", "a3 linker": "a3",
    "a1-a2": "Multiple",
}


def canon_domain(v) -> str:
    s = _clean(v).lower().strip()
    if not s:
        return "Unknown"
    if s in _DOMAIN_CANON:
        return _DOMAIN_CANON[s]
    if s.upper() in {"A1", "A2", "A3", "B", "C1", "C2"}:
        return s.upper()
    if s in {"a1", "a2", "a3"}:
        return s
    return "Unknown"


def _aa1(three) -> str:
    s = _clean(three)
    if not s:
        return ""
    if len(s) == 1:
        return s.upper()
    return THREE_TO_ONE.get(s.capitalize(), "")


def _prop(aa: str, table: dict) -> float:
    return float(table[aa]) if aa in table else np.nan


def build_features(df: pd.DataFrame,
                   include_clinical: bool = True,
                   include_context: bool = True) -> tuple[pd.DataFrame, dict]:
    """Turn allele records into a numeric design matrix.

    ``include_clinical`` and ``include_context`` exist so the ablation study
    can rebuild the matrix without the patient measurements or without the
    reporting-centre proxy, and measure exactly what each layer contributes.

    Returns the matrix and a block map naming which columns belong to which
    biological group; block-wise attribution is reported against that map.
    """
    f = pd.DataFrame(index=df.index)
    blocks: dict[str, list[str]] = {}

    def add(block: str, name: str, values) -> None:
        f[name] = values
        blocks.setdefault(block, []).append(name)

    # -- genotype class ----------------------------------------------------
    effect = df["effect"]
    for lvl in ["Missense", "Nonsense", "Frameshift", "Splice", "InFrame",
                "Silent", "LargeDeletion", "LargeDuplication"]:
        add("genotype", f"effect_{lvl}", (effect == lvl).astype(float))
    add("genotype", "is_null_variant", effect.isin(NULL_EFFECTS).astype(float))
    add("genotype", "is_truncating",
        effect.isin({"Nonsense", "Frameshift", "LargeDeletion",
                     "LargeDuplication"}).astype(float))

    mut_type = df["mut_type"].map(lambda v: _clean(v).lower())
    for lvl in ["point", "deletion", "insertion", "duplication", "indel"]:
        add("genotype", f"muttype_{lvl}",
            mut_type.str.contains(lvl, regex=False).astype(float))

    loc = df["location"].map(lambda v: _clean(v).lower())
    add("genotype", "loc_exon", (loc == "exon").astype(float))
    add("genotype", "loc_intron", (loc == "intron").astype(float))
    add("genotype", "loc_utr", (loc == "utr").astype(float))

    # The curated ``CpG`` column ships empty in this release (every row is
    # "Null"), so the hotspot signature is derived from the nucleotide change
    # instead: C>T and G>A transitions are the deamination products of a
    # methylated CpG and account for most recurrent F8 point mutations.
    nt = df["ntchange"].map(lambda v: _clean(v).upper().replace(" ", ""))
    add("genotype", "cpg_signature", nt.isin({"C>T", "G>A"}).astype(float))
    purine, pyrim = set("AG"), set("CT")
    add("genotype", "is_transition", nt.map(
        lambda s: float(len(s) == 3 and s[1] == ">"
                        and ((s[0] in purine and s[2] in purine)
                             or (s[0] in pyrim and s[2] in pyrim)))))
    add("genotype", "n_bp_log", np.log1p(
        pd.to_numeric(df["n_bp"], errors="coerce").clip(lower=0)))

    # -- domain ------------------------------------------------------------
    dom = df["d_id"].map(canon_domain)
    for lvl in ["A1", "A2", "A3", "B", "C1", "C2", "a1", "a2", "a3",
                "Signal", "UTR", "SpliceSite", "Multiple", "Unknown"]:
        add("domain", f"domain_{lvl}", (dom == lvl).astype(float))
    add("domain", "in_C_domain", dom.isin({"C1", "C2"}).astype(float))
    add("domain", "in_A_domain", dom.isin({"A1", "A2", "A3"}).astype(float))

    # -- position ----------------------------------------------------------
    prot = pd.to_numeric(df["aa_numb"], errors="coerce")
    mature = prot - SIGNAL_PEPTIDE_LEN
    mature = mature.where(mature != 0, np.nan)
    add("position", "mature_pos", mature)
    add("position", "relative_pos", mature / MATURE_LEN)
    add("position", "is_heavy_chain",
        (mature <= HEAVY_LIGHT_BOUNDARY).astype(float))
    add("position", "epitope_distance", mature.map(
        lambda p: nearest_epitope_distance(p) if pd.notna(p) else np.nan))
    add("position", "structural_domain_known", mature.map(
        lambda p: 0.0 if pd.isna(p) or domain_of(p) == "Unknown" else 1.0))

    nuc = pd.to_numeric(df["nuc_numb"], errors="coerce")
    add("position", "nuc_pos", nuc)
    add("position", "nuc_pos_negative", (nuc < 0).astype(float))

    exon = df["e_i_numb"].map(parse_exon_number)
    add("position", "exon_number", exon)
    add("position", "is_exon_14", (exon == B_DOMAIN_EXON).astype(float))
    add("position", "is_last_exon", (exon == LAST_EXON).astype(float))
    # A premature stop escapes nonsense-mediated decay in the final exon, so a
    # truncated but stable protein can still be expressed and tolerised.
    add("position", "nmd_escape",
        ((exon == LAST_EXON)
         & effect.isin({"Nonsense", "Frameshift"})).astype(float))
    # How much of the protein a truncating variant removes.
    trunc = np.where(effect.isin({"Nonsense", "Frameshift"}),
                     1.0 - (mature / MATURE_LEN).clip(0, 1), 0.0)
    add("position", "truncation_fraction", pd.Series(trunc, index=df.index))
    add("position", "locnumb", pd.to_numeric(df["locnumb"], errors="coerce"))

    # -- residue chemistry -------------------------------------------------
    ref = df["aa_first"].map(_aa1)
    alt = df["aa_last"].map(_aa1)
    scorable = [(a, b) if (a in HYDROPATHY and b in HYDROPATHY) else ("", "")
                for a, b in zip(ref, alt)]
    add("chemistry", "is_substitution",
        ((ref != "") & (alt != "") & (ref != alt) & (alt != "*")).astype(float))
    add("chemistry", "grantham",
        [grantham_distance(a, b) if a else np.nan for a, b in scorable])
    add("chemistry", "blosum62",
        [blosum62(a, b) if a else np.nan for a, b in scorable])
    for tag, table in [("hydropathy", HYDROPATHY), ("volume", VOLUME),
                       ("charge", CHARGE), ("polarity", POLARITY)]:
        r = ref.map(lambda a: _prop(a, table))
        add("chemistry", f"ref_{tag}", r)
        add("chemistry", f"delta_{tag}", alt.map(lambda a: _prop(a, table)) - r)
    add("chemistry", "ref_aromatic", ref.map(lambda a: float(a in AROMATIC)))
    add("chemistry", "alt_aromatic", alt.map(lambda a: float(a in AROMATIC)))
    add("chemistry", "ref_is_cysteine", (ref == "C").astype(float))
    add("chemistry", "alt_is_cysteine", (alt == "C").astype(float))
    add("chemistry", "ref_is_proline", (ref == "P").astype(float))
    add("chemistry", "alt_is_proline", (alt == "P").astype(float))
    add("chemistry", "creates_stop", (alt == "*").astype(float))

    # -- patient clinical measurements -------------------------------------
    if include_clinical:
        act = df["fviii_activity"]
        add("clinical", "fviii_activity", act)
        add("clinical", "fviii_activity_log", np.log1p(act.clip(lower=0)))
        add("clinical", "fviii_below_1", (act < 1).astype(float))
        add("clinical", "fviii_1_to_5", ((act >= 1) & (act < 5)).astype(float))
        add("clinical", "fviii_above_5", (act >= 5).astype(float))
        add("clinical", "fviii_measured", act.notna().astype(float))

        sev = df["severity"]
        add("clinical", "severity_ordinal", sev.map(SEVERITY_ORDINAL))
        for lvl in ["Mild", "Moderate", "Severe", "Unknown"]:
            add("clinical", f"severity_{lvl}", (sev == lvl).astype(float))

        ant = df["fviii_antigen"]
        add("clinical", "fviii_antigen", ant)
        add("clinical", "fviii_antigen_measured", ant.notna().astype(float))
        add("clinical", "act_ant_ratio", df["act_ant_ratio"])

        crm = df["crm_type"]
        for lvl in ["I", "II", "NU", "Unknown"]:
            add("clinical", f"crm_{lvl}", (crm == lvl).astype(float))

    # -- reporting context -------------------------------------------------
    if include_context:
        reg = df["region"]
        for lvl in ["europe_other", "east_asia", "south_asia", "middle_east",
                    "latin_america", "north_america", "africa", "oceania",
                    "Unknown"]:
            add("context", f"region_{lvl}", (reg == lvl).astype(float))

    f = f.replace([np.inf, -np.inf], np.nan).astype(float)
    return f, blocks
