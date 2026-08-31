# Stage 2 — patient-level modelling on the EAHAD/HADB cohort

**What changed:** the unit of analysis moved from the *variant* to the
*patient*, and the honest ceiling moved with it — from AUC 0.74–0.75 to
**0.786 ± 0.004**.

Stage 1 established, on the CDC CHAMP variant list, that the published
reference results (97.37% and 99.63% accuracy) are preprocessing artefacts, and
that genotype-only data tops out around 0.74–0.75. That ceiling was a property
of the *data*: CHAMP is a catalogue of variants, with no per-patient factor
level, antigen or CRM typing.

The EAHAD/HADB supplement (Blood Adv. 2024, VTH-2024-000215) supplies exactly
that missing layer.

---

## 1. The dataset

| table | unit | rows | contents |
|---|---|---|---|
| `mmc2` | variant | 6,211 | mutation class, protein consequence, domain, exon, codon and nucleotide change |
| `mmc3` | **allele report (patient)** | 10,064 | the individual's FVIII activity, clinical severity, antigen, CRM type, reporting centre, and the inhibitor outcome |

**4,966 records carry a recorded Yes/No outcome**, across 2,643 variants and
342 studies, at **16.8% prevalence** — which matches published epidemiology
with no adjustment of any kind.

The remaining 5,098 records say "Not reported", "Not", or nothing. They stay
**unlabelled**. Converting them to negatives is the single choice that inflated
the earlier reference results, and §3 shows it doing the same thing here.

### Univariate signal

| mutation class | n | inhibitor rate |
|---|---|---|
| Large deletion | 236 | **48.6%** |
| Nonsense | 594 | 30.1% |
| In-frame | 97 | 27.8% |
| Frameshift | 1,072 | 22.4% |
| Splice | 250 | 17.2% |
| Missense | 2,675 | 8.3% |
| Silent | 30 | 3.3% |

| severity | n | inhibitor rate |
|---|---|---|
| Severe | 2,706 | 24.2% |
| Moderate | 741 | 7.7% |
| Mild | 1,437 | 7.0% |

Textbook immunology: a variant that abolishes FVIII leaves the immune system
with no central tolerance to it, so infused factor arrives as a foreign
antigen. A missense variant still produces full-length protein and risk drops
threefold.

---

## 2. Protocol

- **Unit:** one allele report (a patient), with variant annotation joined in.
- **Grouping:** every split is grouped by `mut_id`. Recurrent variants generate
  up to 104 records each; splitting at random lets a model memorise a variant
  in training and be rewarded for it at test.
- **A second, harsher split groups by study**, to estimate behaviour in a
  reporting centre never seen before.
- **No resampling anywhere.** Class weights only.
- **95 features** in six blocks: genotype (20), domain (16), position (13),
  chemistry (18), clinical (18), context (9).

Two engineering notes: the curated `CpG` column ships empty in this release, so
the hotspot signature is derived from the nucleotide change (C>T and G>A are
the deamination products of a methylated CpG); and factor levels are free text
mixing plain numbers, censored readings (`<1`), ranges (`23 to 40`) and
annotated entries (`9|<1?`), with censored values taking half the bound.

---

## 3. Auditing the derived CSVs

Two convenience files were prepared alongside the raw supplement. They are a
sensible first pass, and they independently reproduce — on a new dataset — the
failure modes stage 1 was built to correct. Both are demonstrated numerically.

### A. Unrecorded outcomes relabelled as negatives

| | |
|---|---|
| variants in the registry | 6,212 |
| variants with a recorded outcome | 2,643 |
| rows in `HemophiliaA_ML_Ready_Inhibitor.csv` | 3,706 |
| **rows labelled negative with no outcome ever recorded** | **1,063** |
| prevalence in that file | 13.3% |
| prevalence among variants actually followed up | 23.5% |

Absence of a report is not a negative result. The substitution pads the
majority class and raises accuracy without improving prediction — the identical
mechanism that took CHAMP's prevalence from 20.1% to 11.4%.

### B. The outcome present as a feature

`HemophiliaA_Merged_MMC2_MMC3.csv` carries `inhibitor_positive_rate`,
`inhibitor_yes_count`, `uinhibitor`, `useverity`, `uclotting`, `uratio`,
`uantigen` and `utype` — all of them the outcome, or `mmc2` summaries of the
very `mmc3` rows being predicted.

| | AUC |
|---|---|
| without them | 0.777 |
| **with `inhibitor_positive_rate` added** | **0.966** |

One column. No new biology. All are listed in `src.hadb.FORBIDDEN`.

### C. Variant-level aggregation — where the experiment contradicted us

The expectation was that collapsing to one row per variant would cost
performance. **It does not: the variant-level model scores higher, 0.804
against 0.777.** That is reported as it came out.

The two numbers are not comparable — different unit, different label, different
n — and predicting a variant's *modal* outcome is an easier question than
predicting one patient's. So the case for patient-level modelling rests on the
measurement that actually supports it:

> Of the **537 variants with two or more recorded outcomes, 124 (23.1%) are
> discordant** — patients carrying an identical variant who differ in whether
> they developed an inhibitor, spanning 1,297 records.

A majority vote hands every one of those patients the same prediction and is
wrong for the minority. A higher AUC on an easier question is not a better
clinical tool.

---

## 4. What each layer is worth

| rung | features | AUC (3 seeds) |
|---|---|---|
| genotype only | 21 | 0.7024 |
| + domain | 37 | 0.7261 |
| + position | 50 | 0.7372 |
| + chemistry (**all genomic**) | 68 | **0.7417** |
| + patient clinical | 86 | 0.7645 |
| + reporting region | 95 | 0.7779 |

Two claims come out of this ladder.

**The genomic rung lands at 0.742** — the CHAMP ceiling, reproduced on a
different registry, curated by a different consortium, from different patients,
through an independently written feature pipeline. Two datasets agreeing on
where genotype-only information runs out is far stronger than either alone.

**The clinical layer adds +0.023 and region a further +0.013.**

### The caveat, stated rather than buried

Baseline factor activity could in principle be a *consequence* of the outcome
rather than a predictor: a circulating inhibitor suppresses measured FVIII
activity. The registry records these as diagnostic baseline values but does not
timestamp them relative to inhibitor detection, so the concern cannot be fully
closed with this data.

That is why the genomic-only rung is reported alongside. **A reader who rejects
the clinical features entirely still has a defensible 0.742 model, validated
across two registries.** The clinical layer is an improvement conditional on
that assumption, not a foundation.

---

## 5. Controls

| control | result | reading |
|---|---|---|
| shuffled labels | **0.5042** | the matrix carries no row identity. Stage 1's CHAMP pipeline reached train AUC **1.000** here, because `HGVS cDNA` was a primary key |
| ungrouped random CV | 0.8098 | what we could have reported by splitting carelessly |
| grouped by variant | 0.7779 | **the 0.032 gap is the leak the protocol closes** |
| grouped by study | 0.7797 | performance holds in an unseen reporting centre |

The leak closed by grouping (0.032) is roughly the size of the entire
clinical-layer gain — a compact demonstration of how easily a protocol choice
is mistaken for a modelling advance.

---

## 6. Model selection

Twelve candidates under variant-grouped 5-fold CV; tuned by randomised search
on grouped folds; combined by greedy forward selection on grouped out-of-fold
scores.

| | CV AUC |
|---|---|
| best single (tuned XGBoost) | 0.7826 |
| **rank ensemble** (XGBoost + ExtraTrees + LightGBM) | **0.7859** |

`sklearn`'s `StackingClassifier` scored **0.24** in the screen — it generates
internal folds at random with no way to pass grouping through, so the same
variant lands on both sides of the meta-learner's boundary and the blend
inverts. The ensemble therefore builds its out-of-fold matrix by hand.

Members are blended by **rank average** rather than probability average,
because a forest's 0.6 and a boosted tree's 0.6 are not the same claim. Ranks
are taken against the *training* distribution stored at fit time, so a
patient's score does not depend on who else is in the request.

---

## 7. Results

**Primary — repeated variant-grouped CV, whole labelled cohort (n = 4,966):**

> ### AUC-ROC 0.7861 ± 0.0040

**Check — single variant-grouped held-out split (n = 974), scored once:**

> AUC-ROC 0.7451, 95% CI [0.6927, 0.7890]

The CV estimate falls inside that interval. Repeated CV is the headline because
a single 20% split holds about 159 positives and carries roughly ±0.04 of
sampling noise.

Brier 0.114, ECE 0.033.

### Operating points

Majority-class baseline accuracy: **83.7%**.

| operating point | threshold | accuracy | sensitivity | specificity | precision | NPV |
|---|---|---|---|---|---|---|
| accuracy-maximising | 0.474 | **86.7%** | 25.8% | 98.5% | 77.4% | 87.2% |
| balanced (Youden) | 0.181 | 76.7% | 61.6% | 79.6% | 37.1% | 91.4% |
| rule-out (80% sens) | 0.131 | 65.6% | 74.2% | 63.9% | 28.6% | 92.7% |
| rule-out (90% sens) | 0.083 | 49.4% | 79.9% | 43.4% | 21.6% | 91.7% |

The accuracy-maximising point beats the do-nothing baseline by 3.0 points while
finding one inhibitor patient in four. It is the least useful column in the
table, and it is the one a rubric that asks for "high accuracy" would select.
The tool ships on the balanced and rule-out points, where NPV above 0.91 means
a patient the model clears is genuinely unlikely to develop an inhibitor.

### Calibrated risk cannot claim certainty

Plain isotonic regression returned exactly **1.0** for the highest-scoring
inputs, because its top step covered a small all-positive training bin — the
app was telling a clinician a patient would *certainly* develop an inhibitor.
Nothing supports that: the worst observed stratum (severe + large deletion)
runs at 53%, the top calibration bin at 60%.

`BoundedIsotonic` clips to the range spanned by Laplace-smoothed decile rates,
**[0.025, 0.520]** here. Ranking is untouched, so AUC is unchanged.

The bounded model tracks the observed epidemiology closely:

| case | predicted | observed rate in that stratum |
|---|---|---|
| severe, large deletion, CRM-I | 41.6% | 53.1% (n = 226) |
| severe, nonsense | 25.9% | 30.6% (n = 568) |
| moderate, splice | 8.3% | 8.1% (n = 37) |
| mild, missense C1 | 3.0% | 6.4% (n = 1,289) |

The model is conservative at both ends — it does not reach the observed 53% for
the worst stratum, and sits below the observed rate for mild missense. That is
the expected consequence of bounding the calibrator and of shrinkage toward
the base rate, and it is the safer direction for the high end.

---

## 8. Cross-registry transfer — and the contamination we had to find

The first run gave HADB → CHAMP **0.879**, against within-CHAMP CV of 0.725.

**A model applied to a cohort it has never seen should not beat that cohort's
own cross-validation.** When it does, the cohort is not unseen.

CHAMP and EAHAD are both compiled from the published literature, and they
compile many of the same papers. Checking directly: **64.5% of CHAMP's
substitution-like labelled variants already appear in HADB.**

One subtlety had to be handled. The matching key needs a reference residue and
a position, which a frameshift does not have — left uncorrected, every
frameshift falls into "novel" by default and the comparison becomes 77%
frameshift against 60% missense, measuring mutation-class composition rather
than novelty. Both strata are therefore restricted to substitution-like
variants.

| | AUC |
|---|---|
| CHAMP variants HADB had seen | 0.9361 |
| **CHAMP variants novel to HADB** | **0.8513** [0.8193, 0.8807] |
| contamination | 0.085 |

And the like-for-like control, on those *exact same novel rows*:

| trained on | AUC on the same 692 novel CHAMP variants |
|---|---|
| CHAMP's own cross-validation | 0.6237 |
| **HADB patient-level records** | **0.8513** |
| **advantage of patient-level training** | **+0.2276** |

Training on 4,966 patient-level records from a European consortium predicts
novel American variants **+0.228 AUC better** than CHAMP predicts them from its
own data. That is the clearest evidence in the project that repeated
patient-level observations are richer supervision than a variant catalogue —
the same conclusion the ablation reached, arrived at independently.

### What relabelling does, one more time

Same fitted model, same CHAMP variants, only the label convention changes:

| | n | prevalence | **majority-class baseline** | model accuracy | AUC |
|---|---|---|---|---|---|
| recorded outcomes only | 2,296 | 0.201 | **79.9%** | 80.1% | 0.879 |
| unrecorded → negative | 4,040 | 0.114 | **88.6%** | 74.7% | 0.855 |

Read the baseline column, not the accuracy column. Relabelling does not make
the model better — AUC in fact falls slightly. What it does is lift the score
available for free, from 79.9% to **88.6%**, by making the majority class
bigger.

That is where a headline like 97% comes from: it is measured on a set whose
no-skill baseline is already 88.6%, and it is reached by tuning the threshold
toward the majority class. Our own model, held at a clinically useful
threshold, scores *below* that baseline on the relabelled set — which is the
honest thing for it to do, and precisely why accuracy is never reported in this
project without the baseline beside it.

---

## 9. Where this lands

**What would move the number further.** Not a better classifier — the
twelve-model spread in the screen is 0.03 wide, so architecture is not the
binding constraint. The missing variables are the ones known to drive inhibitor
development and absent from every public registry: **treatment intensity,
exposure days at first bleed, product type (plasma-derived vs recombinant),
surgery or infection at first exposure, and HLA class II typing.** A
prospective cohort carrying those would plausibly reach 0.85+; no amount of
modelling reaches it from registry data alone.

**What it is not.** A research tool, not a validated medical device. It does
not replace laboratory inhibitor testing or clinical judgement. Its intended
use is prioritising monitoring intensity during the first fifty exposure days —
the window in which most inhibitors appear.

---

## Reproducing

```bash
python .devtools/hadb_screen.py      # model zoo under grouped CV
python .devtools/hadb_ablation.py    # feature ladder + controls
python .devtools/hadb_audit.py       # audit of the derived CSVs
python .devtools/hadb_tune.py        # randomised search + ensembling
python .devtools/hadb_final.py       # fit, calibrate, score held-out, ship
python .devtools/hadb_transfer.py    # cross-registry transfer
python .devtools/build_hadb_notebook.py
python .devtools/execute_notebooks.py Hemophilia_Capstone_HADB.ipynb
```

Artefacts: `reports/hadb_*.json`, `models/hadb_model.joblib`,
`Hemophilia_Capstone_HADB.ipynb`. The app serves the model at `/hadb`.
