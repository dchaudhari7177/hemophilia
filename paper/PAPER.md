# The genotype-only ceiling for factor VIII inhibitor risk prediction: patient-level modelling and cross-registry validation in haemophilia A

**Dipak Chaudhari, Tejas Nagmote, Sneha A, Varsha P**
Department of Computer Science and Engineering, PES University, Bengaluru, India

**Guide:** Prof. Gayathri R S
**Project:** PW_GRS_01

> **Manuscript status.** Results, tables and figures are generated from the
> committed artefacts in `reports/` and are reproducible from the released code.
> The reference list at the end is marked where a citation must be verified
> against the primary source before submission; those entries are named
> descriptively rather than guessed.

---

## Abstract

**Background.** Neutralising alloantibodies ("inhibitors") against infused
factor VIII (FVIII) develop in a substantial minority of people with
haemophilia A and are the most consequential complication of replacement
therapy. No risk-stratification tool is in routine pre-treatment clinical use.
Machine-learning studies on public variant registries have reported accuracies
of 97% and above, which would represent clinically decisive performance.

**Objective.** To establish what is genuinely achievable from registry data, to
determine whether the limiting factor is the model or the data, and to quantify
the methodological mechanisms that produce the published estimates.

**Methods.** We used the EAHAD/HADB *F8* variant resource (Blood Advances 2024
supplement), whose allele-report table contains one row per patient (n =
10,064) rather than one row per variant. 4,966 records carry a recorded
inhibitor outcome across 2,643 variants and 342 reporting studies. We
engineered 95 mechanistic descriptors in six blocks (genotype class, FVIII
domain, protein position, residue chemistry, patient baseline phenotype,
reporting context) and screened twelve model families. **All partitions were
grouped by variant**, so no variant appeared on both sides of any boundary; no
resampling was used at any stage. Unrecorded outcomes were retained as
unlabelled and never treated as negatives. Probabilities were calibrated with a
bounded isotonic regression fitted on training folds only. External validity
was assessed by transfer to the CDC CHAMP registry.

**Results.** A rank-averaged ensemble of XGBoost, extremely randomised trees and
LightGBM achieved **AUC-ROC 0.786 ± 0.004** under repeated variant-grouped
cross-validation (n = 4,966), with a held-out estimate of 0.745 (95% CI
0.693–0.789). The genomic-only rung of the feature ablation reached **0.742**,
independently reproducing the ceiling previously established on CHAMP, a
separately curated registry. Adding the patient's own baseline phenotype
contributed +0.023 AUC. Three controls bounded the result: permuted labels gave
AUC 0.504; ungrouped cross-validation inflated the score by 0.032; and
study-grouped validation retained performance (0.780). Introducing a single
outcome-derived column available in a widely distributed derived file moved AUC
from 0.777 to **0.966**. Cross-registry transfer initially appeared to exceed
within-registry performance; we traced this to **64.5% variant overlap between
CHAMP and EAHAD**, and on genuinely novel variants obtained AUC 0.851
(0.819–0.881) versus 0.624 for CHAMP's own cross-validation on the identical
rows.

**Conclusions.** Genotype-only inhibitor prediction saturates near AUC 0.74 in
two independently curated registries; this is a property of the available data
rather than of model capacity, since twelve model families span only 0.03 AUC.
Patient-level records with per-individual baseline phenotype raise the ceiling
to approximately 0.79. Published accuracies above 95% are attributable to
identifiable preprocessing choices, each of which we reproduce and measure.
Investigators performing cross-registry validation in this domain must account
for substantial shared provenance between CHAMP and EAHAD.

**Keywords:** haemophilia A, factor VIII inhibitors, F8, risk prediction, data
leakage, external validation, model calibration

---

## 1. Introduction

Haemophilia A is an X-linked bleeding disorder caused by variants in *F8*,
managed by replacement of the deficient coagulation factor VIII. In a
substantial minority of patients the immune system recognises infused FVIII as
foreign and produces neutralising alloantibodies, termed inhibitors. Once
inhibitors develop, replacement therapy loses efficacy, bleeding becomes
markedly harder to control, and the patient requires immune tolerance induction
— a prolonged, expensive protocol that does not always succeed.

Inhibitor development is concentrated in the first fifty exposure days, which
makes it, in principle, an ideal target for pre-treatment risk stratification:
the window in which the outcome occurs is known, it is short, and the clinical
response to elevated risk (intensified monitoring, and consideration of product
choice and treatment intensity) is available and low-cost. Despite this, no
genotype-based stratification tool is in routine clinical use.

The mechanistic basis for genotype-based prediction is well established. Variant
classes that abolish endogenous FVIII protein — large deletions, nonsense
variants and frameshifts — are associated with markedly higher inhibitor
incidence than missense variants, which permit expression of a full-length,
if dysfunctional, protein. The immunological interpretation is that a patient
who has never expressed FVIII has established no central tolerance to it, so
infused factor is encountered as a wholly foreign antigen.

Several groups have applied machine learning to publicly available *F8* variant
registries, most prominently the CDC Hemophilia A Mutation Project (CHAMP)
list. Reported accuracies reach 97.37% [R1] and, in unpublished student work,
above 99%. If reproducible as *predictive* performance, such figures would be
clinically decisive.

They are not reproducible. Prior work by this group established that these
estimates arise from three specific preprocessing choices, and placed the
honest ceiling of genotype-only data at approximately AUC 0.74–0.75. That
finding raised the question this paper addresses: **is that ceiling a
limitation of the modelling, or of the data?**

CHAMP is a catalogue of *variants*. It records, for each variant, whether an
inhibitor has ever been reported — but it carries no per-patient factor
activity, no antigen measurement, and no cross-reacting material (CRM) typing.
If the ceiling reflects missing patient-level information rather than
insufficient model capacity, a resource that supplies that information should
move it.

### 1.1 Contributions

1. We build and validate an inhibitor-risk model on the EAHAD/HADB resource at
   the **patient level** — one observation per allele report rather than per
   variant — reaching AUC 0.786 ± 0.004 under strict variant-grouped
   validation. (We are not aware of prior patient-level modelling on this
   resource, but make no priority claim; no systematic review was conducted.)
2. We show that the genotype-only ceiling of ≈0.74 **replicates on an
   independently curated registry**, establishing it as a property of the data
   class rather than an artefact of one dataset or pipeline.
3. We quantify three leakage mechanisms — outcome-derived features, relabelling
   of unrecorded outcomes, and ungrouped splitting — reporting the exact AUC
   attributable to each.
4. We identify and quantify **64.5% variant overlap between the CHAMP and
   EAHAD registries**, which invalidates naive cross-registry validation in this
   domain, and report a corrected external estimate.
5. We report two experiments whose outcomes contradicted our hypotheses, and
   revise the claims rather than the framing.
6. We describe a bounded calibration procedure that prevents the deployed model
   from emitting risk estimates the underlying cohort cannot support.

---

## 2. Related work and the reproducibility problem

Published machine-learning analyses of *F8* variant registries share a common
design: rows are variants, features are registry columns (often label-encoded
directly), the outcome is a reported inhibitor history, and evaluation is a
stratified split or k-fold cross-validation.

Three properties of this design account for the reported performance.

**Identifier features.** Registry columns such as the HGVS cDNA description are
near-unique per row. Label-encoded, they function as primary keys. In our prior
work, a random forest given such a column attained training AUC 1.000 *even
with the outcome labels randomly permuted* — the diagnostic signature of an
identifier rather than a predictor.

**Relabelling of unrecorded outcomes.** Registry inhibitor fields are
tri-state: reported positive, reported negative, and not reported. Mapping "not
reported" to negative converts absence of ascertainment into evidence of
absence. In CHAMP this affects 1,744 of 4,040 rows and reduces apparent
prevalence from 20.1% to 11.4%, raising the majority-class baseline against
which accuracy is measured.

**Resampling before partitioning.** Random over-sampling duplicates minority
rows verbatim. Applied before the train/test split, it places byte-identical
copies of training rows into the test set. The reported 97.37% accuracy [R1] is
obtained under this protocol.

These are not exotic failures; they are the default outcome of applying a
standard tabular-ML recipe to a curated biological registry. Our contribution
here is to measure each, rather than to assert them.

---

## 3. Data

### 3.1 The EAHAD/HADB cohort

The European Association for Haemophilia and Allied Disorders *F8* variant
resource is distributed as two supplementary tables that sit at different levels
of analysis:

| Table | Unit | Rows | Contents |
|---|---|---|---|
| `mmc2` | variant | 6,211 | mutation class, protein consequence, FVIII domain, exon/intron, codon and nucleotide change |
| `mmc3` | **allele report (patient)** | 10,064 | individual baseline FVIII activity, clinical severity, antigen, CRM type, reporting centre and country, source publication, inhibitor outcome |

`mmc3` is the layer absent from CHAMP, and it determines the design of this
study: the unit of analysis is the patient, with variant annotation joined from
`mmc2` on `mut_id`.

### 3.2 Outcome definition

The `Inhibitors` field takes eight surface forms which normalise to three
states. 4,130 records are negative, 836 positive, and 5,098 unrecorded
("Not reported", "Not", or blank).

**Unrecorded outcomes were retained as unlabelled and never mapped to
negative.** The resulting analytic cohort is **4,966 records at 16.83%
prevalence**, which falls within the published range for haemophilia A without
any adjustment — itself evidence that the tri-state field is being read
correctly.

### 3.3 Cohort characteristics

**Table 1.** Baseline characteristics of the 4,966 labelled allele reports
(836 inhibitor-positive, 16.8%), across 2,643 variants and 342 reporting
studies. Median records per variant 1 (maximum 104).

| Characteristic | n | % of cohort | Inhibitor rate |
|---|---|---|---|
| **Mutation class** | | | |
| Missense | 2,677 | 53.9 | 8.3% |
| Frameshift | 1,072 | 21.6 | 22.4% |
| Nonsense | 594 | 12.0 | 30.1% |
| Splice | 250 | 5.0 | 17.2% |
| Large deletion | 237 | 4.8 | **51.9%** |
| In-frame | 97 | 2.0 | 27.8% |
| Silent | 30 | 0.6 | 3.3% |
| **Clinical severity** | | | |
| Severe | 2,718 | 54.7 | 24.1% |
| Moderate | 774 | 15.6 | 8.9% |
| Mild | 1,392 | 28.0 | 6.2% |
| Unknown | 82 | 1.7 | 29.3% |
| **Baseline FVIII activity** | | | |
| < 1 IU/dL | 2,356 | 47.4 | 20.5% |
| 1–2 | 154 | 3.1 | 8.4% |
| 2–5 | 610 | 12.3 | 7.4% |
| 5–15 | 555 | 11.2 | 7.6% |
| 15–40 | 415 | 8.4 | 4.1% |
| > 40 | 30 | 0.6 | 0.0% |
| Not measured | 846 | 17.0 | — |

Median measured activity was 0.50 IU/dL (IQR 0.50–5.00); 1.00 among
inhibitor-negative and 0.50 among inhibitor-positive records. FVIII antigen was
measured in only 649 records (13.1%) and CRM type recorded in 1,312 (26.4%).

The monotone gradient across mutation classes (large deletion → nonsense →
frameshift → splice → missense → silent) and across severity strata reproduces
established immunology and constitutes a prior check on label quality.

### 3.4 The CHAMP comparison cohort

For external validation we used the CDC CHAMP *F8* variant list [R2]: 4,040
variants, of which 2,296 carry a recorded inhibitor outcome at 20.1%
prevalence. CHAMP is variant-level and carries reported clinical severity but
no per-patient measurements.

### 3.5 Audit of redistributed derived files

Two convenience files circulate alongside the raw supplement, prepared by
merging and aggregating the two tables. Both reproduce, on this new dataset,
the failure modes described in §2. We audit them because they are the form in
which many investigators will encounter this resource.

**Relabelling.** The ML-ready file contains 3,706 variant rows, of which
**1,063 carry a negative label although no inhibitor outcome was ever recorded**
for that variant. Apparent prevalence falls from 23.5% (among variants actually
followed up) to 13.3%.

**Outcome-derived features.** The merged file exposes twelve columns in our
forbidden set, including `inhibitor_positive_rate`, `inhibitor_yes_count` and
`uinhibitor` — the outcome itself, or `mmc2` summaries computed from the very
`mmc3` rows being predicted. Adding `inhibitor_positive_rate` alone to our
design matrix moves AUC from **0.777 to 0.966** (Δ = +0.189), with no biological
information added.

---

## 4. Methods

### 4.1 Feature construction

95 numeric descriptors were derived in six declared blocks. Every feature is
either a property of the variant or a measurement recorded at diagnosis; none is
derived from the outcome, and none is row-unique.

| Block | k | Contents |
|---|---|---|
| Genotype | 21 | mutation class indicators, null/truncating flags, CpG deamination signature, transition/transversion, lesion size, exonic/intronic/UTR location |
| Domain | 16 | FVIII domain occupancy (A1, a1, A2, a2, B, a3, A3, C1, C2, signal peptide, splice site) |
| Position | 13 | mature-protein residue, relative position, heavy/light chain, exon number, truncation fraction, NMD-escape indicator, distance to reported inhibitor epitopes |
| Chemistry | 18 | Grantham distance [R3], BLOSUM62 [R4], hydropathy/volume/charge/polarity deltas, cysteine and proline involvement, stop creation |
| Clinical | 18 | patient baseline FVIII activity (raw, log, stratified, measured-indicator), severity ordinal and indicators, antigen, activity/antigen ratio, CRM type |
| Context | 9 | reporting region indicators |

Two engineering decisions required judgement:

**CpG status.** The curated `CpG` column is empty in this release (every row
reads "Null"). The hotspot signature was therefore derived from the nucleotide
change: C>T and G>A transitions are the deamination products of a methylated
CpG dinucleotide and account for most recurrent *F8* point mutations.

**Factor level parsing.** The activity field mixes plain numerals, left-censored
readings (`<1`), ranges (`23 to 40`) and annotated entries (`9|<1?`). Censored
values were assigned half the stated bound, the conventional substitution for a
left-censored assay reading, which preserves the ordering `<1 < 1`.

Missingness is encoded explicitly through `*_measured` indicator features, so
that median imputation of the value does not silently create a measurement
indistinguishable from an observed one.

### 4.2 Validation protocol

**Variant grouping.** 2,643 variants generate 4,966 records, with recurrent
variants contributing up to 104 records each. Under random partitioning a model
can memorise a variant during training and be scored on the same variant at
test — the patient-level form of the identifier leak in §2. **Every partition in
this study is grouped by `mut_id`** using `StratifiedGroupKFold` and
`GroupShuffleSplit`.

**Study grouping.** As a stricter test of transportability, we repeat validation
grouping by source publication. Reporting centres differ in inhibitor screening
practice and in which patients they publish, so study-grouped performance
estimates behaviour in a centre not represented in training.

**No resampling.** Class imbalance is addressed with class weighting
(`class_weight='balanced'`, `scale_pos_weight`) exclusively. No over-sampling,
under-sampling or synthetic minority generation is applied at any stage; the
`imbalanced-learn` package is not a dependency.

**Primary estimand.** Because a single 20% held-out partition contains
approximately 159 positive records and carries roughly ±0.04 of sampling noise,
the primary estimate is **5-fold variant-grouped cross-validation repeated over
three seeds on the entire labelled cohort**. A single grouped held-out
partition, scored once after model and thresholds were fixed, serves as an
independent check.

### 4.3 Models

Twelve families were screened: L2 and elastic-net logistic regression, random
forest, extremely randomised trees, gradient boosting, histogram gradient
boosting, XGBoost [R5], LightGBM [R6], RBF support vector machine, k-nearest
neighbours, and a multilayer perceptron. Leading families were tuned by
randomised search (25 draws) scored on precomputed grouped folds.

**Ensembling.** `scikit-learn`'s `StackingClassifier` cannot be used in this
setting: it generates internal cross-validation folds at random with no
mechanism to propagate grouping, placing the same variant on both sides of the
meta-learner's training boundary. In our screen this produced an *inverted*
model (AUC 0.241). We therefore construct the out-of-fold matrix manually from
grouped folds and compare probability averaging, rank averaging, logistic
stacking, and greedy forward selection with replacement [R7].

Members are combined by **rank average** rather than probability average,
because members are calibrated on different scales. Ranks are computed against
the training score distribution stored at fit time, not within the scoring
batch, so an individual patient's estimate is independent of which other
patients are scored alongside them.

### 4.4 Calibration

Probabilities are calibrated by isotonic regression [R8] fitted on training
out-of-fold scores only.

Unmodified isotonic regression returned exactly 1.0 for the highest-scoring
inputs, because its terminal step covered a small, entirely positive training
bin. A deployed tool reporting certainty of inhibitor development is not
supportable: the highest-risk observed stratum (severe patients with a large
deletion) develops inhibitors at 53.1%, and the highest calibration decile at
59.6%.

We therefore constrain the calibrator. Training scores are partitioned into
deciles; each decile's positive rate is Laplace-smoothed as (k+1)/(n+2); and
predictions are clipped to the interval spanned by those smoothed rates —
**[0.025, 0.520]** in this cohort. The transformation is monotone, so ranking
and hence AUC are unchanged; only the reported magnitude is constrained to what
the cohort supports.

### 4.5 Statistical analysis

Discrimination is reported as AUC-ROC and AUC-PR against the prevalence
baseline. Uncertainty is estimated by stratified bootstrap (2,000 resamples).
Paired model comparison uses DeLong's test [R9]. Calibration is summarised by
the Brier score and expected calibration error. Clinical utility is assessed by
decision-curve net benefit [R10]. Operating thresholds are selected on training
folds by Youden's J [R11], by accuracy maximisation, and at fixed sensitivity
targets.

---

## 5. Results

### 5.1 Model screen

**Table 2.** Twelve families under 5-fold variant-grouped cross-validation on
the training partition (n = 3,992).

| Model | AUC-ROC | AUC-PR | Balanced acc. | Brier |
|---|---|---|---|---|
| Extremely randomised trees | 0.7778 | 0.4568 | 0.7183 | 0.1514 |
| Random forest | 0.7774 | 0.4615 | 0.7176 | 0.1363 |
| LightGBM | 0.7745 | 0.4773 | 0.7065 | 0.1420 |
| XGBoost | 0.7735 | 0.4678 | 0.7110 | 0.1559 |
| Gradient boosting | 0.7705 | 0.4769 | 0.7120 | 0.1165 |
| Histogram gradient boosting | 0.7682 | 0.4592 | 0.7010 | 0.1479 |
| Multilayer perceptron | 0.7622 | 0.4361 | 0.7105 | 0.1202 |
| Elastic-net logistic | 0.7572 | 0.4409 | 0.7002 | 0.1883 |
| L2 logistic | 0.7552 | 0.4372 | 0.6995 | 0.1887 |
| SVM (RBF) | 0.7551 | 0.3997 | 0.7018 | 0.1214 |
| k-nearest neighbours | 0.7507 | 0.4078 | 0.6907 | 0.1244 |
| *Stacking (ungrouped internal CV)* | *0.2412* | *0.1108* | *0.5019* | *0.3492* |

**The eleven valid families span 0.027 AUC.** Model capacity is not the binding
constraint on this problem — a finding that recurs throughout.

Tuning improved the leading families to 0.781–0.784. Greedy forward selection
retained three members (XGBoost, extremely randomised trees, LightGBM) at
out-of-fold AUC 0.7859; rank and mean averaging over all five tuned members gave
0.7860, and the three-member ensemble was retained for inference economy.

### 5.2 Feature ablation

**Table 3.** Contribution of each feature block, repeated variant-grouped CV
over three seeds.

| Rung | k | AUC-ROC | AUC-PR | Δ |
|---|---|---|---|---|
| Genotype class only | 21 | 0.7024 ± 0.0013 | 0.3313 | — |
| + FVIII domain | 37 | 0.7261 ± 0.0033 | 0.4154 | +0.0237 |
| + protein position | 50 | 0.7372 ± 0.0012 | 0.4241 | +0.0111 |
| **+ residue chemistry (all genomic)** | **68** | **0.7417 ± 0.0029** | **0.4368** | **+0.0045** |
| + patient baseline phenotype | 86 | 0.7645 ± 0.0011 | 0.4616 | +0.0228 |
| + reporting region | 95 | 0.7779 ± 0.0025 | 0.4844 | +0.0134 |

Two findings follow.

**The genomic rung reaches 0.7417.** This reproduces the ceiling previously
established on CHAMP — a registry curated by a different consortium, from
different patients, through an independently written feature pipeline. Two
registries agreeing on where genotype-only information is exhausted is
substantially stronger evidence than either alone, and reframes the ceiling from
a limitation of one analysis into a property of this class of data.

**Patient baseline phenotype contributes +0.023 AUC**, confirming the hypothesis
that motivated this study.

### 5.3 Leakage controls

**Table 4.** Controls bounding the primary result.

| Control | Result | Interpretation |
|---|---|---|
| Permuted labels | **0.5042** | design matrix carries no row identity (prior CHAMP pipeline: 1.000) |
| Ungrouped random CV | 0.8098 | — |
| Variant-grouped CV | 0.7779 | **grouping forgoes 0.0319 AUC** |
| Study-grouped CV | 0.7797 ± 0.0004 | performance retained in unseen reporting centres |
| Test rows byte-identical to a training row | 1.03% | over-sampling before splitting yields ≈50% |
| Strongest single feature–outcome correlation | \|r\| = 0.247 | no feature approximates the label |

The 0.032 AUC forgone by grouping is of the same magnitude as the entire
clinical-layer gain (0.023), illustrating how readily a partitioning choice can
be mistaken for a modelling advance.

### 5.4 Primary result and operating characteristics

**Repeated variant-grouped cross-validation, full labelled cohort (n = 4,966):
AUC-ROC 0.7861 ± 0.0040** (per-seed 0.7870, 0.7808, 0.7905).

Held-out partition (n = 974, prevalence 16.3%): AUC-ROC 0.7451 (95% CI
0.6927–0.7890); AUC-PR 0.4552 (95% CI 0.3883–0.5295) against a no-skill
baseline of 0.1632; Brier 0.114; expected calibration error 0.033. The
cross-validated estimate lies within the held-out confidence interval.

**Table 5.** Operating characteristics on the held-out partition. Majority-class
baseline accuracy is **83.68%**.

| Operating point | Threshold | Accuracy | Sens. | Spec. | PPV | NPV | MCC | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy-maximising | 0.474 | **86.65%** | 25.8% | 98.5% | 77.4% | 87.2% | 0.396 | 41 | 12 | 118 | 803 |
| Balanced (Youden) | 0.181 | 76.69% | 61.6% | 79.6% | 37.1% | 91.4% | 0.343 | 98 | 166 | 61 | 649 |
| Rule-out (80% sens. target) | 0.131 | 65.61% | 74.2% | 63.9% | 28.6% | 92.7% | 0.285 | 118 | 294 | 41 | 521 |
| Rule-out (90% sens. target) | 0.083 | 49.38% | 79.9% | 43.4% | 21.6% | 91.7% | 0.176 | 127 | 461 | 32 | 354 |

**Accuracy is reported only alongside its baseline.** The accuracy-maximising
threshold exceeds the no-skill baseline by 2.97 percentage points while
identifying 25.8% of inhibitor-positive patients; it is the least clinically
useful column in Table 5, and the one that an evaluation rubric specifying
"high accuracy" would select. The tool is specified at the balanced and rule-out
points, where negative predictive value exceeds 0.91.

Decision-curve net benefit was 0.088 at a 10% threshold probability (treat-all:
0.070) and 0.058 at 20% (treat-all: −0.046). The model exceeded both treat-all
and treat-none strategies for threshold probabilities between **0.09 and 0.52**,
covering 73% of the swept range. Below a 9% threshold — that is, for a clinician
willing to intensify monitoring for a patient at under one-in-eleven risk —
monitoring every patient is the preferable strategy, and the model offers no
advantage.

Against a genomic-only model on the same held-out patients (AUC 0.7305), the
full model's advantage of 0.0147 did not reach significance (DeLong z = 1.06,
p = 0.29), which is expected at this sample size; the ablation in Table 3,
estimated over the full cohort with repeated resampling, is the better-powered
comparison.

### 5.5 Cross-registry validation, and a registry-overlap artefact

Projecting both registries into a 57-feature intersection, a model trained on
HADB and applied to CHAMP without refitting scored **AUC 0.879** — exceeding
CHAMP's own within-registry cross-validation (0.725).

**A model applied to an unseen cohort should not outperform that cohort's own
cross-validation.** We treated this as a diagnostic rather than a result.

CHAMP and EAHAD are both curated from the published literature, and they curate
substantially the same publications. Matching on (mutation class, mature
residue, reference residue, substituted residue), **1,259 of 1,951 (64.5%) of
CHAMP's substitution-resolvable labelled variants are already present in HADB**.

One methodological subtlety governs this comparison. The matching key requires a
reference residue and position, which frameshift and large structural variants
do not provide; left uncorrected, every frameshift is classified as "novel" by
default, and the resulting comparison contrasts a 77%-frameshift subset with a
60%-missense subset — measuring mutation-class composition rather than novelty.
Both strata are therefore restricted to substitution-like variants.

**Table 6.** Transfer stratified by prior presence in the training registry.

| Stratum | n | Prevalence | AUC-ROC |
|---|---|---|---|
| CHAMP variants present in HADB | 1,259 | 0.137 | 0.9361 |
| **CHAMP variants novel to HADB** | **692** | **0.240** | **0.8513** (0.8193–0.8807) |
| Attributable contamination | | | **0.0848** |

On the identical 692 novel records, CHAMP's own cross-validated model achieved
AUC **0.6237**. Training on 4,966 patient-level European records therefore
predicted novel North American variants **0.2276 AUC better** than CHAMP
predicted them from its own data — independent confirmation, by a different
route, of the ablation's conclusion that repeated patient-level observation is
richer supervision than a variant catalogue.

### 5.6 Reproduction of the relabelling effect

Holding the fitted model and the variant set constant and altering only the
label convention:

| Label convention | n | Prevalence | Majority baseline | Accuracy | AUC |
|---|---|---|---|---|---|
| Recorded outcomes only | 2,296 | 0.201 | 79.9% | 80.1% | 0.879 |
| Unrecorded → negative | 4,040 | 0.114 | **88.6%** | 74.7% | 0.855 |

Discrimination does not improve under relabelling — AUC declines. What changes
is the accuracy obtainable without skill, which rises from 79.9% to 88.6%. A
headline accuracy near 97% is measured on a set whose no-skill baseline is
already 88.6%, and is approached by tuning the decision threshold toward the
enlarged majority class.

### 5.7 Results contrary to hypothesis

We report two experiments whose outcomes contradicted our expectations.

**Variant-level aggregation did not degrade performance.** We predicted that
collapsing records to one row per variant would lose information. It scored
*higher* (0.8038 versus 0.7766). The two estimates are not comparable — they
differ in unit of analysis, label definition and sample size — and predicting a
variant's *modal* outcome is an easier problem than predicting an individual's,
since feature averaging across a variant's records cancels measurement noise
and well-characterised recurrent variants dominate the row set.

The case for patient-level modelling therefore rests on a different measurement.
Among the 537 variants with two or more recorded outcomes, **124 (23.1%) are
discordant** — patients carrying an identical variant who differ in inhibitor
outcome — spanning 1,297 records. Aggregation assigns every such patient the
same prediction and is necessarily wrong for the minority, and it cannot use
per-patient factor activity at all.

**Naive cross-registry transfer overstated external validity**, as detailed in
§5.5.

### 5.8 Feature attribution

Permutation importance on the held-out partition ranked severity ordinal
(0.0084 AUC), East Asian reporting region (0.0073), nucleotide position
(0.0046), log FVIII activity (0.0044) and multi-domain involvement (0.0043)
highest. Aggregated by block: clinical 0.0287, position 0.0211, context 0.0148,
chemistry 0.0116, domain 0.0088, genotype 0.0042.

---

## 6. Discussion

### 6.1 What limits performance

The evidence converges on a single conclusion: **the constraint is
informational, not architectural.** Eleven model families span 0.027 AUC;
tuning contributed 0.005 and ensembling a further 0.003. By comparison, the partitioning
leak we declined to exploit was worth 0.032, and a single outcome-derived
column was worth 0.189.

The genomic ceiling of ≈0.742, now observed in two independently curated
registries, is best interpreted as the information content of *F8* genotype with
respect to inhibitor development. Established determinants that no public
registry records — treatment intensity, exposure days at first bleed, product
class (plasma-derived versus recombinant, cf. [R12]), surgery or infection at
first exposure, and HLA class II typing — are absent by construction. Their
absence, not model capacity, bounds these estimates.

### 6.2 Direction of effect for clinical features

Baseline factor activity could in principle be a *consequence* of the outcome
rather than a predictor of it, since a circulating inhibitor suppresses measured
FVIII activity. The registry documents these as diagnostic baseline values but
does not timestamp them relative to inhibitor detection, so this concern cannot
be fully resolved with these data.

We therefore report the genomic-only rung alongside every headline. **A reader
who rejects the clinical block entirely retains a defensible 0.742 model
validated across two registries.** The clinical contribution is presented as an
improvement conditional on that assumption, not as a foundation.

### 6.3 The reporting-region block

The context block contributes +0.0134 AUC, and East Asian reporting region ranks
second in permutation importance. This warrants explicit caution. The registry
records the *reporting laboratory's* country, not patient ancestry. Observed
regional inhibitor rates vary widely (East Asia 45.5%, n = 308; South Asia
10.8%, n = 297; North America 90.9%, n = 11), and the extreme values rest on
very small strata.

Ancestry is a genuine, established risk modifier for inhibitor development, so
part of this signal may be biological. But it is confounded with centre-level
screening intensity and publication selection, and we cannot separate the two
here. A deployment unwilling to accept that confound can use the model without
the context block at AUC 0.7645, and we recommend this for any use outside the
regions represented in training.

### 6.4 An anomaly in CRM typing

CRM type did not behave as the underlying immunology predicts. CRM-negative
(type I) patients — those with no circulating antigen — showed a *lower*
inhibitor rate (3.8%, n = 156) than CRM-unclassified patients (22.6%, n = 953).
We attribute this to ascertainment: CRM typing is performed selectively, largely
on mild and moderate patients with discrepant one-stage and chromogenic assays,
who are at low baseline inhibitor risk. This is a selection effect in the
registry rather than a contradiction of the immunology, but it means CRM type
should not be interpreted causally in this cohort.

### 6.5 Implications for cross-registry validation

The 64.5% overlap between CHAMP and EAHAD has consequences beyond this study.
Cross-registry transfer is widely regarded as strong evidence of external
validity, and in this domain a naive application of it inflated our own estimate
by 0.085 AUC. Investigators validating *F8* — and plausibly other curated
disease-variant resources compiled from shared literature — should establish
variant-level disjointness before reporting external performance, and should
report the matching procedure, since key resolvability interacts with mutation
class in a way that can silently confound the comparison.

### 6.6 Clinical positioning

The model is a research tool and is not a validated medical device. It does not
replace laboratory inhibitor testing or clinical judgement. Its intended use is
prioritisation of monitoring intensity during the first fifty exposure days.

At the balanced operating point it identifies 61.6% of patients who go on to
develop inhibitors, at a cost of 166 false positives per 974 patients, with NPV
0.914. The clinical consequence of a false positive is intensified monitoring,
not a change in diagnosis or a withheld therapy, which is the asymmetry that
makes a rule-out threshold defensible at this level of discrimination.

---

## 7. Limitations

1. **Registry ascertainment.** Both cohorts are compiled from published reports
   rather than consecutive series. Patients with inhibitors are plausibly more
   likely to be published, and screening intensity varies by centre.
2. **Unresolved temporality** of baseline factor measurements relative to
   inhibitor detection (§6.2).
3. **Absent causal covariates** — treatment intensity, exposure days, product
   class, HLA typing (§6.1).
4. **Reporting region as ancestry proxy** is confounded (§6.3).
5. **Held-out partition is small** (n = 974, 159 positives), giving wide
   intervals; the repeated cross-validated estimate is the primary result for
   this reason.
6. **Single-gene, single-disease scope.** Transfer to haemophilia B (*F9*) was
   not attempted here.
7. **No prospective validation.** All estimates are retrospective and
   registry-based.

---

## 8. Conclusion

Genotype-only prediction of FVIII inhibitor risk saturates near AUC 0.74, a
value we observe independently in two separately curated registries and which
is therefore best understood as a property of the available data rather than of
model capacity. Patient-level records carrying individual baseline phenotype
raise achievable discrimination to 0.786 ± 0.004 under strict variant-grouped
validation, with calibrated probabilities and retained performance in unseen
reporting centres.

Published accuracies above 95% do not represent predictive performance. Each
contributing mechanism — outcome-derived features, relabelling of unrecorded
outcomes, and resampling before partitioning — is reproducible on demand and
quantified here.

Progress beyond approximately 0.79 will not come from better classifiers. It
requires prospective cohorts recording treatment intensity, exposure history,
product class and HLA type. Until such data are available, the honest
contribution of genotype-based modelling is a calibrated rule-out instrument,
not a decisive diagnostic.

---

## Data and code availability

All code, trained artefacts, result files and the executed analysis notebook are
available at `https://github.com/dchaudhari7177/hemophilia`.

The EAHAD/HADB supplementary tables are distributed with the source publication
[R13]. The CHAMP variant list is distributed by the US Centers for Disease
Control and Prevention [R2].

Every result in this paper is regenerated by the scripts in `.devtools/` and
serialised to `reports/*.json`. The analysis notebook executes eight integrity
assertions — including a direct test for the over-sampling signature — and fails
if any does not hold. The test suite comprises 128 tests.

## Author contributions

To be completed by the authors.

## Conflicts of interest

None declared.

---

## References

> **Verification required.** Entries marked ⚠ must be completed against the
> primary source before submission. They are described rather than guessed to
> avoid propagating an incorrect citation.

- **[R1]** ⚠ Singh & Singh (2025). Random-forest classification of *F8*
  variants for inhibitor prediction; reports 97.37% accuracy. *Complete journal,
  volume and DOI from the stage-1 bibliography in `RESULTS.md`.*
- **[R2]** Payne AB, Miller CH, Kelly FM, Soucie JM, Craig Hooper W. The CDC
  Hemophilia A Mutation Project (CHAMP) mutation list: a new online resource.
  *Human Mutation*. 2013;34(2):E2382–91.
- **[R3]** Grantham R. Amino acid difference formula to help explain protein
  evolution. *Science*. 1974;185(4154):862–4.
- **[R4]** Henikoff S, Henikoff JG. Amino acid substitution matrices from
  protein blocks. *PNAS*. 1992;89(22):10915–9.
- **[R5]** Chen T, Guestrin C. XGBoost: a scalable tree boosting system.
  *KDD '16*. 2016:785–94.
- **[R6]** Ke G, Meng Q, Finley T, et al. LightGBM: a highly efficient gradient
  boosting decision tree. *NeurIPS*. 2017.
- **[R7]** Caruana R, Niculescu-Mizil A, Crew G, Ksikes A. Ensemble selection
  from libraries of models. *ICML '04*. 2004.
- **[R8]** Zadrozny B, Elkan C. Transforming classifier scores into accurate
  multiclass probability estimates. *KDD '02*. 2002:694–9.
- **[R9]** DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under
  two or more correlated receiver operating characteristic curves: a
  nonparametric approach. *Biometrics*. 1988;44(3):837–45.
- **[R10]** Vickers AJ, Elkin EB. Decision curve analysis: a novel method for
  evaluating prediction models. *Medical Decision Making*. 2006;26(6):565–74.
- **[R11]** Youden WJ. Index for rating diagnostic tests. *Cancer*.
  1950;3(1):32–5.
- **[R12]** Peyvandi F, Mannucci PM, Garagiola I, et al. A randomized trial of
  factor VIII and neutralizing antibodies in hemophilia A. *New England Journal
  of Medicine*. 2016;374(21):2054–64.
- **[R13]** ⚠ EAHAD/HADB coagulation factor VIII variant resource.
  *Blood Advances* supplement VTH-2024-000215, 2024. *Complete author list,
  volume, pages and DOI from the source article.*
- **[R14]** ⚠ Gouw SC, van den Berg HM, Oldenburg J, et al. *F8* gene mutation
  type and inhibitor development in patients with severe hemophilia A:
  systematic review and meta-analysis. *Blood*. 2012;119(12):2922–34.
  *Verify page range.*
