# Results

**Explainable FVIII Inhibitor Risk Classification in Hemophilia A using F8
Genomic Variant Data**

PES University B.Tech Capstone · Project ID PW_GRS_01
Dipak Chaudhari · Tejas Nagmote · Sneha A · Varsha P
Guide: Prof. Gayathri R S

*Generated from `reports/*.json` on 2026-08-26. Every figure in
this document is read back out of a measurement artefact; none is transcribed.*

---

## 1. Executive summary

The headline results this project was asked to beat do not survive reproduction. Running the reference preprocessing verbatim on a clean stratified split reproduces neither 97.37% nor 99.63%; it produces a model that scores **below the majority-class baseline**. Section 2 shows precisely which three choices manufactured the published numbers.

What this project delivers instead is a model whose performance is real: it is built only from features a treatment centre could supply for a patient it has never seen, it is calibrated, its uncertainty is quantified, and it is validated on a completely separate cohort (hemophilia B, a different gene) that played no part in training.

**Headline numbers**

| Quantity | Value |
|---|---|
| Selected model | DeepMLP |
| Held-out test AUC-ROC (95% CI) | 0.7268 (0.6675–0.7847) |
| Held-out test AUC-PR (95% CI) | 0.4800 (0.3987–0.5709) |
| Prevalence (AUC-PR baseline) | 0.2000 |
| Sensitivity at Youden threshold | 64.13% |
| Specificity | 64.40% |
| Balanced accuracy | 64.27% |
| MCC | 0.2318 |
| Brier score (calibrated) | 0.1355 |
| Expected calibration error | 0.0431 |
| External AUC, CHBMP F9 (95% CI) | 0.7504 (0.6709–0.8237) |

## 2. Why the reference results do not reproduce

Each row below runs the reference preprocessing and changes exactly one thing. All use the same Random Forest the classical-ML reference reports as its best model.

| Experiment | Train AUC | Test AUC | Test accuracy | Majority-class accuracy |
|---|---|---|---|---|
| Reference pipeline, verbatim | 1.0000 | 0.6391 | 86.39% | 88.61% |
| Identifier columns only | 1.0000 | 0.5832 | 85.77% | 88.61% |
| Biology only, identifiers removed | 0.9254 | 0.6326 | 75.62% | 88.61% |
| Reference features, unrecorded outcomes dropped | 1.0000 | 0.7062 | 79.35% | 80.00% |
| Labels shuffled | 1.0000 | 0.4355 | 86.76% | 87.75% |
| Novel-variant (position-blocked) split | 1.0000 | 0.6467 | 88.24% | 87.50% |
| Over-sampling applied before the split | 1.0000 | 0.9938 | 95.25% | 50.00% |

### 2.1 Three separate defects

**(a) Over-sampling before the split.** The classical reference applies Random Over-Sampling to the whole dataset and only then runs stratified k-fold. Because over-sampling duplicates minority rows verbatim, **50.0% of the evaluation rows are byte-identical copies of training rows** (716 of 1432). Under that protocol the same Random Forest scores 95.25% accuracy and 0.9938 AUC — which is where the published 97.37% comes from. Under a clean split the identical model scores 86.39%.

**(b) Unrecorded outcomes relabelled as negative.** CHAMP records 1744 variants whose inhibitor status was never reported — 43% of the database. The reference maps them to 0. That drops apparent prevalence from 20.08% — which matches the 20–40% the reference's own introduction quotes — to 11.41%, and inflates accuracy by 7.04% purely by padding the majority class.

**(c) Identifier columns used as features.** `HGVS cDNA` takes 4,038 distinct values across 4,050 rows, and among the 2,296 *labelled* rows it has **no duplicates at all** — it is a row index. Label-encoding it hands the model a lookup key. The signature is visible in every row of the table above: training AUC pinned at 1.0000 while test AUC sits near chance. The permutation control makes it unambiguous — with the labels shuffled, training AUC stays at 1.0000 while test AUC falls to 0.4355. A model that fits noise perfectly is memorising, not learning.

Correcting only the label handling, and keeping every other reference choice, moves test AUC to 0.7062. That is the honest starting point this project builds on.

## 3. What replaced the identifier columns

The HGVS string is discarded, but what it *means* is kept. The parser turns each variant into mechanistic descriptors grouped into seven biological blocks:

| Block | What it encodes |
|---|---|
| `consequence` | missense / nonsense / frameshift / splice / structural class, null-mutation flag, event span |
| `position` | FVIII domain, heavy vs light chain, B-domain membership, distance to each known inhibitor epitope, exon geometry, hotspot density |
| `truncation` | premature stop position, fraction of protein lost, NMD escape, which domains are removed |
| `chemistry` | Grantham distance, BLOSUM62, changes in hydropathy, volume, charge and polarity |
| `nucleotide` | transition vs transversion, CpG signature, reference and alternate base, frame preservation |
| `splicing` | intronic offset, canonical vs extended splice site, donor vs acceptor side |
| `clinical` | FVIII activity stratum, variable expressivity, poly-A context, null×severe interaction |

**Positional features are deliberately coarse.** Genomic position is biologically real but at full resolution it is near-unique, which reintroduces the identifier problem in numeric form. Positions are therefore snapped to a 40-bin grid (~58 residues per bin — finer than a FVIII domain, far coarser than one variant). The measured cost of doing this is **0.0000 AUC** (0.7372 → 0.7372). That the cost is zero is the cleanest available evidence that the fine resolution was carrying identity, not biology. A regression test now fails if any engineered feature becomes near-unique again.

### 3.1 What is the engineering actually worth?

Feature engineering is easy to justify after the fact, so it is worth measuring rather than describing. All rows below use the same ExtraTrees model and the same 5-fold protocol on the training split.

| Feature set | k | AUC-ROC |
|---|---|---|
| null-mutation flag alone | 1 | 0.6481 |
| variant type only | 9 | 0.6778 |
| clinical severity only | 8 | 0.6391 |
| variant type + severity | 17 | 0.6985 |
| all features | 135 | 0.7372 |

The two variables any clinician already has — variant type and FVIII activity stratum — reach 0.6985 on their own. The full engineered set adds **+0.0387 AUC** on top. That is a real but modest gain, and stating it that way is more useful than implying the mechanistic features carry the model.

**Leave-one-block-out.** Cost of removing each biological block from the full set (full-set AUC 0.7372):

| Block removed | Features dropped | AUC without | Cost |
|---|---|---|---|
| clinical | 16 | 0.7226 | +0.0145 |
| chemistry | 15 | 0.7283 | +0.0088 |
| nucleotide | 14 | 0.7326 | +0.0046 |
| splicing | 9 | 0.7344 | +0.0028 |
| position | 40 | 0.7346 | +0.0026 |
| consequence | 25 | 0.7356 | +0.0016 |
| truncation | 16 | 0.7358 | +0.0014 |

A block whose removal costs nothing is not contributing, however good the biological story behind it sounds. Those are reported here rather than quietly retained.

**Feature count.** With 110 top-ranked features the model reaches AUC 0.7373; the sweep across 10, 20, 30, 50, 80, 110, all features is in `reports/ablation.json`. Top-ranked features: `is_null_mutation`, `null_and_severe`, `vtype_missense`, `severity_mild`, `severity_severe`, `mech_deletion`, `severity_ordinal`, `cdna_pos_norm`.

## 4. Model comparison

RepeatedStratifiedKFold(5x3) on 1836 patients (369 inhibitor-positive), 135 features. Position-blocked cross-validation holds out contiguous stretches of F8, so it measures generalisation to a region of the gene the model has never seen — the situation when a novel mutation is found.

| Model | CV AUC-ROC | CV AUC-PR | MCC | Position-blocked AUC |
|---|---|---|---|---|
| ExtraTrees | 0.7394 ± 0.0298 | 0.4706 | 0.2982 | 0.7009 ± 0.0305 |
| StackedEnsemble | 0.7355 ± 0.0309 | 0.4561 | 0.3055 | — |
| RandomForest | 0.7347 ± 0.0308 | 0.4555 | 0.2976 | 0.7057 ± 0.0314 |
| DeepMLP | 0.7346 ± 0.0314 | 0.4392 | 0.3007 | 0.7167 ± 0.0361 |
| WeightedAverageEnsemble | 0.7331 ± 0.0347 | 0.4521 | 0.3026 | — |
| BioBlockAttention | 0.7266 ± 0.0347 | 0.4413 | 0.2862 | 0.7094 ± 0.0374 |
| TabTransformer | 0.7250 ± 0.0288 | 0.4210 | 0.2857 | 0.7135 ± 0.0284 |
| ElasticNetLR | 0.7213 ± 0.0356 | 0.4164 | 0.2872 | 0.6967 ± 0.0318 |
| CNN1D | 0.7192 ± 0.0228 | 0.3885 | 0.2875 | 0.7145 ± 0.0375 |
| XGBoost | 0.7161 ± 0.0365 | 0.4198 | 0.2739 | 0.6854 ± 0.0399 |
| GradientBoosting | 0.7155 ± 0.0344 | 0.4169 | 0.2886 | 0.6928 ± 0.0352 |
| LogisticRegression | 0.7113 ± 0.0378 | 0.4000 | 0.2844 | 0.6919 ± 0.0280 |
| LightGBM | 0.7085 ± 0.0361 | 0.4146 | 0.2743 | 0.6759 ± 0.0361 |
| ResidualMLP | 0.6918 ± 0.0341 | 0.3986 | 0.2578 | 0.6724 ± 0.0446 |

AUC-PR baseline (prevalence) is 0.201.

### 4.1 Nested hyperparameter search

The reference works tune with `GridSearchCV` and report the best cross-validated score. That score *can* be optimistically biased, because the folds that chose the hyperparameters also graded them. Running the search inside an outer loop it never sees measures the size of that bias directly rather than assuming it.

| Model | Nested (honest) AUC | Inner best AUC | Tuning optimism |
|---|---|---|---|
| LogisticRegression | 0.7361 ± 0.0351 | 0.7320 | -0.0041 |
| ExtraTrees | 0.7357 ± 0.0316 | 0.7313 | -0.0044 |
| LightGBM | 0.7285 ± 0.0287 | 0.7238 | -0.0047 |
| RandomForest | 0.7258 ± 0.0306 | 0.7281 | +0.0023 |
| XGBoost | 0.7246 ± 0.0303 | 0.7264 | +0.0018 |

**This experiment did not find what it was set up to find, and that is reported rather than dropped.** Tuning optimism is negligible here: the largest gap in either direction is 0.0047 AUC, and three of the five models score *higher* on the honest outer loop than on the inner one. With a search space this small relative to 1,836 training rows, the inner-loop estimate is a fair one. So this particular criticism does not apply to the reference works — their `GridSearchCV` scores are not inflated by the tuning itself. The defects documented in §2 are quite sufficient on their own without adding one the data does not support.

A second observation from the same table: tuned logistic regression reaches the top of this list. On 369 events a penalised linear model is genuinely competitive with everything more elaborate.

### 4.2 Which differences are real?

The spread from best to worst in the table above is about 0.035 AUC and the fold-to-fold standard deviation is about 0.03. A ranking alone would therefore invite a claim the data cannot support. Each model below is tested against **ExtraTrees** by DeLong's test on the shared out-of-fold predictions.

| Model | Pooled OOF AUC | Δ vs best | p | Verdict |
|---|---|---|---|---|
| ExtraTrees | 0.7412 | 0.0000 | 1.0000 | -- |
| RandomForest | 0.7372 | +0.0040 | 0.3014 | indistinguishable |
| StackedEnsemble | 0.7370 | +0.0042 | 0.1391 | indistinguishable |
| WeightedAverageEnsemble | 0.7357 | +0.0054 | 0.1888 | indistinguishable |
| DeepMLP | 0.7338 | +0.0074 | 0.3194 | indistinguishable |
| BioBlockAttention | 0.7287 | +0.0125 | 0.0980 | indistinguishable |
| ElasticNetLR | 0.7249 | +0.0162 | 0.0515 | indistinguishable |
| TabTransformer | 0.7249 | +0.0162 | 0.0257 | significantly worse |
| GradientBoosting | 0.7211 | +0.0201 | 0.0077 | significantly worse |
| CNN1D | 0.7198 | +0.0214 | 0.0141 | significantly worse |
| XGBoost | 0.7183 | +0.0229 | 0.0022 | significantly worse |
| LogisticRegression | 0.7149 | +0.0262 | 0.0053 | significantly worse |
| LightGBM | 0.7138 | +0.0274 | 0.0004 | significantly worse |
| ResidualMLP | 0.7061 | +0.0351 | 0.0004 | significantly worse |

**6 of 13 competing models cannot be separated from ExtraTrees at p<0.05 on these out-of-fold predictions. Selecting on the third decimal place of AUC would be selecting on noise.**

The result is two tiers rather than one winner. A top group — bagged forests, both ensembles, the deep MLP, the block-attention network and penalised logistic regression — cannot be told apart. Below it sits a group that genuinely is worse, and it is worth noting what is in that group: the boosted models and three of the four reference deep architectures, with the deepest of them (ResidualMLP) last. On 369 events, capacity is not the binding constraint and adding it costs rather than pays.

### 4.3 Which model gets shipped, and why

**Rule, fixed before looking at the answer:** among models DeLong cannot separate from the best on repeated CV (p >= 0.05), ship the highest position-blocked AUC.

Repeated CV and position-blocked CV disagree, and the disagreement is informative. A random split can put residue 490 in training and residue 491 in test — neighbouring positions in the same epitope, which is closer to interpolation than prediction. Blocking removes that, and it is the situation a treatment centre is actually in when a newly sequenced patient carries an uncatalogued variant.

| Model in the statistically-tied tier | Repeated-CV AUC | Blocked AUC |
|---|---|---|
| DeepMLP | — | 0.7167 |
| BioBlockAttention | — | 0.7094 |
| RandomForest | — | 0.7057 |
| ExtraTrees | — | 0.7009 |
| ElasticNetLR | — | 0.6967 |

Repeated CV ranks ExtraTrees first, but the difference from DeepMLP is not statistically significant (p = 0.319401). Under the blocked protocol -- generalising to a stretch of F8 never seen in training, which is the clinical case -- DeepMLP scores 0.7167 against 0.7009.

The rule therefore ships **DeepMLP** rather than ExtraTrees, which had the higher headline AUC. Choosing the other way round would have meant picking the protocol that flattered the number — the kind of choice §2 of this report exists to call out.

## 5. Final model on the held-out test set

`DeepMLP`, isotonic-calibrated, trained on 1836 patients and evaluated once on 460 held-out patients (92 events). Both decision thresholds were chosen on out-of-fold predictions from the training set; the test set was not used to select anything.

| Metric | Youden threshold | 90%-sensitivity threshold |
|---|---|---|
| Threshold | 0.2034 | 0.1055 |
| AUC-ROC | 0.7268 | 0.7268 |
| AUC-PR | 0.4800 | 0.4800 |
| Sensitivity | 64.13% | 86.96% |
| Specificity | 64.40% | 38.59% |
| Precision (PPV) | 31.05% | 26.14% |
| NPV | 87.78% | 92.21% |
| Balanced accuracy | 64.27% | 62.77% |
| F1 | 0.4184 | 0.4020 |
| MCC | 0.2318 | 0.2165 |
| Brier | 0.1355 | 0.1355 |
| Calibration error | 0.0431 | 0.0431 |
| Net benefit @ 20% | 0.0560 | 0.0560 |
| Confusion (TP/FP/FN/TN) | 59/131/33/237 | 80/226/12/142 |

### 5.1 Calibration

A risk score used to decide whether to start inhibitor-aware prophylaxis has to mean what it says: among patients scored at 30%, about 30% should develop inhibitors. Neither reference work reports this. Isotonic calibration moves the Brier score from 0.2539 to 0.1355 and the expected calibration error from 0.2719 to 0.0431.

The decision curve in `reports/figures/03_performance_panel.png` shows where the model beats both default strategies (test everyone / test no one) in net benefit — the range of clinical thresholds over which using it is better than not using it.

### 5.2 Accuracy, and why it is reported with a baseline

Accuracy is the metric review panels usually ask for, so it is reported here — next to the number a model gets for never predicting an inhibitor at all. On a 20.00%-prevalence outcome the second figure is not a formality: it is most of the first one.

| Operating point | Accuracy | Sensitivity | Specificity | Cases caught / missed |
|---|---|---|---|---|
| Balanced (Youden's J) | 64.35% | 64.13% | 64.40% | 59 / 33 |
| High sensitivity (90%) | 48.26% | 86.96% | 38.59% | 80 / 12 |
| Accuracy-maximising | 83.04% | 17.39% | 99.46% | 16 / 76 |
| *Predict "no inhibitor" for everyone* | *80.00%* | *0.00%* | *100.00%* | *0 / 92* |

The accuracy-maximising operating point reaches **83.04%** against a no-skill baseline of **80.00%** — a margin of **+0.0304**. It gets there by declining to predict inhibitors: it catches 16 of 92 cases. That is the arithmetic of an imbalanced outcome, not a property of this particular model, and it is why the tool ships on the balanced and high-sensitivity points instead.

Two consequences worth stating plainly. **No threshold anywhere on the curve reaches 85% accuracy** — the maximum is the figure above. And a version of this label that counts unrecorded outcomes as negatives lifts the no-skill baseline to 88.6%, at which point an accuracy target in the high eighties is met by a model that does nothing. Section 10 works through a dataset where exactly that happens.

The metrics that cannot be gamed this way — AUC-ROC, AUC-PR against prevalence, balanced accuracy, MCC — are the ones this project leads with, and they are in the table above this one.

## 6. Does it work where it would be used?

Inhibitor prophylaxis decisions are made almost entirely in **severe** hemophilia A. A model with a respectable overall AUC that sits at chance inside the severe stratum would be useless in clinic, and the overall number would never reveal it. Neither reference work reports subgroup performance.

| Subgroup | n | Events | Prevalence | AUC-ROC (95% CI) | Sens. | Spec. |
|---|---|---|---|---|---|---|
| All patients | 460 | 92 | 20.00% | 0.727 (0.662-0.784) | 64.13% | 64.13% |
| Severe phenotype | 286 | 74 | 25.87% | 0.694 (0.623-0.760) | 78.38% | 41.04% |
| Moderate phenotype | 59 | 5 | 8.47% | nan (None) | nan% | nan% |
| Mild phenotype | 89 | 9 | 10.11% | nan (None) | nan% | nan% |
| Null variants | 243 | 72 | 29.63% | 0.652 (0.573-0.726) | 81.94% | 22.81% |
| Non-null variants | 217 | 20 | 9.22% | 0.616 (0.480-0.736) | — | 100.00% |
| Missense only | 210 | 19 | 9.05% | 0.656 (0.537-0.762) | — | 100.00% |
| Truncating only | 177 | 49 | 27.68% | 0.541 (0.438-0.643) | 75.51% | 22.66% |
| Large structural | 31 | 19 | 61.29% | 0.882 (0.732-0.983) | 100.00% | — |
| Light chain | 157 | 33 | 21.02% | 0.693 (0.592-0.789) | 57.58% | 71.77% |
| Heavy chain | 235 | 37 | 15.74% | 0.656 (0.559-0.738) | 51.35% | 67.68% |

*Strata with fewer than 10 events in either class are listed with their counts but no AUC, because an estimate from that few events would not be stable.*

### 6.1 The most important caveat in this report

The overall AUC of 0.727 is not evenly distributed. Inside the **severe** stratum — where essentially every prophylaxis decision is actually made — it falls to 0.694. Inside **null variants** it is 0.652. Inside **truncating variants alone** it is 0.541, which is indistinguishable from chance.

The reading is uncomfortable but clear: most of the model's apparent discrimination comes from separating null variants from non-null ones — and a clinician already knows that from the variant type without any model at all. Within the high-risk group, where a tool would actually add information, this model adds very little.

That is a limit of the data rather than of the fitting. Whether a particular severe, null-variant patient develops an inhibitor depends on treatment intensity, product type, age at first exposure and HLA haplotype — none of which CHAMP records. No model can recover from a database what was never in it, and a report that showed only the pooled 0.727 would have concealed exactly the thing a reviewer most needs to know.

The one stratum with strong discrimination is **large structural variants** (AUC 0.882), but on 31 patients with 19 events the interval is wide and it should be treated as a signal to follow up, not a result.

## 7. External validation: zero-shot transfer to hemophilia B

The F8 model is applied unchanged to 351 hemophilia **B** patients from the CDC CHBMP database (40 inhibitor-positive, 11.40% prevalence). F9 is a different gene coding a different protein, and no F9 patient took any part in training, feature fitting or threshold selection.

Nothing F8-specific can transfer. What can transfer is the underlying immunology: a null variant abolishes the protein, so the patient was never tolerised to the factor they are later infused with. A model that survives this transfer has learned that mechanism; a model that memorised F8 collapses to chance. No reference work attempts this test.

> **What this does and does not show.** The positional features are computed with FVIII coordinates. Factor IX is a 415-residue mature protein against FVIII's 2,332, so every F9 variant lands in the low-numbered bins and the domain, epitope-distance and truncation-extent features carry no real information here -- they are near-constant across the cohort and therefore contribute almost nothing to the ranking. What is being tested is whether the *consequence-class* signal (null vs missense, truncating vs in-frame, splice disruption) plus clinical severity transfers across genes. That is the intended test, and it is the part of the model that should be gene-agnostic; but the result should not be read as evidence that the FVIII structural features generalise.

| Metric | Value |
|---|---|
| AUC-ROC (95% CI) | 0.7504 (0.6709–0.8237) |
| AUC-PR | 0.4094 (baseline 0.1140) |
| Sensitivity | 87.50% |
| Specificity | 50.80% |
| Balanced accuracy | 69.15% |
| MCC | 0.2441 |

## 8. The 1,744 unrecorded outcomes

**Is the missingness informative?** A classifier trained to predict *whether* a variant's inhibitor status was recorded reaches AUC 0.6009. Interpretation: informative missingness -- the labelled subset is not a random sample of CHAMP and this limits external generalisation.

**What does relabelling them cost?** Scoring the unlabelled pool with the trained model gives a mean predicted risk of 0.3326 and flags 502 of 1744 rows as likely positive. Setting all of them to 0, as the reference does, therefore injects on the order of 502 false negatives straight into the training signal.

**Does using them help?** Self-training over the pool moves held-out AUC from 0.7504 to 0.7418 (DeLong p = 0.168153). The difference is not statistically significant, and is reported as such rather than claimed as an improvement.

## 9. Comparison with the reference works

Comparing raw accuracy across different label definitions and different splitting protocols is meaningless, so the table below states the protocol alongside every number.

| Work | Reported | Protocol | Reproduces? |
|---|---|---|---|
| Singh & Singh (2025), Random Forest | 97.37% accuracy | Random Over-Sampling applied before stratified k-fold | No — see §2 |
| Prior capstone notebook, Deep MLP v2 | 99.63% accuracy, AUC 0.9999 | all columns label-encoded; unrecorded outcomes set to 0 | No — see §2 |
| **This project (DeepMLP)** | AUC 0.7268, balanced accuracy 64.27% | single held-out test set, thresholds fixed on training folds, identifier columns excluded, unrecorded outcomes excluded | Yes — plus external cohort |

The like-for-like comparison is the one that matters: with the same honest labels and the same clean split, the reference's feature set reaches AUC 0.7062. This project's feature set and model reach 0.7268 on the same task.

## 10. A dataset that appeared to solve the problem

Section 12 says the binding constraint is the absence of patient-level covariates. A collaborator supplied `Final_Fused_Dataset.csv`: CHAMP with five of those covariates appended — age at diagnosis, ethnicity, treatment regimen, exposure days, family history — and it reports accuracy in the high eighties. It was tested properly rather than adopted.

### 9.1 Where the high accuracy comes from

The file's `Inhibitor_Status` column maps CHAMP's 1,731 unrecorded outcomes to 0 — the defect documented in §2. That moves prevalence from 20.1% to 11.4%, and with it the no-skill baseline:

| | Accuracy |
|---|---|
| Predict "no inhibitor" for every patient | **88.55%** |
| Trained model on the supplied label | **89.58%** |
| Margin over doing nothing | **+0.99 points** |
| Inhibitor cases actually caught | **13%** (16 flagged of 806) |

The 89.6% falls inside the range a rubric might ask for. It is also, in substance, what a model scores for learning to say "no".

### 9.2 The clinical columns are simulated

CHAMP rows are published *variants*, not patients — one row aggregates every case ever reported with that mutation — so there is no key on which per-patient clinical data could have been joined. Four independent checks agree the block was generated:

| Check | Finding |
|---|---|
| `Patient_ID` format | random UUID4 on 100% of rows |
| `Ethnicity` association | inhibitor rate flat across all five groups, spread 2.06 points, chi-square p = 0.955 |
| `Family_History` effect | odds ratio 3.12 against a published 3.0 |
| Age vs exposure days | r = 0.864 |

The ethnicity result is decisive. Roughly two-fold higher inhibitor risk in Black and Hispanic patients is among the most reproducible non-genetic findings in this field, replicated across CDC surveillance, MLOF and UKHCDO. A real cohort of 4,026 patients would show it. A column drawn from a fixed multinomial produces exactly the flat line observed.

### 9.3 Evaluated properly, they add nothing

Same folds, same held-out patients, same leakage-free genomic featuriser, honest labels throughout. The only difference between arms is whether the clinical block is present:

| Arm | Features | CV AUC | Held-out AUC (95% CI) |
|---|---|---|---|
| Genomic only | 135 | 0.7394 ± 0.0288 | 0.7432 (0.6834–0.8013) |
| Clinical only | 13 | 0.6076 ± 0.0319 | 0.6438 (0.5817–0.7052) |
| Genomic + clinical | 148 | 0.7506 ± 0.0310 | 0.7390 (0.6773–0.7982) |

Adding the clinical block changes held-out AUC by -0.0042 — DeLong p = 0.69114. Cross-validation AUC rises while held-out AUC falls, which is what fitting injected noise looks like.

So the dataset offers no real gain, and the accuracy it advertises is the baseline in disguise. Read the other way round it is still useful: it is a serviceable power analysis showing what *real* registry covariates would need to look like, and it supports the conclusion in section 12 that the ceiling here is data rather than method. Reported as a simulation, which is what it is.

## 11. Pipeline integrity checks

The claim that these numbers are trustworthy is worth no more than the checks behind it, so each property is verified mechanically and the result is written to `reports/integrity.json` rather than asserted in prose.

| Check | Result |
|---|---|
| no resampling of the training set | pass |
| imbalance handled by weighting, not resampling | pass |
| imputer and scaler live inside the CV pipeline | pass |
| featuriser never reads the outcome | pass |
| no feature behaves like a row identifier | pass |
| test set is scored, never fitted or tuned on | pass |
| 'Not reported' is never relabelled as 'no inhibitor' | pass |
| accuracy reported with its majority-class baseline | pass |

**8 of 8 passed.**

Two are worth spelling out. *No resampling*: class imbalance is handled by weighting the objective, never by duplicating or synthesising patients — the reference pipeline's Random Over-Sampling is what put half of its own test set into its training data. *Featuriser is label-blind*: scrambling the outcome and re-fitting produces a byte-identical feature matrix, so the engineering cannot have absorbed the answer.

## 12. Limitations

These are stated because a model for clinical use is only as trustworthy as its
declared boundaries.

1. **Most of the discrimination is null-versus-non-null, which is already
   known.** Section 6.1 is the limitation that matters most. Pooled AUC is
   0.727, but inside the severe stratum it is 0.694 and inside truncating
   variants alone it is 0.541 — chance. The model largely reproduces a
   distinction the variant type already gives a clinician for free, and adds
   little within the high-risk group where a tool would actually change
   management.
2. **CHAMP is a variant catalogue, not a patient registry.** Each row is a
   distinct mutation whose outcome is summarised across everyone reported to
   carry it. Individual patients are not resolvable, so the label carries
   irreducible noise and the unit of analysis is the variant.
3. **The strongest known risk factors are absent.** Treatment intensity,
   product type, exposure days, HLA haplotype, family history and ethnicity all
   drive inhibitor development and none are in the data. A genomic-only model
   has a ceiling well below what the reference works advertise, and the
   performance here should be read against that ceiling.
4. **Reporting bias.** CHAMP aggregates published case reports, which
   over-represent unusual variants and outcomes worth publishing.
5. **The external cohort is small.** CHBMP contributes 351 labelled patients
   with 40 events, so its confidence interval is wide. It establishes that
   transfer happens, not how well.
6. **Not a medical device.** Research and educational use only. It does not
   replace clinical judgement or laboratory inhibitor testing.

---

## 13. Reproducing this document

```bash
python scripts/fetch_data.py
python -m src.train --stage all
python -m src.figures
python -m src.report
python -m pytest
```

Figures land in `reports/figures/`, measurements in `reports/*.json`, and this
document is regenerated from them.
