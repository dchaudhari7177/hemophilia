# Architecture and design rationale

This document explains *why* the system is shaped the way it is. The numbers
live in [`../RESULTS.md`](../RESULTS.md); this is the reasoning behind them.

---

## 1. The clinical problem

A boy is diagnosed with severe hemophilia A. He will start prophylactic Factor
VIII infusions within months. Somewhere between 25% and 40% of patients like him
will develop neutralising antibodies — inhibitors — usually inside the first 50
exposure days. When that happens, standard prophylaxis stops working, annual
treatment cost rises from roughly $200,000 to over $1,000,000, and the patient
faces bypassing agents or immune tolerance induction.

Today inhibitor status is discovered **reactively**, by assay, after the
antibodies already exist. If risk could be estimated at the point of genetic
diagnosis, a high-risk child could be started on a different regimen — earlier
emicizumab, a modified product choice, tighter monitoring — before the immune
system has ever seen infused FVIII.

That is the decision this model is built to inform. It is not a diagnostic; it
is a triage signal that arrives months before the assay could.

## 2. Why the target is hard, and why that matters

Inhibitor development is multi-factorial. The F8 variant is one input. Treatment
intensity, product type, age at first exposure, surgery or bleeds during early
exposure, HLA haplotype, immune-regulatory polymorphisms, family history and
ethnicity are all established contributors — and **none of them are in CHAMP**.

This has a direct consequence for what "good" means. A genomics-only model has a
ceiling set by how much of the variance the mutation itself explains. Published
odds ratios for null versus missense variants sit around 3–5, which corresponds
to an AUC in the mid-0.7s, not the high 0.9s. A model on this data reporting
AUC 0.9999 is not a breakthrough; it is a bug.

That reasoning is what motivated the audit in §2 of the results, and it is worth
stating as a design principle: **when a result is far better than the biology
allows, the first hypothesis is leakage, not insight.**

## 3. Pipeline

```
CDC CHAMP (F8)                          CDC CHBMP (F9)
  4,040 variants                          1,399 variants
        |                                       |
        v                                       |
  datasets.py   tri-state labels                |
  Yes / No / Not-reported                       |
        |                                       |
   +----+-----------------+                     |
   |                      |                     |
 labelled (2,296)   unlabelled (1,744)          |
   |                      |                     |
   |                      v                     |
   |             semisupervised.py              |
   |             missingness probe              |
   |             self-training pool             |
   v                                            v
                    features.py
       hgvs_parser.py  ->  biology.py  ->  135 features
                   in 7 biological blocks
                            |
        +-------------------+-------------------+
        |                   |                   |
   models.py           tuning.py           evaluate.py
   classical zoo       nested search       bootstrap CIs
   4 reference DNNs    monotone priors     DeLong tests
   BioBlockAttention                       calibration
        |                                  net benefit
        v                                       |
   train.py  --stage {audit,cv,blocked,final,ssl,external}
        |
        +--> models/final_model.joblib --> predict.py --> app.py
        +--> reports/*.json ------------> figures.py
                                     \--> report.py --> RESULTS.md
```

## 4. The five design decisions that define this project

### 4.1 Discard identity, keep mechanism

`HGVS cDNA` takes 4,038 distinct values across 4,050 rows, and among the 2,296
labelled rows it has **no duplicates whatsoever**. Label-encoding it gives the
model a primary key.

The parser therefore throws the string away and keeps what it means: which
domain the residue sits in, how far it is from a known inhibitor epitope, how
much protein a stop codon removes, whether the transcript escapes
nonsense-mediated decay, how chemically drastic a substitution is, how close to
a splice junction the change falls.

Positional features are then **quantised to a 40-bin grid**, roughly 58 residues
per bin. Position is real biology, but at nucleotide resolution it is once again
near-unique, and a tree ensemble will split it down to the individual patient.
The measured cost of this coarsening is 0.0000 AUC — which is itself the
cleanest evidence available that the fine resolution was carrying identity
rather than signal. A regression test now fails if any feature becomes
near-unique again.

### 4.2 "Not reported" is not "no"

1,744 CHAMP rows — 43% of the database — have no recorded inhibitor outcome.
Mapping them to 0 asserts something the data does not say, and it pushes
apparent prevalence from 20.1% (which matches the published epidemiology) to
11.4%, inflating accuracy purely by padding the majority class.

They are kept as a third state. `ReportingBiasProbe` first checks whether their
missingness is informative; `SelfTrainingSSL` then uses their features under
capped, down-weighted pseudo-labels. They are never scored against.

### 4.3 Priors instead of more parameters

With 369 training events and 135 features, the binding constraint is data, not
model capacity. Two priors are supplied rather than learned:

- **Feature blocks.** The features are grouped into seven biological axes, and
  `BioBlockAttentionNet` encodes each block separately before a gated attention
  layer weights them. The network does not have to rediscover from 369 events
  that "residue chemistry" and "splice offset" are different kinds of evidence.
- **Monotone constraints.** A null variant cannot lower inhibitor risk relative
  to a missense; a severe phenotype cannot lower it relative to a mild one.
  These directions are pinned in the gradient-boosted model. It costs a little
  training fit and buys a model that cannot contradict established immunology —
  which is also what makes it defensible in front of a clinician.

### 4.4 Validate the way the model will be used

Three protocols, each answering a different question:

| Protocol | Question it answers |
|---|---|
| Repeated stratified CV | How well does it do on variants like the ones it trained on? |
| Position-blocked CV | How well does it do on a stretch of F8 it has never seen? |
| Zero-shot CHBMP (F9) | Has it learned mutation-class immunology, or just F8? |

The third is the strongest test in the project. F9 is a different gene coding a
different protein, so nothing F8-specific can transfer. What can transfer is the
mechanism: a null variant abolishes the protein, so the patient was never
tolerised to the factor they are later infused with. A memorising model scores
chance on this test.

### 4.5 A probability, not a verdict

A risk score used to choose a prophylaxis regimen has to mean what it says: of
the patients scored at 30%, about 30% should go on to develop inhibitors.
Discrimination alone does not deliver that, so the final model is
isotonic-calibrated and reported with Brier score, expected calibration error
and a reliability curve.

Two operating points are exposed rather than one, because the right threshold
depends on the clinical question. Youden's J balances the two error types.
The 90%-sensitivity point is for the setting where a missed high-risk child
costs far more than an extra assay. Both are fixed on training-fold predictions;
the test set selects nothing.

Decision-curve analysis then answers the question a clinician actually asks:
over what range of thresholds is using this model better than testing everyone
or testing no one?

## 5. What the models are

| Model | Why it is here |
|---|---|
| Logistic regression (L1/L2) | Interpretable floor; on 369 events a linear model is a serious competitor |
| Random Forest / Extra Trees | The reference paper's best classical model, and its natural sibling |
| Gradient Boosting / LightGBM / XGBoost | Standard strong tabular baselines |
| Deep MLP (SELU + AlphaDropout) | Reference architecture 1 |
| Residual MLP | Reference architecture 2 |
| Multi-scale 1D-CNN | Reference architecture 3 |
| TabTransformer | Reference architecture 4 |
| **BioBlockAttentionNet** | This project's contribution: block-wise encoding with gated attention over biological axes, giving per-patient attribution as part of the forward pass rather than as a post-hoc approximation |

All four reference architectures are reimplemented so the comparison is
like-for-like: same features, same folds, same metrics. Where they lose, they
lose on the merits rather than because they were given a worse pipeline.

## 6. Explainability, and its limits

SHAP is kept because it is the field standard and the reference results have to
be comparable to something. Two things are added:

- **Block attribution.** Summing |SHAP| within a biological block is stable
  under the feature correlation that makes per-feature SHAP noisy: credit can
  shuffle between `vtype_nonsense` and `is_truncating` without changing the
  total assigned to "molecular consequence".
- **Intrinsic attention.** `BioBlockAttentionNet` emits its block weights as
  part of the forward pass. That is not an approximation of the model — it *is*
  the model, so it cannot disagree with what the network computed.

Agreement between the two rankings is reported. A disagreement is a reason to
distrust the explanation, and saying so is more useful than presenting a single
confident-looking bar chart.

## 7. What would move this forward

The ceiling here is data, not method. In rough order of expected value:

1. **Patient-level registry data.** PedNet, the ATHN dataset or MLOF would add
   exposure days, product type and treatment intensity — the variables that
   actually dominate the outcome.
2. **HLA typing.** Class II haplotype is among the strongest non-genomic
   predictors and is absent here entirely.
3. **Structural modelling.** An AlphaFold-derived FVIII structure would replace
   linear sequence distance to an epitope with true spatial proximity and
   solvent accessibility.
4. **Prospective validation.** Everything here is retrospective and
   variant-level. A prospective, patient-level cohort is what would be needed
   before this could inform care.
