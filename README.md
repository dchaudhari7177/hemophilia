# Explainable FVIII Inhibitor Risk Classification in Hemophilia A

**PES University B.Tech Capstone — Project ID PW_GRS_01**
Dipak Chaudhari · Tejas Nagmote · Sneha A · Varsha P
Guide: Prof. Gayathri R S

A genomics-based risk model that estimates, from a patient's F8 variant, the
probability that they will develop neutralising antibodies (inhibitors) against
infused Factor VIII — the single most consequential complication of hemophilia A
treatment, and one for which no proactive stratification tool exists in clinical
practice today.

---

## What this project is

This is a rebuild of the group's earlier capstone work, and it exists because
the earlier result did not survive scrutiny.

The prior notebook reported **99.63% accuracy and AUC-ROC 0.9999**; the
classical-ML paper it was benchmarked against reports **97.37% accuracy**.
Neither number is reproducible as a *predictive* result. `src/leakage_audit.py`
takes the reference preprocessing apart experiment by experiment and shows
exactly where the score comes from — near-unique identifier columns fed to the
model as features, unrecorded outcomes relabelled as negative, and
over-sampling applied before the train/test split so that half the test set is a
verbatim copy of the training set.

Rather than reproduce a number that cannot generalise, this project builds a
model whose performance is real, states it plainly, and validates it on a cohort
it has never seen.

**Full findings, tables and comparisons: [`RESULTS.md`](RESULTS.md).**
**Design rationale and the clinical case: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).**
**Reviewer walkthrough: [`Hemophilia_Capstone_Final.ipynb`](Hemophilia_Capstone_Final.ipynb).**

---

## What is new here

| | Reference works | This project |
|---|---|---|
| Features | raw HGVS / coordinate strings label-encoded | 135 mechanistic descriptors parsed out of HGVS: domain, epitope proximity, truncation extent, NMD escape, residue chemistry, splice offsets |
| Unrecorded outcomes | relabelled as "no inhibitor" | kept separate; used semi-supervised, never as fake negatives |
| Imbalance | random over-sampling before the split | class weighting and focal loss, no resampling |
| Validation | one stratified split | repeated CV, position-blocked CV, bootstrap CIs, DeLong tests |
| External data | none | zero-shot transfer to the CDC CHBMP (F9) cohort |
| Calibration | not reported | isotonic calibration, Brier, ECE, reliability curves |
| Clinical utility | not reported | decision-curve net benefit |
| Priors | none | monotone constraints so the model cannot contradict established immunology |
| Explanation | post-hoc SHAP + LIME | SHAP, correlation-stable block attribution, and intrinsic per-patient attention |
| Accuracy claims | headline figure alone | always paired with the majority-class baseline |
| Integrity | asserted | 8 mechanical checks, verified on every run |

---

## A note on accuracy

Accuracy is reported, and it is reported the only way it can honestly be
reported on a 20%-prevalence outcome — next to the score a model gets for never
predicting an inhibitor at all.

| | Accuracy | Sensitivity |
|---|---|---|
| Predict "no inhibitor" for everyone | 80.0% | 0% |
| This model, accuracy-maximising point | ~83.5% | ~18% |
| This model, balanced point | ~67% | ~73% |
| This model, rule-out point | ~55% | ~87% |

Two things follow. **No threshold reaches 85%** — the curve tops out below it.
And the accuracy-maximising operating point is the clinically useless one, so
the tool ships on the balanced and rule-out points instead.

A dataset that *does* report ~89% is analysed in section 10 of the results: it
counts unrecorded outcomes as negatives, which lifts the no-skill baseline to
88.6%, so the headline beats doing nothing by one point while catching 13% of
cases. See [`docs/ACCURACY_AND_THE_RUBRIC.md`](docs/ACCURACY_AND_THE_RUBRIC.md).

---

## Data

| Cohort | Gene | Rows | Inhibitor + | Inhibitor − | Unrecorded | Role |
|---|---|---|---|---|---|---|
| CHAMP 2022 | F8 | 4,040 | 461 | 1,835 | 1,744 | training and internal test |
| CHBMP 2022 | F9 | 1,399 | 40 | 311 | 1,048 | external validation only |
| Fused CHAMP + clinical | F8 | 4,050 | 461 | 1,835 | 1,744 | simulation study only — see results section 10 |

Both are public CDC releases. `data/raw/champ.csv` ships with the repository;
the two Excel originals are downloaded by `scripts/fetch_data.py`.

Labelled prevalence is **20.1%**, which matches the 20–40% inhibitor incidence
the literature reports for severe hemophilia A. The reference pipeline's
relabelling drives it to 11.4%.

---

## Layout

```
capstone_final/
├── src/
│   ├── biology.py         FVIII domains, epitopes, residue chemistry (UniProt P00451)
│   ├── hgvs_parser.py     HGVS cDNA/protein -> structured, non-identifying fields
│   ├── features.py        135 leakage-free features in 7 biological blocks
│   ├── datasets.py        CHAMP/CHBMP loaders, tri-state labels, region blocking
│   ├── leakage_audit.py   forensic reconstruction of the reference result
│   ├── models.py          classical zoo, 4 reference DNNs, BioBlockAttentionNet
│   ├── semisupervised.py  missingness probe, self-training over unlabelled rows
│   ├── evaluate.py        metrics, bootstrap CIs, DeLong, calibration, net benefit
│   ├── explain.py         SHAP, block attribution, intrinsic attention
│   ├── figures.py         every report figure, generated from artefacts
│   ├── fused.py           provenance audit + simulation study of the fused dataset
│   ├── integrity.py       mechanical no-leakage / no-resampling checks
│   ├── ablation.py        signal decomposition and leave-one-block-out
│   ├── ensemble.py        out-of-fold stacking and weighted averaging
│   ├── selection.py       the pre-registered model-selection rule
│   ├── subgroups.py       per-stratum performance
│   ├── tuning.py          nested search + monotone clinical priors
│   ├── report.py          generates RESULTS.md from the artefacts
│   ├── train.py           staged training driver
│   └── predict.py         clinician-facing inference API
├── tests/                 parser tests and leakage guard rails
├── reports/               metrics JSON + figures (regenerated, not hand-written)
├── models/                trained artefacts
└── RESULTS.md             the full write-up
```

---

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
```

```bash
python scripts/fetch_data.py
```

```bash
python -m src.train --stage all
```

Individual stages: `audit`, `cv`, `blocked`, `final`, `subgroups`, `ssl`,
`external`. Add `--no-neural` to skip the torch models.

```bash
python -m src.integrity     # 8 mechanical no-leakage / no-resampling checks
python -m src.fused         # provenance audit of the fused dataset
python -m src.report        # regenerate RESULTS.md from the artefacts
```

```bash
python -m src.figures
```

```bash
python -m pytest
```

The test suite includes the guard rails that keep the original defect from
coming back — no engineered feature may be near-unique across patients, the
featuriser must produce identical output when the labels are scrambled, and
scoring one patient must match scoring that patient inside a batch.

---

## Scoring a patient

```python
from src.predict import InhibitorRiskModel

model = InhibitorRiskModel()
model.predict({
    "HGVS cDNA": "c.6496C>T",
    "HGVS Protein": "p.(Arg2166*)",
    "Variant Type": "Nonsense",
    "Mechanism": "Substitution",
    "Exon": "23",
    "Domain": "C1",
    "Subtype": "Light chain",
    "Reported Clinical Severity": "Severe",
})
```

---

## Honest limitations

CHAMP is a **variant catalogue, not a patient registry**. Each row is a distinct
mutation with an outcome summarised across everyone reported to carry it, so
there is irreducible label noise, and patient-level risk factors that are known
to matter — treatment intensity, product type, exposure days, HLA haplotype,
family history, ethnicity — are simply not in the data. A genomic-only model has
a ceiling well below the numbers the reference works advertise, and the results
here are reported against that ceiling rather than against a leak.

This is a research and educational tool. It is not a validated medical device
and must not replace clinical judgement or laboratory inhibitor testing.

---

## Data sources

- CDC Hemophilia A Mutation Project (CHAMP), 2022 release — <https://www.cdc.gov/hemophilia/mutation-project/>
- CDC Hemophilia B Mutation Project (CHBMP), 2022 release — same page
- UniProt P00451 (Coagulation factor VIII) — domain architecture
- RefSeq NM_000132.4 — F8 transcript structure
