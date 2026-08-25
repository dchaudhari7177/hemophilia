# TRIPOD-style methods checklist

Clinical prediction models are expected to report a specific set of items
(TRIPOD, Collins et al. 2015). The checklist is reproduced here with the
location of each item, and — where an item is *not* met — a plain statement of
that rather than a silent omission.

| # | Item | Status | Where |
|---|---|---|---|
| 1 | Title identifies the model, target population and outcome | ✅ | `README.md` |
| 2 | Abstract summarises objective, data, methods, results | ✅ | `RESULTS.md` §1 |
| 3a | Background and rationale | ✅ | `docs/ARCHITECTURE.md` §1 |
| 3b | Specific objectives | ✅ | `docs/ARCHITECTURE.md` §1 |
| 4a | Source of data | ✅ | `README.md`, CDC CHAMP / CHBMP 2022 |
| 4b | Dates of data collection | ⚠️ partial | CHAMP aggregates literature 1984–2022; per-patient dates unavailable |
| 5a | Study setting | ⚠️ | Not a cohort — a curated variant catalogue built from published case reports |
| 5b | Eligibility criteria | ✅ | `src/datasets.py` — rows with a recorded Yes/No inhibitor outcome |
| 5c | Treatment received | ❌ **not available** | CHAMP records no treatment data. This is the single largest limitation; see `RESULTS.md` §10 |
| 6a | Outcome definition | ✅ | `History of Inhibitor`, tri-state; `src/datasets.py` |
| 6b | Outcome assessed blind to predictors | ⚠️ | Unknown — outcomes come from the source publications |
| 7a | Predictors defined | ✅ | `src/features.py`, `RESULTS.md` §3 |
| 7b | Predictors assessed blind to outcome | ✅ | Enforced by test `test_featuriser_never_reads_the_outcome` |
| 8 | Sample size | ✅ | 2,296 labelled patients, 461 events; `RESULTS.md` §1 |
| 9 | Missing data handling | ✅ | Median imputation inside each CV fold; unrecorded outcomes excluded, not imputed; `RESULTS.md` §8 |
| 10a | Predictor handling in analysis | ✅ | `RESULTS.md` §3; positional features quantised |
| 10b | Model-building procedure | ✅ | `src/train.py`, `src/tuning.py` |
| 10c | Model validation | ✅ | Repeated CV, position-blocked CV, held-out test, external cohort |
| 10d | Performance measures | ✅ | Discrimination, calibration and clinical utility; `src/evaluate.py` |
| 11 | Risk groups | ✅ | Four bands anchored to epidemiology; `src/predict.py` |
| 12 | Development vs validation differences | ✅ | `RESULTS.md` §7 states which features cannot cross genes |
| 13a | Participant flow | ✅ | 4,040 → 2,296 labelled → 1,836 train / 460 test |
| 13b | Participant characteristics | ✅ | `RESULTS.md` §6, figure 04 |
| 13c | Outcome events in validation | ✅ | `RESULTS.md` §7 |
| 14a | Number of participants and events per predictor | ✅ | 369 training events / 135 features ≈ 2.7 EPV — low, and stated as a limitation |
| 14b | Unadjusted association of predictors | ✅ | Figure 04, `RESULTS.md` §3.1 |
| 15a | Full model presented | ✅ | `models/final_model.joblib`, reproducible from `src/train.py` |
| 15b | How to use the model | ✅ | `src/predict.py`, `app.py` |
| 16 | Performance with confidence intervals | ✅ | Bootstrap CIs on every headline metric |
| 17 | Model updating | n/a | No updating performed |
| 18 | Limitations | ✅ | `RESULTS.md` §10 |
| 19a | Validation interpretation | ✅ | `RESULTS.md` §7 with explicit caveat |
| 19b | Overall interpretation | ✅ | `RESULTS.md` §1, §9 |
| 20 | Implications for practice | ✅ | `docs/ARCHITECTURE.md` §1, §4.5 |
| 21 | Supplementary information | ✅ | `reports/*.json`, all figures regenerable |
| 22 | Funding | ✅ | None; PES University capstone |

## The items that are not met

Three are worth stating outright rather than leaving in a table cell.

**5c — no treatment data.** Inhibitor development depends heavily on treatment
intensity, product type and age at first exposure. CHAMP has none of it. Any
model built on this database is estimating the *genomic component* of risk only,
and its ceiling is set accordingly.

**5a — not a cohort.** CHAMP is a catalogue of variants assembled from published
case reports, so the unit of analysis is the variant, not the patient, and
publication bias applies: unusual variants and notable outcomes are
over-represented relative to routine ones.

**14a — 2.7 events per predictor.** The conventional guidance for clinical
prediction models is 10 or more. This is why the analysis leans on
regularisation, monotone priors and repeated cross-validation rather than on a
single split, and why the confidence intervals are reported rather than the
point estimates alone.
