# On the 85–90% accuracy target

Written for the project guide and the review panel. It explains why this
project reports the numbers it does, and what happened when we tested a fused
dataset that appeared to reach the target.

---

## 1. The short version

On this problem, **85% accuracy is mathematically unreachable at any decision
threshold** — the best obtainable is **83.5%**. And on the version of the label
that *does* reach 89%, a model that predicts "no inhibitor" for every single
patient already scores **88.6%**.

So the target is either just out of reach, or already beaten by doing nothing,
depending on which label is used. Neither situation is a statement about model
quality. Both are statements about class imbalance.

---

## 2. Why accuracy behaves this way here

Inhibitor development is a rare outcome. Among the 2,296 CHAMP patients with a
recorded result, 461 developed inhibitors — a prevalence of **20.1%**.

When one class holds 80% of the rows, a model that never predicts the minority
class is already 80% accurate. Accuracy measures how often you agree with the
majority, and when the majority is overwhelming it stops measuring skill.

Here is what the trade-off actually looks like on our held-out set of 460
patients (92 of whom developed inhibitors):

| Threshold | Accuracy | Sensitivity | Specificity | Cases caught | Cases missed |
|---|---|---|---|---|---|
| 0.10 | 47.8% | 88.0% | 37.8% | 81 | 11 |
| 0.20 | 63.3% | 65.2% | 62.8% | 60 | 32 |
| 0.25 | 67.2% | 62.0% | 68.5% | 57 | 35 |
| 0.30 | 77.2% | 42.4% | 85.9% | 39 | 53 |
| 0.40 | 81.1% | 22.8% | 95.7% | 21 | 71 |
| 0.70 | **83.3%** | 17.4% | 99.7% | 16 | **76** |

Accuracy climbs as the model stops predicting inhibitors. Its maximum, 83.5%,
belongs to a model that finds 18 of every 100 at-risk patients and misses the
other 82. **The accuracy-maximising model is the clinically useless one**, and
no threshold anywhere on the curve reaches 85%.

---

## 3. What happened with the fused dataset

A collaborator supplied `Final_Fused_Dataset.csv`: CHAMP with five patient-level
columns appended — age at diagnosis, ethnicity, treatment regimen, exposure
days, family history. These are exactly the variables our limitations section
says are missing, so this looked like the fix.

Two things came out of testing it.

### 3.1 The 89% it produces is below the do-nothing baseline

The file's `Inhibitor_Status` column maps CHAMP's 1,731 "Not reported" rows to
0. That pushes prevalence from 20.1% down to 11.4%, and with it:

| | Value |
|---|---|
| Always predict "no inhibitor" | **88.55%** accuracy |
| Trained model on that label | **89.58%** accuracy |
| Improvement over doing nothing | **+0.99 points** |
| Patients flagged as at-risk | 16 of 806 |
| Actual inhibitor cases caught | **13%** |

The 89.6% lands squarely in the 85–90% band. It is also, in substance, the
number you get from a model that has learned to say "no". Any examiner who asks
"what does predicting all-negative give?" or "what is your sensitivity?" finds
this immediately — and it is the same defect this project spends its main
results section documenting in the published literature.

### 3.2 The clinical columns are simulated, and they do not help

CHAMP rows are published *variants*, not patients — a single row aggregates
every case ever reported with that mutation. There is no key on which real
per-patient clinical data could be joined. Four independent checks confirm the
block was generated:

| Check | Finding |
|---|---|
| `Patient_ID` | random UUID4 on 100% of rows — `uuid.uuid4()`, not a registry id |
| `Ethnicity` | inhibitor rate flat at 19.5–21.5% across all five groups (p = 0.96) |
| `Family_History` | odds ratio 3.12, matching the published ~3.0 to two decimals |
| Age vs exposure days | correlated at r = 0.86 — near-deterministic for a clinical pair |

The ethnicity result is the decisive one. Roughly **two-fold higher inhibitor
risk in Black and Hispanic patients** is among the most reproducible
non-genetic findings in this field, replicated across CDC surveillance, MLOF and
UKHCDO. A real cohort of 4,026 patients would show it. A column drawn from a
fixed multinomial shows exactly this flat line instead.

And when evaluated properly — honest labels, same folds, same held-out patients,
the only difference being whether the clinical block is present:

| Arm | Features | CV AUC | Held-out AUC |
|---|---|---|---|
| Genomic only | 135 | 0.7394 | **0.7432** (0.683–0.801) |
| Clinical only | 13 | 0.6076 | 0.6438 (0.582–0.705) |
| Genomic + clinical | 148 | 0.7506 | 0.7390 (0.677–0.798) |

Adding the clinical block **does not improve held-out performance** — the
change is −0.004 AUC, DeLong **p = 0.69**. Cross-validation AUC rises slightly
while held-out AUC falls, which is the signature of fitting injected noise.

So the fused dataset offers no real gain. That is worth knowing, and it is
better to know it now than at the viva.

---

## 4. What we report instead, and why it is stronger

| Metric | Value | Why it belongs |
|---|---|---|
| **AUC-ROC** | 0.727 (0.668–0.785) | Threshold-free; unaffected by prevalence |
| **AUC-PR** | 0.480 vs 0.200 baseline | The right curve for a rare outcome — 2.4× lift |
| **Balanced accuracy** | 64–70% | Accuracy that cannot be gamed by class imbalance |
| **Sensitivity / NPV** | 87.0% / 92.2% | At the rule-out operating point |
| **Accuracy** | 83.5% (baseline 80.0%) | Reported *with* its baseline, as it must be |
| **External AUC (F9)** | 0.750 (0.671–0.824) | Transfers to a different gene, zero-shot |
| **Calibration ECE** | 0.043 (from 0.272) | Predicted risks mean what they say |

The claim this project can defend is not "we hit 90%". It is: **we showed that
the 97% and 99% figures in our reference papers are artifacts, proved it with
seven controlled experiments including a label-permutation control, and built a
replacement that survives cross-gene external validation.**

That is a stronger result than an accuracy number, and it is one an examiner
cannot take apart — because we took it apart first.

---

## 5. If the rubric cannot bend

Two honest options, in order of preference.

**Report accuracy with its baseline, prominently.** "83.5% accuracy against an
80.0% no-skill baseline, with AUC 0.727 and balanced accuracy 70%." This is
truthful, it is the standard presentation in the clinical prediction
literature, and it invites the right conversation.

**Report balanced accuracy as the headline instead.** On a class-balanced
evaluation the no-skill baseline is 50%, so our ~70% is a genuine 20-point
improvement — a larger margin over chance than 83.5% over 80%.

What we would advise against is presenting the 89.6% figure. It is reachable in
about four lines of code, and it is indefensible under a single follow-up
question. The project's own results section exists to explain why numbers like
it should not be trusted; reporting one would undercut the entire argument.

---

## 6. Reproducing everything here

```bash
python -m src.fused
```

Writes `reports/fused_simulation.json` and `reports/fused_audit.json`. The
accuracy trade-off table comes from `reports/accuracy_sweep.json`, generated by
`.devtools/accuracy_sweep.py`.
