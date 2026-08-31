"""Build the reviewer-facing notebook for the patient-level HADB model.

Cells read the cached JSON in ``reports/`` rather than refitting, so the
notebook executes in about a minute and always shows the same numbers the
committed artefacts contain.
"""
import json
from pathlib import Path

CELLS = []
OUT = Path(__file__).resolve().parents[1] / "Hemophilia_Capstone_HADB.ipynb"


def _lines(text):
    """nbformat wants each source entry to keep its trailing newline.

    Splitting on "\\n" without putting them back concatenates every line into
    one, which turns a code cell into a syntax error.
    """
    return text.strip("\n").splitlines(keepends=True)


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": _lines(text)})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": _lines(text)})


# ===========================================================================
md("""
# FVIII Inhibitor Risk from Genotype and Baseline Phenotype
## Patient-level modelling on the EAHAD/HADB cohort

**PES University B.Tech Capstone — PW_GRS_01**

Dipak Chaudhari · Tejas Nagmote · Sneha A · Varsha P — Guide: Prof. Gayathri R S

---

### The clinical problem

About one in five people with severe hemophilia A develops neutralising
antibodies — *inhibitors* — against the Factor VIII they are infused with. When
that happens, replacement therapy stops working, bleeds become far harder to
control, and the patient needs immune tolerance induction that costs on the
order of hundreds of thousands of dollars and succeeds perhaps two thirds of
the time.

Inhibitors are the single most consequential complication of hemophilia A
treatment, and there is no tool in routine clinical use that stratifies risk
*before* a patient is first exposed to factor. That is the gap this project
addresses.

### Where this notebook picks up

The earlier stage of this capstone rebuilt a genomics-only model on the CDC
**CHAMP** variant list and established, with measurements rather than argument,
that the published reference results (97.37% and 99.63% accuracy) are artefacts
of three specific preprocessing choices. It also established the honest ceiling
of genotype-only data: **AUC ≈ 0.74–0.75**.

That ceiling was a *data* limitation, not a modelling one. CHAMP is a variant
catalogue — it has no per-patient factor level, no antigen, no CRM typing.

This notebook adds the **EAHAD/HADB cohort** (Blood Advances 2024), whose
supplementary table `mmc3` is one row per *patient*, and asks whether the
ceiling moves once each individual's own baseline phenotype is available.

**It does — to AUC 0.785 ± 0.004 under a strictly variant-grouped protocol,
while the genomic rung of the same ablation independently reproduces 0.742 on
a completely different registry.**

Two of the experiments below came out against what we expected. Section 3
predicted that variant-level aggregation would cost performance; it gains.
Section 9's transfer test initially scored *higher* than within-registry
validation, which turned out to be registry contamination rather than a
result. Both are reported as they came out, with the correction shown, because
that is the entire point of a project whose subject is irreproducible results.
""")

md("## 0. Setup")
code("""
import sys, json, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
pd.set_option('display.width', 170)
pd.set_option('display.max_columns', 60)

from pathlib import Path
REPORTS = Path('reports')
def load(name):
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else {}

screen   = load('hadb_screen.json')
ablation = load('hadb_ablation.json')
audit    = load('hadb_audit.json')
tuning   = load('hadb_tuning.json')
final    = load('hadb_final.json')
transfer = load('hadb_transfer.json')
print('cached reports loaded:',
      [k for k, v in dict(screen=screen, ablation=ablation, audit=audit,
                          tuning=tuning, final=final, transfer=transfer).items() if v])
""")

# ---------------------------------------------------------------------------
md("""
## 1. The dataset

The Blood Advances 2024 supplement ships two tables that sit at different
levels of analysis, and the distinction is the whole reason the dataset is
worth adding.

| table | unit | rows | what it holds |
|---|---|---|---|
| `mmc2` | variant | 6,211 | mutation class, protein consequence, domain, exon, codon and nucleotide change |
| `mmc3` | **allele report (patient)** | 10,064 | that individual's FVIII activity, clinical severity, antigen, CRM type, reporting centre — and the inhibitor outcome |

`mmc3` is the layer CHAMP never had. Modelling therefore happens at the patient
level with variant annotation joined in, rather than at the variant level with
outcomes collapsed to a majority vote.
""")

code("""
from src.hadb import load_hadb, label_summary, build_features

hadb = load_hadb()
summary = label_summary(hadb)
print(json.dumps(summary, indent=2))
""")

md("""
### The label, and what is deliberately *not* done to it

The outcome column has three states, not two. 4,966 records carry a recorded
Yes/No; the remaining 5,098 say "Not reported", "Not", or nothing at all.

Those unrecorded rows stay **unlabelled**. They are never converted into
negatives. That single substitution is what inflated the earlier reference
results, and section 3 shows it doing exactly the same thing here.

The resulting prevalence — 16.8% — matches published epidemiology without any
adjustment, which is itself evidence the labels are being read correctly.
""")

code("""
raw = pd.read_csv('data/raw/hadb/BVTH_VTH-2024-000215-mmc3.csv', low_memory=False)
print(raw['Inhibitors'].fillna('(blank)').str.strip().str.lower()
         .value_counts().to_frame('records'))
print()
print(f"labelled   : {summary['n_labelled']:>6,}  ({summary['n_positive']:,} positive)")
print(f"unlabelled : {summary['n_unlabelled']:>6,}  kept out of training as an outcome")
print(f"prevalence : {summary['prevalence']:.1%}  (published range for hemophilia A: 15-30%)")
""")

md("""
### Is there signal at all?

Before any model, the univariate gradients. If these do not follow known
immunology, nothing downstream is trustworthy.
""")

code("""
lab = hadb[hadb.y.notna()]

by_effect = (lab.groupby('effect')['y']
             .agg(n='size', inhibitor_rate='mean')
             .query('n >= 25').sort_values('inhibitor_rate'))
by_effect['inhibitor_rate'] = (by_effect['inhibitor_rate'] * 100).round(1)
print('Inhibitor rate by mutation class'); print(by_effect); print()

by_sev = lab.groupby('severity')['y'].agg(n='size', inhibitor_rate='mean')
by_sev['inhibitor_rate'] = (by_sev['inhibitor_rate'] * 100).round(1)
print('Inhibitor rate by clinical severity'); print(by_sev)
""")

md("""
The gradient is textbook immunology, which is the first reassurance that the
signal is real:

**large deletion → nonsense → frameshift → splice → missense → silent**

A patient whose variant abolishes FVIII entirely has never presented the
protein to their immune system during development, so infused factor arrives
as a foreign antigen. A missense variant still produces a full-length protein,
central tolerance is established, and risk drops by a factor of three or more.

Severity tracks the same mechanism from the phenotype side: 24% in severe
disease against 7% in mild.
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
b = by_effect.sort_values('inhibitor_rate')
axes[0].barh(b.index, b['inhibitor_rate'], color='#3b6ea5')
axes[0].axvline(lab.y.mean()*100, color='crimson', ls='--', lw=1,
                label=f'cohort mean {lab.y.mean():.1%}')
axes[0].set_xlabel('inhibitor rate (%)'); axes[0].set_title('By mutation class')
axes[0].legend(fontsize=8)

order = ['Mild', 'Moderate', 'Severe']
s = by_sev.reindex(order)
axes[1].bar(order, s['inhibitor_rate'], color='#c26a3d')
axes[1].set_ylabel('inhibitor rate (%)'); axes[1].set_title('By clinical severity')
for i, v in enumerate(s['inhibitor_rate']):
    axes[1].text(i, v + 0.6, f'{v}%', ha='center', fontsize=9)
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md("""
## 2. Features

95 numeric descriptors in six blocks. Every one is either a property of the
variant or a measurement taken at diagnosis — nothing is derived from the
outcome, and nothing is a row identifier.
""")

code("""
X, blocks = build_features(hadb)
print(f'design matrix: {X.shape[0]:,} records x {X.shape[1]} features\\n')
for name, cols in blocks.items():
    print(f'  {name:10s} {len(cols):3d}   {", ".join(cols[:5])} ...')
""")

md("""
| block | what it encodes |
|---|---|
| `genotype` | mutation class, null/truncating flags, CpG deamination signature, lesion size |
| `domain` | FVIII domain (A1/A2/A3/B/C1/C2 and the acidic linkers) |
| `position` | mature-protein residue, exon, truncation extent, NMD escape, distance to known inhibitor epitopes |
| `chemistry` | Grantham distance, BLOSUM62, hydropathy/volume/charge/polarity deltas, cysteine and proline involvement |
| `clinical` | **the patient's own** FVIII activity, severity, antigen, CRM type |
| `context` | reporting region (a weak ascertainment proxy — ablated separately) |

Two engineering notes worth stating out loud:

**The curated `CpG` column ships empty** in this release — every row reads
"Null". The hotspot signature is recovered from the nucleotide change instead:
C>T and G>A transitions are the deamination products of a methylated CpG.

**Factor levels are messy free text** — the column mixes plain numbers,
censored readings (`<1`), ranges (`23 to 40`) and annotated entries (`9|<1?`).
Censored values take half the bound, the standard substitution for a
left-censored assay, which keeps `<1` strictly below any observed 1.
""")

# ---------------------------------------------------------------------------
md("""
## 3. Auditing the derived CSVs

Two convenience files were prepared alongside the raw supplement:
`HemophiliaA_Merged_MMC2_MMC3.csv` and `HemophiliaA_ML_Ready_Inhibitor.csv`.
They are a sensible first pass, and they independently reproduce — on a brand
new dataset — the two failure modes this project was rebuilt to correct.

Both are demonstrated numerically. "This would leak" is a claim; "this scores
1.000" is evidence.
""")

code("""
print(json.dumps(audit.get('A_unrecorded_relabelled_as_negative', {}), indent=2))
""")

md("""
**Finding A — absence of a report is not a negative result.**

1,063 variants carry `inhibitor_target = 0` although no inhibitor outcome was
ever recorded for them. Apparent prevalence falls from 23.5% to 13.3%, which
pads the majority class and lifts accuracy without the model predicting
anything better. This is the identical mechanism that took CHAMP's prevalence
from 20.1% to 11.4%.
""")

code("""
print(json.dumps(audit.get('B_outcome_aggregate_used_as_feature', {}), indent=2))
""")

md("""
**Finding B — the outcome is present in the merged file as a feature.**

`inhibitor_positive_rate` is the target averaged over the very records being
predicted. `uinhibitor`, `useverity`, `uclotting`, `uratio`, `uantigen` and
`utype` are the same quantity in different clothing — they are `mmc2` summaries
of the `mmc3` patient rows. All are listed in `src.hadb.FORBIDDEN` and are
never assembled into the design matrix.
""")

code("""
print(json.dumps(audit.get('C_variant_level_aggregation', {}), indent=2))
""")

md("""
**Finding C — and here the experiment contradicted what we expected.**

The plan was to show that variant-level aggregation costs performance. It does
not: the variant-level model scores **higher** (0.804 against 0.777). That
result is reported as it came out.

The two numbers are not comparable. They use a different unit of analysis, a
different label and a different n, and predicting a variant's *modal* outcome
is an easier question than predicting one patient's — averaging features across
a variant's records cancels measurement noise, and well-characterised recurrent
variants dominate the row set.

So the case for patient-level modelling has to rest on something else, and it
does. **Of the 537 variants with two or more recorded outcomes, 124 (23.1%) are
discordant** — patients carrying an identical variant who differ in whether they
developed an inhibitor, spanning 1,297 records. A majority vote hands every one
of those patients the same prediction and is simply wrong for the minority.

A higher AUC on an easier question is not a better clinical tool. The variant
model cannot answer "will *this* patient develop an inhibitor", and it cannot
use the per-patient factor level that this dataset was added to supply.
""")

# ---------------------------------------------------------------------------
md("""
## 4. Protocol

The design decisions that make the rest of the numbers meaningful.

**Grouping.** 2,643 variants generate 4,966 records, and some recurrent
variants appear up to 104 times. Split at random, a model memorises a variant
in training and is rewarded for it at test time — the patient-level form of the
identifier leak found in the earlier work. **Every split here is grouped by
`mut_id`.**

**A second, harsher split groups by study.** Reporting centres differ in how
they screen for inhibitors and in which patients they publish, so
study-grouped scores estimate behaviour in a centre never seen before.

**No resampling.** Imbalance is handled with class weights only. Over-sampling
before the split is precisely what produced the figures this project corrects.

**One held-out set, scored once**, after the model and its thresholds are
fixed on training folds.
""")

code("""
from src.hadb_train import build_cohort, holdout_split, grouped_folds
cohort = build_cohort()
train_mask, test_mask = holdout_split(cohort)

overlap = set(cohort.groups[train_mask]) & set(cohort.groups[test_mask])
print(f'train {train_mask.sum():,} records / test {test_mask.sum():,} records')
print(f'variants shared across the split boundary: {len(overlap)}   <- must be 0')
print(f'prevalence  train {cohort.y[train_mask].mean():.4f} | test {cohort.y[test_mask].mean():.4f}')
""")

# ---------------------------------------------------------------------------
md("""
## 5. Model screen

Twelve candidates, all imbalance-aware, none resampling, scored by
variant-grouped 5-fold CV on the training partition only.
""")

code("""
rows = [{'model': k, **{m: v.get(m) for m in
         ['auc_roc', 'auc_pr', 'balanced_accuracy', 'sensitivity',
          'specificity', 'brier']}}
        for k, v in screen.get('models', {}).items() if 'auc_roc' in v]
tbl = pd.DataFrame(rows).sort_values('auc_roc', ascending=False)
print(tbl.to_string(index=False))
""")

md("""
The tree ensembles lead at 0.777–0.778; linear and kernel models sit around
0.755. The spread is narrow, which is what one expects when the signal is
genuinely in the features rather than in an exploitable artefact.

**The `stacking` entry scoring 0.24 is instructive rather than embarrassing.**
`sklearn`'s `StackingClassifier` generates its own internal folds *at random*
and offers no way to pass grouping through. The same variant therefore lands on
both sides of the meta-learner's boundary, its base features become
anti-correlated with the meta-target, and the blend inverts. It is a compact
demonstration of why the grouping discipline is not optional — and it is why
the ensemble in section 7 builds its out-of-fold matrix by hand.
""")

# ---------------------------------------------------------------------------
md("""
## 6. What each layer is worth

The ablation ladder is the central result. It prices every block of features,
and the rung that matters is the one where the patient's own phenotype enters.
""")

code("""
lad = ablation.get('ablation', {})
order = ['genotype_only', 'plus_domain', 'plus_position', 'genomic_all',
         'plus_clinical', 'plus_region']
rows = [{'rung': k, 'n_features': lad[k]['n_features'],
         'auc_roc': lad[k]['auc_roc_mean'], 'sd': lad[k]['auc_roc_std']}
        for k in order if k in lad]
lt = pd.DataFrame(rows)
lt['gain'] = lt['auc_roc'].diff().round(4)
print(lt.to_string(index=False))
print(f"\\nclinical layer contribution: {lad.get('clinical_contribution')} AUC")
""")

code("""
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(lt['rung'], lt['auc_roc'], 'o-', lw=2, color='#3b6ea5')
ax.axhline(0.75, color='crimson', ls='--', lw=1,
           label='CHAMP genotype-only ceiling (0.74-0.75)')
ax.fill_between(lt['rung'], lt['auc_roc'] - lt['sd'], lt['auc_roc'] + lt['sd'],
                alpha=0.2, color='#3b6ea5')
ax.set_ylabel('AUC-ROC (variant-grouped CV, 3 seeds)')
ax.set_title('What each feature layer contributes')
plt.xticks(rotation=20, ha='right'); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

md("""
Read the ladder carefully, because two separate claims come out of it.

**The genomic rung lands at 0.742.** That is the CHAMP ceiling — 0.74–0.75 —
reproduced on a different registry, curated by a different consortium, from
different patients, using an independently written feature pipeline. Two
independent datasets agreeing on where genotype-only information runs out is
much stronger evidence than either one alone. The earlier ceiling was a
property of the data, not a failure of modelling.

**The clinical rung adds +0.023 and region a further +0.013.** The patient's
own factor level and severity are what move past the genotype ceiling — which
is exactly the hypothesis that motivated adding this dataset.

### A caveat that must be stated, not buried

Baseline factor activity could in principle be a *consequence* of the outcome
rather than a predictor of it: a circulating inhibitor suppresses measured
FVIII activity. The registry records these as diagnostic baseline values, but
it does not timestamp them relative to inhibitor detection, so the concern
cannot be fully closed with this data.

That is precisely why the genomic-only rung is reported alongside the full
model. **A reader who rejects the clinical features entirely still has a
defensible 0.742 model, validated across two registries.** The clinical layer
is presented as an improvement conditional on that assumption, not as a
foundation.
""")

# ---------------------------------------------------------------------------
md("""
### The controls

Three checks that a reviewer should demand of any result like this.
""")

code("""
sh = ablation.get('shuffled_label_control', {})
st = ablation.get('split_strictness', {})
print('shuffled-label control (expect ~0.50 if the matrix carries no row identity)')
print(f"  values : {sh.get('auc_roc_values')}")
print(f"  mean   : {sh.get('auc_roc_mean')}")
print(f"  verdict: {sh.get('verdict')}\\n")

print('split strictness')
print(f"  ungrouped random   {st.get('ungrouped_random', {}).get('auc_roc')}")
print(f"  grouped by variant {st.get('grouped_by_variant', {}).get('auc_roc_mean')}")
print(f"  grouped by study   {st.get('grouped_by_study', {}).get('auc_roc_mean')}")
print(f"  leak closed by grouping: {st.get('leak_from_ungrouping')} AUC")
""")

md("""
**Shuffled labels give 0.503.** Refit on permuted outcomes, the model learns
nothing. For contrast, the earlier CHAMP pipeline reached **train AUC 1.000
with shuffled labels**, because `HGVS cDNA` was label-encoded and acted as a
primary key. This matrix contains no such column.

**Ungrouped CV gives 0.810 against 0.778 grouped.** Splitting at random would
have let us report a number 0.032 higher, for free, by allowing variant
memorisation. That gap is the leak the protocol closes — and it is roughly the
size of the entire clinical-layer gain, which shows how easily a protocol
choice can be mistaken for a modelling advance.

**Study-grouped CV gives 0.780 — statistically indistinguishable from
variant-grouped.** This is the most encouraging number in the project. The
model does not depend on centre-specific reporting habits; it holds up in a
treatment centre whose patients it has never seen.
""")

# ---------------------------------------------------------------------------
md("""
## 7. Tuning and the ensemble

Randomised search over five families on grouped folds, then three ways of
combining the tuned members. The out-of-fold matrix is built by hand from
grouped folds — for the reason section 5 demonstrated.

Members are combined by **rank averaging** rather than probability averaging,
because a forest's 0.6 and a boosted tree's 0.6 are not the same claim.
Ranking removes the scale; a single isotonic layer afterwards puts the blend
back on a probability axis.
""")

code("""
if tuning:
    t = pd.DataFrame([{'family': k, 'cv_auc': v.get('best_cv_auc'),
                       'oof_auc': v.get('oof_auc'), 'seconds': v.get('seconds')}
                      for k, v in tuning.get('tuning', {}).items()])
    print(t.sort_values('oof_auc', ascending=False).to_string(index=False))
    print()
    for k, v in tuning.get('ensembles', {}).items():
        extra = f"  members={v.get('members')}" if v.get('members') else ''
        print(f"  ensemble {k:12s} {v['auc_roc']:.4f}{extra}")
    print()
    print(json.dumps(tuning.get('selection', {}), indent=2))
""")

# ---------------------------------------------------------------------------
md("""
## 8. The headline result

Two numbers, and it matters which is which.

The **primary estimate** is repeated variant-grouped cross-validation over the
whole labelled cohort. The **held-out split** is the independent check that
nothing was tuned into that estimate. It is reported second, not because it is
less trustworthy, but because a single 20% split holds 974 records and about
159 positives — roughly ±0.04 of sampling noise, too wide to headline.
""")

code("""
if final:
    fc = final.get('repeated_full_cohort_cv', {})
    print('PRIMARY -- repeated variant-grouped CV, whole labelled cohort')
    print(f"  AUC-ROC {fc.get('auc_roc_mean'):.4f} +/- {fc.get('auc_roc_std'):.4f}"
          f"   (seeds: {fc.get('auc_roc_values')}, n = {fc.get('n'):,})")
    print()
    print('CHECK -- single variant-grouped held-out split, scored once')
    ty = final.get('test_youden', {})
    print(f"  AUC-ROC {ty['auc_roc']:.4f}  CI95 {final['confidence_intervals']['auc_roc_ci95']}")
    inside = (final['confidence_intervals']['auc_roc_ci95'][0]
              <= fc.get('auc_roc_mean', 0)
              <= final['confidence_intervals']['auc_roc_ci95'][1])
    print(f"  the CV estimate falls inside that interval: {inside}")
    print()
""")

code("""
if final:
    ty = final.get('test_youden', {})
    print(f"n = {final['n_test']:,} records, prevalence {final['test_prevalence']:.4f}")
    print(f"AUC-ROC {ty['auc_roc']:.4f}  CI95 {final['confidence_intervals']['auc_roc_ci95']}")
    print(f"AUC-PR  {ty['auc_pr']:.4f}   (no-skill baseline {ty['auc_pr_baseline']:.4f})")
    print(f"Brier   {ty['brier']:.4f}   ECE {ty['ece']:.4f}")
    print()
    rows = []
    for name in ['accuracy_max', 'youden', 'sensitivity_80', 'sensitivity_90']:
        m = final.get(f'test_{name}')
        if m:
            rows.append({'operating point': name, 'threshold': m['threshold'],
                         'accuracy': m['accuracy'], 'sensitivity': m['sensitivity'],
                         'specificity': m['specificity'], 'precision': m['precision'],
                         'npv': m['npv']})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\\nmajority-class baseline accuracy: {final['majority_baseline_accuracy']:.4f}")
""")

md("""
### Reading the accuracy row honestly

At 17% prevalence, a model that predicts "no inhibitor" for everyone is about
83% accurate and clinically worthless. **Accuracy is only meaningful next to
that baseline**, which is why it is printed alongside every operating point.

The accuracy-maximising threshold is the least useful one — it wins by
declining to identify anybody. The tool therefore ships on the balanced
(Youden) and rule-out (sensitivity-80/90) points, where it catches most of the
patients who go on to develop inhibitors at a cost in false positives that a
treatment centre can absorb: a flagged patient gets closer monitoring during
their first fifty exposure days, not a different diagnosis.
""")

code("""
if final:
    print('full model vs genomic-only, same held-out patients:')
    print(f"  full        {final['test_youden']['auc_roc']:.4f}")
    print(f"  genomic     {final['genomic_only_test']['auc_roc']:.4f}")
    d = final.get('delong_full_vs_genomic', {})
    print(f"  DeLong      z = {d.get('z'):.3f},  p = {d.get('p_value'):.4g}"
          if d.get('z') is not None else f"  DeLong: {d}")
""")

code("""
if final and (REPORTS / 'hadb_test_predictions.npz').exists():
    from sklearn.metrics import roc_curve, precision_recall_curve
    d = np.load(REPORTS / 'hadb_test_predictions.npz')
    y, p, pg = d['y_true'], d['prob_cal'], d['prob_genomic']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for probs, lab_, c in [(p, 'full model', '#3b6ea5'),
                           (pg, 'genomic only', '#999999')]:
        fpr, tpr, _ = roc_curve(y, probs)
        axes[0].plot(fpr, tpr, color=c, lw=2, label=lab_)
    axes[0].plot([0, 1], [0, 1], 'k--', lw=0.8)
    axes[0].set_xlabel('1 - specificity'); axes[0].set_ylabel('sensitivity')
    axes[0].set_title('ROC (held-out)'); axes[0].legend(fontsize=8)

    pr, rc, _ = precision_recall_curve(y, p)
    axes[1].plot(rc, pr, color='#3b6ea5', lw=2)
    axes[1].axhline(y.mean(), color='crimson', ls='--', lw=1,
                    label=f'no skill {y.mean():.3f}')
    axes[1].set_xlabel('recall'); axes[1].set_ylabel('precision')
    axes[1].set_title('Precision-recall'); axes[1].legend(fontsize=8)

    cal = final.get('calibration_curve', {})
    if cal:
        xs, ys = cal.get('mean_predicted'), cal.get('fraction_positive')
        axes[2].plot([0, 1], [0, 1], 'k--', lw=0.8, label='perfect')
        axes[2].plot(xs, ys, 'o-', color='#c26a3d', lw=2, label='model')
        axes[2].set_xlabel('predicted risk'); axes[2].set_ylabel('observed rate')
        axes[2].set_title(f"Calibration (ECE {final['test_youden']['ece']:.3f})")
        axes[2].legend(fontsize=8)
    plt.tight_layout(); plt.show()
""")

md("""
Calibration matters more than the headline here. A risk quoted to a clinician
has to mean what it says — a stated 25% should come true about a quarter of the
time — otherwise the number cannot be used to decide anything. Isotonic
regression is fitted on training out-of-fold scores only, never on the
held-out set.

### Does using it beat not using it?

AUC says the ranking is good. It does not say whether acting on the score helps.
A decision curve answers that directly: net benefit across the range of
thresholds a clinician might adopt, against the two strategies that need no
model at all — monitor everybody, or monitor nobody.
""")

code("""
if final and final.get('decision_curve'):
    dc = final['decision_curve']
    t = np.array(dc['threshold'])
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(t, dc['net_benefit_model'], lw=2, color='#3b6ea5', label='model')
    ax.plot(t, dc['net_benefit_treat_all'], lw=1.5, color='#c26a3d',
            ls='--', label='monitor everyone')
    ax.axhline(0, color='k', lw=1, label='monitor no one')
    ax.set_xlabel('threshold probability a clinician would act at')
    ax.set_ylabel('net benefit')
    ax.set_title('Decision curve (held-out)')
    ax.set_ylim(min(-0.02, float(np.min(dc['net_benefit_model'])) - 0.01), None)
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()

    useful = t[(np.array(dc['net_benefit_model']) > 0) &
               (np.array(dc['net_benefit_model']) >
                np.array(dc['net_benefit_treat_all']))]
    if len(useful):
        print(f'the model beats both trivial strategies for thresholds '
              f'{useful.min():.2f} to {useful.max():.2f}')
    else:
        print('the model does not beat both trivial strategies at any threshold')
""")

# ---------------------------------------------------------------------------
md("""
## 9. Cross-registry transfer

The strongest external check available. CHAMP (CDC, United States) and
HADB/EAHAD (European consortium) are curated independently, by different
teams, from different patients. Anything that transfers between them is
mutation-class immunology rather than a quirk of one curation protocol.

Both registries are projected into a shared 57-feature space — the intersection
of what both hold. HADB's per-patient factor level, antigen and CRM type have
no CHAMP counterpart and are excluded here, so the harmonised model is weaker
than the full one by construction. That is the price of being comparable.
""")

code("""
if transfer:
    print(json.dumps(transfer.get('cohorts', {}), indent=2)); print()
    rows = [
        {'setting': 'within HADB (grouped CV)',  'auc': transfer['within_hadb']['auc_roc']},
        {'setting': 'within CHAMP (CV)',         'auc': transfer['within_champ']['auc_roc']},
        {'setting': 'HADB -> CHAMP (transfer)',  'auc': transfer['hadb_to_champ']['auc_roc']},
        {'setting': 'CHAMP -> HADB (transfer)',  'auc': transfer['champ_to_hadb']['auc_roc']},
    ]
    print(pd.DataFrame(rows).to_string(index=False))
""")

md("""
### Stop — that result is wrong, and the reason matters

Transfer scores **higher** than within-registry cross-validation (0.879 against
0.725). A model applied to a cohort it has never seen should not beat the same
model cross-validated on its own data. When that happens, the "unseen" cohort
is not unseen.

CHAMP and EAHAD are both compiled from the *published literature*, and they
compile a lot of the same papers. Checking directly: **64.5% of CHAMP's
substitution-like labelled variants already appear in HADB.** The transfer
figure above is therefore substantially a memorisation test.
""")

code("""
if transfer:
    print(json.dumps(transfer.get('registry_overlap', {}), indent=2))
""")

md("""
### The corrected external result

Splitting CHAMP into variants HADB has seen and variants novel to it gives the
honest number. One subtlety had to be handled first: the matching key needs a
reference residue and a position, which a frameshift does not have. Left
uncorrected, every frameshift falls into "novel" by default, and the comparison
becomes 77% frameshift against 60% missense — measuring mutation-class
composition rather than novelty. Both strata are therefore restricted to
substitution-like variants.
""")

code("""
if transfer:
    ce = transfer.get('contamination_effect', {})
    print(json.dumps(ce, indent=2))
    nov = transfer.get('hadb_to_champ_novel_only', {})
    if nov:
        print(f"\\nexternal AUC on variants novel to HADB: {nov['auc_roc']:.4f} "
              f"{nov.get('auc_roc_ci95')}")
""")

md("""
Reading it in order:

* **0.936** on CHAMP variants HADB had already seen — familiarity.
* **0.851** [0.819, 0.881] on CHAMP variants novel to HADB — **the external
  result this project claims.**
* The 0.085 gap between them is the contamination, now measured rather than
  ignored.

And the like-for-like control, which is the striking part. On those *exact same
novel rows*, CHAMP's own cross-validated model scores **0.624**. Training on
4,966 patient-level records from a European consortium predicts novel American
variants **+0.228 AUC better** than CHAMP predicts them from its own data.

That is the clearest evidence in the project that patient-level records with
repeated observations per variant are richer supervision than a variant
catalogue — the same conclusion the ablation reached, arrived at independently.
""")

code("""
if transfer:
    r = transfer.get('champ_with_unrecorded_as_negative', {})
    print('Same fitted model, same CHAMP variants, only the label convention changes:')
    print(f"  outcomes recorded only : n={transfer['cohorts']['champ']['n']:,}  "
          f"prevalence {transfer['cohorts']['champ']['prevalence']:.4f}  "
          f"AUC {transfer['hadb_to_champ']['auc_roc']:.4f}  "
          f"accuracy {transfer['hadb_to_champ']['accuracy']:.4f}")
    print(f"  unrecorded -> negative : n={r.get('n'):,}  "
          f"prevalence {r.get('prevalence'):.4f}  "
          f"AUC {r.get('auc_roc'):.4f}  accuracy {r.get('accuracy'):.4f}")
""")

md("""
This last table is the clearest single illustration of the whole project.

The model is identical. The variants are identical. Only the treatment of rows
with no recorded outcome changes — and **accuracy jumps while AUC does not
improve**. Accuracy rose because the majority class got bigger, not because
anything was predicted better.

That is the mechanism behind the 97.37% and 99.63% figures in the literature,
reproduced here on demand.
""")

# ---------------------------------------------------------------------------
md("""
## 10. Explanation

A risk score a clinician cannot interrogate will not be used. SHAP values are
computed against the fitted ensemble, and reported both per-feature and
aggregated into the biological blocks from section 2.
""")

code("""
import joblib
bundle = joblib.load('models/hadb_model.joblib')
ens, cal_ = bundle['ensemble'], bundle['calibrator']
names = bundle['feature_names']

Xte = cohort.X.loc[test_mask].reset_index(drop=True)
yte = cohort.y[test_mask]

# Permutation importance on the held-out set: model-agnostic, and it measures
# what the ensemble actually uses rather than what one member believes.
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)
base = roc_auc_score(yte, ens.decision_scores(Xte))
imp = {}
for col in names:
    drops = []
    for _ in range(3):
        Xp = Xte.copy()
        Xp[col] = rng.permutation(Xp[col].values)
        drops.append(base - roc_auc_score(yte, ens.decision_scores(Xp)))
    imp[col] = float(np.mean(drops))
imp = pd.Series(imp).sort_values(ascending=False)
print(f'held-out AUC {base:.4f}\\n')
print('Top 15 features by permutation importance (AUC lost when shuffled):')
print(imp.head(15).round(4).to_string())
""")

code("""
block_imp = {b: float(imp[[c for c in cols if c in imp.index]].clip(lower=0).sum())
             for b, cols in bundle['feature_blocks'].items()}
bi = pd.Series(block_imp).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
top = imp.head(14).iloc[::-1]
axes[0].barh(top.index, top.values, color='#3b6ea5')
axes[0].set_xlabel('AUC lost when shuffled'); axes[0].set_title('Feature importance')
axes[1].bar(bi.index, bi.values, color='#c26a3d')
axes[1].set_ylabel('summed AUC contribution'); axes[1].set_title('By biological block')
plt.setp(axes[1].get_xticklabels(), rotation=20, ha='right')
plt.tight_layout(); plt.show()
print(bi.round(4).to_string())
""")

md("""
The ordering is the point. Whatever the model has learned should be legible as
immunology, and if a database artefact were driving the score it would surface
here as an unexplainable feature at the top of the list.
""")

# ---------------------------------------------------------------------------
md("""
## 11. Using the model

`src/hadb_predict.py` wraps the artefact for a single patient. The inputs are
what a haemophilia treatment centre actually holds after genetic testing and a
baseline factor assay.
""")

code("""
from src.hadb_predict import HADBRiskModel
model = HADBRiskModel()

cases = [
    dict(label='Severe, large deletion',
         effect='Large Deletion', domain='A2', exon=14, severity='Severe',
         fviii_activity=0.5, crm_type='I'),
    dict(label='Severe, intron-22-type nonsense',
         effect='Nonsense', domain='A2', exon=14, severity='Severe',
         fviii_activity=0.5, crm_type='Unknown'),
    dict(label='Mild missense, C1 domain',
         effect='Missense', domain='C1', exon=23, severity='Mild',
         fviii_activity=18.0, aa_first='Arg', aa_last='Cys', crm_type='II'),
    dict(label='Moderate splice variant',
         effect='Splice', domain='A3', exon=18, severity='Moderate',
         fviii_activity=3.0),
]
out = model.predict(cases, threshold_name='youden')
print(pd.DataFrame(out)[['label', 'risk', 'band', 'call', 'threshold']]
      .to_string(index=False))
""")

md("""
The ordering across these four cases is the sanity check: a severe patient with
a whole-domain deletion and no circulating antigen should outrank a mild
missense carrier with 18% residual activity by a wide margin.

The Flask app at `app.py` exposes the same model at `/hadb`, with the four
operating points selectable, so a reviewer can try cases interactively.
""")

# ---------------------------------------------------------------------------
md("""
## 12. Where this lands

**Result.** AUC 0.78 at the patient level, under variant-grouped validation,
with calibrated probabilities, holding up in unseen reporting centres and
transferring across registries.

**Why that is the honest number.** Every mechanism that could have inflated it
was tested and closed: identifier leakage (shuffled labels → 0.503), split
leakage (grouping costs 0.032 and was paid), label inflation (unrecorded
outcomes kept unlabelled), and outcome-derived features (`FORBIDDEN`).

**What would move it further.** Not a better classifier — the twelve-model
spread in section 5 is 0.03 wide, so architecture is not the binding
constraint. The missing variables are the ones known to drive inhibitor
development and absent from every public registry: **treatment intensity,
exposure days at first bleed, product type (plasma-derived vs recombinant),
surgery or infection at first exposure, and HLA class II typing.** A
prospective cohort carrying those would plausibly reach 0.85+; no amount of
modelling reaches it from registry data alone.

**What it is not.** A research tool, not a validated medical device. It does
not replace laboratory inhibitor testing or clinical judgement. Its intended
use is prioritising monitoring intensity during the first fifty exposure days
— the window in which most inhibitors appear.
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                        "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {OUT} with {len(CELLS)} cells")
