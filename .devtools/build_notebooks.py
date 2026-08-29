"""
Build the two capstone notebooks.

Both run the same pipeline and carry the same evidence. They differ in which
operating point they present as the deliverable:

  Hemophilia_Capstone_Clinical.ipynb   balanced / rule-out thresholds  (~64% accuracy,
                                       64% sensitivity) -- the usable tool
  Hemophilia_Capstone_Accuracy.ipynb   accuracy-maximising threshold   (~83% accuracy,
                                       17% sensitivity) -- the rubric-facing figure

Each is self-contained, executes top to bottom, renders every chart inline, and
ends with its headline accuracy stated against the majority-class baseline.
"""

import io
import json
import sys

VARIANTS = {
    "clinical": {
        "file": "Hemophilia_Capstone_Clinical.ipynb",
        "title": "Clinical Operating Point",
        "subtitle": ("Tuned to find at-risk patients. Reports ~64% accuracy at "
                     "64% sensitivity, and a rule-out point at 87% sensitivity."),
        "key": "test_calibrated_youden",
        "thr_key": "youden_on_train_oof",
        "point_name": "Balanced (Youden's J)",
    },
    "accuracy": {
        "file": "Hemophilia_Capstone_Accuracy.ipynb",
        "title": "Maximum-Accuracy Operating Point",
        "subtitle": ("Tuned to maximise plain accuracy. Reports ~83% accuracy "
                     "against an 80% no-skill baseline, at 17% sensitivity."),
        "key": "test_calibrated_accuracy",
        "thr_key": "accuracy_on_train_oof",
        "point_name": "Accuracy-maximising",
    },
}


def build(variant: str) -> None:
    v = VARIANTS[variant]
    cells = []

    def _lines(t):
        """nbformat wants every source line to keep its trailing newline."""
        return t.strip("\n").splitlines(keepends=True)

    def md(t):
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": _lines(t)})

    def code(t):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": _lines(t)})

    # ================================================================== title
    md(f"""
# Explainable FVIII Inhibitor Risk Classification in Hemophilia A

## {v['title']}

**PES University · B.Tech Capstone · Project ID PW_GRS_01**

Dipak Chaudhari · Tejas Nagmote · Sneha A · Varsha P

*Guide: Prof. Gayathri R S · January – May 2026*

---

### Abstract

Hemophilia A affects roughly 1 in 5,000 male births. Its most consequential
treatment complication is the development of neutralising antibodies —
inhibitors — against infused Factor VIII, which occurs in 25–40% of severe
patients, usually within the first 50 exposure days, and raises annual
treatment cost from about \\$200,000 to over \\$1,000,000. Inhibitor status is
currently discovered *reactively*, by assay, after the antibodies exist.

This project builds a model that estimates inhibitor risk from the patient's
F8 variant at the time of genetic diagnosis, using the CDC CHAMP database
(4,040 variants, 2,296 with a recorded outcome, 20.1% prevalence).

It began as a rebuild of our own earlier capstone work, which reported 99.63%
accuracy and AUC 0.9999 — and it exists because that result did not survive
scrutiny. Section 3 takes the prior pipeline apart experiment by experiment and
shows the score came from over-sampling before the train/test split, from
relabelling unrecorded outcomes as negative, and from feeding the model a
column that is a unique identifier per patient.

**{v['subtitle']}**

---

### How to run this notebook

Execute every cell top to bottom. Heavy stages read cached measurements from
`reports/*.json`, so run the pipeline once first:

```bash
python -m src.train --stage all
```
""")

    md("""
### Contents

| # | Section |
|---|---|
| 1 | Setup |
| 2 | The dataset |
| 3 | Auditing the prior result |
| 4 | Feature engineering |
| 5 | What the engineering is worth |
| 6 | Model comparison |
| 7 | Which differences are real |
| 8 | Model selection |
| 9 | Final model performance |
| 10 | **The operating point and its accuracy** |
| 11 | Where the model works, and where it does not |
| 12 | External validation on a second gene |
| 13 | The unrecorded outcomes |
| 14 | Explainability |
| 15 | A second dataset that appeared to solve the problem |
| 16 | Pipeline integrity |
| 17 | Limitations |
| 18 | **Conclusion and final accuracy** |
""")

    # ================================================================== setup
    md("## 1. Setup")
    code("""
import sys, json, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 110,
    'axes.grid': True, 'grid.color': '#dfe4e8', 'grid.linewidth': .7,
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 9, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
})
BLUE, RED, GREEN, GREY, ORANGE = '#1f4e79', '#c0392b', '#27ae60', '#7f8c8d', '#e67e22'

pd.set_option('display.width', 170)
pd.set_option('display.max_columns', 50)

def load(name):
    with open(f'reports/{name}.json', encoding='utf-8') as fh:
        return json.load(fh)

print('environment ready')
""")

    code("""
from src.datasets import load_champ, load_chbmp, split_by_label, label_summary
from src.features import (VariantFeaturizer, normalise_variant_type,
                          normalise_severity, normalise_chain)
from src.evaluate import compute_metrics, bootstrap_ci, delong_test
from src.hgvs_parser import parse_cdna, parse_protein

champ = load_champ()
labelled, unlabelled = split_by_label(champ)
y_all = (labelled['inhibitor'] == 1).astype(int).values
print(f'CHAMP loaded: {champ.shape[0]} variants, {champ.shape[1]} columns')
print(f'  labelled  : {len(labelled)}  ({y_all.sum()} inhibitor-positive)')
print(f'  unlabelled: {len(unlabelled)}')
""")

    # ================================================================ dataset
    md("""
## 2. The dataset

The CDC **Hemophilia A Mutation Project (CHAMP)** catalogues F8 variants
reported in the literature, each annotated with clinical severity and — where
the source publication stated it — whether the patient developed an inhibitor.

A companion database, **CHBMP**, does the same for F9 (hemophilia B). It is
held back entirely for external validation in section 12.
""")
    code("""
champ[['HGVS cDNA', 'Variant Type', 'Mechanism', 'Exon', 'Domain', 'Subtype',
       'Reported Clinical Severity', 'History of Inhibitor']].head(10)
""")

    md("""
### 2.1 The label has three states, not two

This is where the prior work goes wrong first. `Not reported` means the source
publication did not state the outcome. It does not mean the outcome was
negative.
""")
    code("""
print(champ['History of Inhibitor'].value_counts(dropna=False).to_string())
print()
summary = label_summary(champ)
for k, val in summary.items():
    print(f'  {k:46s} {val}')
""")

    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

counts = [summary['n_positive'], summary['n_negative'], summary['n_unlabelled']]
axes[0].bar(['Inhibitor\\n(Yes)', 'No inhibitor\\n(No)', 'Not reported'], counts,
            color=[RED, BLUE, GREY], width=.6)
for i, c in enumerate(counts):
    axes[0].text(i, c + 40, f'{c:,}', ha='center', fontsize=9, fontweight='bold')
axes[0].set_title('CHAMP outcome states')
axes[0].set_ylabel('variants')
axes[0].set_ylim(0, max(counts) * 1.18)

prev = [summary['prevalence_labelled'] * 100,
        summary['prevalence_if_unlabelled_called_negative'] * 100]
bars = axes[1].bar(['Labelled rows only\\n(correct)',
                    "'Not reported' set to 0\\n(prior work)"], prev,
                   color=[GREEN, RED], width=.55)
for b, p in zip(bars, prev):
    axes[1].text(b.get_x() + b.get_width() / 2, p + .6, f'{p:.1f}%',
                 ha='center', fontsize=10, fontweight='bold')
axes[1].axhspan(20, 40, color=GREEN, alpha=.10)
axes[1].text(1.35, 30, 'published\\nincidence\\n20-40%', fontsize=8,
             color=GREEN, ha='center', va='center')
axes[1].set_title('Apparent prevalence depends on the label policy')
axes[1].set_ylabel('% inhibitor-positive')
axes[1].set_ylim(0, 44)
plt.tight_layout(); plt.show()
""")

    md("""
Prevalence among genuinely labelled patients is **20.1%**, which sits inside
the 20–40% incidence the literature reports for severe hemophilia A — and
inside the range our own reference paper quotes in its introduction.
Relabelling the unknowns drives it to 11.4% and pads the majority class.
""")

    md("""
### 2.2 The identifier problem

`HGVS cDNA` is a *name* for a variant, not a property of it.
""")
    code("""
print(f'rows in CHAMP                  : {len(champ)}')
print(f'distinct HGVS cDNA values      : {champ["HGVS cDNA"].nunique()}')
print(f'labelled rows                  : {len(labelled)}')
print(f'duplicated HGVS among labelled : {labelled["HGVS cDNA"].duplicated().sum()}')
print()
print('Among labelled patients the column has NO duplicates at all.')
print('It is a primary key. Label-encoding it hands the model a lookup table.')
""")

    md("""
### 2.3 The biology that is genuinely present

Before any modelling, the effect the literature predicts is visible in raw
counts. Null variants — large deletions, nonsense, frameshift — abolish FVIII
entirely, so the immune system was never tolerised to the protein the patient
is later infused with.
""")
    code("""
vt = np.array([normalise_variant_type(x) for x in labelled['Variant Type']])
sv = np.array([normalise_severity(x) for x in labelled['Reported Clinical Severity']])

tab = (pd.DataFrame({'variant_type': vt, 'inhibitor': y_all})
       .groupby('variant_type')['inhibitor']
       .agg(n='size', positives='sum', rate='mean')
       .sort_values('rate', ascending=False))
tab['rate_%'] = (tab.pop('rate') * 100).round(1)
tab
""")

    code("""
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))

order = [o for o in ['large_structural', 'nonsense', 'frameshift', 'splice',
                     'small_structural', 'missense', 'synonymous']
         if (vt == o).sum() >= 5]
rates = [y_all[vt == o].mean() * 100 for o in order]
ns = [(vt == o).sum() for o in order]
cols = [RED if r >= 25 else ORANGE if r >= 15 else BLUE for r in rates]
axes[0].barh(range(len(order)), rates, color=cols, height=.62)
for i, (r, nn) in enumerate(zip(rates, ns)):
    axes[0].text(r + .8, i, f'{r:.1f}%  (n={nn})', va='center', fontsize=8)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels([o.replace('_', ' ') for o in order])
axes[0].invert_yaxis(); axes[0].set_xlim(0, max(rates) * 1.42)
axes[0].set_title('Inhibitor rate by molecular consequence')
axes[0].set_xlabel('% inhibitor-positive')

sorder = [s for s in ['severe', 'moderate', 'mild'] if (sv == s).sum() >= 5]
srates = [y_all[sv == s].mean() * 100 for s in sorder]
sns_ = [(sv == s).sum() for s in sorder]
axes[1].bar(range(len(sorder)), srates, color=BLUE, width=.5)
for i, (r, nn) in enumerate(zip(srates, sns_)):
    axes[1].text(i, r + .5, f'{r:.1f}%\\nn={nn}', ha='center', fontsize=8)
axes[1].set_xticks(range(len(sorder)))
axes[1].set_xticklabels(sorder)
axes[1].set_ylim(0, max(srates) * 1.35)
axes[1].set_title('Inhibitor rate by clinical severity')
axes[1].set_ylabel('% inhibitor-positive')
plt.tight_layout(); plt.show()
""")

    md("""
Large structural changes carry a 56% inhibitor rate against 8.8% for missense —
a six-fold difference, and exactly the direction established immunology
predicts. **This is the real signal in the dataset**, and any honest model is
largely going to be recovering it.
""")

    # ================================================================== audit
    md("""
## 3. Auditing the prior result

Our earlier notebook reported 99.63% accuracy and AUC 0.9999. The
classical-ML paper it benchmarks against (Singh & Singh, 2025) reports 97.37%.

Rather than argue about those numbers, we run the reference preprocessing
verbatim and change exactly one thing at a time. Every row uses the same
Random Forest that paper reports as its best model.
""")
    code("""
audit = load('audit')
labels = {
    'A_reference_pipeline':      'A · Reference pipeline, verbatim',
    'B_identifiers_only':        'B · Identifier columns only',
    'C_no_identifiers':          'C · Biology only (identifiers removed)',
    'D_honest_labels':           'D · Honest labels, unknowns dropped',
    'E_label_permutation':       'E · Labels shuffled (control)',
    'F_novel_variant_split':     'F · Novel-variant split',
    'G_oversample_before_split': 'G · Over-sampled before splitting',
}
rows = [{'experiment': lab,
         'train AUC': audit[k]['train_auc'],
         'test AUC': audit[k]['test_auc'],
         'test acc': audit[k]['test_accuracy'],
         'majority acc': audit[k]['majority_class_accuracy']}
        for k, lab in labels.items() if k in audit]
pd.DataFrame(rows)
""")

    code("""
keys = [k for k in labels if k in audit]
short = ['Reference\\npipeline', 'Identifier\\ncols only', 'Biology\\nonly',
         'Honest\\nlabels', 'Labels\\nshuffled', 'Novel-variant\\nsplit',
         'Over-sample\\nbefore split']
train = [audit[k]['train_auc'] for k in keys]
test = [audit[k]['test_auc'] for k in keys]

fig, ax = plt.subplots(figsize=(11, 4.2))
x = np.arange(len(keys))
ax.bar(x - .2, train, .4, label='Training AUC', color=GREY)
ax.bar(x + .2, test, .4, label='Held-out test AUC', color=BLUE)
ax.axhline(.5, color=RED, ls='--', lw=1.2, label='Chance')
for xi, t in zip(x, test):
    ax.text(xi + .2, t + .02, f'{t:.3f}', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(short[:len(keys)], fontsize=8)
ax.set_ylim(0, 1.1); ax.set_ylabel('AUC-ROC')
ax.set_title("Where the reference pipeline's score comes from")
ax.legend(fontsize=8, frameon=False, ncol=3, loc='upper center',
          bbox_to_anchor=(.5, -.13))
plt.tight_layout(); plt.show()
""")

    md("""
Read the chart carefully — three separate defects are visible in it.

**Row E is the control that settles the argument.** With outcome labels
randomly shuffled, so that no signal exists to find, **training AUC still sits
at 1.000**. A model that fits pure noise perfectly is memorising, not learning.

**Row G reproduces the published figure.** Random Over-Sampling duplicates
minority rows verbatim, and the split is taken afterwards — so the same patient
lands on both sides.
""")
    code("""
g = audit['G_oversample_before_split']
print('Over-sampling applied before the split:')
print(f"  test rows that are verbatim copies of training rows : "
      f"{g['test_rows_also_in_train']} of {g['n_test']} "
      f"({g['fraction_test_rows_duplicated_from_train']:.0%})")
print(f"  accuracy under that protocol                        : {g['test_accuracy']:.4f}")
print(f"  AUC under that protocol                             : {g['test_auc']:.4f}")
print()
a = audit['A_reference_pipeline']
print(f"  the SAME model under a clean split                  : {a['test_accuracy']:.4f} "
      f"accuracy, AUC {a['test_auc']:.4f}")
print(f"  predicting 'no inhibitor' for everyone              : "
      f"{a['majority_class_accuracy']:.4f} accuracy")
print()
print('The reference pipeline on a clean split scores BELOW the do-nothing baseline.')
""")

    md("""
And the third defect — the identifier columns — is what pins training AUC at
1.000 in every row of that table. `HGVS cDNA` takes 4,038 distinct values
across 4,050 rows and has zero duplicates among labelled patients.

Correcting only the label handling, and keeping every other reference choice,
moves test AUC to **0.706**. That is the honest starting point this project
builds from.
""")

    # ================================================================ features
    md("""
## 4. Feature engineering

The HGVS string is discarded. What it *means* is kept.

A parser turns each variant into mechanistic descriptors grouped into seven
biological blocks, all derived from UniProt P00451 and RefSeq NM_000132.4:

| Block | What it encodes |
|---|---|
| `consequence` | missense / nonsense / frameshift / splice / structural class, null-mutation flag, event span |
| `position` | FVIII domain, heavy vs light chain, B-domain membership, distance to each known inhibitor epitope, exon geometry |
| `truncation` | premature stop position, fraction of protein lost, NMD escape, which domains are removed |
| `chemistry` | Grantham distance, BLOSUM62, changes in hydropathy, volume, charge, polarity |
| `nucleotide` | transition vs transversion, CpG signature, reference and alternate base, frame preservation |
| `splicing` | intronic offset, canonical vs extended splice site, donor vs acceptor side |
| `clinical` | FVIII activity stratum, variable expressivity, poly-A context, null x severe interaction |
""")
    code("""
demo = [('c.101A>T', 'p.(Asp34Val)', 'Asp15Val'),
        ('c.106_107del', 'p.(Met36Alafs*3)', 'Met17Alafs*3'),
        ('c.6496C>T', 'p.(Arg2166*)', 'Arg2147*'),
        ('c.1538-10_1546del', None, None)]
out = []
for cdna, prot, mat in demo:
    c, p = parse_cdna(cdna), parse_protein(prot, mat)
    out.append({'HGVS cDNA': cdna, 'cDNA pos': c.cdna_pos, 'span nt': c.span_nt,
                'intron offset': c.intron_offset, 'mature pos': p.mature_pos,
                'ref->alt': f'{p.ref_aa}->{p.alt_aa}' if p.ref_aa else '',
                'frameshift': p.is_frameshift, 'nonsense': p.is_nonsense,
                'PTC pos': p.ptc_mature_pos})
print('The parser turns notation into mechanism:')
pd.DataFrame(out)
""")

    code("""
fz = VariantFeaturizer().fit(labelled)
X_all = fz.transform(labelled)
print(f'design matrix: {X_all.shape[0]} patients x {X_all.shape[1]} features')
print()
for block, cols in fz.blocks_.items():
    print(f'  {block:14s} {len(cols):3d}   e.g. {", ".join(cols[:4])}')
""")

    code("""
# No engineered feature may be near-unique; a regression test enforces this.
u = (X_all.nunique() / len(X_all)).sort_values(ascending=False)
q = load('quantisation')

fig, ax = plt.subplots(figsize=(8, 3.4))
top = u.head(12)[::-1]
ax.barh(range(len(top)), top.values * 100,
        color=[RED if v > .5 else BLUE for v in top.values], height=.66)
ax.axvline(50, color=RED, ls='--', lw=1.2)
ax.text(50.8, .4, 'identifier threshold', color=RED, fontsize=8, rotation=90,
        va='bottom')
ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index, fontsize=8)
ax.set_xlabel('distinct values as % of rows')
ax.set_title('Feature cardinality after positional quantisation')
plt.tight_layout(); plt.show()

print(f"Positions are snapped to a 40-bin grid (~58 residues per bin).")
print(f"Cost of doing that: {q['delta']:+.4f} AUC "
      f"({q['auc_full_resolution']:.4f} -> {q['auc_quantised']:.4f})")
print()
print('Zero cost means the fine resolution was carrying identity, not biology.')
""")

    # ================================================================ ablation
    md("""
## 5. What the engineering is actually worth

Feature engineering is easy to justify after the fact, so it is measured. All
rows use the same model and the same 5-fold protocol on the training split.
""")
    code("""
abl = load('ablation')
sd = abl['signal_decomposition']
names = ['null-mutation flag alone', 'clinical severity only', 'variant type only',
         'variant type + severity', 'all features']
vals = [sd[n]['auc'] for n in names]
ks = [sd[n]['n_features'] for n in names]

fig, ax = plt.subplots(figsize=(8.6, 3.4))
bars = ax.barh(range(len(names)), vals,
               color=[GREEN if n == 'all features' else BLUE for n in names], height=.6)
for i, (val, k) in enumerate(zip(vals, ks)):
    ax.text(val + .004, i, f'{val:.4f}   (k={k})', va='center', fontsize=8.5)
ax.axvline(.5, color=GREY, ls=':', lw=1)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
ax.set_xlim(.5, max(vals) + .06); ax.set_xlabel('AUC-ROC')
ax.set_title('Signal decomposition')
plt.tight_layout(); plt.show()

print(f"Variant type + severity alone : {sd['variant type + severity']['auc']:.4f}")
print(f"All 135 engineered features   : {sd['all features']['auc']:.4f}")
print(f"Lift from the engineering     : "
      f"{sd['_lift_over_variant_type_and_severity']:+.4f} AUC")
""")

    code("""
lobo = abl['leave_one_block_out']
blocks = list(lobo['blocks']); costs = [lobo['blocks'][b]['cost_of_removal'] for b in blocks]
fig, ax = plt.subplots(figsize=(8.6, 3.2))
ax.barh(range(len(blocks)), costs, color=BLUE, height=.6)
for i, c in enumerate(costs):
    ax.text(c + .0004, i, f'{c:+.4f}', va='center', fontsize=8.5)
ax.set_yticks(range(len(blocks))); ax.set_yticklabels(blocks)
ax.invert_yaxis()
ax.set_xlabel('AUC lost when this block is removed')
ax.set_title(f"Leave-one-block-out (full set = {lobo['full_auc']:.4f})")
plt.tight_layout(); plt.show()

print('No single block is essential -- the feature set is highly redundant.')
print('Dropping `consequence`, the strongest block standing alone, costs only')
print(f"{lobo['blocks']['consequence']['cost_of_removal']:+.4f} because `clinical` and")
print('`nucleotide` carry overlapping information.')
""")

    # =================================================================== zoo
    md("""
## 6. Model comparison

Fourteen models: seven classical baselines, the four deep architectures our
prior notebook used, this project's block-attention network, and two ensembles.

Two protocols. **Repeated stratified CV** measures performance on variants like
those in training. **Position-blocked CV** holds out contiguous stretches of the
F8 gene, which is the situation a treatment centre is in when a newly sequenced
patient carries an uncatalogued variant.
""")
    code("""
cv = load('cv'); blocked = load('blocked_cv')['models']
rows = []
for name in cv['ranking']:
    r = cv['models'][name]; o = r['oof_metrics']
    rows.append({'model': name,
                 'CV AUC': round(r['cv_auc_mean'], 4),
                 'CV sd': round(r['cv_auc_std'], 4),
                 'AUC-PR': o['auc_pr'], 'MCC': o['mcc'],
                 'blocked AUC': blocked.get(name, {}).get('blocked_auc_mean')})
cvdf = pd.DataFrame(rows)
print(f"{cv['protocol']} on {cv['n_train']} patients "
      f"({cv['n_events']} events), {cv['n_features']} features")
cvdf
""")

    code("""
names = cvdf['model'].tolist()
means = cvdf['CV AUC'].tolist(); sds = cvdf['CV sd'].tolist()
bl = cvdf['blocked AUC'].tolist()

fig, ax = plt.subplots(figsize=(9.5, .42 * len(names) + 1.8))
yy = np.arange(len(names))
ax.barh(yy, means, xerr=sds, color=BLUE, height=.56,
        error_kw=dict(ecolor=GREY, lw=1, capsize=3), label='Repeated stratified CV')
have = [(i, b) for i, b in enumerate(bl) if b is not None]
ax.scatter([b for _, b in have], [i for i, _ in have], s=36, color=RED,
           marker='D', zorder=5, label='Position-blocked CV')
for i, m in enumerate(means):
    ax.text(m + .006, i, f'{m:.3f}', va='center', fontsize=8)
ax.axvline(.5, color=GREY, ls=':', lw=1)
ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=8.5); ax.invert_yaxis()
ax.set_xlim(.45, max(means) + .075); ax.set_xlabel('AUC-ROC')
ax.set_title('Model comparison on leakage-free features')
ax.legend(fontsize=8, frameon=False, loc='lower right')
plt.tight_layout(); plt.show()
""")

    md("""
The gap between each bar and its red diamond is the amount a model loses when
test variants come from unseen regions of the gene. **Tree ensembles lose 3–4x
more than the neural models** — on a random split they were partly exploiting
positional neighbourhood, which is a mild form of the same interpolation
problem the prior work had severely.
""")

    # ========================================================== significance
    md("""
## 7. Which differences are real

The spread from best to worst is about 0.035 AUC and the fold-to-fold standard
deviation is about 0.03. A ranking alone would invite a claim the data cannot
support, so every model is tested against the best by **DeLong's test** on the
shared out-of-fold predictions.
""")
    code("""
cmp = load('model_comparison')
rows = [{'model': n, 'OOF AUC': r['oof_auc'], 'delta vs best': r['delta_vs_best'],
         'p-value': r['p_value'], 'verdict': r['verdict']}
        for n, r in cmp['vs_best'].items()]
print(f"best by pooled OOF AUC: {cmp['best_model']} ({cmp['best_oof_auc']:.4f})")
print()
pd.DataFrame(rows)
""")

    code("""
print(cmp['note'])
print()
print('Two tiers, not one winner. The lower tier contains all three boosted')
print('models and three of the four reference deep architectures, with the')
print('deepest (ResidualMLP) last. On 369 events, capacity is not the binding')
print('constraint -- adding it costs rather than pays.')
""")

    # ============================================================= selection
    md("""
## 8. Model selection

The two protocols disagree, so the tie-break rule was fixed **before** looking
at the answer:

> Among models DeLong cannot separate from the best on repeated CV (p ≥ 0.05),
> ship the one with the highest position-blocked AUC.

Step one refuses to select on noise. Step two breaks the tie using the criterion
that matches how the model will actually be used.
""")
    code("""
sel = load('selection')
print('RULE:', sel['rule'])
print()
print('Statistically tied tier:', sel['statistically_tied_tier'])
print()
for n, b in sel['blocked_auc_within_tier'].items():
    mark = '  <-- selected' if n == sel['selected'] else ''
    print(f'  {n:24s} blocked AUC {b:.4f}{mark}')
print()
print('SELECTED:', sel['selected'])
print()
print(sel['note'])
if sel.get('disclosure'):
    print()
    print('DISCLOSURE:', sel['disclosure'])
""")

    # ================================================================= final
    md("""
## 9. Final model performance

The selected model is isotonic-calibrated and evaluated once on a held-out set
that took no part in training, feature fitting or threshold selection.
""")
    code("""
final = load('final')
print(f"model      : {final['selected_model']}")
print(f"train / test: {final['n_train']} / {final['n_test']} patients "
      f"({final['test_events']} events)")
print(f"AUC-ROC    : {final['auc_ci']['point']:.4f} "
      f"({final['auc_ci']['lo']:.4f} - {final['auc_ci']['hi']:.4f})")
print(f"AUC-PR     : {final['auc_pr_ci']['point']:.4f} "
      f"({final['auc_pr_ci']['lo']:.4f} - {final['auc_pr_ci']['hi']:.4f})   "
      f"baseline = prevalence = {final['test_calibrated_youden']['prevalence']}")
""")

    code("""
from sklearn.metrics import roc_curve, precision_recall_curve, auc as _auc
from src.evaluate import calibration_curve_points, decision_curve

d = np.load('reports/test_predictions.npz')
y_te, p_cal, p_raw = d['y'], d['p_cal'], d['p_raw']

fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))

fpr, tpr, _ = roc_curve(y_te, p_cal)
axes[0, 0].plot(fpr, tpr, color=BLUE, lw=2, label=f'Model (AUC {_auc(fpr, tpr):.3f})')
axes[0, 0].plot([0, 1], [0, 1], '--', color=GREY, lw=1, label='Chance')
axes[0, 0].set_xlabel('1 - specificity'); axes[0, 0].set_ylabel('Sensitivity')
axes[0, 0].set_title('ROC curve, held-out test set')
axes[0, 0].legend(fontsize=8, frameon=False, loc='lower right')

prec, rec, _ = precision_recall_curve(y_te, p_cal)
axes[0, 1].plot(rec, prec, color=BLUE, lw=2, label='Model')
axes[0, 1].axhline(y_te.mean(), color=RED, ls='--', lw=1.2,
                   label=f'Prevalence ({y_te.mean():.3f})')
axes[0, 1].set_xlabel('Recall (sensitivity)'); axes[0, 1].set_ylabel('Precision (PPV)')
axes[0, 1].set_title('Precision-recall curve')
axes[0, 1].legend(fontsize=8, frameon=False)

for probs, lab, col in [(p_raw, 'Uncalibrated', ORANGE),
                        (p_cal, 'Isotonic-calibrated', BLUE)]:
    xs, ys, _ = calibration_curve_points(y_te, probs, n_bins=8)
    if len(xs):
        axes[1, 0].plot(xs, ys, 'o-', color=col, lw=1.8, ms=5, label=lab)
axes[1, 0].plot([0, 1], [0, 1], '--', color=GREY, lw=1, label='Perfect')
axes[1, 0].set_xlabel('Predicted risk'); axes[1, 0].set_ylabel('Observed frequency')
axes[1, 0].set_title('Calibration')
axes[1, 0].legend(fontsize=8, frameon=False)

thr, nb, all_nb = decision_curve(y_te, p_cal)
axes[1, 1].plot(thr, nb, color=BLUE, lw=2, label='Model')
axes[1, 1].plot(thr, all_nb, '--', color=ORANGE, lw=1.4, label='Test everyone')
axes[1, 1].axhline(0, color=GREY, ls=':', lw=1, label='Test no one')
axes[1, 1].set_ylim(min(-.02, nb.min() - .01), max(nb.max(), .05) * 1.35)
axes[1, 1].set_xlabel('Threshold probability'); axes[1, 1].set_ylabel('Net benefit')
axes[1, 1].set_title('Decision curve (clinical net benefit)')
axes[1, 1].legend(fontsize=8, frameon=False)

plt.tight_layout(); plt.show()
""")

    code("""
ce = final['calibration_effect']
print('Calibration matters for a score used to choose a prophylaxis regimen:')
print(f"  Brier  uncalibrated -> calibrated : {ce['brier_uncalibrated']:.4f} -> "
      f"{ce['brier_calibrated']:.4f}")
print(f"  ECE    uncalibrated -> calibrated : {ce['ece_uncalibrated']:.4f} -> "
      f"{ce['ece_calibrated']:.4f}")
print()
print('Of the patients scored at 30%, about 30% should develop inhibitors.')
print('Neither reference work reports this.')
""")

    # ======================================================= OPERATING POINT
    md(f"""
## 10. The operating point: {v['point_name']}

This is where the two versions of this notebook differ. Everything above is
identical; what follows is the choice of decision threshold, and it changes
the reported accuracy completely while leaving the model untouched.

All thresholds are fixed on **calibrated training-fold predictions**. The test
set selects nothing.
""")
    code("""
you = final['test_calibrated_youden']
sen = final['test_calibrated_sens90']
acc = final['test_calibrated_accuracy']
ctx = final['accuracy_context']

rows = []
for label, m in [('Balanced (Youden J)', you), ('Rule-out (90% sens)', sen),
                 ('Accuracy-maximising', acc)]:
    rows.append({'operating point': label, 'threshold': m['threshold'],
                 'accuracy': m['accuracy'], 'sensitivity': m['sensitivity'],
                 'specificity': m['specificity'], 'precision': m['precision'],
                 'NPV': m['npv'], 'MCC': m['mcc'],
                 'caught': m['tp'], 'missed': m['fn']})
rows.append({'operating point': 'Predict "no" for everyone',
             'threshold': 1.0, 'accuracy': ctx['majority_class_accuracy'],
             'sensitivity': 0.0, 'specificity': 1.0, 'precision': 0.0,
             'NPV': round(1 - ctx['prevalence'], 4), 'MCC': 0.0,
             'caught': 0, 'missed': int(you['tp'] + you['fn'])})
pd.DataFrame(rows)
""")

    code("""
from sklearn.metrics import accuracy_score

grid = np.linspace(0.01, 0.99, 200)
accs = [accuracy_score(y_te, (p_cal >= t).astype(int)) for t in grid]
senss = [((p_cal >= t) & (y_te == 1)).sum() / y_te.sum() for t in grid]

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(grid, np.array(accs) * 100, color=BLUE, lw=2, label='Accuracy')
ax.plot(grid, np.array(senss) * 100, color=RED, lw=2, label='Sensitivity')
ax.axhline(ctx['majority_class_accuracy'] * 100, color=GREY, ls='--', lw=1.4,
           label=f"All-negative baseline ({ctx['majority_class_accuracy']*100:.1f}%)")
ax.axhline(85, color=ORANGE, ls=':', lw=1.4, label='85% target')

SHOWN_THR = final['thresholds']['%%THR_KEY%%']
ax.axvline(SHOWN_THR, color=GREEN, lw=1.8)
ax.text(SHOWN_THR + .012, 96, 'this notebook\\'s\\noperating point',
        fontsize=8, color=GREEN, fontweight='bold')

ax.set_xlabel('Decision threshold'); ax.set_ylabel('%')
ax.set_ylim(0, 104)
ax.set_title('Accuracy and sensitivity trade off against each other')
ax.legend(fontsize=8, frameon=False, loc='center right')
plt.tight_layout(); plt.show()

print('Accuracy rises only as the model stops predicting inhibitors.')
print('The 85% line is never crossed at any threshold.')
""".replace("%%THR_KEY%%", v["thr_key"]))

    code(f"""
from sklearn.metrics import confusion_matrix

m = final['{v["key"]}']
cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])

fig, ax = plt.subplots(figsize=(4.6, 4))
im = ax.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{{cm[i, j]}}', ha='center', va='center', fontsize=16,
                fontweight='bold',
                color='white' if cm[i, j] > cm.max() * .55 else '#131a21')
ax.set_xticks([0, 1]); ax.set_xticklabels(['predicted\\nno inhibitor', 'predicted\\ninhibitor'])
ax.set_yticks([0, 1]); ax.set_yticklabels(['actual\\nno inhibitor', 'actual\\ninhibitor'])
ax.set_title("Confusion matrix — {v['point_name']}")
ax.grid(False)
plt.tight_layout(); plt.show()

print(f"threshold        : {{m['threshold']}}")
print(f"ACCURACY         : {{m['accuracy']:.4f}}")
print(f"  baseline       : {{ctx['majority_class_accuracy']:.4f}}  (predict 'no' for everyone)")
print(f"  margin         : {{m['accuracy'] - ctx['majority_class_accuracy']:+.4f}}")
print(f"sensitivity      : {{m['sensitivity']:.4f}}   ({{m['tp']}} of {{m['tp'] + m['fn']}} cases caught)")
print(f"specificity      : {{m['specificity']:.4f}}")
print(f"balanced accuracy: {{m['balanced_accuracy']:.4f}}")
print(f"MCC              : {{m['mcc']:.4f}}")
""")

    if variant == "clinical":
        md("""
### What this operating point is for

This threshold balances the two error types. It catches **59 of 92** at-risk
patients at 64% specificity, and the rule-out threshold in the table above
reaches **87% sensitivity with 92.2% negative predictive value**.

Accuracy here is **64.35%** — *below* the 80% you get from predicting "no
inhibitor" for everyone. That is not a defect of the model; it is what happens
when a model actually predicts the minority class on an imbalanced outcome.
Accuracy rewards agreeing with the majority, and this threshold deliberately
does not.

The metrics that describe this model honestly are AUC-ROC 0.727, AUC-PR 0.480
against a 0.200 baseline, balanced accuracy 64.3%, and MCC 0.232.

**This is the version that would be deployed.** The companion notebook,
`Hemophilia_Capstone_Accuracy.ipynb`, shows the same model at the
accuracy-maximising threshold.
""")
    else:
        md("""
### What this operating point is for

This threshold maximises plain accuracy, which is the metric review panels
most often ask for. It reaches **83.04%** against an all-negative baseline of
**80.00%** — a genuine margin of **+3.04 points** over doing nothing.

It is important to be equally clear about the cost. At this threshold the model
flags 16 patients and **misses 76 of 92 actual cases**. Specificity is 99.5%
because it almost never predicts the positive class. On a 20%-prevalence
outcome the accuracy-maximising model is close to the do-nothing model, and
that is arithmetic rather than a property of this particular classifier.

Note also that **no threshold anywhere on the curve reaches 85%** — the chart
above shows the accuracy line topping out below that target.

The metrics that cannot be gamed this way — AUC-ROC 0.727, AUC-PR 0.480 against
a 0.200 baseline, MCC 0.348 — are unchanged, because the model is unchanged.

**The companion notebook, `Hemophilia_Capstone_Clinical.ipynb`, shows the same
model at the threshold that would actually be deployed.**
""")

    # ============================================================= subgroups
    md("""
## 11. Where the model works, and where it does not

A pooled AUC hides the question a clinician actually asks: does this work for
the patients I would use it on? Neither reference work reports this.
""")
    code("""
sub = load('subgroups')
sg = pd.DataFrame(sub['subgroups'])
sg[['subgroup', 'n', 'events', 'prevalence', 'auc_roc', 'auc_ci', 'note']]
""")

    code("""
ok = sg[sg['auc_roc'].notna()].copy()
fig, ax = plt.subplots(figsize=(9, .42 * len(ok) + 1.5))
cols = [RED if a < .60 else ORANGE if a < .70 else BLUE for a in ok['auc_roc']]
ax.barh(range(len(ok)), ok['auc_roc'], color=cols, height=.6)
ax.axvline(.5, color=RED, ls='--', lw=1.4, label='Chance')
for i, (a, n, e) in enumerate(zip(ok['auc_roc'], ok['n'], ok['events'])):
    ax.text(a + .008, i, f'{a:.3f}  (n={n}, {e} events)', va='center', fontsize=8)
ax.set_yticks(range(len(ok))); ax.set_yticklabels(ok['subgroup'], fontsize=9)
ax.invert_yaxis(); ax.set_xlim(.45, 1.0); ax.set_xlabel('AUC-ROC')
ax.set_title('Performance inside each clinical stratum')
ax.legend(fontsize=8, frameon=False, loc='lower right')
plt.tight_layout(); plt.show()
""")

    md("""
### The most important caveat in this project

The overall AUC of 0.727 is not evenly distributed. Inside the **severe**
stratum — where essentially every prophylaxis decision is made — it falls to
**0.694**. Inside **truncating variants alone** it is **0.541**, which is
indistinguishable from chance.

The reading is uncomfortable and clear: most of the model's apparent
discrimination comes from separating null variants from non-null ones, and a
clinician already knows that from the variant type without any model. Within
the high-risk group, where a tool would actually change management, this model
adds very little.

That is a limit of the data rather than of the fitting. Whether a particular
severe, null-variant patient develops an inhibitor depends on treatment
intensity, product type, age at first exposure and HLA haplotype — none of
which CHAMP records.
""")

    # ============================================================== external
    md("""
## 12. External validation on a second gene

The F8 model is applied unchanged to hemophilia **B** patients from the CDC
CHBMP database. Different gene, different protein, and no F9 patient took any
part in training, feature fitting or threshold selection.

Nothing F8-specific can transfer. What can is the immunology: a null variant
abolishes the protein, so the patient was never tolerised to the factor they
are later infused with. A model that memorised F8 would score chance here.
""")
    code("""
ext = load('external')
print('cohort      :', ext['cohort'])
print(f"patients    : {ext['n_scored']}  events: {ext['n_events']}  "
      f"prevalence: {ext['prevalence']:.1%}")
print(f"AUC-ROC     : {ext['auc_ci']['point']:.4f} "
      f"({ext['auc_ci']['lo']:.4f} - {ext['auc_ci']['hi']:.4f})")
print()
pd.Series(ext['metrics'])[['auc_roc', 'auc_pr', 'auc_pr_baseline', 'sensitivity',
                           'specificity', 'balanced_accuracy', 'mcc']]
""")

    code("""
fig, ax = plt.subplots(figsize=(6.4, 4))
labels = ['CHAMP (F8)\\ninternal test', 'CHBMP (F9)\\nzero-shot transfer']
vals = [final['auc_ci']['point'], ext['auc_ci']['point']]
los = [final['auc_ci']['lo'], ext['auc_ci']['lo']]
his = [final['auc_ci']['hi'], ext['auc_ci']['hi']]
err = np.array([[v - l for v, l in zip(vals, los)], [h - v for v, h in zip(vals, his)]])
ax.bar([0, 1], vals, .45, yerr=err, color=[BLUE, GREEN],
       error_kw=dict(ecolor=GREY, lw=1.2, capsize=6))
ax.axhline(.5, color=RED, ls='--', lw=1.2, label='Chance')
for xi, val in zip([0, 1], vals):
    ax.text(xi, val + .02, f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1.0); ax.set_ylabel('AUC-ROC (95% CI)')
ax.set_title('Internal and cross-gene external validation')
ax.legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.show()

print(ext['caveat'])
""")

    # =================================================================== ssl
    md("""
## 13. The 1,744 unrecorded outcomes

Rather than relabelling them, they are used properly — and the cost of the
reference pipeline's choice is quantified.
""")
    code("""
ssl = load('ssl')
p = ssl['reporting_bias_probe']; u = ssl['unlabelled_risk_profile']
print(f"Missingness probe AUC : {p['reporting_auc']:.4f}")
print(f"  interpretation      : {p['interpretation']}")
print()
print(f"Unlabelled pool ({u['n_unlabelled']} rows):")
print(f"  mean predicted risk : {u['mean_predicted_risk']:.4f}")
print(f"  flagged at 0.5      : {u['predicted_positive_at_0.5']} "
      f"({u['predicted_positive_fraction_at_0.5']:.1%})")
print()
print(f"So relabelling them 0 injects roughly {u['predicted_positive_at_0.5']} "
      f"false negatives into training.")
print()
print(f"Self-training: {ssl['supervised_test_auc']:.4f} -> "
      f"{ssl['semisupervised_test_auc']:.4f}  "
      f"(DeLong p = {ssl['delong_ssl_vs_supervised']['p_value']})")
print('Not a significant improvement -- reported as such rather than claimed.')
""")

    # ============================================================== explain
    md("""
## 14. Explainability

SHAP is retained because it is the field standard, and two things are added:
block-level attribution (stable under the feature correlation that makes
per-feature SHAP noisy) and, for the attention network, intrinsic per-patient
weights that are part of the forward pass rather than a post-hoc approximation.
""")
    code("""
from src.predict import InhibitorRiskModel

model = InhibitorRiskModel()

high_risk = {'HGVS cDNA': 'c.6496C>T', 'HGVS Protein': 'p.(Arg2166*)',
             'Variant Type': 'Nonsense', 'Mechanism': 'Substitution',
             'Exon': '23', 'Domain': 'C1', 'Subtype': 'Light chain',
             'In Poly A': 'N', 'Reported Clinical Severity': 'Severe'}
low_risk = {'HGVS cDNA': 'c.103T>C', 'HGVS Protein': 'p.(Tyr35His)',
            'Variant Type': 'Missense', 'Mechanism': 'Substitution',
            'Exon': '1', 'Domain': 'A1', 'Subtype': 'Heavy chain',
            'In Poly A': 'N', 'Reported Clinical Severity': 'Mild'}

for name, rec in [('nonsense / severe / C1', high_risk),
                  ('missense / mild / A1', low_risk)]:
    r = model.predict(rec)[0]
    print(f"{name:26s} risk {r['probability']:.4f}   band {r['risk_band']:<10s} "
          f"{r['prediction']}")
""")

    code("""
expl = model.explain(high_risk, top=8)
print('Why the nonsense / severe patient scores high:')
expl
""")

    code("""
e = expl.iloc[::-1]
fig, ax = plt.subplots(figsize=(8.4, 3.6))
ax.barh(range(len(e)), e['shap'],
        color=[RED if s > 0 else GREEN for s in e['shap']], height=.62)
ax.axvline(0, color='#131a21', lw=1)
ax.set_yticks(range(len(e))); ax.set_yticklabels(e['feature'], fontsize=8.5)
ax.set_xlabel('SHAP value  (left = lowers risk, right = raises risk)')
ax.set_title('Per-patient attribution — nonsense / severe / C1')
plt.tight_layout(); plt.show()
""")

    # ================================================================ fused
    md("""
## 15. A second dataset that appeared to solve the problem

Section 17 says the binding constraint is the absence of patient-level
covariates. A collaborator supplied exactly those — CHAMP with age at
diagnosis, ethnicity, treatment regimen, exposure days and family history
appended — in a file reporting accuracy in the high eighties.

It was tested rather than adopted.
""")
    code("""
fa = load('fused_audit'); prov = fa['provenance']
print('PROVENANCE CHECKS')
print(f"  Patient_ID matching UUID4    : "
      f"{prov['patient_id']['fraction_matching_uuid4']:.0%}")
print(f"  Ethnicity chi-square p       : {prov['ethnicity']['chi2_p']} "
      f"(spread {prov['ethnicity']['spread_pct']} points)")
print(f"  Family history odds ratio    : {prov['family_history']['odds_ratio']} "
      f"vs published {prov['family_history']['published_odds_ratio']}")
print(f"  Age vs exposure-days r       : {prov['age_vs_exposure']['pearson_r']}")
print()
print('Ethnicity inhibitor rate by group (%):')
for g, r in prov['ethnicity']['inhibitor_rate_pct'].items():
    print(f'    {g:12s} {r}')
print()
print('Roughly 2x higher inhibitor risk in Black and Hispanic patients is among')
print('the most reproducible non-genetic findings in this field (CDC, MLOF,')
print('UKHCDO). A real cohort of 4,026 would show it. This one is flat.')
""")

    code("""
sim = load('fused_simulation')
rows = [{'arm': lab, 'features': sim[k]['n_features'],
         'CV AUC': sim[k]['cv_auc_mean'],
         'held-out AUC': sim[k]['test_auc_ci']['point'],
         '95% CI': f"{sim[k]['test_auc_ci']['lo']:.3f}-{sim[k]['test_auc_ci']['hi']:.3f}"}
        for k, lab in [('genomic_only', 'Genomic only'),
                       ('clinical_only', 'Clinical only'),
                       ('genomic_plus_clinical', 'Genomic + clinical')]]
print('Same folds, same held-out patients, same featuriser.')
print('The only difference is whether the clinical block is present.')
print()
display(pd.DataFrame(rows))
print()
print('DeLong (genomic+clinical vs genomic):', sim['_delong_gain'])
print()
print('Adding the clinical block does NOT improve held-out performance.')
print('CV AUC rises while held-out AUC falls -- fitting injected noise.')
""")

    code("""
fig, ax = plt.subplots(figsize=(7.6, 3.6))
arms = ['Genomic\\nonly', 'Clinical\\nonly', 'Genomic +\\nclinical']
cvv = [sim[k]['cv_auc_mean'] for k in
       ['genomic_only', 'clinical_only', 'genomic_plus_clinical']]
tev = [sim[k]['test_auc_ci']['point'] for k in
       ['genomic_only', 'clinical_only', 'genomic_plus_clinical']]
x = np.arange(3)
ax.bar(x - .2, cvv, .4, color=GREY, label='Cross-validation AUC')
ax.bar(x + .2, tev, .4, color=BLUE, label='Held-out AUC')
for xi, (c, t) in enumerate(zip(cvv, tev)):
    ax.text(xi - .2, c + .006, f'{c:.3f}', ha='center', fontsize=8)
    ax.text(xi + .2, t + .006, f'{t:.3f}', ha='center', fontsize=8)
ax.axhline(.5, color=RED, ls='--', lw=1.2)
ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=9)
ax.set_ylim(.45, .82); ax.set_ylabel('AUC-ROC')
ax.set_title('The simulated clinical covariates add nothing')
ax.legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.show()
""")

    md("""
The file's label also counts CHAMP's 1,731 unrecorded outcomes as negatives,
which moves prevalence to 11.4% and lifts the all-negative baseline to
**88.55%**. A model trained on it scores **89.58%** — one point better than
doing nothing, while catching **13%** of actual cases.

That is how a dataset reaches "high-eighties accuracy" without a working model.
Read the other way round the file is still useful: it is a serviceable power
analysis for what real registry covariates would need to be, and it is reported
here as a simulation, which is what it is.
""")

    # ============================================================ integrity
    md("""
## 16. Pipeline integrity

The claim that these numbers are trustworthy is worth no more than the checks
behind it, so each property is verified mechanically rather than asserted.
These also run as part of the test suite.
""")
    code("""
integ = load('integrity')
rows = [{'check': r.get('check', k),
         'result': {True: 'PASS', False: 'FAIL', None: 'skip'}[r.get('passed')]}
        for k, r in integ.items() if not k.startswith('_')]
display(pd.DataFrame(rows))
print()
s = integ['_summary']
print(f"{s['passed']} of {s['n_checks']} checks passed.")
print()
print('Two worth spelling out:')
print('  - No resampling. Imbalance is handled by weighting the objective,')
print('    never by duplicating or synthesising patients. The reference')
print("    pipeline's over-sampling put half its test set into its training data.")
print('  - The featuriser is label-blind. Scrambling the outcome and refitting')
print('    produces a byte-identical feature matrix, so the engineering cannot')
print('    have absorbed the answer.')
""")

    # ========================================================== limitations
    md("""
## 17. Limitations

Stated because a model for clinical use is only as trustworthy as its declared
boundaries.

1. **Most of the discrimination is null-versus-non-null, which is already
   known.** Pooled AUC is 0.727, but inside the severe stratum it is 0.694 and
   inside truncating variants alone it is 0.541 — chance. The model largely
   reproduces a distinction the variant type already gives a clinician for free.
2. **CHAMP is a variant catalogue, not a patient registry.** Each row is a
   distinct mutation whose outcome is summarised across everyone reported to
   carry it, so the label carries irreducible noise.
3. **The strongest known risk factors are absent.** Treatment intensity,
   product type, exposure days, HLA haplotype, family history and ethnicity all
   drive inhibitor development, and none are in the data.
4. **Informative missingness.** The probe in section 13 reaches AUC 0.601, so
   the labelled subset is not a random sample of CHAMP.
5. **Reporting bias.** CHAMP aggregates published case reports, which
   over-represent unusual variants and notable outcomes.
6. **The external cohort is small.** CHBMP contributes 351 patients with 40
   events, so its interval is wide. It establishes that transfer happens, not
   how well.
7. **Not a medical device.** Research and educational use only. It does not
   replace clinical judgement or laboratory inhibitor testing.
""")

    # ============================================================ conclusion
    md("## 18. Conclusion")
    code("""
print('=' * 68)
print('  FINAL RESULTS')
print('=' * 68)
print()
print(f"  Dataset            CHAMP (CDC), {len(labelled)} labelled patients")
print(f"  Prevalence         {y_all.mean():.1%} inhibitor-positive")
print(f"  Features           {X_all.shape[1]} leakage-free biological descriptors")
print(f"  Model              {final['selected_model']}, isotonic-calibrated")
print(f"  Held-out test      {final['n_test']} patients, {final['test_events']} events")
print()
print(f"  AUC-ROC            {final['auc_ci']['point']:.4f}  "
      f"({final['auc_ci']['lo']:.4f} - {final['auc_ci']['hi']:.4f})")
print(f"  AUC-PR             {final['auc_pr_ci']['point']:.4f}  "
      f"(baseline {you['prevalence']})")
print(f"  External AUC (F9)  {ext['auc_ci']['point']:.4f}  "
      f"({ext['auc_ci']['lo']:.4f} - {ext['auc_ci']['hi']:.4f})")
print(f"  Calibration ECE    {you['ece']:.4f}  (from {ce['ece_uncalibrated']:.4f})")
print()
""")

    code(f"""
m = final['{v["key"]}']
print('-' * 68)
print("  OPERATING POINT: {v['point_name']}")
print('-' * 68)
print(f"  Threshold          {{m['threshold']}}")
print()
print(f"  ACCURACY           {{m['accuracy']:.4f}}   ({{m['accuracy']*100:.2f}}%)")
print(f"  Baseline           {{ctx['majority_class_accuracy']:.4f}}   "
      f"(predict 'no inhibitor' for everyone)")
print(f"  Margin             {{m['accuracy'] - ctx['majority_class_accuracy']:+.4f}}")
print()
print(f"  Sensitivity        {{m['sensitivity']:.4f}}   "
      f"({{m['tp']}} of {{m['tp'] + m['fn']}} cases caught)")
print(f"  Specificity        {{m['specificity']:.4f}}")
print(f"  Balanced accuracy  {{m['balanced_accuracy']:.4f}}")
print(f"  Precision / NPV    {{m['precision']:.4f}} / {{m['npv']:.4f}}")
print(f"  MCC                {{m['mcc']:.4f}}")
print('=' * 68)
""")

    if variant == "clinical":
        md("""
### What this project claims

Not "we reached 90% accuracy" — that number is not reachable on this problem
without a label policy we consider indefensible, and section 15 shows what
happens to a dataset that appears to reach it.

What it claims is this: **the 97.37% and 99.63% figures in our reference works
are artifacts, we proved it with seven controlled experiments including a
label-permutation control, and we built a replacement that survives cross-gene
external validation at AUC 0.750.**

At this operating point the model catches 64% of at-risk patients, and at the
rule-out threshold 87% with a 92.2% negative predictive value. Accuracy of
64.35% is lower than the do-nothing baseline precisely *because* the model
predicts the minority class, which is the entire point of building it.

### Future work

The ceiling here is data, not method. In order of expected value: patient-level
registry data (PedNet, ATHN, MLOF) for exposure days, product type and
treatment intensity; HLA class II typing; an AlphaFold-derived FVIII structure
to replace linear sequence distance with true spatial proximity; and
prospective, patient-level validation before any clinical use.
""")
    else:
        md("""
### What this project claims

**83.04% accuracy against an 80.00% no-skill baseline** — a real margin of
+3.04 points — with AUC-ROC 0.727, AUC-PR 0.480 against a 0.200 baseline, and
cross-gene external validation at AUC 0.750.

The accuracy figure is reported here with its baseline attached, always,
because on a 20%-prevalence outcome it means very little without one. At this
threshold the model misses 76 of 92 cases, so the companion notebook's
operating point is the one that would be deployed.

The larger claim does not rest on accuracy at all: **the 97.37% and 99.63%
figures in our reference works are artifacts, we proved it with seven
controlled experiments including a label-permutation control, and we built a
replacement that survives external validation on a different gene.**

### Future work

The ceiling here is data, not method. In order of expected value: patient-level
registry data (PedNet, ATHN, MLOF) for exposure days, product type and
treatment intensity; HLA class II typing; an AlphaFold-derived FVIII structure
to replace linear sequence distance with true spatial proximity; and
prospective, patient-level validation before any clinical use.
""")

    md("""
---

### References

1. V. K. Singh and M. P. Singh, "Predicting Inhibitor Development in Hemophilia
   'A' using Machine Learning," *Curr. Pharm. Biotechnol.*, 26(13), 2014–2030, 2025.
2. J. M. Payne *et al.*, "The CDC Hemophilia A Mutation Project (CHAMP) mutation
   list," *Hum. Mutat.*, 34(2), 2013.
3. T. Li *et al.*, "The CDC Hemophilia B mutation project mutation list,"
   *Mol. Genet. Genomic Med.*, 1(4), 2013.
4. E. Berntorp *et al.*, "Haemophilia," *Nat. Rev. Dis. Primers*, 7(1):45, 2021.
5. T.-Y. Lin *et al.*, "Focal Loss for Dense Object Detection," *ICCV*, 2017.
6. S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model
   Predictions," *NeurIPS*, 2017.
7. E. R. DeLong *et al.*, "Comparing the Areas under Two or More Correlated ROC
   Curves," *Biometrics*, 44(3), 1988.
8. A. J. Vickers and E. B. Elkin, "Decision Curve Analysis," *Med. Decis.
   Making*, 26(6), 2006.
9. G. S. Collins *et al.*, "Transparent Reporting of a multivariable prediction
   model (TRIPOD)," *BMJ*, 350:g7594, 2015.
10. W. J. Youden, "Index for rating diagnostic tests," *Cancer*, 3(1), 1950.

**Data**: CDC CHAMP and CHBMP 2022 releases · UniProt P00451 · RefSeq NM_000132.4

**Code**: <https://github.com/dchaudhari7177/hemophilia>
""")

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with io.open(v["file"], "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, indent=1)
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"wrote {v['file']}  ({len(cells)} cells, {n_code} code)")


if __name__ == "__main__":
    for name in (sys.argv[1:] or list(VARIANTS)):
        build(name)
