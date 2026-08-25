"""Build the reviewer-facing notebook from a list of markdown/code cells."""
import io
import json

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip().split("\n")})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip().split("\n")})


md("""
# Explainable FVIII Inhibitor Risk Classification in Hemophilia A

**PES University B.Tech Capstone — PW_GRS_01**

Dipak Chaudhari · Tejas Nagmote · Sneha A · Varsha P — Guide: Prof. Gayathri R S

This notebook walks through the project end to end: what the reference results
claimed, why they do not reproduce, what was built instead, and how it performs
on data it has never seen.

Run every cell top to bottom. The heavy stages read cached measurements from
`reports/`, so run `python -m src.train --stage all` once first.
""")

md("## 0. Setup")
code("""
import sys, json, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd
from IPython.display import Image, display
pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 40)

from src.datasets import load_champ, load_chbmp, split_by_label, label_summary
from src.features import VariantFeaturizer, normalise_variant_type
print('ready')
""")

md("""
## 1. The data

CHAMP is the CDC Hemophilia A Mutation Project: a curated catalogue of F8
variants, each annotated with clinical severity and — where the source
publication reported it — whether the patient developed an inhibitor against
infused Factor VIII.
""")
code("""
champ = load_champ()
print('CHAMP shape:', champ.shape)
champ[['HGVS cDNA', 'Variant Type', 'Mechanism', 'Exon', 'Domain',
       'Reported Clinical Severity', 'History of Inhibitor']].head(8)
""")

md("""
### 1.1 The label has three states, not two

This is the first place the reference pipeline goes wrong. `Not reported` means
the source publication did not state the outcome. It does not mean the outcome
was negative.
""")
code("""
print(champ['History of Inhibitor'].value_counts(dropna=False).to_string())
print()
for k, v in label_summary(champ).items():
    print(f'{k:48s} {v}')
""")

md("""
Prevalence among genuinely labelled patients is **20.1%**, matching the 20–40%
inhibitor incidence the literature reports for severe hemophilia A. Relabelling
the unknowns as negative drives it to 11.4% and inflates accuracy purely by
padding the majority class.
""")

md("""
### 1.2 The identifier problem

`HGVS cDNA` is a *name* for the variant, not a property of it.
""")
code("""
labelled, unlabelled = split_by_label(champ)
print('rows in CHAMP                  :', len(champ))
print('distinct HGVS cDNA values      :', champ['HGVS cDNA'].nunique())
print('labelled rows                  :', len(labelled))
print('duplicated HGVS among labelled :', labelled['HGVS cDNA'].duplicated().sum())
print()
print('Among labelled patients the column is a primary key.')
print('Label-encoding it hands the model a lookup table.')
""")

md("""
### 1.3 The biology that is genuinely present

Before any modelling, the effect the literature predicts is visible in the raw
counts: null variants abolish FVIII entirely, so the immune system was never
tolerised to the protein the patient is later infused with.
""")
code("""
y = (labelled['inhibitor'] == 1).astype(int).values
vt = np.array([normalise_variant_type(v) for v in labelled['Variant Type']])
tab = (pd.DataFrame({'variant_type': vt, 'inhibitor': y})
         .groupby('variant_type')['inhibitor']
         .agg(n='size', positives='sum', rate='mean')
         .sort_values('rate', ascending=False))
tab['rate_pct'] = (tab.pop('rate') * 100).round(1)
tab
""")

md("## 2. Auditing the reference result")
md("""
The prior notebook reported 99.63% accuracy and AUC 0.9999; the classical-ML
paper it benchmarks against reports 97.37% accuracy. Rather than argue about
those numbers, we run the reference preprocessing verbatim and change one thing
at a time.
""")
code("""
audit = json.load(open('reports/audit.json'))
labels = {
    'A_reference_pipeline': 'Reference pipeline, verbatim',
    'B_identifiers_only': 'Identifier columns only',
    'C_no_identifiers': 'Biology only (identifiers removed)',
    'D_honest_labels': 'Reference features, unknowns dropped',
    'E_label_permutation': 'Labels shuffled (control)',
    'F_novel_variant_split': 'Novel-variant split',
    'G_oversample_before_split': 'Over-sampled before splitting',
}
rows = [{'experiment': lab,
         'train AUC': audit[k]['train_auc'],
         'test AUC': audit[k]['test_auc'],
         'test acc': audit[k]['test_accuracy'],
         'majority acc': audit[k]['majority_class_accuracy']}
        for k, lab in labels.items() if k in audit]
pd.DataFrame(rows)
""")

md("""
Read that table carefully.

* **Row A** — the reference pipeline on a clean split scores 0.639 AUC, and its
  86.4% accuracy is *below* the 88.6% you get by predicting "no inhibitor" for
  every patient.
* **Row E** — with the labels shuffled, training AUC stays pinned at 1.0. A
  model that fits pure noise perfectly is memorising, not learning.
* **Row G** — over-sampling before the split yields 95.3% accuracy and 0.994
  AUC. That is where the published 97.37% comes from.
""")
code("""
g = audit['G_oversample_before_split']
print('test rows that are verbatim copies of training rows: '
      f"{g['test_rows_also_in_train']} of {g['n_test']} "
      f"({g['fraction_test_rows_duplicated_from_train']:.0%})")
""")
code("""
display(Image('reports/figures/01_leakage_audit.png'))
""")

md("## 3. The replacement feature set")
md("""
The HGVS string is discarded; what it *means* is kept, grouped into seven
biological blocks.
""")
code("""
fz = VariantFeaturizer().fit(champ)
X = fz.transform(champ)
print('design matrix:', X.shape)
print()
for block, cols in fz.blocks_.items():
    print(f'{block:14s} {len(cols):3d}   e.g. {", ".join(cols[:4])}')
""")
code("""
# No engineered feature may be near-unique; a regression test enforces this.
u = X.nunique().sort_values(ascending=False)
print('highest-cardinality features, as a fraction of rows:')
print((u.head(6) / len(X)).round(3).to_string())
print()
q = json.load(open('reports/quantisation.json'))
print(f"cost of quantising position to a 40-bin grid: {q['delta']:+.4f} AUC")
""")
code("""
display(Image('reports/figures/04_cohort_biology.png'))
""")

md("## 4. Model comparison")
code("""
cv = json.load(open('reports/cv.json'))
blocked = json.load(open('reports/blocked_cv.json'))['models']
rows = []
for name in cv['ranking']:
    r = cv['models'][name]
    o = r['oof_metrics']
    rows.append({'model': name,
                 'CV AUC': f"{r['cv_auc_mean']:.4f} +/- {r['cv_auc_std']:.4f}",
                 'AUC-PR': o['auc_pr'], 'MCC': o['mcc'],
                 'blocked AUC': blocked.get(name, {}).get('blocked_auc_mean')})
pd.DataFrame(rows)
""")
code("""
display(Image('reports/figures/02_model_comparison.png'))
""")

md("## 5. Final model on held-out data")
code("""
final = json.load(open('reports/final.json'))
print('selected model :', final['selected_model'])
print('test AUC (CI)  :', final['auc_ci'])
print('thresholds     :', final['thresholds'])
print()
keys = ['auc_roc', 'auc_pr', 'sensitivity', 'specificity', 'precision', 'npv',
        'balanced_accuracy', 'mcc', 'brier', 'ece']
pd.DataFrame({'Youden': final['test_calibrated_youden'],
              '90% sensitivity': final['test_calibrated_sens90']}).T[keys]
""")
code("""
display(Image('reports/figures/03_performance_panel.png'))
""")

md("## 6. External validation — zero-shot transfer to hemophilia B")
md("""
The F8 model is applied unchanged to hemophilia **B** patients from the CDC
CHBMP database. Different gene, different protein, and no F9 patient took any
part in training, feature fitting or threshold selection. Only mutation-class
immunology can transfer; a model that memorised F8 scores chance here.
""")
code("""
ext = json.load(open('reports/external.json'))
print('cohort      :', ext['cohort'])
print('patients    :', ext['n_scored'], '| events:', ext['n_events'],
      f"| prevalence: {ext['prevalence']:.1%}")
print('AUC (95% CI):', ext['auc_ci'])
pd.Series(ext['metrics'])[['auc_roc', 'auc_pr', 'sensitivity', 'specificity',
                           'balanced_accuracy', 'mcc']]
""")
code("""
display(Image('reports/figures/06_external_validation.png'))
""")

md("## 7. The 1,744 unrecorded outcomes")
code("""
ssl = json.load(open('reports/ssl.json'))
print('missingness probe:')
for k, v in ssl['reporting_bias_probe'].items():
    print(f'  {k:22s} {v}')
print()
print('unlabelled pool:')
for k, v in ssl['unlabelled_risk_profile'].items():
    print(f'  {k:44s} {v}')
print()
print(f"supervised AUC     : {ssl['supervised_test_auc']}")
print(f"semi-supervised AUC: {ssl['semisupervised_test_auc']}")
print('DeLong             :', ssl['delong_ssl_vs_supervised'])
""")

md("## 8. Scoring a patient")
code("""
from src.predict import InhibitorRiskModel

model = InhibitorRiskModel()
patient = {
    'HGVS cDNA': 'c.6496C>T', 'HGVS Protein': 'p.(Arg2166*)',
    'Variant Type': 'Nonsense', 'Mechanism': 'Substitution',
    'Exon': '23', 'Domain': 'C1', 'Subtype': 'Light chain',
    'In Poly A': 'N', 'Reported Clinical Severity': 'Severe',
}
print(json.dumps(model.predict(patient)[0], indent=2))
""")
code("""
model.explain(patient)
""")

md("""
## 9. Conclusions

1. The reference results do not reproduce. Three separate defects — over-sampling
   before the split, relabelling unrecorded outcomes as negative, and feeding
   identifier columns to the model — account for the published numbers between
   them.
2. A leakage-free, biology-informed model on the same database performs at the
   level the underlying biology supports, and reports that level plainly.
3. It is calibrated, it survives a position-blocked split, and it transfers
   zero-shot to a different gene — evidence that it learned mutation-class
   immunology rather than this particular table.
4. The ceiling here is the data, not the method. CHAMP contains no treatment
   intensity, no exposure days and no HLA typing. Those are what a next version
   needs.

Full write-up: `RESULTS.md`. Design rationale: `docs/ARCHITECTURE.md`.
""")

notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with io.open("Hemophilia_Capstone_Final.ipynb", "w", encoding="utf-8",
             newline="\n") as fh:
    json.dump(notebook, fh, indent=1)
print(f"wrote Hemophilia_Capstone_Final.ipynb with {len(CELLS)} cells")
