"""Can more feature engineering move the needle? Measured, not assumed."""
import sys, warnings
sys.path.insert(0, '.'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier

from src.train import prepare
from src.models import build_pipeline
from src.features import parse_exon

d = prepare()
X, y, tr = d['X'], d['y'], d['train_idx']
lab = d['labelled']
cv = StratifiedKFold(5, shuffle=True, random_state=42)
et = lambda: ExtraTreesClassifier(n_estimators=600, min_samples_leaf=4,
                                  max_features='sqrt', class_weight='balanced',
                                  n_jobs=-1, random_state=42)

def score(M, label):
    p = cross_val_predict(build_pipeline(et()), M[tr], y[tr], cv=cv,
                          method='predict_proba')[:, 1]
    a = roc_auc_score(y[tr], p)
    print(f'{label:38s} k={M.shape[1]:3d}  AUC {a:.4f}', flush=True)
    return a

base = score(X, 'baseline (current 135 features)')

# 1. exon identity as one-hot -- exon 14 is the B domain, 26 is terminal
ex = np.array([parse_exon(v) for v in lab['Exon']], dtype=float)
onehot = np.zeros((len(lab), 27))
for i, e in enumerate(ex):
    if not np.isnan(e) and 1 <= e <= 26:
        onehot[i, int(e)] = 1.0
    else:
        onehot[i, 0] = 1.0
score(np.hstack([X, onehot]), '+ exon one-hot (27)')

# 2. pairwise products of the strongest predictors
names = d['feature_names']
idx = {n: i for i, n in enumerate(names)}
key = [n for n in ['is_null_mutation', 'severity_ordinal', 'severity_severe',
                   'vtype_missense', 'is_light_chain', 'in_b_domain',
                   'fraction_protein_lost', 'exon_norm', 'mature_pos_norm']
       if n in idx]
Xi = X[:, [idx[n] for n in key]]
prods = np.hstack([(Xi[:, [i]] * Xi[:, [j]]) for i in range(len(key))
                   for j in range(i + 1, len(key))])
score(np.hstack([X, prods]), f'+ pairwise interactions ({prods.shape[1]})')

# 3. both
score(np.hstack([X, onehot, prods]), '+ exon one-hot + interactions')
