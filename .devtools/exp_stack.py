"""Screen stack compositions on the cached out-of-fold prediction matrix.

Trees and neural nets make different errors; the current stack is all-classical
and therefore highly correlated. This screens which member sets are worth a
full nested run.
"""
import sys, warnings, itertools
sys.path.insert(0, '.'); warnings.filterwarnings('ignore')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

d = np.load('reports/cv_oof.npz')
y = d['y']
names = [k for k in d.files if k != 'y']
OOF = {n: d[n] for n in names}
cv = StratifiedKFold(5, shuffle=True, random_state=42)

print('individual out-of-fold AUCs:')
singles = sorted(((roc_auc_score(y, OOF[n]), n) for n in names), reverse=True)
for a, n in singles:
    print(f'  {n:26s} {a:.4f}')

# how correlated are the members?
print('\ncorrelation between the top classical and top neural model:')
print(f"  ExtraTrees vs DeepMLP : {np.corrcoef(OOF['ExtraTrees'], OOF['DeepMLP'])[0,1]:.3f}")
print(f"  ExtraTrees vs RandomForest : {np.corrcoef(OOF['ExtraTrees'], OOF['RandomForest'])[0,1]:.3f}")

def stack_auc(members, C=1.0):
    M = np.column_stack([OOF[m] for m in members])
    p = cross_val_predict(
        LogisticRegression(C=C, max_iter=5000, class_weight='balanced',
                           random_state=42), M, y, cv=cv, method='predict_proba')[:, 1]
    return roc_auc_score(y, p)

def rank_avg(members):
    from scipy.stats import rankdata
    R = np.mean([rankdata(OOF[m]) for m in members], axis=0)
    return roc_auc_score(y, R)

CLASSICAL = ['LogisticRegression', 'ExtraTrees', 'RandomForest', 'LightGBM', 'XGBoost']
NEURAL = ['DeepMLP', 'BioBlockAttention', 'TabTransformer', 'CNN1D']

configs = {
    'current stack (5 classical)': CLASSICAL,
    'classical + DeepMLP': CLASSICAL + ['DeepMLP'],
    'classical + all 4 DNNs': CLASSICAL + NEURAL,
    'ET + RF + DeepMLP + BioBlock': ['ExtraTrees', 'RandomForest', 'DeepMLP', 'BioBlockAttention'],
    'ET + DeepMLP': ['ExtraTrees', 'DeepMLP'],
    'ET + DeepMLP + TabTransformer': ['ExtraTrees', 'DeepMLP', 'TabTransformer'],
    'ET + DeepMLP + BioBlock + LR': ['ExtraTrees', 'DeepMLP', 'BioBlockAttention', 'LogisticRegression'],
    'diverse 6': ['ExtraTrees', 'RandomForest', 'LogisticRegression',
                  'DeepMLP', 'BioBlockAttention', 'TabTransformer'],
    'all 14': names,
}
print(f"\n{'stack composition':34s}{'LR meta':>10s}{'rank-avg':>10s}")
best = (0, None)
for label, members in configs.items():
    members = [m for m in members if m in OOF]
    a, r = stack_auc(members), rank_avg(members)
    print(f'{label:34s}{a:10.4f}{r:10.4f}', flush=True)
    for v, kind in ((a, 'LR'), (r, 'rank')):
        if v > best[0]:
            best = (v, f'{label} [{kind}]')
print(f'\nbest screened: {best[1]}  AUC {best[0]:.4f}')
print(f"single best model for reference: {singles[0][1]} {singles[0][0]:.4f}")
