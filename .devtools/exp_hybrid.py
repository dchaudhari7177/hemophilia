"""Honest nested evaluation of the hybrid candidates."""
import sys, time, warnings, json
sys.path.insert(0, '.'); warnings.filterwarnings('ignore')
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

from src.train import prepare
from src.models import build_pipeline, classical_models, neural_models
from src.hybrid import build_hybrids

d = prepare(); X, y, tr = d['X'], d['y'], d['train_idx']
Xtr, ytr = X[tr], y[tr]
cv = StratifiedKFold(5, shuffle=True, random_state=42)

cands = {'ExtraTrees (current best single)': classical_models()['ExtraTrees'],
         'DeepMLP': neural_models(d['blocks'])['DeepMLP']}
cands.update(build_hybrids(d['blocks']))

print(f"{'model':34s}{'CV AUC':>10s}{'sd':>8s}{'secs':>8s}")
res = {}
for name, mdl in cands.items():
    t0 = time.time(); aucs = []; oof = np.zeros(len(ytr))
    for a, b in cv.split(Xtr, ytr):
        pipe = build_pipeline(mdl).fit(Xtr[a], ytr[a])
        p = pipe.predict_proba(Xtr[b])[:, 1]
        oof[b] = p
        aucs.append(roc_auc_score(ytr[b], p))
    grid = np.unique(np.round(oof, 3))
    thr = float(max(grid, key=lambda t: accuracy_score(ytr, (oof >= t).astype(int))))
    res[name] = {'cv_auc_mean': round(float(np.mean(aucs)), 4),
                 'cv_auc_std': round(float(np.std(aucs)), 4),
                 'oof_auc': round(float(roc_auc_score(ytr, oof)), 4),
                 'oof_best_accuracy': round(float(accuracy_score(ytr, (oof >= thr).astype(int))), 4),
                 'seconds': round(time.time() - t0, 1)}
    r = res[name]
    print(f"{name:34s}{r['cv_auc_mean']:10.4f}{r['cv_auc_std']:8.4f}{r['seconds']:8.1f}", flush=True)

json.dump(res, open('reports/hybrid_screen.json', 'w'), indent=2)
best = max(res, key=lambda k: res[k]['cv_auc_mean'])
print(f"\nbest by CV AUC: {best}  {res[best]['cv_auc_mean']:.4f}")
print(f"train-OOF accuracy ceiling for that model: {res[best]['oof_best_accuracy']:.4f}")
