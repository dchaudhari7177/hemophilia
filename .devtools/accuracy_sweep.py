"""Best held-out accuracy per model, with the threshold fixed on train folds."""
import sys, json, warnings
sys.path.insert(0, '.')
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score

from src.train import prepare
from src.models import build_pipeline, classical_models

d = prepare(); X, y = d['X'], d['y']; tr, te = d['train_idx'], d['test_idx']
zoo = dict(classical_models())
cv = StratifiedKFold(5, shuffle=True, random_state=42)
base = float(max(y[te].mean(), 1 - y[te].mean()))
print(f"all-negative baseline on the held-out set = {base:.4f}", flush=True)

rows = []
for name, mdl in zoo.items():
    oof = cross_val_predict(build_pipeline(mdl), X[tr], y[tr], cv=cv,
                            method='predict_proba')[:, 1]
    grid = np.unique(np.round(oof, 3))
    thr = float(max(grid, key=lambda t: accuracy_score(y[tr], (oof >= t).astype(int))))
    p = build_pipeline(mdl).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    pred = (p >= thr).astype(int)
    acc = float(accuracy_score(y[te], pred))
    sens = float(((pred == 1) & (y[te] == 1)).sum() / y[te].sum())
    spec = float(((pred == 0) & (y[te] == 0)).sum() / (y[te] == 0).sum())
    auc = float(roc_auc_score(y[te], p))
    rows.append({"model": name, "threshold": round(thr, 4),
                 "test_accuracy": round(acc, 4), "sensitivity": round(sens, 4),
                 "specificity": round(spec, 4), "auc": round(auc, 4),
                 "beats_baseline_by": round(acc - base, 4)})
    print(f"{name:22s} acc {acc:.4f}  sens {sens:.3f}  spec {spec:.3f}  auc {auc:.4f}", flush=True)

rows.sort(key=lambda r: -r["test_accuracy"])
json.dump({"baseline_accuracy": round(base, 4), "models": rows},
          open("reports/accuracy_sweep.json", "w"), indent=2)
print("\nbest:", rows[0]["model"], rows[0]["test_accuracy"])
