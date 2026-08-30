"""Fit the CV-selected hybrid, calibrate it, and score the held-out set once."""
import sys, json, warnings
sys.path.insert(0, '.'); warnings.filterwarnings('ignore')
import numpy as np, joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score

from src.train import prepare
from src.models import build_pipeline, RANDOM_STATE
from src.hybrid import build_hybrids
from src.evaluate import (compute_metrics, bootstrap_ci, delong_test,
                          youden_threshold, threshold_at_sensitivity,
                          accuracy_threshold)

d = prepare(); X, y = d['X'], d['y']; tr, te = d['train_idx'], d['test_idx']
mdl = build_hybrids(d['blocks'])['Hybrid_RankAvg']
inner = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

cal = CalibratedClassifierCV(build_pipeline(mdl), method='isotonic', cv=inner)
cal.fit(X[tr], y[tr])
p_te = cal.predict_proba(X[te])[:, 1]

oof = cross_val_predict(
    CalibratedClassifierCV(build_pipeline(mdl), method='isotonic', cv=inner),
    X[tr], y[tr], cv=inner, method='predict_proba')[:, 1]
thr_y = youden_threshold(y[tr], oof)
thr_s = threshold_at_sensitivity(y[tr], oof, 0.90)
thr_a = accuracy_threshold(y[tr], oof)

prev = float(y[te].mean()); base = max(prev, 1 - prev)
out = {
    'model': 'Hybrid_RankAvg (ExtraTrees + DeepMLP, rank-averaged)',
    'n_train': int(len(tr)), 'n_test': int(len(te)), 'test_events': int(y[te].sum()),
    'thresholds': {'youden_on_train_oof': round(thr_y, 4),
                   'sensitivity90_on_train_oof': round(thr_s, 4),
                   'accuracy_on_train_oof': round(thr_a, 4)},
    'test_calibrated_youden': compute_metrics(y[te], p_te, thr_y),
    'test_calibrated_sens90': compute_metrics(y[te], p_te, thr_s),
    'test_calibrated_accuracy': compute_metrics(y[te], p_te, thr_a),
    'auc_ci': bootstrap_ci(y[te], p_te, 'auc_roc'),
    'auc_pr_ci': bootstrap_ci(y[te], p_te, 'auc_pr'),
    'accuracy_context': {
        'majority_class_accuracy': round(base, 4),
        'model_accuracy': compute_metrics(y[te], p_te, thr_a)['accuracy'],
        'prevalence': round(prev, 4),
    },
}
prev_pred = np.load('reports/test_predictions.npz')
out['delong_vs_deepmlp'] = delong_test(y[te], p_te, prev_pred['p_cal'])

json.dump(out, open('reports/hybrid_final.json', 'w'), indent=2, default=float)
np.savez('reports/hybrid_test_predictions.npz', y=y[te], p_cal=p_te)
joblib.dump({'model': cal, 'featurizer': d['featurizer'],
             'thresholds': out['thresholds'],
             'feature_names': d['feature_names'],
             'shap_background': X[np.random.default_rng(42).choice(tr, 200, replace=False)]},
            'models/hybrid_model.joblib')

print(f"HYBRID: {out['model']}")
print(f"  AUC      {out['auc_ci']['point']:.4f} ({out['auc_ci']['lo']:.4f}-{out['auc_ci']['hi']:.4f})")
print(f"  AUC-PR   {out['auc_pr_ci']['point']:.4f}")
print(f"  DeLong vs shipped DeepMLP: {out['delong_vs_deepmlp']}")
print()
for label, k in [('Youden', 'test_calibrated_youden'),
                 ('Sens90', 'test_calibrated_sens90'),
                 ('AccOpt', 'test_calibrated_accuracy')]:
    m = out[k]
    print(f"  {label:8s} thr {m['threshold']:.4f}  acc {m['accuracy']:.4f}  "
          f"sens {m['sensitivity']:.4f}  spec {m['specificity']:.4f}  "
          f"mcc {m['mcc']:.4f}  ({m['tp']}/{m['tp']+m['fn']} caught)")
print(f"\n  baseline (all-negative) accuracy: {base:.4f}")
