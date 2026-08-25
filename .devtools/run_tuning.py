import sys, json, time, warnings
sys.path.insert(0, '.')
warnings.filterwarnings("ignore")
import numpy as np
from src.train import prepare
from src.tuning import search_spaces, nested_search

d = prepare()
X, y = d['X'][d['train_idx']], d['y'][d['train_idx']]
out = {}
for name, (est, space) in search_spaces().items():
    t0 = time.time()
    r = nested_search(X, y, name, est, space, n_iter=25, inner_splits=3)
    r.pop('_estimator', None)
    r['seconds'] = round(time.time() - t0, 1)
    out[name] = r
    print(f"{name:20s} nested {r['nested_auc_mean']:.4f}+/-{r['nested_auc_std']:.4f}  "
          f"inner {r['inner_best_auc']:.4f}  optimism {r['optimism_from_tuning']:+.4f}  "
          f"({r['seconds']}s)", flush=True)
json.dump(out, open('reports/tuning.json', 'w'), indent=2, default=float)
print("saved reports/tuning.json")
