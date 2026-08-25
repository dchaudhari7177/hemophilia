"""Ablations and nested hyperparameter search, written into reports/."""
import json
import sys
import time
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from src import ablation, train
from src.tuning import nested_search, search_spaces


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


data = train.prepare()
X = data["X"][data["train_idx"]]
y = data["y"][data["train_idx"]]
names = data["feature_names"]
blocks = data["blocks"]
log(f"supplementary experiments on X={X.shape}, {int(y.sum())} events")

# ---- ablations ------------------------------------------------------------
log("ABLATION -- signal decomposition, leave-one-block-out, feature sweep")
abl = ablation.run(X, y, names, blocks)
json.dump(abl, open("reports/ablation.json", "w"), indent=2, default=float)
log("  -> reports/ablation.json")

for k, v in abl["signal_decomposition"].items():
    if k.startswith("_"):
        continue
    log(f"    {k:34s} k={v['n_features']:3d}  AUC {v['auc']:.4f}")
log(f"    lift over variant-type+severity: "
    f"{abl['signal_decomposition']['_lift_over_variant_type_and_severity']:+.4f}")
for k, v in abl["leave_one_block_out"]["blocks"].items():
    log(f"    drop {k:14s} AUC {v['auc_without']:.4f}  "
        f"cost {v['cost_of_removal']:+.4f}")
log(f"    best feature count: {abl['feature_count_sweep']['best_k']} "
    f"(AUC {abl['feature_count_sweep']['best_auc']:.4f})")

# ---- nested hyperparameter search -----------------------------------------
log("TUNING -- nested search (inner tunes, outer grades)")
out = {}
for name, (est, space) in search_spaces().items():
    t0 = time.time()
    r = nested_search(X, y, name, est, space, n_iter=20, inner_splits=3,
                      outer_splits=5)
    r.pop("_estimator", None)
    r["seconds"] = round(time.time() - t0, 1)
    out[name] = r
    log(f"    {name:20s} nested {r['nested_auc_mean']:.4f}"
        f"+/-{r['nested_auc_std']:.4f}  inner {r['inner_best_auc']:.4f}  "
        f"optimism {r['optimism_from_tuning']:+.4f}  ({r['seconds']}s)")
json.dump(out, open("reports/tuning.json", "w"), indent=2, default=float)
log("  -> reports/tuning.json")
log("SUPPLEMENTARY COMPLETE")
