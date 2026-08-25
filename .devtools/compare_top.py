"""DeLong tests between the top cross-validated models.

The spread between the best and worst model in this zoo is about 0.05 AUC and
the fold-to-fold standard deviation is about 0.03. Most of these models are
therefore statistically indistinguishable, and a ranking table alone would
invite a claim the data cannot support. This pits the top models against each
other on the shared out-of-fold predictions and reports p-values.
"""
import json
import sys

sys.path.insert(0, ".")
import numpy as np

from src.evaluate import delong_test

d = np.load("reports/cv_oof.npz")
y = d["y"]
names = [k for k in d.files if k != "y"]
aucs = {}
from sklearn.metrics import roc_auc_score
for n in names:
    aucs[n] = float(roc_auc_score(y, d[n]))
ranked = sorted(aucs, key=lambda n: -aucs[n])

best = ranked[0]
out = {"best_model": best, "best_oof_auc": round(aucs[best], 4), "vs_best": {}}
print(f"best by pooled OOF AUC: {best} ({aucs[best]:.4f})\n")
print(f"{'model':22s}{'OOF AUC':>9s}{'delta':>9s}{'p':>10s}  verdict")
for n in ranked:
    r = delong_test(y, d[best], d[n])
    p = r["p_value"]
    verdict = ("--" if n == best else
               "significantly worse" if p is not None and p < 0.05 else
               "indistinguishable")
    out["vs_best"][n] = {"oof_auc": round(aucs[n], 4),
                         "delta_vs_best": r["delta"], "p_value": p,
                         "verdict": verdict}
    print(f"{n:22s}{aucs[n]:9.4f}{r['delta']:9.4f}"
          f"{'' if p is None else format(p, '10.4f')}  {verdict}")

n_indist = sum(1 for v in out["vs_best"].values()
               if v["verdict"] == "indistinguishable")
out["n_statistically_indistinguishable_from_best"] = n_indist
out["note"] = (
    f"{n_indist} of {len(ranked) - 1} competing models cannot be separated from "
    f"{best} at p<0.05 on these out-of-fold predictions. Selecting on the third "
    f"decimal place of AUC would be selecting on noise.")
json.dump(out, open("reports/model_comparison.json", "w"), indent=2, default=float)
print("\n" + out["note"])
print("  -> reports/model_comparison.json")
