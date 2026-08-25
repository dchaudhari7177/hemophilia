"""
Choosing which model to ship.

Two protocols disagree, and the disagreement is informative rather than
annoying:

* **Repeated stratified CV** ranks by performance on variants drawn from the
  same pool as training. Tree ensembles do well here, partly because a random
  split can put residue 490 in training and residue 491 in test -- neighbouring
  positions in the same epitope, which is closer to interpolation than to
  prediction.
* **Position-blocked CV** holds out contiguous stretches of F8 entirely. This is
  the situation a treatment centre is actually in: a newly sequenced patient
  carries a variant nobody has catalogued.

Picking whichever protocol flatters a favourite model would be exactly the kind
of choice this project was built to call out. The rule used instead is stated up
front and applied mechanically:

    1. Take every model that DeLong cannot separate from the best by
       repeated-CV out-of-fold AUC (p >= 0.05). These are the models the
       internal data cannot tell apart.
    2. Among those, ship the one with the highest position-blocked AUC.

Step 1 refuses to select on noise. Step 2 breaks the resulting tie using the
criterion that matches how the model will be used. Both are decided before
looking at the held-out test set, which is scored exactly once.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

ALPHA = 0.05


def _load(name: str):
    p = REPORTS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def select_model(cv=None, blocked=None, comparison=None) -> dict:
    cv = cv or _load("cv")
    blocked = blocked or _load("blocked_cv")
    comparison = comparison or _load("model_comparison")

    if not cv:
        return {"selected": "LightGBM", "rule": "fallback -- no cv.json found"}

    cv_best = cv["ranking"][0]
    if not comparison or not blocked:
        return {"selected": cv_best,
                "rule": "highest repeated-CV AUC (no blocked/DeLong data)"}

    # Step 1 -- the models internal CV cannot separate from the best.
    tier = [name for name, r in comparison["vs_best"].items()
            if r["verdict"] in {"--", "indistinguishable"}]

    # Step 2 -- break the tie on generalisation to unseen regions of the gene.
    bl = blocked["models"]
    scored = [(n, bl[n]["blocked_auc_mean"]) for n in tier if n in bl]
    if not scored:
        return {"selected": cv_best, "rule": "no blocked scores for the top tier"}
    scored.sort(key=lambda kv: -kv[1])
    winner, winner_blocked = scored[0]

    return {
        "selected": winner,
        "rule": ("among models DeLong cannot separate from the best on repeated "
                 "CV (p >= 0.05), ship the highest position-blocked AUC"),
        "statistically_tied_tier": tier,
        "blocked_auc_within_tier": {n: round(v, 4) for n, v in scored},
        "best_by_repeated_cv": cv_best,
        "best_by_repeated_cv_auc": comparison["best_oof_auc"],
        "selected_repeated_cv_auc": comparison["vs_best"][winner]["oof_auc"],
        "selected_blocked_auc": round(winner_blocked, 4),
        "changed_from_cv_winner": winner != cv_best,
        "note": (
            f"Repeated CV ranks {cv_best} first, but the difference from "
            f"{winner} is not statistically significant "
            f"(p = {comparison['vs_best'][winner]['p_value']}). Under the "
            f"blocked protocol -- generalising to a stretch of F8 never seen in "
            f"training, which is the clinical case -- {winner} scores "
            f"{winner_blocked:.4f} against {bl.get(cv_best, {}).get('blocked_auc_mean')}."
            if winner != cv_best else
            f"{winner} leads on both protocols, so the tie-break did not change "
            f"the choice."),
    }


def save(result: dict) -> Path:
    path = REPORTS / "selection.json"
    path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    return path


if __name__ == "__main__":
    r = select_model()
    save(r)
    print(json.dumps(r, indent=2))
