"""Apply the pre-stated selection rule, then refit and re-report from it.

The main pipeline fits its final model on the repeated-CV winner. That ranking
is only available part-way through, so the blocked-CV tie-break in
``src.selection`` has to be applied afterwards, once both protocols have run.
"""
import json
import sys
import time
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from src import figures, report, selection, train


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


choice = selection.select_model()
selection.save(choice)
log(f"SELECTION: {choice['selected']}  "
    f"(repeated-CV winner was {choice.get('best_by_repeated_cv')}; "
    f"changed = {choice.get('changed_from_cv_winner')})")
log(f"  tier: {choice.get('statistically_tied_tier')}")
log(f"  blocked AUC within tier: {choice.get('blocked_auc_within_tier')}")

data = train.prepare()
log("refitting the final model on the selected estimator")
train.stage_final(data, best_name=choice["selected"])
train.stage_subgroups(data)
train.stage_ssl(data)
train.stage_external(data)
log("rebuilding figures")
figures.build_all()
log("rebuilding RESULTS.md")
report.build()
log("FINALISE COMPLETE")
