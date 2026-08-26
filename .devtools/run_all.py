"""Run every stage end to end, unbuffered, into reports/run.log."""
import json
import sys
import time
import warnings

sys.path.insert(0, '.')
warnings.filterwarnings("ignore")

from src import figures, integrity, report, selection, train
from src.fused import audit_provenance, clinical_effect_sizes, load_fused, run_simulation

t0 = time.time()
data = train.prepare()
train._log(f"X={data['X'].shape}  {int(data['y'].sum())} events / "
           f"{len(data['y'])} labelled  {len(data['unlabelled'])} unlabelled")

train.stage_audit()
train.stage_cv(data, include_neural=True)
train.stage_blocked(data, include_neural=True)

# the shipped model is chosen by the pre-registered rule, which needs both
# cross-validation protocols to have finished
choice = selection.select_model()
selection.save(choice)
train._log(f"SELECTION: {choice['selected']} "
           f"(repeated-CV winner {choice.get('best_by_repeated_cv')}; "
           f"changed={choice.get('changed_from_cv_winner')})")

train.stage_final(data, best_name=choice["selected"])
train.stage_subgroups(data)
train.stage_ssl(data)
train.stage_external(data)

train._log("FUSED -- provenance audit and simulation study")
fused = load_fused()
json.dump({"provenance": audit_provenance(fused),
           "clinical_effect_sizes": clinical_effect_sizes(fused)},
          open("reports/fused_audit.json", "w"), indent=2, default=str)
json.dump(run_simulation(), open("reports/fused_simulation.json", "w"),
          indent=2, default=float)
train._log("  -> reports/fused_audit.json, reports/fused_simulation.json")

train._log("INTEGRITY -- mechanical pipeline checks")
integ = integrity.run_all()
json.dump(integ, open("reports/integrity.json", "w"), indent=2, default=str)
train._log(f"  {integ['_summary']['passed']}/{integ['_summary']['n_checks']} "
           f"checks passed; failed: {integ['_summary']['failed_checks']}")

train._log("building figures")
figures.build_all()
train._log("building RESULTS.md")
report.build()
train._log(f"ALL STAGES COMPLETE in {time.time() - t0:.0f}s")
