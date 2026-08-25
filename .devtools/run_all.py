"""Run every stage end to end, unbuffered, into reports/run.log."""
import sys, time, warnings
sys.path.insert(0, '.')
warnings.filterwarnings("ignore")

from src import train, figures, report

t0 = time.time()
data = train.prepare()
train._log(f"X={data['X'].shape}  {int(data['y'].sum())} events / "
           f"{len(data['y'])} labelled  {len(data['unlabelled'])} unlabelled")

train.stage_audit()
train.stage_cv(data, include_neural=True)
train.stage_blocked(data, include_neural=True)
train.stage_final(data)
train.stage_ssl(data)
train.stage_external(data)
train._log("building figures")
figures.build_all()
train._log("building RESULTS.md")
report.build()
train._log(f"ALL STAGES COMPLETE in {time.time() - t0:.0f}s")
