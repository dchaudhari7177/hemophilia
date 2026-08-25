"""
Generate ``RESULTS.md`` from the measurement artefacts.

Nothing in the write-up is typed by hand. Every table and every number is read
back out of ``reports/*.json``, so the document cannot drift away from what the
code actually measured, and re-running the pipeline re-writes the report.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUT = ROOT / "RESULTS.md"

# Numbers quoted from the works this project benchmarks against.
REFERENCE_CLAIMS = {
    "Singh & Singh (2025), Random Forest": {
        "accuracy": 97.37, "auc": None,
        "protocol": "Random Over-Sampling applied before stratified k-fold",
    },
    "Prior capstone notebook, Deep MLP v2": {
        "accuracy": 99.63, "auc": 0.9999,
        "protocol": "all columns label-encoded; unrecorded outcomes set to 0",
    },
}


def _load(name: str):
    p = REPORTS / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _pct(x, digits=2):
    return "—" if x is None else f"{100 * float(x):.{digits}f}%"


def _num(x, digits=4):
    return "—" if x is None else f"{float(x):.{digits}f}"


def _ci(d):
    if not d or d.get("lo") is None:
        return "—"
    return f"{d['point']:.4f} ({d['lo']:.4f}–{d['hi']:.4f})"
