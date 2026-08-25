"""
Training driver.

Stages (run with ``python -m src.train --stage <name>``):

  audit     reproduce the reference pipeline and measure where its score comes from
  cv        repeated stratified cross-validation over the whole model zoo
  blocked   position-blocked cross-validation (generalising to unseen regions)
  final     fit the selected model, calibrate it, evaluate on the held-out test set
  ssl       missing-data probe and semi-supervised use of the unlabelled pool
  external  transfer the F8 model onto the CHBMP F9 cohort
  all       every stage, in order

Every stage writes a JSON file into ``reports/`` so the write-up is generated
from measurements rather than transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     train_test_split)

from .datasets import (LABEL_UNKNOWN, load_champ, load_chbmp, label_summary,
                       protein_region_blocks, split_by_label)
from .evaluate import (bootstrap_ci, compute_metrics, delong_test,
                       threshold_at_sensitivity, youden_threshold)
from .features import VariantFeaturizer, block_index
from .models import (RANDOM_STATE, build_pipeline, classical_models,
                     neural_models)
from .semisupervised import (ReportingBiasProbe, SelfTrainingSSL,
                             estimate_unlabelled_prevalence)

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
TEST_SIZE = 0.20
N_SPLITS = 5
N_REPEATS = 3


def _save(name: str, obj) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    print(f"  -> {path.relative_to(ROOT)}")
    return path


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Shared data preparation
# ---------------------------------------------------------------------------
def prepare(seed: int = RANDOM_STATE):
    """Load CHAMP, featurise, and carve off a held-out test set.

    The featuriser is fitted on the training rows only. It never sees a label,
    so this is belt-and-braces rather than a strict requirement, but it keeps
    the pipeline defensible under the strictest reading.
    """
    champ = load_champ()
    labelled, unlabelled = split_by_label(champ)
    y = (labelled["inhibitor"] == 1).astype(int).values

    idx = np.arange(len(labelled))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE, stratify=y,
                                      random_state=seed)

    fz = VariantFeaturizer().fit(labelled.iloc[tr_idx])
    X = fz.transform(labelled).values.astype(float)
    X_unl = fz.transform(unlabelled).values.astype(float)

    return {
        "champ": champ,
        "labelled": labelled,
        "unlabelled": unlabelled,
        "featurizer": fz,
        "X": X,
        "y": y,
        "X_unlabelled": X_unl,
        "train_idx": tr_idx,
        "test_idx": te_idx,
        "blocks": block_index(fz.blocks_, fz.columns_),
        "feature_names": fz.columns_,
    }


def all_models(blocks, include_neural: bool = True) -> dict:
    zoo = dict(classical_models())
    if include_neural:
        zoo.update(neural_models(blocks))
    return zoo
