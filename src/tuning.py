"""
Hyperparameter search and clinically-constrained gradient boosting.

Two ideas beyond a plain grid search.

**Nested search.** The reference works tune with ``GridSearchCV`` and then
report the best cross-validated score. That score is optimistically biased:
the same folds chose the hyperparameters and graded them. Here the search runs
in an inner loop and is scored on an outer loop it never touched, so the
reported number is what a new cohort should expect.

**Monotone constraints.** Some directions are not in question. A null variant
cannot lower inhibitor risk relative to a missense; a severe phenotype cannot
lower it relative to a mild one. Gradient boosting is free to learn the
opposite from noise in a 369-event training set, and on small data it often
does. Pinning the sign of these features costs a little training fit and buys
a model that cannot contradict established immunology -- which also makes it
defensible in front of a clinician.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict)

from .models import RANDOM_STATE, build_pipeline

warnings.filterwarnings("ignore")
