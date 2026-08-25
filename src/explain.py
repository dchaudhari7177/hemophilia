"""
Explainability.

The reference works pair SHAP with LIME. Both are post-hoc: they fit a
surrogate around a fixed model and report what the surrogate says. That is
useful, but it has two well-known failure modes on a dataset like this one --
correlated features split their credit arbitrarily, and LIME's local linear fit
is unstable across reruns on sparse binary columns.

This module keeps SHAP (it is the field standard and the reference results have
to be comparable to something) and adds two things the references lack:

* **Block attribution.** Because the features are grouped into seven biological
  axes, SHAP values can be summed within a block. Block-level attribution is
  stable under feature correlation in a way that per-feature attribution is
  not: shuffling credit between ``vtype_nonsense`` and ``is_truncating`` does
  not change the total assigned to "molecular consequence".

* **Intrinsic attention.** ``BioBlockAttentionNet`` emits a weight per
  biological axis per patient as part of the forward pass. That is not an
  approximation of the model -- it *is* the model -- so it cannot disagree with
  what the network actually computed.

Agreement between the post-hoc and intrinsic rankings is reported, since a
disagreement is a reason to distrust the explanation rather than the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
