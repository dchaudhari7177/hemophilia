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


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def shap_values(pipeline, X: np.ndarray, feature_names: list[str],
                max_rows: int = 500, random_state: int = 42):
    """SHAP values for a fitted ``Pipeline(prep, clf)``.

    The explainer is run on the *transformed* matrix, because that is what the
    classifier actually sees; column names survive the transform unchanged
    since imputation and scaling are both column-wise.
    """
    import shap

    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    Xt = prep.transform(X)
    if len(Xt) > max_rows:
        rng = np.random.default_rng(random_state)
        Xt = Xt[rng.choice(len(Xt), max_rows, replace=False)]

    try:
        explainer = shap.TreeExplainer(clf)
        vals = explainer.shap_values(Xt)
        if isinstance(vals, list):
            vals = vals[1]
        elif vals.ndim == 3:
            vals = vals[:, :, 1]
    except (TypeError, ValueError, AttributeError):
        bg = shap.kmeans(Xt, min(50, len(Xt)))
        explainer = shap.KernelExplainer(
            lambda z: clf.predict_proba(z)[:, 1], bg)
        vals = explainer.shap_values(Xt, nsamples=100, silent=True)

    return np.asarray(vals), Xt


def global_importance(vals: np.ndarray, feature_names: list[str],
                      top: int = 25) -> pd.DataFrame:
    imp = np.abs(vals).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp})
    df["share"] = df["mean_abs_shap"] / df["mean_abs_shap"].sum()
    return df.sort_values("mean_abs_shap", ascending=False).head(top).reset_index(drop=True)


def block_attribution(vals: np.ndarray, feature_names: list[str],
                      blocks: dict[str, list[int]]) -> pd.DataFrame:
    """Sum |SHAP| within each biological block.

    Stable under the feature correlation that makes per-feature SHAP noisy:
    credit can move between two collinear columns of the same block without
    changing the block total.
    """
    total = np.abs(vals).mean(axis=0)
    rows = []
    for name, idx in blocks.items():
        idx = [i for i in idx if i < len(total)]
        if not idx:
            continue
        rows.append({"block": name,
                     "mean_abs_shap": float(total[idx].sum()),
                     "n_features": len(idx)})
    df = pd.DataFrame(rows)
    df["share"] = df["mean_abs_shap"] / df["mean_abs_shap"].sum()
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
