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
                max_rows: int = 500, random_state: int = 42,
                background: np.ndarray | None = None):
    """SHAP values for a fitted estimator.

    The explainer runs on the *transformed* matrix, because that is what the
    classifier actually sees; column names survive the transform unchanged
    since imputation and scaling are both column-wise.

    ``background`` matters more than it looks. Shapley values are defined
    against a reference distribution — they answer "how far did this feature
    move the prediction away from what it would be for a typical patient".
    Handing the model-agnostic explainer a background built from the single
    row being explained leaves it nothing to vary, so every attribution comes
    back exactly zero, silently and without an error. A real sample of
    training patients has to be supplied for a per-patient explanation to
    carry any information at all.
    """
    import shap

    # The selected model may be a Pipeline(prep, clf) or a bare estimator such
    # as an ensemble that does its own preprocessing. Handle both rather than
    # assuming the shape.
    steps = getattr(pipeline, "named_steps", None)
    if steps and "prep" in steps and "clf" in steps:
        prep, clf = steps["prep"], steps["clf"]
        Xt = prep.transform(X)
        bg_raw = None if background is None else prep.transform(background)
    else:
        clf = pipeline
        Xt = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        bg_raw = (None if background is None
                  else np.nan_to_num(np.asarray(background, dtype=float), nan=0.0))

    if len(Xt) > max_rows:
        rng = np.random.default_rng(random_state)
        Xt = Xt[rng.choice(len(Xt), max_rows, replace=False)]

    try:
        explainer = shap.TreeExplainer(clf)
        vals = explainer.shap_values(Xt)
        if isinstance(vals, list):
            vals = vals[1]
        elif np.asarray(vals).ndim == 3:
            vals = np.asarray(vals)[:, :, 1]
        return np.asarray(vals), Xt
    except Exception:            # not a tree model -- fall through
        pass

    # Model-agnostic fallback. Prefer the supplied background; only fall back
    # to the explained rows when there are enough of them to vary.
    if bg_raw is not None and len(bg_raw) >= 10:
        source = bg_raw
    elif len(Xt) >= 10:
        source = Xt
    else:
        raise ValueError(
            "SHAP needs a background of at least 10 patients to produce "
            "non-zero attributions; pass `background=` (the saved artefact "
            "carries one as 'shap_background').")

    bg = shap.kmeans(source, min(25, len(source)))
    explainer = shap.KernelExplainer(lambda z: clf.predict_proba(z)[:, 1], bg)
    vals = explainer.shap_values(Xt, nsamples=200, silent=True)
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


# ---------------------------------------------------------------------------
# Intrinsic block attention
# ---------------------------------------------------------------------------
def attention_profile(torch_clf, X: np.ndarray) -> pd.DataFrame:
    """Mean attention weight the network places on each biological axis."""
    import torch

    model = torch_clf.model_
    model.eval()
    with torch.no_grad():
        model(torch.from_numpy(np.asarray(X, dtype=np.float32)))
    a = model.last_attention_.numpy()
    return pd.DataFrame({
        "block": model.block_names,
        "mean_attention": a.mean(axis=0),
        "std_attention": a.std(axis=0),
    }).sort_values("mean_attention", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------
def lime_explainer(pipeline, X_background: np.ndarray, feature_names: list[str],
                   random_state: int = 42):
    """Build a LIME tabular explainer over the transformed feature space.

    LIME perturbs one patient's feature vector, scores the perturbations with
    the real model, and fits a sparse linear surrogate to that local
    neighbourhood. It answers a different question from SHAP: not "how is
    credit allocated across features" but "what does the decision boundary look
    like immediately around this patient".

    Both are reported because they disagree in an informative way. SHAP is
    exact for the model and additive by construction; LIME is an approximation
    but shows local linearity. Where they agree, an explanation is safe to act
    on. Where they diverge, the patient sits somewhere the model's response is
    sharply non-linear -- which is worth a clinician knowing.
    """
    from lime.lime_tabular import LimeTabularExplainer

    steps = getattr(pipeline, "named_steps", None)
    prep = steps.get("prep") if steps else None
    bg = (prep.transform(X_background) if prep is not None
          else np.nan_to_num(np.asarray(X_background, dtype=float), nan=0.0))

    return LimeTabularExplainer(
        training_data=bg,
        feature_names=list(feature_names),
        class_names=["no inhibitor", "inhibitor"],
        discretize_continuous=True,
        random_state=random_state,
        mode="classification",
    )


def lime_explain_patient(pipeline, explainer, X_row: np.ndarray,
                         feature_names: list[str], top: int = 8,
                         n_samples: int = 5000) -> pd.DataFrame:
    """LIME attribution for one patient, as a ranked table."""
    steps = getattr(pipeline, "named_steps", None)
    if steps and "prep" in steps and "clf" in steps:
        prep, clf = steps["prep"], steps["clf"]
        row = prep.transform(np.asarray(X_row, dtype=float).reshape(1, -1))[0]
        predict = clf.predict_proba
    else:
        row = np.nan_to_num(np.asarray(X_row, dtype=float).reshape(-1), nan=0.0)
        predict = pipeline.predict_proba

    exp = explainer.explain_instance(row, predict, num_features=top,
                                     num_samples=n_samples, labels=(1,))
    rows = []
    for rule, weight in exp.as_list(label=1):
        rows.append({"condition": rule, "weight": round(float(weight), 5),
                     "direction": "increases risk" if weight > 0
                                  else "decreases risk"})
    df = pd.DataFrame(rows)
    return df.reindex(df["weight"].abs().sort_values(ascending=False).index
                      ).reset_index(drop=True)


def shap_lime_agreement(shap_row: pd.DataFrame, lime_df: pd.DataFrame,
                        feature_names: list[str]) -> dict:
    """How far do the two explanations agree on which features matter?

    Compared as sets rather than as ranked lists, because LIME reports
    discretised conditions ("severity_ordinal > 1.20") rather than bare feature
    names, so the orderings are not directly comparable.
    """
    def _feature_of(condition: str) -> str | None:
        hits = [f for f in feature_names if f in condition]
        return max(hits, key=len) if hits else None

    lime_feats = {f for f in (_feature_of(c) for c in lime_df["condition"]) if f}
    shap_feats = set(shap_row["feature"])
    if not lime_feats or not shap_feats:
        return {"overlap": None}
    inter = lime_feats & shap_feats
    return {
        "shap_top_features": sorted(shap_feats),
        "lime_top_features": sorted(lime_feats),
        "shared": sorted(inter),
        "jaccard": round(len(inter) / len(lime_feats | shap_feats), 3),
        "overlap": round(len(inter) / min(len(lime_feats), len(shap_feats)), 3),
    }


def rank_agreement(a: pd.DataFrame, b: pd.DataFrame, key: str = "block") -> dict:
    """Spearman correlation between two block rankings."""
    from scipy.stats import spearmanr

    merged = a[[key]].assign(rank_a=range(len(a))).merge(
        b[[key]].assign(rank_b=range(len(b))), on=key, how="inner")
    if len(merged) < 3:
        return {"n": len(merged), "spearman": None, "p_value": None}
    rho, p = spearmanr(merged["rank_a"], merged["rank_b"])
    return {"n": int(len(merged)), "spearman": round(float(rho), 4),
            "p_value": round(float(p), 5)}


# ---------------------------------------------------------------------------
# Per-patient explanation
# ---------------------------------------------------------------------------
def explain_patient(vals: np.ndarray, Xt: np.ndarray, feature_names: list[str],
                    row: int, top: int = 8) -> pd.DataFrame:
    """Ranked drivers for one patient, with the direction of each effect."""
    v = vals[row]
    order = np.argsort(-np.abs(v))[:top]
    return pd.DataFrame({
        "feature": [feature_names[i] for i in order],
        "value": [float(Xt[row, i]) for i in order],
        "shap": [float(v[i]) for i in order],
        "direction": ["increases risk" if v[i] > 0 else "decreases risk"
                      for i in order],
    })
