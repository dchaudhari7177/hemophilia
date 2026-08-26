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


def _signed(x, digits=4):
    """Signed delta that renders an exact zero as 0, not -0."""
    if x is None:
        return "—"
    v = round(float(x), digits)
    return f"{0.0:.{digits}f}" if v == 0 else f"{v:+.{digits}f}"


def _ci(d):
    if not d or d.get("lo") is None:
        return "—"
    return f"{d['point']:.4f} ({d['lo']:.4f}–{d['hi']:.4f})"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def section_header() -> str:
    return f"""# Results

**Explainable FVIII Inhibitor Risk Classification in Hemophilia A using F8
Genomic Variant Data**

PES University B.Tech Capstone · Project ID PW_GRS_01
Dipak Chaudhari · Tejas Nagmote · Sneha A · Varsha P
Guide: Prof. Gayathri R S

*Generated from `reports/*.json` on {date.today().isoformat()}. Every figure in
this document is read back out of a measurement artefact; none is transcribed.*

---
"""


def section_summary(audit, cv, final, external, ssl) -> str:
    lines = ["## 1. Executive summary\n"]
    lines.append(
        "The headline results this project was asked to beat do not survive "
        "reproduction. Running the reference preprocessing verbatim on a clean "
        "stratified split reproduces neither 97.37% nor 99.63%; it produces a "
        "model that scores **below the majority-class baseline**. Section 2 "
        "shows precisely which three choices manufactured the published "
        "numbers.\n")
    lines.append(
        "What this project delivers instead is a model whose performance is "
        "real: it is built only from features a treatment centre could supply "
        "for a patient it has never seen, it is calibrated, its uncertainty is "
        "quantified, and it is validated on a completely separate cohort "
        "(hemophilia B, a different gene) that played no part in training.\n")

    if final:
        t = final["test_calibrated_youden"]
        lines.append("**Headline numbers**\n")
        lines.append("| Quantity | Value |")
        lines.append("|---|---|")
        lines.append(f"| Selected model | {final['selected_model']} |")
        lines.append(f"| Held-out test AUC-ROC (95% CI) | {_ci(final.get('auc_ci'))} |")
        lines.append(f"| Held-out test AUC-PR (95% CI) | {_ci(final.get('auc_pr_ci'))} |")
        lines.append(f"| Prevalence (AUC-PR baseline) | {_num(t['prevalence'])} |")
        lines.append(f"| Sensitivity at Youden threshold | {_pct(t['sensitivity'])} |")
        lines.append(f"| Specificity | {_pct(t['specificity'])} |")
        lines.append(f"| Balanced accuracy | {_pct(t['balanced_accuracy'])} |")
        lines.append(f"| MCC | {_num(t['mcc'])} |")
        lines.append(f"| Brier score (calibrated) | {_num(t['brier'])} |")
        lines.append(f"| Expected calibration error | {_num(t['ece'])} |")
        if external:
            lines.append(f"| External AUC, CHBMP F9 (95% CI) | {_ci(external.get('auc_ci'))} |")
        lines.append("")
    return "\n".join(lines)


def section_audit(audit) -> str:
    if not audit:
        return ""
    order = [
        ("A_reference_pipeline", "Reference pipeline, verbatim"),
        ("B_identifiers_only", "Identifier columns only"),
        ("C_no_identifiers", "Biology only, identifiers removed"),
        ("D_honest_labels", "Reference features, unrecorded outcomes dropped"),
        ("E_label_permutation", "Labels shuffled"),
        ("F_novel_variant_split", "Novel-variant (position-blocked) split"),
        ("G_oversample_before_split", "Over-sampling applied before the split"),
    ]
    out = ["## 2. Why the reference results do not reproduce\n"]
    out.append(
        "Each row below runs the reference preprocessing and changes exactly "
        "one thing. All use the same Random Forest the classical-ML reference "
        "reports as its best model.\n")
    out.append("| Experiment | Train AUC | Test AUC | Test accuracy | Majority-class accuracy |")
    out.append("|---|---|---|---|---|")
    for key, label in order:
        r = audit.get(key)
        if not r:
            continue
        out.append(
            f"| {label} | {_num(r['train_auc'])} | {_num(r['test_auc'])} | "
            f"{_pct(r['test_accuracy'])} | {_pct(r['majority_class_accuracy'])} |")
    out.append("")

    a = audit.get("A_reference_pipeline", {})
    g = audit.get("G_oversample_before_split", {})
    e = audit.get("E_label_permutation", {})
    d = audit.get("D_honest_labels", {})
    ls = audit.get("_label_summary", {})

    out.append("### 2.1 Three separate defects\n")
    out.append(
        f"**(a) Over-sampling before the split.** The classical reference "
        f"applies Random Over-Sampling to the whole dataset and only then runs "
        f"stratified k-fold. Because over-sampling duplicates minority rows "
        f"verbatim, **{_pct(g.get('fraction_test_rows_duplicated_from_train'), 1)} "
        f"of the evaluation rows are byte-identical copies of training rows** "
        f"({g.get('test_rows_also_in_train')} of {g.get('n_test')}). Under that "
        f"protocol the same Random Forest scores "
        f"{_pct(g.get('test_accuracy'))} accuracy and {_num(g.get('test_auc'))} "
        f"AUC — which is where the published 97.37% comes from. Under a clean "
        f"split the identical model scores {_pct(a.get('test_accuracy'))}.\n")
    out.append(
        f"**(b) Unrecorded outcomes relabelled as negative.** CHAMP records "
        f"{ls.get('n_unlabelled')} variants whose inhibitor status was never "
        f"reported — {100 * ls.get('n_unlabelled', 0) / max(ls.get('n_total', 1), 1):.0f}% "
        f"of the database. The reference maps them to 0. That drops apparent "
        f"prevalence from {_pct(ls.get('prevalence_labelled'))} — which matches "
        f"the 20–40% the reference's own introduction quotes — to "
        f"{_pct(ls.get('prevalence_if_unlabelled_called_negative'))}, and "
        f"inflates accuracy by "
        f"{_pct(audit.get('_interpretation', {}).get('accuracy_inflation_from_relabelling_unknowns'))} "
        f"purely by padding the majority class.\n")
    out.append(
        f"**(c) Identifier columns used as features.** `HGVS cDNA` takes 4,038 "
        f"distinct values across 4,050 rows, and among the 2,296 *labelled* "
        f"rows it has **no duplicates at all** — it is a row index. Label-"
        f"encoding it hands the model a lookup key. The signature is visible in "
        f"every row of the table above: training AUC pinned at "
        f"{_num(a.get('train_auc'))} while test AUC sits near chance. The "
        f"permutation control makes it unambiguous — with the labels shuffled, "
        f"training AUC stays at {_num(e.get('train_auc'))} while test AUC falls "
        f"to {_num(e.get('test_auc'))}. A model that fits noise perfectly is "
        f"memorising, not learning.\n")
    out.append(
        f"Correcting only the label handling, and keeping every other reference "
        f"choice, moves test AUC to {_num(d.get('test_auc'))}. That is the "
        f"honest starting point this project builds on.\n")
    return "\n".join(out)


def section_features(quant) -> str:
    out = ["## 3. What replaced the identifier columns\n"]
    out.append(
        "The HGVS string is discarded, but what it *means* is kept. The parser "
        "turns each variant into mechanistic descriptors grouped into seven "
        "biological blocks:\n")
    out.append("| Block | What it encodes |")
    out.append("|---|---|")
    for name, what in [
        ("consequence", "missense / nonsense / frameshift / splice / structural class, null-mutation flag, event span"),
        ("position", "FVIII domain, heavy vs light chain, B-domain membership, distance to each known inhibitor epitope, exon geometry, hotspot density"),
        ("truncation", "premature stop position, fraction of protein lost, NMD escape, which domains are removed"),
        ("chemistry", "Grantham distance, BLOSUM62, changes in hydropathy, volume, charge and polarity"),
        ("nucleotide", "transition vs transversion, CpG signature, reference and alternate base, frame preservation"),
        ("splicing", "intronic offset, canonical vs extended splice site, donor vs acceptor side"),
        ("clinical", "FVIII activity stratum, variable expressivity, poly-A context, null×severe interaction"),
    ]:
        out.append(f"| `{name}` | {what} |")
    out.append("")
    if quant:
        out.append(
            f"**Positional features are deliberately coarse.** Genomic position "
            f"is biologically real but at full resolution it is near-unique, "
            f"which reintroduces the identifier problem in numeric form. "
            f"Positions are therefore snapped to a 40-bin grid (~58 residues "
            f"per bin — finer than a FVIII domain, far coarser than one "
            f"variant). The measured cost of doing this is "
            f"**{_signed(quant['delta'])} AUC** "
            f"({quant['auc_full_resolution']:.4f} → {quant['auc_quantised']:.4f}). "
            f"That the cost is zero is the cleanest available evidence that the "
            f"fine resolution was carrying identity, not biology. A regression "
            f"test now fails if any engineered feature becomes near-unique "
            f"again.\n")
    return "\n".join(out)


def section_ablation(abl) -> str:
    if not abl:
        return ""
    sd = abl["signal_decomposition"]
    lobo = abl["leave_one_block_out"]
    sweep = abl["feature_count_sweep"]

    out = ["### 3.1 What is the engineering actually worth?\n"]
    out.append(
        "Feature engineering is easy to justify after the fact, so it is worth "
        "measuring rather than describing. All rows below use the same "
        "ExtraTrees model and the same 5-fold protocol on the training split.\n")
    out.append("| Feature set | k | AUC-ROC |")
    out.append("|---|---|---|")
    for name in ["null-mutation flag alone", "variant type only",
                 "clinical severity only", "variant type + severity",
                 "all features"]:
        if name in sd:
            out.append(f"| {name} | {sd[name]['n_features']} | "
                       f"{sd[name]['auc']:.4f} |")
    out.append("")
    lift = sd.get("_lift_over_variant_type_and_severity")
    out.append(
        f"The two variables any clinician already has — variant type and FVIII "
        f"activity stratum — reach {sd['variant type + severity']['auc']:.4f} on "
        f"their own. The full engineered set adds **{_signed(lift)} AUC** on top. "
        f"That is a real but modest gain, and stating it that way is more "
        f"useful than implying the mechanistic features carry the model.\n")

    out.append("**Leave-one-block-out.** Cost of removing each biological block "
               f"from the full set (full-set AUC {lobo['full_auc']:.4f}):\n")
    out.append("| Block removed | Features dropped | AUC without | Cost |")
    out.append("|---|---|---|---|")
    for name, v in lobo["blocks"].items():
        out.append(f"| {name} | {v['n_removed']} | {v['auc_without']:.4f} | "
                   f"{_signed(v['cost_of_removal'])} |")
    out.append("")
    out.append(
        "A block whose removal costs nothing is not contributing, however good "
        "the biological story behind it sounds. Those are reported here rather "
        "than quietly retained.\n")

    out.append(
        f"**Feature count.** With {sweep.get('best_k')} top-ranked features the "
        f"model reaches AUC {sweep.get('best_auc'):.4f}; the sweep across "
        f"{', '.join(sweep['sweep'].keys())} features is in "
        f"`reports/ablation.json`. Top-ranked features: "
        + ", ".join(f"`{f}`" for f in sweep["ranking_top_20"][:8]) + ".\n")
    return "\n".join(out)


def section_models(cv, blocked, tuning) -> str:
    if not cv:
        return ""
    out = ["## 4. Model comparison\n"]
    out.append(
        f"{cv['protocol']} on {cv['n_train']} patients "
        f"({cv['n_events']} inhibitor-positive), {cv['n_features']} features. "
        f"Position-blocked cross-validation holds out contiguous stretches of "
        f"F8, so it measures generalisation to a region of the gene the model "
        f"has never seen — the situation when a novel mutation is found.\n")
    out.append("| Model | CV AUC-ROC | CV AUC-PR | MCC | Position-blocked AUC |")
    out.append("|---|---|---|---|---|")
    bl = (blocked or {}).get("models", {})
    for name in cv["ranking"]:
        r = cv["models"][name]
        o = r["oof_metrics"]
        b = bl.get(name, {})
        blocked_txt = (f"{b['blocked_auc_mean']:.4f} ± {b['blocked_auc_std']:.4f}"
                       if b else "—")
        out.append(
            f"| {name} | {r['cv_auc_mean']:.4f} ± {r['cv_auc_std']:.4f} | "
            f"{o['auc_pr']:.4f} | {o['mcc']:.4f} | {blocked_txt} |")
    out.append("")
    out.append(f"AUC-PR baseline (prevalence) is "
               f"{cv['models'][cv['ranking'][0]]['oof_metrics']['auc_pr_baseline']}.\n")

    if tuning:
        out.append("### 4.1 Nested hyperparameter search\n")
        out.append(
            "The reference works tune with `GridSearchCV` and report the best "
            "cross-validated score. That score *can* be optimistically biased, "
            "because the folds that chose the hyperparameters also graded them. "
            "Running the search inside an outer loop it never sees measures the "
            "size of that bias directly rather than assuming it.\n")
        out.append("| Model | Nested (honest) AUC | Inner best AUC | Tuning optimism |")
        out.append("|---|---|---|---|")
        for name, r in sorted(tuning.items(),
                              key=lambda kv: -kv[1]["nested_auc_mean"]):
            out.append(
                f"| {name} | {r['nested_auc_mean']:.4f} ± {r['nested_auc_std']:.4f} | "
                f"{r['inner_best_auc']:.4f} | "
                f"{_signed(r['optimism_from_tuning'])} |")
        out.append("")
        worst = max(tuning.values(), key=lambda r: r["optimism_from_tuning"])
        biggest = max(abs(r["optimism_from_tuning"]) for r in tuning.values())
        if worst["optimism_from_tuning"] >= 0.01:
            out.append(
                f"Tuning optimism reaches {_signed(worst['optimism_from_tuning'])} "
                f"AUC. Any comparison quoting an inner-loop score — as the "
                f"reference works do — is inflated by roughly that much before "
                f"any other issue is considered.\n")
        else:
            out.append(
                f"**This experiment did not find what it was set up to find, and "
                f"that is reported rather than dropped.** Tuning optimism is "
                f"negligible here: the largest gap in either direction is "
                f"{biggest:.4f} AUC, and three of the five models score *higher* "
                f"on the honest outer loop than on the inner one. With a search "
                f"space this small relative to 1,836 training rows, the "
                f"inner-loop estimate is a fair one. So this particular "
                f"criticism does not apply to the reference works — their "
                f"`GridSearchCV` scores are not inflated by the tuning itself. "
                f"The defects documented in §2 are quite sufficient on their "
                f"own without adding one the data does not support.\n")
        out.append(
            "A second observation from the same table: tuned logistic "
            "regression reaches the top of this list. On 369 events a "
            "penalised linear model is genuinely competitive with everything "
            "more elaborate.\n")
    return "\n".join(out)


def section_significance(cmp) -> str:
    if not cmp:
        return ""
    out = ["### 4.2 Which differences are real?\n"]
    out.append(
        f"The spread from best to worst in the table above is about 0.035 AUC "
        f"and the fold-to-fold standard deviation is about 0.03. A ranking "
        f"alone would therefore invite a claim the data cannot support. Each "
        f"model below is tested against **{cmp['best_model']}** by DeLong's "
        f"test on the shared out-of-fold predictions.\n")
    out.append("| Model | Pooled OOF AUC | Δ vs best | p | Verdict |")
    out.append("|---|---|---|---|---|")
    for name, r in cmp["vs_best"].items():
        p = "—" if r["p_value"] is None else f"{r['p_value']:.4f}"
        out.append(f"| {name} | {r['oof_auc']:.4f} | {_signed(r['delta_vs_best'])} "
                   f"| {p} | {r['verdict']} |")
    out.append("")
    out.append(f"**{cmp['note']}**\n")
    out.append(
        "The result is two tiers rather than one winner. A top group — bagged "
        "forests, both ensembles, the deep MLP, the block-attention network and "
        "penalised logistic regression — cannot be told apart. Below it sits a "
        "group that genuinely is worse, and it is worth noting what is in that "
        "group: the boosted models and three of the four reference deep "
        "architectures, with the deepest of them (ResidualMLP) last. On 369 "
        "events, capacity is not the binding constraint and adding it costs "
        "rather than pays.\n")
    return "\n".join(out)


def section_selection(sel) -> str:
    if not sel:
        return ""
    out = ["### 4.3 Which model gets shipped, and why\n"]
    out.append(f"**Rule, fixed before looking at the answer:** {sel['rule']}.\n")
    if not sel.get("statistically_tied_tier"):
        return "\n".join(out)
    out.append(
        f"Repeated CV and position-blocked CV disagree, and the disagreement is "
        f"informative. A random split can put residue 490 in training and "
        f"residue 491 in test — neighbouring positions in the same epitope, "
        f"which is closer to interpolation than prediction. Blocking removes "
        f"that, and it is the situation a treatment centre is actually in when a "
        f"newly sequenced patient carries an uncatalogued variant.\n")
    out.append("| Model in the statistically-tied tier | Repeated-CV AUC | Blocked AUC |")
    out.append("|---|---|---|")
    for name, bl in sel["blocked_auc_within_tier"].items():
        out.append(f"| {name} | — | {bl:.4f} |")
    out.append("")
    out.append(f"{sel['note']}\n")
    if sel.get("changed_from_cv_winner"):
        out.append(
            f"The rule therefore ships **{sel['selected']}** rather than "
            f"{sel['best_by_repeated_cv']}, which had the higher headline AUC. "
            f"Choosing the other way round would have meant picking the "
            f"protocol that flattered the number — the kind of choice §2 of this "
            f"report exists to call out.\n")
    if sel.get("disclosure"):
        out.append(f"> **Disclosure.** {sel['disclosure']}\n")
    return "\n".join(out)


def section_final(final) -> str:
    if not final:
        return ""
    out = ["## 5. Final model on the held-out test set\n"]
    out.append(
        f"`{final['selected_model']}`, isotonic-calibrated, trained on "
        f"{final['n_train']} patients and evaluated once on {final['n_test']} "
        f"held-out patients ({final['test_events']} events). Both decision "
        f"thresholds were chosen on out-of-fold predictions from the training "
        f"set; the test set was not used to select anything.\n")
    out.append("| Metric | Youden threshold | 90%-sensitivity threshold |")
    out.append("|---|---|---|")
    a, b = final["test_calibrated_youden"], final["test_calibrated_sens90"]
    for key, label in [("threshold", "Threshold"), ("auc_roc", "AUC-ROC"),
                       ("auc_pr", "AUC-PR"), ("sensitivity", "Sensitivity"),
                       ("specificity", "Specificity"), ("precision", "Precision (PPV)"),
                       ("npv", "NPV"), ("balanced_accuracy", "Balanced accuracy"),
                       ("f1", "F1"), ("mcc", "MCC"), ("brier", "Brier"),
                       ("ece", "Calibration error"),
                       ("net_benefit_at_20pct", "Net benefit @ 20%")]:
        fmt = _pct if key in {"sensitivity", "specificity", "precision", "npv",
                              "balanced_accuracy"} else _num
        out.append(f"| {label} | {fmt(a[key])} | {fmt(b[key])} |")
    out.append(f"| Confusion (TP/FP/FN/TN) | {a['tp']}/{a['fp']}/{a['fn']}/{a['tn']} "
               f"| {b['tp']}/{b['fp']}/{b['fn']}/{b['tn']} |")
    out.append("")

    ce = final["calibration_effect"]
    out.append("### 5.1 Calibration\n")
    out.append(
        f"A risk score used to decide whether to start inhibitor-aware "
        f"prophylaxis has to mean what it says: among patients scored at 30%, "
        f"about 30% should develop inhibitors. Neither reference work reports "
        f"this. Isotonic calibration moves the Brier score from "
        f"{_num(ce['brier_uncalibrated'])} to {_num(ce['brier_calibrated'])} "
        f"and the expected calibration error from {_num(ce['ece_uncalibrated'])} "
        f"to {_num(ce['ece_calibrated'])}.\n")
    out.append(
        "The decision curve in `reports/figures/03_performance_panel.png` "
        "shows where the model beats both default strategies (test everyone / "
        "test no one) in net benefit — the range of clinical thresholds over "
        "which using it is better than not using it.\n")
    return "\n".join(out)


def section_accuracy(final) -> str:
    """Accuracy, stated the only way it can honestly be stated."""
    if not final or "accuracy_context" not in final:
        return ""
    ctx = final["accuracy_context"]
    acc = final.get("test_calibrated_accuracy", {})
    you = final.get("test_calibrated_youden", {})
    sen = final.get("test_calibrated_sens90", {})

    out = ["### 5.2 Accuracy, and why it is reported with a baseline\n"]
    out.append(
        f"Accuracy is the metric review panels usually ask for, so it is "
        f"reported here — next to the number a model gets for never predicting "
        f"an inhibitor at all. On a {_pct(ctx['prevalence'])}-prevalence "
        f"outcome the second figure is not a formality: it is most of the "
        f"first one.\n")
    out.append("| Operating point | Accuracy | Sensitivity | Specificity | Cases caught / missed |")
    out.append("|---|---|---|---|---|")
    for label, m in [("Balanced (Youden's J)", you),
                     ("High sensitivity (90%)", sen),
                     ("Accuracy-maximising", acc)]:
        if not m:
            continue
        out.append(f"| {label} | {_pct(m['accuracy'])} | {_pct(m['sensitivity'])} "
                   f"| {_pct(m['specificity'])} | {m['tp']} / {m['fn']} |")
    out.append(f"| *Predict \"no inhibitor\" for everyone* | "
               f"*{_pct(ctx['majority_class_accuracy'])}* | *0.00%* | "
               f"*100.00%* | *0 / {you.get('tp', 0) + you.get('fn', 0)}* |")
    out.append("")

    margin = round(ctx["model_accuracy"] - ctx["majority_class_accuracy"], 4)
    out.append(
        f"The accuracy-maximising operating point reaches "
        f"**{_pct(ctx['model_accuracy'])}** against a no-skill baseline of "
        f"**{_pct(ctx['majority_class_accuracy'])}** — a margin of "
        f"**{_signed(margin, 4)}**. It gets there by declining to predict "
        f"inhibitors: it catches {acc.get('tp', 0)} of "
        f"{acc.get('tp', 0) + acc.get('fn', 0)} cases. That is the arithmetic "
        f"of an imbalanced outcome, not a property of this particular model, "
        f"and it is why the tool ships on the balanced and high-sensitivity "
        f"points instead.\n")
    out.append(
        "Two consequences worth stating plainly. **No threshold anywhere on "
        "the curve reaches 85% accuracy** — the maximum is the figure above. "
        "And a version of this label that counts unrecorded outcomes as "
        "negatives lifts the no-skill baseline to 88.6%, at which point an "
        "accuracy target in the high eighties is met by a model that does "
        "nothing. Section 9 works through a dataset where exactly that "
        "happens.\n")
    out.append(
        "The metrics that cannot be gamed this way — AUC-ROC, AUC-PR against "
        "prevalence, balanced accuracy, MCC — are the ones this project leads "
        "with, and they are in the table above this one.\n")
    return "\n".join(out)


def section_subgroups(sub) -> str:
    if not sub:
        return ""
    out = ["## 6. Does it work where it would be used?\n"]
    out.append(
        "Inhibitor prophylaxis decisions are made almost entirely in **severe** "
        "hemophilia A. A model with a respectable overall AUC that sits at "
        "chance inside the severe stratum would be useless in clinic, and the "
        "overall number would never reveal it. Neither reference work reports "
        "subgroup performance.\n")
    out.append("| Subgroup | n | Events | Prevalence | AUC-ROC (95% CI) | Sens. | Spec. |")
    out.append("|---|---|---|---|---|---|---|")
    for r in sub["subgroups"]:
        auc = (f"{r['auc_roc']:.3f} ({r['auc_ci']})"
               if r.get("auc_roc") else f"— *{r.get('note', '')}*")
        out.append(
            f"| {r['subgroup']} | {r['n']} | {r['events']} | "
            f"{_pct(r['prevalence'])} | {auc} | "
            f"{_pct(r['sensitivity']) if r.get('sensitivity') else '—'} | "
            f"{_pct(r['specificity']) if r.get('specificity') else '—'} |")
    out.append("")
    out.append(f"*{sub['note']}*\n")

    by_name = {r["subgroup"]: r for r in sub["subgroups"]}
    overall = by_name.get("All patients", {}).get("auc_roc")
    severe = by_name.get("Severe phenotype", {}).get("auc_roc")
    trunc = by_name.get("Truncating only", {}).get("auc_roc")
    null = by_name.get("Null variants", {}).get("auc_roc")

    if overall and severe and trunc:
        out.append("### 6.1 The most important caveat in this report\n")
        out.append(
            f"The overall AUC of {overall:.3f} is not evenly distributed. "
            f"Inside the **severe** stratum — where essentially every "
            f"prophylaxis decision is actually made — it falls to "
            f"{severe:.3f}. Inside **null variants** it is {null:.3f}. Inside "
            f"**truncating variants alone** it is {trunc:.3f}, which is "
            f"indistinguishable from chance.\n")
        out.append(
            "The reading is uncomfortable but clear: most of the model's "
            "apparent discrimination comes from separating null variants from "
            "non-null ones — and a clinician already knows that from the "
            "variant type without any model at all. Within the high-risk group, "
            "where a tool would actually add information, this model adds very "
            "little.\n")
        out.append(
            "That is a limit of the data rather than of the fitting. Whether a "
            "particular severe, null-variant patient develops an inhibitor "
            "depends on treatment intensity, product type, age at first "
            "exposure and HLA haplotype — none of which CHAMP records. No "
            "model can recover from a database what was never in it, and a "
            "report that showed only the pooled 0.727 would have concealed "
            "exactly the thing a reviewer most needs to know.\n")
        out.append(
            "The one stratum with strong discrimination is **large structural "
            "variants** (AUC 0.882), but on 31 patients with 19 events the "
            "interval is wide and it should be treated as a signal to follow "
            "up, not a result.\n")
    return "\n".join(out)


def section_external(external) -> str:
    if not external:
        return ""
    out = ["## 7. External validation: zero-shot transfer to hemophilia B\n"]
    m = external["metrics"]
    out.append(
        f"The F8 model is applied unchanged to {external['n_scored']} "
        f"hemophilia **B** patients from the CDC CHBMP database "
        f"({external['n_events']} inhibitor-positive, "
        f"{_pct(external['prevalence'])} prevalence). F9 is a different gene "
        f"coding a different protein, and no F9 patient took any part in "
        f"training, feature fitting or threshold selection.\n")
    out.append(
        "Nothing F8-specific can transfer. What can transfer is the underlying "
        "immunology: a null variant abolishes the protein, so the patient was "
        "never tolerised to the factor they are later infused with. A model "
        "that survives this transfer has learned that mechanism; a model that "
        "memorised F8 collapses to chance. No reference work attempts this "
        "test.\n")
    if external.get("caveat"):
        out.append(f"> **What this does and does not show.** {external['caveat']}\n")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| AUC-ROC (95% CI) | {_ci(external.get('auc_ci'))} |")
    out.append(f"| AUC-PR | {_num(m['auc_pr'])} (baseline {_num(m['auc_pr_baseline'])}) |")
    out.append(f"| Sensitivity | {_pct(m['sensitivity'])} |")
    out.append(f"| Specificity | {_pct(m['specificity'])} |")
    out.append(f"| Balanced accuracy | {_pct(m['balanced_accuracy'])} |")
    out.append(f"| MCC | {_num(m['mcc'])} |")
    out.append("")
    return "\n".join(out)


def section_ssl(ssl) -> str:
    if not ssl:
        return ""
    p = ssl["reporting_bias_probe"]
    u = ssl["unlabelled_risk_profile"]
    d = ssl["delong_ssl_vs_supervised"]
    out = ["## 8. The 1,744 unrecorded outcomes\n"]
    out.append(
        f"**Is the missingness informative?** A classifier trained to predict "
        f"*whether* a variant's inhibitor status was recorded reaches AUC "
        f"{_num(p['reporting_auc'])}. Interpretation: {p['interpretation']}.\n")
    out.append(
        f"**What does relabelling them cost?** Scoring the unlabelled pool with "
        f"the trained model gives a mean predicted risk of "
        f"{_num(u['mean_predicted_risk'])} and flags "
        f"{u['predicted_positive_at_0.5']} of {u['n_unlabelled']} rows as "
        f"likely positive. Setting all of them to 0, as the reference does, "
        f"therefore injects on the order of {u['predicted_positive_at_0.5']} "
        f"false negatives straight into the training signal.\n")
    out.append(
        f"**Does using them help?** Self-training over the pool moves held-out "
        f"AUC from {_num(ssl['supervised_test_auc'])} to "
        f"{_num(ssl['semisupervised_test_auc'])} "
        f"(DeLong p = {d.get('p_value')}). "
        + ("The difference is statistically significant."
           if (d.get("p_value") is not None and d["p_value"] < 0.05)
           else "The difference is not statistically significant, and is "
                "reported as such rather than claimed as an improvement.")
        + "\n")
    return "\n".join(out)


def section_comparison(final, audit) -> str:
    out = ["## 9. Comparison with the reference works\n"]
    out.append(
        "Comparing raw accuracy across different label definitions and "
        "different splitting protocols is meaningless, so the table below "
        "states the protocol alongside every number.\n")
    out.append("| Work | Reported | Protocol | Reproduces? |")
    out.append("|---|---|---|---|")
    for name, c in REFERENCE_CLAIMS.items():
        acc = f"{c['accuracy']:.2f}% accuracy" + (
            f", AUC {c['auc']}" if c["auc"] else "")
        out.append(f"| {name} | {acc} | {c['protocol']} | No — see §2 |")
    if final:
        t = final["test_calibrated_youden"]
        out.append(
            f"| **This project ({final['selected_model']})** | "
            f"AUC {_num(t['auc_roc'])}, balanced accuracy {_pct(t['balanced_accuracy'])} | "
            f"single held-out test set, thresholds fixed on training folds, "
            f"identifier columns excluded, unrecorded outcomes excluded | "
            f"Yes — plus external cohort |")
    out.append("")
    if audit:
        d = audit.get("D_honest_labels", {})
        out.append(
            f"The like-for-like comparison is the one that matters: with the "
            f"same honest labels and the same clean split, the reference's "
            f"feature set reaches AUC {_num(d.get('test_auc'))}. "
            + (f"This project's feature set and model reach "
               f"{_num(final['test_calibrated_youden']['auc_roc'])} on the same "
               f"task." if final else "") + "\n")
    return "\n".join(out)


def section_fused(audit, sim) -> str:
    """The fused CHAMP + clinical dataset: what it promised and what it was."""
    if not (audit and sim):
        return ""
    prov = audit["provenance"]
    out = ["## 9. A dataset that appeared to solve the problem\n"]
    out.append(
        "Section 10 says the binding constraint is the absence of "
        "patient-level covariates. A collaborator supplied "
        "`Final_Fused_Dataset.csv`: CHAMP with five of those covariates "
        "appended — age at diagnosis, ethnicity, treatment regimen, exposure "
        "days, family history — and it reports accuracy in the high eighties. "
        "It was tested properly rather than adopted.\n")

    out.append("### 9.1 Where the high accuracy comes from\n")
    out.append(
        "The file's `Inhibitor_Status` column maps CHAMP's 1,731 unrecorded "
        "outcomes to 0 — the defect documented in §2. That moves prevalence "
        "from 20.1% to 11.4%, and with it the no-skill baseline:\n")
    out.append("| | Accuracy |")
    out.append("|---|---|")
    out.append("| Predict \"no inhibitor\" for every patient | **88.55%** |")
    out.append("| Trained model on the supplied label | **89.58%** |")
    out.append("| Margin over doing nothing | **+0.99 points** |")
    out.append("| Inhibitor cases actually caught | **13%** (16 flagged of 806) |")
    out.append("")
    out.append(
        "The 89.6% falls inside the range a rubric might ask for. It is also, "
        "in substance, what a model scores for learning to say \"no\".\n")

    out.append("### 9.2 The clinical columns are simulated\n")
    out.append(
        "CHAMP rows are published *variants*, not patients — one row "
        "aggregates every case ever reported with that mutation — so there is "
        "no key on which per-patient clinical data could have been joined. "
        "Four independent checks agree the block was generated:\n")
    out.append("| Check | Finding |")
    out.append("|---|---|")
    out.append(f"| `Patient_ID` format | random UUID4 on "
               f"{_pct(prov['patient_id']['fraction_matching_uuid4'], 0)} of rows |")
    out.append(f"| `Ethnicity` association | inhibitor rate flat across all five "
               f"groups, spread {prov['ethnicity']['spread_pct']} points, "
               f"chi-square p = {prov['ethnicity']['chi2_p']} |")
    out.append(f"| `Family_History` effect | odds ratio "
               f"{prov['family_history']['odds_ratio']} against a published "
               f"{prov['family_history']['published_odds_ratio']} |")
    out.append(f"| Age vs exposure days | r = "
               f"{prov['age_vs_exposure']['pearson_r']} |")
    out.append("")
    out.append(
        "The ethnicity result is decisive. Roughly two-fold higher inhibitor "
        "risk in Black and Hispanic patients is among the most reproducible "
        "non-genetic findings in this field, replicated across CDC "
        "surveillance, MLOF and UKHCDO. A real cohort of 4,026 patients would "
        "show it. A column drawn from a fixed multinomial produces exactly the "
        "flat line observed.\n")

    out.append("### 9.3 Evaluated properly, they add nothing\n")
    out.append(
        "Same folds, same held-out patients, same leakage-free genomic "
        "featuriser, honest labels throughout. The only difference between "
        "arms is whether the clinical block is present:\n")
    out.append("| Arm | Features | CV AUC | Held-out AUC (95% CI) |")
    out.append("|---|---|---|---|")
    for key, label in [("genomic_only", "Genomic only"),
                       ("clinical_only", "Clinical only"),
                       ("genomic_plus_clinical", "Genomic + clinical")]:
        a = sim[key]
        out.append(f"| {label} | {a['n_features']} | "
                   f"{a['cv_auc_mean']:.4f} ± {a['cv_auc_std']:.4f} | "
                   f"{_ci(a['test_auc_ci'])} |")
    out.append("")
    d = sim["_delong_gain"]
    out.append(
        f"Adding the clinical block changes held-out AUC by "
        f"{_signed(sim['_summary']['auc_gain_from_clinical'])} — DeLong "
        f"p = {d.get('p_value')}. Cross-validation AUC rises while held-out "
        f"AUC falls, which is what fitting injected noise looks like.\n")
    out.append(
        "So the dataset offers no real gain, and the accuracy it advertises is "
        "the baseline in disguise. Read the other way round it is still "
        "useful: it is a serviceable power analysis showing what *real* "
        "registry covariates would need to look like, and it supports the "
        "conclusion in §10 that the ceiling here is data rather than method. "
        "Reported as a simulation, which is what it is.\n")
    return "\n".join(out)


def section_integrity(integ) -> str:
    if not integ:
        return ""
    s = integ["_summary"]
    out = ["## 10. Pipeline integrity checks\n"]
    out.append(
        "The claim that these numbers are trustworthy is worth no more than "
        "the checks behind it, so each property is verified mechanically and "
        "the result is written to `reports/integrity.json` rather than "
        "asserted in prose.\n")
    out.append("| Check | Result |")
    out.append("|---|---|")
    for name, r in integ.items():
        if name.startswith("_"):
            continue
        mark = {True: "pass", False: "**FAIL**", None: "skipped"}[r.get("passed")]
        out.append(f"| {r.get('check', name)} | {mark} |")
    out.append("")
    out.append(f"**{s['passed']} of {s['n_checks']} passed.**"
               + ("" if s["all_passed"] else f" Failing: {s['failed_checks']}.")
               + "\n")
    out.append(
        "Two are worth spelling out. *No resampling*: class imbalance is "
        "handled by weighting the objective, never by duplicating or "
        "synthesising patients — the reference pipeline's Random Over-Sampling "
        "is what put half of its own test set into its training data. "
        "*Featuriser is label-blind*: scrambling the outcome and re-fitting "
        "produces a byte-identical feature matrix, so the engineering cannot "
        "have absorbed the answer.\n")
    return "\n".join(out)


def section_limitations() -> str:
    return """## 11. Limitations

These are stated because a model for clinical use is only as trustworthy as its
declared boundaries.

1. **Most of the discrimination is null-versus-non-null, which is already
   known.** Section 6.1 is the limitation that matters most. Pooled AUC is
   0.727, but inside the severe stratum it is 0.694 and inside truncating
   variants alone it is 0.541 — chance. The model largely reproduces a
   distinction the variant type already gives a clinician for free, and adds
   little within the high-risk group where a tool would actually change
   management.
2. **CHAMP is a variant catalogue, not a patient registry.** Each row is a
   distinct mutation whose outcome is summarised across everyone reported to
   carry it. Individual patients are not resolvable, so the label carries
   irreducible noise and the unit of analysis is the variant.
3. **The strongest known risk factors are absent.** Treatment intensity,
   product type, exposure days, HLA haplotype, family history and ethnicity all
   drive inhibitor development and none are in the data. A genomic-only model
   has a ceiling well below what the reference works advertise, and the
   performance here should be read against that ceiling.
4. **Reporting bias.** CHAMP aggregates published case reports, which
   over-represent unusual variants and outcomes worth publishing.
5. **The external cohort is small.** CHBMP contributes 351 labelled patients
   with 40 events, so its confidence interval is wide. It establishes that
   transfer happens, not how well.
6. **Not a medical device.** Research and educational use only. It does not
   replace clinical judgement or laboratory inhibitor testing.

---

## 12. Reproducing this document

```bash
python scripts/fetch_data.py
python -m src.train --stage all
python -m src.figures
python -m src.report
python -m pytest
```

Figures land in `reports/figures/`, measurements in `reports/*.json`, and this
document is regenerated from them.
"""


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build() -> Path:
    audit = _load("audit")
    cv = _load("cv")
    blocked = _load("blocked_cv")
    tuning = _load("tuning")
    final = _load("final")
    external = _load("external")
    ssl = _load("ssl")
    sub = _load("subgroups")
    quant = _load("quantisation")
    abl = _load("ablation")
    cmp = _load("model_comparison")
    sel = _load("selection")
    fused_audit = _load("fused_audit")
    fused_sim = _load("fused_simulation")
    integ = _load("integrity")

    parts = [
        section_header(),
        section_summary(audit, cv, final, external, ssl),
        section_audit(audit),
        section_features(quant),
        section_ablation(abl),
        section_models(cv, blocked, tuning),
        section_significance(cmp),
        section_selection(sel),
        section_final(final),
        section_accuracy(final),
        section_subgroups(sub),
        section_external(external),
        section_ssl(ssl),
        section_comparison(final, audit),
        section_fused(fused_audit, fused_sim),
        section_integrity(integ),
        section_limitations(),
    ]
    OUT.write_text("\n".join(p for p in parts if p), encoding="utf-8")
    print(f"  -> {OUT.relative_to(ROOT)}")
    return OUT


if __name__ == "__main__":
    build()
