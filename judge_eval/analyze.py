"""Compare judge vs. human labels and write results/report.md + an audit sample.

Usage:
  python -m judge_eval.analyze --judgments results/judgments_v1_gpt-4o.jsonl
  python -m judge_eval.analyze --judgments ... --mlflow     # also log metrics to MLflow
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from judge_eval.llm import CRITERIA, read_jsonl
from judge_eval.metrics import LABELS, agreement, bootstrap_ci, rate_ci, weighted_kappa

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AUDIT_TAXONOMY = (
    "judge_lenient_unsupported | judge_penalized_paraphrase | judge_confused_criteria | "
    "rubric_ambiguous | reference_incomplete | human_label_error | other"
)


def fmt(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def confusion_png(h, j, title, path):
    m = pd.crosstab(pd.Categorical(h, LABELS), pd.Categorical(j, LABELS), dropna=False).values
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(m, cmap="Blues")
    for (r, c), v in np.ndenumerate(m):
        ax.text(c, r, int(v), ha="center", va="center", fontsize=8)
    ax.set_xticks(range(5), LABELS)
    ax.set_yticks(range(5), LABELS)
    ax.set_xlabel("Judge")
    ax.set_ylabel("Human")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval_set.jsonl")
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--labels", default="results/human_labels.csv")
    ap.add_argument("--audit-n", type=int, default=60)
    ap.add_argument("--mlflow", action="store_true")
    a = ap.parse_args()

    tag = os.path.basename(a.judgments).replace("judgments_", "").replace(".jsonl", "")
    outdir = f"results/{tag}"
    os.makedirs(outdir, exist_ok=True)

    items = pd.DataFrame(read_jsonl(a.data))
    items["resp_words"] = items["response"].str.split().str.len()
    J_all = pd.DataFrame(read_jsonl(a.judgments))
    J = J_all[J_all.repeat == 0]
    H = pd.read_csv(a.labels, dtype={"id": str})
    df = items.merge(J, on="id").merge(H, on="id", suffixes=("_j", "_h"))
    n = len(df)
    if n == 0:
        raise SystemExit("No items have both a judgment and a human label yet.")

    md = [f"# Judge validation report: {tag}\n", f"Items with judge + human label: **{n}**\n"]
    metrics = {}

    # 1. Judge vs human agreement per criterion
    md.append("## 1. Judge vs. human agreement\n")
    md.append("| Criterion | QWK [95% CI] | Exact | Within ±1 | Spearman | Judge − human |")
    md.append("|---|---|---|---|---|---|")
    for c in CRITERIA:
        j, h = df[f"{c}_j"].values, df[f"{c}_h"].values
        s = agreement(j, h)
        lo, hi = bootstrap_ci(j, h)
        md.append(f"| {c} | {fmt(s['qwk'])} [{fmt(lo)}, {fmt(hi)}] | {s['exact']:.0%} | {s['within_1']:.0%} "
                  f"| {fmt(s['spearman'])} | {s['mean_diff']:+.2f} |")
        confusion_png(h, j, c, f"{outdir}/confusion_{c}.png")
        metrics.update({f"{c}_qwk": s["qwk"], f"{c}_exact": s["exact"], f"{c}_within1": s["within_1"]})
    jm, hm = df[[f"{c}_j" for c in CRITERIA]].mean(1), df[[f"{c}_h" for c in CRITERIA]].mean(1)
    r = spearmanr(jm, hm).correlation
    md.append(f"\nMean-of-criteria Spearman ρ (judge vs. human): **{fmt(r)}**. Confusion matrices: `{outdir}/`.\n")
    metrics["overall_spearman"] = r

    # 2. Hallucination flag agreement
    jflag = (df["unsupported_claims"].str.len() > 0).astype(int)
    hflag = df["hallucination"].astype(int)
    tp = int(((jflag == 1) & (hflag == 1)).sum())
    prec = tp / max(jflag.sum(), 1)
    rec = tp / max(hflag.sum(), 1)
    md.append("## 2. Unsupported-claim detection (human label = truth)\n")
    md.append(f"Judge precision **{prec:.0%}**, recall **{rec:.0%}** "
              f"(human flagged {int(hflag.sum())}, judge flagged {int(jflag.sum())}).\n")
    metrics.update({"halluc_precision": prec, "halluc_recall": rec})

    # 3. Known-error control set
    inj = df[df.condition == "injected_error"]
    md.append("## 3. Injected-error control set\n")
    if len(inj):
        md.append("| Rater | Caught (factual_accuracy ≤ 3) [95% CI] | Caught strict (≤ 2) |")
        md.append("|---|---|---|")
        for who, col in [("Judge", "factual_accuracy_j"), ("Human", "factual_accuracy_h")]:
            p, lo, hi = rate_ci(int((inj[col] <= 3).sum()), len(inj))
            p2 = (inj[col] <= 2).mean()
            md.append(f"| {who} | {p:.0%} [{lo:.0%}, {hi:.0%}] | {p2:.0%} |")
            metrics[f"{who.lower()}_injected_caught"] = p
        md.append(f"\nn = {len(inj)} responses with one known inserted error.\n")

    # 4. Scores by condition (sanity: grounded should score highest)
    md.append("## 4. Mean scores by condition (judge / human)\n")
    md.append("| Condition | n | " + " | ".join(CRITERIA) + " |")
    md.append("|---|---|" + "---|" * len(CRITERIA))
    for cond, g in df.groupby("condition"):
        cells = [f"{g[f'{c}_j'].mean():.2f} / {g[f'{c}_h'].mean():.2f}" for c in CRITERIA]
        md.append(f"| {cond} | {len(g)} | " + " | ".join(cells) + " |")

    # 5. Length bias
    md.append("\n## 5. Length bias\n")
    md.append("Spearman ρ between response word count and score. A judge ρ much higher than the human ρ means the judge rewards length.\n")
    md.append("| Criterion | Judge ρ | Human ρ | ρ(length, judge − human) |")
    md.append("|---|---|---|---|")
    for c in CRITERIA:
        rj = spearmanr(df.resp_words, df[f"{c}_j"]).correlation
        rh = spearmanr(df.resp_words, df[f"{c}_h"]).correlation
        rd = spearmanr(df.resp_words, df[f"{c}_j"] - df[f"{c}_h"]).correlation
        md.append(f"| {c} | {fmt(rj)} | {fmt(rh)} | {fmt(rd)} |")
        metrics[f"{c}_length_bias"] = rd

    # 6. Judge stability (test-retest across repeats)
    if J_all.repeat.nunique() > 1:
        md.append("\n## 6. Judge stability (repeat 0 vs repeat 1)\n")
        p = J_all.pivot_table(index="id", columns="repeat", values=CRITERIA)
        md.append("| Criterion | QWK | Exact |")
        md.append("|---|---|---|")
        for c in CRITERIA:
            x = p[c][[0, 1]].dropna()
            md.append(f"| {c} | {fmt(weighted_kappa(x[0].astype(int), x[1].astype(int)))} | {(x[0] == x[1]).mean():.0%} |")

    # 7. Intra-rater reliability (your own consistency)
    rl = a.labels.replace(".csv", "_relabel.csv")
    if os.path.exists(rl):
        R = pd.read_csv(rl, dtype={"id": str}).merge(H, on="id", suffixes=("_2", "_1"))
        md.append(f"\n## 7. Intra-rater reliability (n = {len(R)} re-labeled items)\n")
        md.append("| Criterion | QWK [95% CI] | Exact |")
        md.append("|---|---|---|")
        for c in CRITERIA:
            s = agreement(R[f"{c}_2"], R[f"{c}_1"])
            lo, hi = bootstrap_ci(R[f"{c}_2"].values, R[f"{c}_1"].values)
            md.append(f"| {c} | {fmt(s['qwk'])} [{fmt(lo)}, {fmt(hi)}] | {s['exact']:.0%} |")
            metrics[f"{c}_intra_qwk"] = s["qwk"]
        md.append("\nJudge–human agreement close to your own test–retest agreement means the judge is near the human ceiling.\n")

    # Audit sample: every large disagreement + stratified random agreements
    df["max_gap"] = np.max(np.abs(df[[f"{c}_j" for c in CRITERIA]].values - df[[f"{c}_h" for c in CRITERIA]].values), 1)
    big = df[df.max_gap >= 2]
    rest = df[df.max_gap < 2]
    k = max(a.audit_n - len(big), 0)
    rand = rest.groupby("condition", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(1, k // 4)), random_state=0)) if k and len(rest) else rest.head(0)
    audit = pd.concat([big.assign(audit_reason="disagreement>=2"), rand.assign(audit_reason="random_agreement")])
    cols = ["id", "audit_reason", "condition", "question", "response", "reference"] + \
           [f"{c}_{s}" for c in CRITERIA for s in "jh"] + ["rationales", "unsupported_claims", "notes"]
    audit = audit[cols].assign(error_category="", root_cause="", rubric_fix="")
    audit.to_csv(f"{outdir}/audit_sample.csv", index=False)
    md.append(f"\n## Audit sample\n\n{len(big)} items with a ≥2-point disagreement + {len(rand)} random agreements → "
              f"`{outdir}/audit_sample.csv`. Fill `error_category` using: {AUDIT_TAXONOMY}.\n")

    open(f"{outdir}/report.md", "w").write("\n".join(md))
    print("\n".join(md))
    print(f"\nWrote {outdir}/report.md")

    if a.mlflow:
        import mlflow
        with mlflow.start_run(run_name=tag):
            mlflow.log_params({"judgments": a.judgments, "n_items": n})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
            mlflow.log_artifacts(outdir)


if __name__ == "__main__":
    main()
