"""Blind human labeling CLI. You never see the condition or the judge's scores.

Usage:
  python -m judge_eval.label                     # label (resumable; Ctrl+C or 'q' to stop anytime)
  python -m judge_eval.label --relabel 0.2       # re-label a random 20% subset (do this >=7 days later)
"""
import argparse
import csv
import os
import random
import shutil
import textwrap
from datetime import datetime

import yaml

from judge_eval.llm import CRITERIA, read_jsonl

W = min(shutil.get_terminal_size().columns, 110)
FIELDS = ["id", *CRITERIA, "hallucination", "notes", "seconds", "labeled_at"]


def wrap(t, n=None):
    t = t if n is None or len(t) <= n else t[:n] + " ...[truncated, type 'c' to show full context]"
    return "\n".join(textwrap.fill(p, W) for p in t.split("\n"))


def ask_score(name, anchors):
    while True:
        v = input(f"  {name} (1-5, ?=anchors, q=quit): ").strip().lower()
        if v == "q":
            raise KeyboardInterrupt
        if v == "?":
            for s in sorted(anchors, reverse=True):
                print(f"     {s} = {anchors[s]}")
            continue
        if v in {"1", "2", "3", "4", "5"}:
            return int(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval_set.jsonl")
    ap.add_argument("--rubric", default="rubric/rubric_v1.yaml")
    ap.add_argument("--out", default="results/human_labels.csv")
    ap.add_argument("--relabel", type=float, default=None, help="fraction of labeled items to re-label")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rubric = yaml.safe_load(open(a.rubric))
    items = {r["id"]: r for r in read_jsonl(a.data)}
    order = list(items)
    random.Random(a.seed).shuffle(order)  # fixed random order, unrelated to condition

    out = a.out
    if a.relabel:
        first = [r["id"] for r in csv.DictReader(open(a.out))]
        k = round(len(first) * a.relabel)
        order = random.Random(a.seed + 1).sample(first, k)
        out = a.out.replace(".csv", "_relabel.csv")
        print(f"Re-labeling {k} items blind to your earlier labels -> {out}")

    done = {r["id"] for r in csv.DictReader(open(out))} if os.path.exists(out) else set()
    todo = [i for i in order if i not in done]
    new = not os.path.exists(out)
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()

    try:
        for n, iid in enumerate(todo, 1):
            it = items[iid]
            t0 = datetime.now()
            print("\n" + "=" * W)
            print(f"[{len(done) + n}/{len(order)}]  id={iid}")
            print("\nQUESTION:\n" + wrap(it["question"]))
            print("\nCONTEXT:\n" + wrap(it["context"], 900))
            print("\nREFERENCE:\n" + wrap(it["reference"]))
            print("\nRESPONSE:\n" + wrap(it["response"]))
            print("-" * W)
            if input("  Enter to score, 'c' for full context: ").strip().lower() == "c":
                print(wrap(it["context"]))
            row = {"id": iid}
            for c in CRITERIA:
                row[c] = ask_score(c, rubric["criteria"][c]["anchors"])
            row["hallucination"] = int(input("  any claim unsupported by context? (y/N): ").strip().lower() == "y")
            row["notes"] = input("  notes (optional): ").strip()
            row["seconds"] = round((datetime.now() - t0).total_seconds())
            row["labeled_at"] = datetime.now().isoformat(timespec="seconds")
            w.writerow(row)
            f.flush()
    except (KeyboardInterrupt, EOFError):
        print("\nSaved. Run again to resume.")
    finally:
        f.close()


if __name__ == "__main__":
    main()
