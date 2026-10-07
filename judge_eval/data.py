"""Build the evaluation set: PubMedQA questions + candidate responses under 4 conditions.

Conditions (balanced, assigned at random with a fixed seed):
  grounded        - model answers using the abstract
  closed_book     - model answers WITHOUT the abstract (natural hallucination risk)
  incomplete      - model answers from the abstract in one short sentence
  injected_error  - grounded answer, then ONE subtle factual error is inserted
                    (known-error control set; the inserted error is recorded)

Usage:
  python -m judge_eval.data --n 400 --model gpt-4o-mini
"""
import argparse
import json
import os
import random

from judge_eval.llm import append_jsonl, chat, get_client, read_jsonl

CONDITIONS = ["grounded", "closed_book", "incomplete", "injected_error"]

SYS = "You are a medical research assistant. Answer the question accurately in 3-5 sentences."


def load_pubmedqa(n, seed):
    from datasets import load_dataset

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    rows = [
        {
            "id": str(r["pubid"]),
            "question": r["question"],
            "context": "\n".join(r["context"]["contexts"]),
            "reference": r["long_answer"],
            "final_decision": r["final_decision"],
        }
        for r in ds
    ]
    rng = random.Random(seed)
    # stratify by gold yes/no/maybe so the sample mirrors the dataset
    by = {}
    for r in rows:
        by.setdefault(r["final_decision"], []).append(r)
    out = []
    for k, group in by.items():
        rng.shuffle(group)
        out += group[: round(n * len(group) / len(rows))]
    rng.shuffle(out)
    out = out[:n]
    conds = (CONDITIONS * (len(out) // 4 + 1))[: len(out)]
    rng.shuffle(conds)
    for r, c in zip(out, conds):
        r["condition"] = c
    return out


def generate(client, model, item):
    q, ctx = item["question"], item["context"]
    c = item["condition"]
    if c == "closed_book":
        user = f"Question: {q}"
    elif c == "incomplete":
        user = f"Abstract:\n{ctx}\n\nQuestion: {q}\n\nAnswer in ONE short sentence only, without the supporting findings."
    else:
        user = f"Abstract:\n{ctx}\n\nQuestion: {q}\n\nAnswer using only the abstract."
    resp, _ = chat(client, model, [{"role": "system", "content": SYS}, {"role": "user", "content": user}], temperature=0.7)
    item["response"] = resp
    item["injected_error"] = None
    if c == "injected_error":
        prompt = (
            "Rewrite the answer below so it contains exactly ONE subtle but clear factual error "
            "(e.g., reverse a finding, change a number, or swap a group). Keep everything else identical. "
            'Return JSON: {"answer": "...", "error_description": "..."}\n\n'
            f"Abstract:\n{ctx}\n\nAnswer:\n{resp}"
        )
        out, _ = chat(client, model, [{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        j = json.loads(out)
        item["response"] = j["answer"]
        item["injected_error"] = j["error_description"]
    return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--model", default=os.getenv("GEN_MODEL", "gpt-4o-mini"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/eval_set.jsonl")
    a = ap.parse_args()

    done = {r["id"] for r in read_jsonl(a.out)} if os.path.exists(a.out) else set()
    items = [r for r in load_pubmedqa(a.n, a.seed) if r["id"] not in done]
    print(f"{len(done)} already generated, {len(items)} to go")
    client = get_client()
    for i, item in enumerate(items, 1):
        item["gen_model"] = a.model
        append_jsonl(a.out, generate(client, a.model, item))
        if i % 20 == 0:
            print(f"  {i}/{len(items)}")
    print(f"done -> {a.out}")


if __name__ == "__main__":
    main()
