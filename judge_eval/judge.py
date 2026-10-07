"""LLM-as-a-Judge: score each response on the rubric with strict structured JSON output.

Usage:
  python -m judge_eval.judge --rubric rubric/rubric_v1.yaml --model gpt-4o
  python -m judge_eval.judge --rubric rubric/rubric_v1.yaml --repeats 3 --limit 100   # stability check
"""
import argparse
import json
import os
import time

import yaml

from judge_eval.llm import CRITERIA, append_jsonl, chat, get_client, read_jsonl


def build_system_prompt(rubric):
    lines = [
        "You are an expert medical evaluator. Grade the RESPONSE to a biomedical research question.",
        "You are given the QUESTION, the CONTEXT (study abstract), and a REFERENCE answer written by the study authors.",
        "Score each criterion independently on a 1-5 scale. For each criterion, write a short rationale FIRST, then the score.",
        "",
    ]
    for name, c in rubric["criteria"].items():
        lines.append(f"## {name}: {c['question']}")
        for s in sorted(c["anchors"], reverse=True):
            lines.append(f"  {s} = {c['anchors'][s]}")
        lines.append("")
    lines.append("Rules:")
    lines += [f"- {r}" for r in rubric["rules"]]
    return "\n".join(lines)


def json_schema():
    crit = {
        "type": "object",
        "properties": {"rationale": {"type": "string"}, "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]}},
        "required": ["rationale", "score"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "judgment",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    **{c: crit for c in CRITERIA},
                    "unsupported_claims": {"type": "array", "items": {"type": "string"}},
                },
                "required": CRITERIA + ["unsupported_claims"],
                "additionalProperties": False,
            },
        },
    }


def validate(j):
    for c in CRITERIA:
        s = j[c]["score"]
        if not isinstance(s, int) or not 1 <= s <= 5:
            raise ValueError(f"bad score for {c}: {s}")
    return j


def judge_one(client, model, system, item):
    user = (
        f"QUESTION:\n{item['question']}\n\nCONTEXT:\n{item['context']}\n\n"
        f"REFERENCE:\n{item['reference']}\n\nRESPONSE:\n{item['response']}"
    )
    t0 = time.time()
    out, usage = chat(
        client, model, [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0, response_format=json_schema(),
    )
    j = validate(json.loads(out))
    return j, time.time() - t0, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubric", default="rubric/rubric_v1.yaml")
    ap.add_argument("--model", default=os.getenv("JUDGE_MODEL", "gpt-4o"))
    ap.add_argument("--data", default="data/eval_set.jsonl")
    ap.add_argument("--repeats", type=int, default=1, help=">1 measures judge test-retest stability")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    rubric = yaml.safe_load(open(a.rubric))
    system = build_system_prompt(rubric)
    out_path = f"results/judgments_{rubric['version']}_{a.model}.jsonl"
    done = {(r["id"], r["repeat"]) for r in read_jsonl(out_path)} if os.path.exists(out_path) else set()

    items = read_jsonl(a.data)[: a.limit]
    client = get_client()
    todo = [(it, k) for k in range(a.repeats) for it in items if (it["id"], k) not in done]
    print(f"{len(todo)} judgments to run -> {out_path}")
    for i, (it, k) in enumerate(todo, 1):
        try:
            j, secs, usage = judge_one(client, a.model, system, it)
        except Exception as e:
            print(f"  FAILED {it['id']}: {e}")
            continue
        append_jsonl(out_path, {
            "id": it["id"], "repeat": k, "rubric_version": rubric["version"], "judge_model": a.model,
            **{c: j[c]["score"] for c in CRITERIA},
            "rationales": {c: j[c]["rationale"] for c in CRITERIA},
            "unsupported_claims": j["unsupported_claims"],
            "latency_s": round(secs, 2),
            "tokens": getattr(usage, "total_tokens", None),
        })
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}")
    print("done")


if __name__ == "__main__":
    main()
