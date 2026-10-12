# LLM-as-a-Judge Evaluation Framework for Medical QA

Can an LLM judge grade medical answers the way a careful human expert does? This project builds a rubric-based judge for PubMedQA and the full protocol to validate it against blind human labels: weighted Cohen's κ, a known-error control set, length-bias tests, and an error-taxonomy audit.

> **Status:** the judge, rubric, blind labeling tool and analysis pipeline are built and tested. Human labeling is in progress; results will be published in `results/` when complete.

## Why this matters for an AI product

Any product that uses an LLM to grade, tag or summarise needs an answer to "how often is it wrong, and in which direction?" before anyone acts on its output. This repo is that answer for one task, built so the same protocol can be reused:

- **The judge can't cheat.** It never sees which condition produced an answer, writes its rationale before its score, and runs at temperature 0 with schema-validated JSON.
- **Known-error controls.** A quarter of answers have one deliberately injected factual error, so the catch rate is measured, not assumed.
- **Bias checks.** Length bias and test–retest stability are measured, because LLM judges are known to favour longer answers.
- **No fitting to the judge.** Rubric versions are compared on the *same* human labels; disagreements go through an error-taxonomy audit to decide what to change.

## Design

| Component | What it does |
|---|---|
| `judge_eval/data.py` | Samples 400 PubMedQA (pqa_labeled) questions, stratified by gold yes/no/maybe. Generates one answer per question under one of 4 balanced conditions: `grounded`, `closed_book`, `incomplete`, `injected_error` (one known factual error inserted; the error is recorded). |
| `rubric/rubric_v1.yaml` | 4 criteria (factual accuracy, groundedness, completeness, relevance), 1–5 anchors, and rules that separate "incomplete" from "inaccurate" and "true" from "supported". |
| `judge_eval/judge.py` | Judge with strict JSON-schema output (rationale before score), temperature 0, retries, validation, and `--repeats` for test–retest stability. The judge never sees the condition. |
| `judge_eval/label.py` | Blind, resumable terminal labeling tool in fixed random order. `--relabel 0.2` re-labels a 20% subset to measure your own consistency. |
| `judge_eval/analyze.py` | Per-criterion QWK with bootstrap 95% CIs, exact/±1 agreement, confusion matrices, unsupported-claim precision/recall, injected-error catch rate (Wilson CI), scores by condition, length bias, judge stability, intra-rater κ. It also writes an audit sample for root-cause analysis. Optional MLflow logging. |

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env        # add your OpenAI or Azure OpenAI key, then: export $(cat .env | xargs)

python -m judge_eval.data --n 400                                    # ~$1, a few minutes
python -m judge_eval.judge --rubric rubric/rubric_v1.yaml            # gpt-4o judge, ~$3-6
python -m judge_eval.judge --rubric rubric/rubric_v1.yaml --repeats 2 --limit 100   # stability
python -m judge_eval.label                                           # human labeller: ~6-8 h, resumable
python -m judge_eval.analyze --judgments results/judgments_v1_gpt-4o.jsonl
# >= 7 days later:
python -m judge_eval.label --relabel 0.2
python -m judge_eval.analyze --judgments results/judgments_v1_gpt-4o.jsonl
pytest -q
```

Costs are rough estimates; check your usage dashboard.

## Iterate (this is the part that makes it science)

1. Open `results/<tag>/audit_sample.csv` and fill in `error_category`, `root_cause` and `rubric_fix` for each disagreement.
2. Count the categories. The largest one tells you what to fix.
3. Write `rubric/rubric_v2.yaml` (set `version: v2`) to address it: clearer anchors, a borderline example, or a new rule.
4. Re-run `judge` and `analyze` with v2 and compare QWK and CIs. Use the **same** human labels; never re-label to fit the judge.

## Labeling protocol (keep yourself honest)

- Label before you look at any judge output.
- Score each criterion separately, using `?` to check the anchors whenever you're unsure.
- Record a note on hard cases. Those notes feed the audit.
- Do the re-label pass at least a week later, without looking at your first labels.

## Limitations

- There's a single human annotator, so judge-vs-human agreement is capped by your own consistency. That's why intra-rater κ is reported.
- The PubMedQA reference answers are the abstracts' conclusions. They can be incomplete, which the `reference_incomplete` audit category tracks.
- The judge and the generator come from the same model family, which can bias scoring. Try a different judge model as an extension.
