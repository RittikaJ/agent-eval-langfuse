# Invoice Agent Evaluation POC

A small OpenAI Agents SDK agent for learning Langfuse evaluation.
The agent extracts a text invoice, calls one calculator tool, and returns a
structured Invoice. Python validates it; evaluation compares it with an expected answer.

## Setup

Python 3.12 is the tested version. Run from this repository folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
cp .env.example .env
```

If .env already exists, edit it rather than replacing it. Set OPENAI_API_KEY.
For tracing, set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL.
The default model is gpt-4.1-mini, configurable through OPENAI_MODEL.

## Run

```bash
python run.py --case invoice-001 --trace
python setup_dataset.py             # preview and review the 30 cases
python setup_dataset.py --upload    # upload reviewed cases
python evaluate.py --experiment baseline
python -m unittest discover -s tests -v
```

run.py and evaluate.py call real models and incur provider costs. Omit --trace
from run.py for local results only. evaluate.py always creates a Langfuse experiment.
Offline tests use scripted model responses and in-memory
traces. They check the code, not extraction accuracy or Langfuse ingestion.

## Files

- agent.py: all agent instructions, invoice fields, calculator, and validation.
- observability.py: Langfuse setup and tracing wrapper.
- evaluate.py: output checks, tool checks, OpenAI judge, and experiment summaries.
- setup_dataset.py: preview/upload the 30-case Langfuse dataset.
- run.py: run cases, upload scores, and save local results.
- data/cases.json: 30 synthetic invoice inputs.
- data/expected_outputs.json: expected results used only by evaluation.
- tests/: offline behavior and tracing checks.

## Evaluation

Scores are field_accuracy, arithmetic_correct, issues_exact, status_correct,
calculator_used, calculator_arguments_correct, and task_success.
The separate groundedness judge returns 0 or 1 with an explanation; judge failures
are evaluator errors, never zero groundedness scores. Named experiments record
per-case scores, averages with coverage, deterministic pass rate, and error counts.
A needs_review result can pass all scores
when it correctly identifies a problem. Review reference labels before making
model-quality claims. The 30 synthetic cases cover valid invoices (8), wrong totals (5), missing fields
(5), decimal/rounding cases (4), ambiguous dates/currencies (4), and embedded
instructions (4). Expected validation statuses are valid (14) and needs_review (16).

This version supports quantity times unit price items in one currency with two
decimal places. Taxes, discounts, shipping, credit notes, and multiple currencies
are outside the POC. Currency validation checks format, not the ISO registry.
The prompt states the scope; there is no comprehensive unsupported-input detector.

## Verification and next steps

All 19 offline tests pass. They cover all 30 case paths, calculator arguments, structured judge
verdicts and errors, summaries, and tracing. They do not establish model quality.
The 30-case live baseline completed on 2026-10-10: 23 passed (76.7%), six
returned outputs that failed deterministic checks, and one had an execution
error. No evaluator/API errors occurred, but manual inspection found mistaken
judge verdicts. Dataset items and cloud scores were retrieved and verified.
See [the experiment](https://jp.cloud.langfuse.com/project/cmuznb50z0009ad0gi3oq6htn/datasets/cmv15oll20011ad0kp9ircvfs/runs/3e04ce28-2a0e-4edf-894a-59d73b536344). Langfuse project access is required to view these results.
Human annotation, prompt management, online evaluation, feedback, and judge
calibration remain later exercises.

Local results are stored in results/ and ignored by Git, along with .env and
.venv. requirements.txt lists direct dependencies; requirements.lock records the
tested environment. This folder is the Git repository root. Article drafts and illustrations are kept locally and excluded from the repository.
