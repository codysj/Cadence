# Pre-Op Triage Take-Home

`triage_submission(...)` in `core.py` takes one patient submission package and returns a
decision (`READY | NEEDS_FOLLOW_UP | NOT_CLEARED`), the issues with their categories, and
an evidence checklist a pre-op nurse can verify.

A model reads the free-text documents and returns quoted facts. Plain code checks every
quote against its document, applies the policy and writes the output. The approach,
assumptions, results and limits are in [WRITEUP.md](WRITEUP.md).

## Setup

1. Install [`uv`](https://docs.astral.sh/uv/) and confirm it runs:

```bash
uv --version
```

2. Copy `.env.example` to `.env` and put your key in it.

| Provider | Key in `.env` | `MODEL` | Status |
|---|---|---|---|
| OpenRouter | `OPENROUTER_API_KEY` | `openai/gpt-6-luna` | The path this was run and tested on |
| OpenAI | `OPENAI_API_KEY` | an OpenAI model id | Covered by mocked tests only; not run live |

If both keys are set, OpenRouter is used. A missing or rejected key, or a model id the
provider does not know, stops the run with a message. It does not produce 50 outputs.

## Run

Run these from this folder. They are the same commands the Makefile wraps, written out
so they also work where `make` is not installed (Windows PowerShell included).

1. Triage all 50 sample cases. This writes `data/baseline_outputs.jsonl` and, because
   `.env` sets `TRIAGE_FACTS_DIR`, saves what the reader returned for each case in
   `data/reader_facts/`.

```bash
uv run --env-file .env run_baseline.py --model openai/gpt-6-luna
```

2. Score the run against the labels. This uses the harness's own metric functions from
   `run_evals.py` and needs no API key. It also compares the reader's facts one by one
   with the reference facts in `data/gold_facts/`.

```bash
uv run offline_eval.py score --facts data/reader_facts
```

3. Browse the results case by case in the report viewer. Press `q` to quit.

```bash
uv run view_report.py --report data/offline_report.json
```

4. Run the harness's determinism check: the same case, ten fresh model calls.

```bash
uv run --env-file .env run_evals.py --determinism --model openai/gpt-6-luna --report data/determinism_report.json
```

5. Run the unit tests. They need no key and make no model calls.

```bash
uv run --with "openai>=2.0.0" --with "pydantic>=2.8.0" --with "pytest>=8.0.0" python -m pytest tests
```

With `make`: `make baseline`, `make offline`, `make report REPORT=data/offline_report.json`,
`make determinism`, `make test`. The Makefile was edited on a machine without `make`, so
those targets are untested; the commands above are the ones that were run.

### Scoring through OpenAI's Evals API

`make evals` (`run_evals.py` without `--determinism`) scores through OpenAI's Evals API,
as in the starter. It needs an `OPENAI_API_KEY` and does not work with an OpenRouter key.
It was not run for this submission. Step 2 above applies the same three metrics locally.

## Re-score without a key

The readings behind the numbers in the write-up are committed in `data/reader_facts/`,
so steps 2 and 3 work straight after cloning, with no key and no model call.

## What is where

| File | What it does |
|---|---|
| `core.py` | Entry point: validate the package, read the documents, apply the rules |
| `models.py` | The input package, the checklist rows and the output |
| `facts.py` | What the reader is shown and the facts it must return |
| `reader.py` | The one model call, its prompt, one retry, and failing closed |
| `rules.py` | The policy as pure code: seven checks, decision, issues, explanation |
| `offline_eval.py` | `view`, `check` and `score` for saved facts; no model calls |
| `data/gold_facts/` | Reference facts for the 50 sample cases |
| `data/reader_facts/` | What the reader returned in the run reported in the write-up |
| `run_baseline.py`, `run_evals.py`, `view_report.py` | The starter's harness |

`run_baseline.py` has one change from the starter: it no longer validates every record
before starting, so one malformed record fails closed for that case instead of stopping
the batch. `run_evals.py` and `view_report.py` are unchanged.

## Outputs

- Triage outputs: `data/baseline_outputs.jsonl`
- Saved readings: `data/reader_facts/`
- Offline score report: `data/offline_report.json`
- Determinism report: `data/determinism_report.json`
- Evals API report, if you run `make evals`: `data/eval_report.json`
