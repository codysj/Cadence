# Pre-Op Triage Take-Home

`triage_submission(...)` in `preop-exercise/core.py` takes one patient submission package and returns a
decision (`READY | NEEDS_FOLLOW_UP | NOT_CLEARED`), the issues with their categories, and
an evidence checklist a pre-op nurse can verify.

A model reads the free-text documents and returns quoted facts. Plain code checks every
quote against its document, applies the policy and writes the output. The approach,
assumptions, results and limits are in [WRITEUP.md](preop-exercise/WRITEUP.md).

## Setup

Install [`uv`](https://docs.astral.sh/uv/) and confirm it runs:

```bash
uv --version
```

Run everything below from the `preop-exercise/` folder:

```bash
cd preop-exercise
```

The commands are the ones the Makefile wraps, written out so they also work where `make`
is not installed (Windows PowerShell included).

## Run without a key: re-score the recorded run

The readings behind the numbers in the write-up are committed in `data/recorded_run/`.
Scoring them needs no key and makes no model call:

```bash
uv run offline_eval.py score --facts data/recorded_run
```

This applies the harness's own metric functions from `run_evals.py`, and compares the
reader's facts one by one with the reference facts in `data/gold_facts/`. Then browse the
results case by case in the report viewer (press `q` to quit):

```bash
uv run view_report.py --report data/offline_report.json
```

The unit tests need no key either:

```bash
uv run --with "openai>=2.0.0" --with "pydantic>=2.8.0" --with "pytest>=8.0.0" python -m pytest tests
```

## Run with a key: a fresh run

Copy `.env.example` to `.env` and put your key in it. The model is not set in `.env`; it
is passed with `--model`, and the id must be one your provider knows.

| Provider | Key in `.env` | `--model` | Status |
|---|---|---|---|
| OpenRouter | `OPENROUTER_API_KEY` | `openai/gpt-6-luna` | The path this was run and tested on |
| OpenAI | `OPENAI_API_KEY` | `gpt-6-luna` | Covered by mocked tests only; not run live |

If both keys are set, OpenRouter is used. OpenAI rejects the OpenRouter id
`openai/gpt-6-luna`, so with an OpenAI key use `gpt-6-luna`. A missing or rejected key,
a model id the provider does not know, or an account with no credit left stops the run
with a message. It does not produce 50 outputs.

1. Triage all 50 sample cases. This writes `data/baseline_outputs.jsonl` and, because
   `.env` sets `TRIAGE_FACTS_DIR`, saves what the reader returned for each case in
   `data/reader_facts/`. That folder is ignored by git, so your runs never touch the
   recorded run.

With an OpenRouter key:

```bash
uv run --env-file .env run_baseline.py --model openai/gpt-6-luna
```

With an OpenAI key:

```bash
uv run --env-file .env run_baseline.py --model gpt-6-luna
```

2. Score your run against the labels, and open it in the viewer as above:

```bash
uv run offline_eval.py score --facts data/reader_facts
```

3. Run the harness's determinism check: the same case, ten fresh model calls. Each call
   saves its reading over that case's file in `data/reader_facts/`, so score first. Use
   the same `--model` as in step 1.

```bash
uv run --env-file .env run_evals.py --determinism --model openai/gpt-6-luna --report data/determinism_report.json
```

With `make`: `make baseline`, `make offline`, `make report REPORT=data/offline_report.json`,
`make determinism`, `make test`. `MODEL` defaults to `openai/gpt-6-luna`; with an OpenAI
key add `MODEL=gpt-6-luna`. The Makefile was edited on a machine without `make`, so those
targets are untested; the commands above are the ones that were run.

### Scoring through OpenAI's Evals API

`make evals` (`run_evals.py` without `--determinism`) scores through OpenAI's Evals API,
as in the starter. It needs an `OPENAI_API_KEY` and does not work with an OpenRouter key.
It was not run for this submission. `offline_eval.py score` applies the same three
metrics locally.

## What is where

Paths are relative to `preop-exercise/`.

| File | What it does |
|---|---|
| `core.py` | Entry point: validate the package, read the documents, apply the rules |
| `models.py` | The input package, the checklist rows and the output |
| `facts.py` | What the reader is shown and the facts it must return |
| `reader.py` | The one model call, its prompt, one retry, and failing closed |
| `rules.py` | The policy as pure code: seven checks, decision, issues, explanation |
| `offline_eval.py` | `view`, `check` and `score` for saved facts; no model calls |
| `data/gold_facts/` | Reference facts for the 50 sample cases |
| `data/recorded_run/` | What the reader returned in the run reported in the write-up, and that run's determinism report |
| `data/reader_facts/` | What the reader returns in your own runs; ignored by git |
| `run_baseline.py`, `run_evals.py`, `view_report.py` | The starter's harness |
| `../docs/session-export/` | The exported coding-agent session, with one email address redacted |

`run_baseline.py` has one change from the starter: it no longer validates every record
before starting, so one malformed record fails closed for that case instead of stopping
the batch. `run_evals.py` and `view_report.py` are unchanged.

## Outputs

- Triage outputs: `data/baseline_outputs.jsonl`
- Saved readings from your runs: `data/reader_facts/` (ignored by git)
- Offline score report: `data/offline_report.json`
- Determinism report: `data/determinism_report.json`
- Evals API report, if you run `make evals`: `data/eval_report.json`
