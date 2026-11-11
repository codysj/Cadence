INPUT ?= data/patients_sample_50.jsonl
OUTPUT ?= data/baseline_outputs.jsonl
REPORT ?= data/eval_report.json
DETERMINISM_REPORT ?= data/determinism_report.json
# An OpenRouter model id, the provider this was tested on. With an OpenAI key, pass an
# OpenAI model id instead: make baseline MODEL=<id>
MODEL ?= openai/gpt-6-luna
FACTS ?= data/reader_facts
OFFLINE_REPORT ?= data/offline_report.json

# Load API keys from .env when it exists (see .env.example). uv stops with an error
# if it is told to load a file that is not there, so the flag is added only then.
ENV_FILE := $(if $(wildcard .env),--env-file .env,)

.PHONY: baseline evals offline determinism score report test all clean

baseline:
	uv run $(ENV_FILE) run_baseline.py \
		--input $(INPUT) \
		--output $(OUTPUT) \
		--model $(MODEL)

evals:
	uv run $(ENV_FILE) run_evals.py \
		--input $(INPUT) \
		--outputs $(OUTPUT) \
		--report $(REPORT)

# Score the readings saved by `make baseline` with the harness's own metric code.
# Needs no API key. `make evals` does the same through OpenAI's Evals API and needs one.
offline:
	uv run offline_eval.py score --facts $(FACTS) --report $(OFFLINE_REPORT)

determinism:
	uv run $(ENV_FILE) run_evals.py \
		--determinism \
		--input $(INPUT) \
		--model $(MODEL) \
		--report $(DETERMINISM_REPORT)

score:
	@python3 -c 'import json; r=json.load(open("$(REPORT)")); s=(r.get("primary_score",{}) or {}).get("value_pct"); print(s if s is not None else r.get("local_metrics_summary",{}).get("aggregate_local_score_pct", 0.0))'

report:
	uv run view_report.py --report $(REPORT)

test:
	uv run \
		--with 'openai>=2.0.0' \
		--with 'pydantic>=2.8.0' \
		--with 'pytest>=8.0.0' \
		python -m pytest tests

all: baseline evals determinism score

clean:
	rm -f data/baseline_outputs.jsonl \
		data/eval_report.json \
		data/determinism_report.json
