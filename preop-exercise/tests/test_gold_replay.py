"""Stage 3 check: given a careful reading of each case, the rules reproduce the label.

data/gold_facts/ holds reference facts for all 50 sample cases: what a careful, literal
reader finds in each case's documents. They were written from the documents alone,
without sight of the labels, and each must leave nothing open: every quote is found in
the document it cites and nothing written in a document goes unreported.

This tests the code side only. It says nothing about how well a model reads, and it
exercises the paths where facts are complete; the doubt paths (unverified quotes,
skipped documents, unordered sources) are covered by the synthetic tests in test_rules.py.
The rules were also built knowing these 50 labels, so this is a consistency check, not
evidence of how the system does on unseen cases.
"""

from __future__ import annotations

import json

import pytest

from facts import submission_hash
from models import PatientSubmission
from offline_eval import GOLD, load_facts, load_rows, open_items, score_rows
from rules import evaluate

ROWS = load_rows()
CASE_IDS = [row["case_id"] for row in ROWS]

# Cases where the rules, given the reference facts, are expected to differ from the
# label, with the reason. Empty: there are none.
KNOWN_DISAGREEMENTS: dict[str, str] = {}


def test_there_are_reference_facts_for_all_fifty_cases() -> None:
    assert len(ROWS) == 50
    assert sorted(path.stem for path in GOLD.glob("*.json")) == sorted(CASE_IDS)


@pytest.mark.parametrize("row", ROWS, ids=CASE_IDS)
def test_reference_facts_belong_to_this_submission(row: dict) -> None:
    saved = json.loads((GOLD / f"{row['case_id']}.json").read_text(encoding="utf-8"))
    assert saved["case_id"] == row["case_id"]
    assert saved["submission_sha256"] == submission_hash(row["submission"])


@pytest.mark.parametrize("row", ROWS, ids=CASE_IDS)
def test_reference_facts_leave_nothing_open(row: dict) -> None:
    submission = PatientSubmission.model_validate(row["submission"])
    assert open_items(submission, load_facts(GOLD, row)) == []


@pytest.mark.parametrize("row", ROWS, ids=CASE_IDS)
def test_rules_reproduce_the_label_from_reference_facts(row: dict) -> None:
    if row["case_id"] in KNOWN_DISAGREEMENTS:
        pytest.skip(KNOWN_DISAGREEMENTS[row["case_id"]])
    output = evaluate(PatientSubmission.model_validate(row["submission"]), load_facts(GOLD, row))

    assert output.decision == row["label"]["decision"]
    assert sorted({issue.category for issue in output.issues}) == sorted(set(row["label"]["categories"]))


def test_offline_scorer_agrees_with_the_harness_metrics() -> None:
    records = score_rows(ROWS, GOLD)

    assert [r["record_index"] for r in records] == list(range(50))
    for metric in ("json_schema_valid", "decision_match_oracle", "issue_categories_match_oracle"):
        assert sum(r["metrics"][metric] for r in records) == 50 - len(KNOWN_DISAGREEMENTS)


def test_missing_facts_file_is_scored_as_a_failure_not_a_crash(tmp_path) -> None:
    record = score_rows(ROWS[:1], tmp_path)[0]

    assert record["parsed_output"] is None
    assert record["baseline_error"].startswith("FileNotFoundError")
    assert record["aggregate_local_score"] == 0.0
