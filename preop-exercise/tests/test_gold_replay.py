"""Stage 3 check: given a careful reading of each case, the rules reproduce the label.

data/gold_facts/ holds reference facts for all 50 sample cases: what a careful, literal
reader finds in each case's documents. Each must leave nothing open: every quote is
found in the document it cites and nothing written in a document goes unreported.

How independent they are: 49 were written by subagents that never saw labels, following
a reading guide written after studying the labels. That guide settles the judgement
calls (what counts as a concrete or a deferred instruction, which titles are consents)
with examples close to wording from labelled cases. case_00016 was written knowing its
label. So agreement with the labels here was partly built in by the guide.

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


def test_offline_scores_are_the_harness_own_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """The offline scorer does not re-implement scoring: it calls run_evals. Swapping the
    harness's metric function for a marker changes what the offline scorer reports."""

    import run_evals

    records = score_rows(ROWS, GOLD)
    assert [r["record_index"] for r in records] == list(range(50))
    for record, row in zip(records, ROWS):
        direct = run_evals._local_metrics_for_row(run_evals.CaseLabel.model_validate(row["label"]), record["parsed_output"])
        assert record["metrics"] == direct["metrics"]
        assert record["aggregate_local_score"] == direct["aggregate_local_score"]
    assert run_evals._summarize_local_rows(records)["aggregate_local_score_pct"] == 100.0

    marker = {"parsed_output": None, "parse_error": "marker", "metrics": dict.fromkeys(run_evals.METRIC_WEIGHTS, False),
              "aggregate_local_score": -1.0, "expected_categories": [], "actual_categories": [], "actual_decision": "INVALID"}
    monkeypatch.setattr(run_evals, "_local_metrics_for_row", lambda **kwargs: marker)
    assert score_rows(ROWS[:1], GOLD)[0]["aggregate_local_score"] == -1.0


def test_the_harness_fails_a_case_for_one_extra_category() -> None:
    """Why a false UNKNOWN is expensive: the category set must match exactly."""

    from offline_eval import score_outputs

    row = next(r for r in ROWS if r["label"]["decision"] == "NEEDS_FOLLOW_UP" and len(r["label"]["categories"]) == 1)
    output = evaluate(PatientSubmission.model_validate(row["submission"]), load_facts(GOLD, row))
    extra = output.model_copy(update={"issues": [*output.issues, output.issues[0].model_copy(update={"category": "MISSING_REQUIRED_DATA"})]})
    if row["label"]["categories"] == ["MISSING_REQUIRED_DATA"]:
        extra = output.model_copy(update={"issues": [*output.issues, output.issues[0].model_copy(update={"category": "REQUIRED_TESTING"})]})

    metrics = score_outputs([row], [(extra, None)])[0]["metrics"]
    assert metrics == {"json_schema_valid": True, "decision_match_oracle": True, "issue_categories_match_oracle": False}


def test_missing_facts_file_is_scored_as_a_failure_not_a_crash(tmp_path) -> None:
    ready = next(r for r in ROWS if r["label"]["decision"] == "READY")
    record = score_rows([ready], tmp_path)[0]

    assert record["parsed_output"] is None
    assert record["baseline_error"].startswith("FileNotFoundError")
    assert record["metrics"]["json_schema_valid"] is False
    assert record["metrics"]["decision_match_oracle"] is False
    # A property of the harness, not of this code: with no output at all, a READY label's
    # empty category list "matches", so a crashed READY case still earns one metric in three.
    # My own re-implementation scored this 0; calling the harness shows what it really does.
    assert record["metrics"]["issue_categories_match_oracle"] is True
    assert round(record["aggregate_local_score"], 1) == 33.3


def test_facts_are_compared_one_by_one_with_the_reference() -> None:
    """Stage 4: a reading is scored on what it found, missed, added and judged differently."""

    from offline_eval import compare_facts

    row = next(r for r in ROWS if r["case_id"] == "case_00016")
    reference = load_facts(GOLD, row)
    assert all(c["found"] == c["reference"] and not (c["missed"] or c["extra"] or c["differ"]) for c in compare_facts(reference, reference).values())

    changed = reference.model_dump()
    dropped = changed["note_blood_pressures"].pop()  # the 08:30 recheck
    changed["note_blood_pressures"][0]["time"] = "11:11"
    changed["consents"][0]["patient_signed"] = "unclear"
    changed["note_temperatures"].append({"doc_id": dropped["doc_id"], "value": 39.0, "unit": "C", "time": None, "quote": "x"})
    result = compare_facts(reference, type(reference).model_validate(changed))

    pressures = result["blood pressure readings"]
    assert (pressures["reference"], pressures["found"]) == (3, 2)
    assert pressures["missed"] == [(dropped["doc_id"], 137, 78)]
    assert len(pressures["differ"]) == 1 and pressures["differ"][0][2] == ("11:11",)
    assert result["consents"]["differ"][0][1:] == (("yes", "yes"), ("unclear", "yes"))
    assert result["temperature readings"]["extra"] == [(dropped["doc_id"], 102.2)]
    assert result["H&P documents"]["found"] == 2 and not result["H&P documents"]["differ"]
