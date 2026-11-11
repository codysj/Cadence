"""A record that breaks the input contract fails closed for that case only."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import triage_submission

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "patients_sample_50.jsonl"


@pytest.fixture
def rows() -> list[dict]:
    with SAMPLE.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle][:2]


@pytest.fixture
def no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any attempt to build a client fails the test."""

    def refuse(**kwargs: object) -> None:
        raise AssertionError("the model must not be called for an unreadable submission")

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=refuse))


def test_unknown_field_gives_a_fail_closed_result_without_a_model_call(rows: list[dict], no_model: None) -> None:
    """Bug: strict validation raised, and the batch runner validated every record up front,
    so one unknown field in one record left all 50 cases without an output."""

    drifted = copy.deepcopy(rows[0]["submission"])
    drifted["labs"][0]["result_status"] = drifted["labs"][0].pop("status")

    output = triage_submission(drifted, model="unused")

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert [issue.category for issue in output.issues] == ["MISSING_REQUIRED_DATA"]
    assert "labs.0.result_status" in output.issues[0].evidence
    assert all(check.status == "UNKNOWN" for check in output.checks) and len(output.checks) == 7
    assert output.explanation.startswith("NEEDS_FOLLOW_UP: SYSTEM: the submission could not be read")
    assert output.system_failure == "invalid_submission"


def test_the_validation_message_names_fields_not_patient_values(rows: list[dict], no_model: None) -> None:
    drifted = copy.deepcopy(rows[0]["submission"])
    drifted["patient"]["nickname"] = "Zebediah-Unique-Value"

    output = triage_submission(drifted, model="unused")

    assert "patient.nickname" in output.issues[0].evidence
    assert "Zebediah-Unique-Value" not in output.model_dump_json()


def test_one_bad_record_does_not_stop_the_batch_from_loading(rows: list[dict], tmp_path: Path) -> None:
    from run_baseline import load_cases

    rows[0]["submission"]["lab_results"] = rows[0]["submission"].pop("labs")
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    cases = load_cases(path)

    assert [case.case_id for case in cases] == [rows[0]["case_id"], rows[1]["case_id"]]
    assert "lab_results" in cases[0].submission  # passed through raw; triage_submission judges it
