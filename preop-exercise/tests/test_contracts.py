"""Stage 1 checks: the input contract, the vocabularies and the reader's contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from models import CHECK_ORDER, PatientSubmission
from facts import Facts
from rules import LAB_STATUSES, RISK_LEVELS, VITAL_TYPES, anticoagulants_named, lab_test

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "patients_sample_50.jsonl"


@pytest.fixture(scope="module")
def submissions() -> list[dict]:
    with SAMPLE.open(encoding="utf-8") as handle:
        return [json.loads(line)["submission"] for line in handle if line.strip()]


# --- input contract ----------------------------------------------------------


def test_all_sample_submissions_pass_strict_validation(submissions: list[dict]) -> None:
    assert len(submissions) == 50
    for submission in submissions:
        PatientSubmission.model_validate(submission)


def test_renamed_top_level_field_is_an_error_not_an_empty_list(submissions: list[dict]) -> None:
    drifted = copy.deepcopy(submissions[0])
    drifted["lab_results"] = drifted.pop("labs")

    with pytest.raises(ValidationError):
        PatientSubmission.model_validate(drifted)


def test_unknown_nested_field_is_an_error(submissions: list[dict]) -> None:
    drifted = copy.deepcopy(submissions[0])
    drifted["labs"][0]["result_status"] = drifted["labs"][0].pop("status")

    with pytest.raises(ValidationError):
        PatientSubmission.model_validate(drifted)


def test_unfamiliar_values_are_accepted_for_the_rules_to_classify(submissions: list[dict]) -> None:
    drifted = copy.deepcopy(submissions[0])
    drifted["labs"][0]["code"] = "HEMOGRAM"
    drifted["labs"][0]["status"] = "preliminary"
    drifted["procedure"]["procedure_risk"] = "INTERMEDIATE"

    parsed = PatientSubmission.model_validate(drifted)

    assert lab_test(parsed.labs[0].code) is None
    assert parsed.labs[0].status not in LAB_STATUSES
    assert parsed.procedure.procedure_risk not in RISK_LEVELS


# --- vocabularies ------------------------------------------------------------


def test_every_value_in_the_sample_is_recognised(submissions: list[dict]) -> None:
    """Drift detection starts from zero: nothing in the 50 cases is unrecognised."""

    for s in submissions:
        assert s["procedure"]["procedure_risk"] in (*RISK_LEVELS, None)
        for lab in s["labs"]:
            assert lab_test(lab["code"]) is not None, lab["code"]
            assert lab["status"] in LAB_STATUSES, lab["status"]
        for vital in s["vitals"]:
            assert vital["type"] in VITAL_TYPES, vital["type"]


def test_lab_codes_map_to_the_test_they_are() -> None:
    assert lab_test("CBC") == lab_test("LAB-CBC") == lab_test(" lab-cbc ") == "CBC"
    assert lab_test("CMP") == lab_test("LAB-CMP") == "CMP"
    assert lab_test("BMP") == lab_test("LAB-BMP") == "OTHER"  # a BMP is not a CMP
    assert lab_test("HEMOGRAM") is None
    assert lab_test(None) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("warfarin (Coumadin) 5 mg PO daily", {"warfarin"}),
        ("Coumadin 5 mg PO daily (INR goal 2-3)", {"warfarin"}),
        ("Xarelto 20 mg PO daily with evening meal", {"rivaroxaban"}),
        ("Eliquis 5 mg PO BID", {"apixaban"}),
        ("dabigatran (Pradaxa) 150 mg PO BID", {"dabigatran"}),
        ("edoxaban (Savaysa) 60 mg PO daily", {"edoxaban"}),
        ("Hold warfarin 5 days; bridge with enoxaparin", {"warfarin", "enoxaparin"}),
        ("ASA 81 mg PO daily", set()),
        ("Aspirin EC 81 mg PO daily", set()),
        ("clopidogrel 75 mg PO daily", set()),
        ("atorvastatin 40 mg PO nightly", set()),
        (None, set()),
    ],
)
def test_anticoagulant_names(text: str | None, expected: set[str]) -> None:
    assert anticoagulants_named(text) == expected


# --- reader contract ---------------------------------------------------------


def _objects(schema: object):
    if isinstance(schema, dict):
        if "properties" in schema:
            yield schema
        for value in schema.values():
            yield from _objects(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _objects(value)


def test_facts_schema_meets_strict_structured_output_rules() -> None:
    """Strict mode needs every property required and no additional properties, at every level."""

    objects = list(_objects(Facts.model_json_schema()))
    assert len(objects) >= 7
    for obj in objects:
        assert obj.get("additionalProperties") is False, obj.get("title")
        assert set(obj.get("required", [])) == set(obj["properties"]), obj.get("title")


def test_facts_reject_missing_and_extra_fields() -> None:
    empty = {
        "hp_candidates": [],
        "consents": [],
        "note_blood_pressures": [],
        "note_temperatures": [],
        "anticoagulant_mentions": [],
        "medication_list_anticoagulants": [],
        "anticoagulation_plans": [],
    }
    Facts.model_validate(empty)

    with pytest.raises(ValidationError):
        Facts.model_validate({k: v for k, v in empty.items() if k != "consents"})
    with pytest.raises(ValidationError):
        Facts.model_validate({**empty, "decision": "READY"})


def test_checklist_has_seven_rows_in_a_fixed_order() -> None:
    assert CHECK_ORDER == (
        "hp", "consent", "cbc", "cmp", "anticoagulation", "blood_pressure", "temperature"
    )
