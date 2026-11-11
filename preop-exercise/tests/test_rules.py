"""Stage 2 checks: the rules on synthetic boundaries, and the safety property.

Every case here is built by hand, so these tests cannot overfit the 50 sample cases.
Each starts from one READY patient and changes a single thing.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from models import CHECK_ORDER, PatientSubmission
from facts import Facts
from rules import _norm, evaluate

PROCEDURE = date(2026, 3, 20)


def ago(days: int, clock: str | None = None) -> str:
    day = (PROCEDURE - timedelta(days=days)).isoformat()
    return f"{day}T{clock}:00Z" if clock else day


def ready_case() -> tuple[dict, dict]:
    """A LOW-risk patient with everything in order."""

    submission = {
        "procedure": {
            "case_id": "synthetic",
            "procedure_type": "knee arthroscopy",
            "procedure_risk": "LOW",
            "procedure_date": PROCEDURE.isoformat(),
        },
        "vitals": [
            {"type": "blood_pressure", "systolic": 120, "diastolic": 80, "date": ago(2, "09:00"), "source": "clinic"},
            {"type": "temperature", "value_f": 98.6, "date": ago(2, "09:05"), "source": "clinic"},
        ],
        "labs": [{"id": "lab-cbc-1", "code": "CBC", "effective_at": ago(10, "08:00"), "status": "final"}],
        "medications": [],
        "documents": [
            {"doc_id": "hp1", "type": "H&P", "date": ago(10),
             "text": "PREOPERATIVE HISTORY AND PHYSICAL\nPlanned procedure: knee arthroscopy"},
            {"doc_id": "consent1", "type": "Surgical Consent", "date": ago(5),
             "text": "Procedure: knee arthroscopy\nSigned by patient: Pat Example\n"
                     "Physician attestation: Electronically signed by Dr. Example"},
            {"doc_id": "note1", "type": "Nursing Note", "date": ago(1), "text": "PRE-OP NURSING ASSESSMENT"},
        ],
    }
    facts = {
        "hp_candidates": [{"doc_id": "hp1", "is_preop_evaluation": True, "quote": "PREOPERATIVE HISTORY AND PHYSICAL"}],
        "consents": [{
            "doc_id": "consent1",
            "patient_signed": "yes", "signature_quote": "Signed by patient: Pat Example",
            "matches_scheduled_procedure": "yes", "procedure_quote": "Procedure: knee arthroscopy",
        }],
        "note_blood_pressures": [],
        "note_temperatures": [],
        "anticoagulant_mentions": [],
        "medication_list_anticoagulants": [],
        "anticoagulation_plans": [],
    }
    return submission, facts


def run(submission: dict, facts: dict):
    return evaluate(PatientSubmission.model_validate(submission), Facts.model_validate(facts))


def statuses(output) -> dict[str, str]:
    return {check.id: check.status for check in output.checks}


def categories(output) -> set[str]:
    return {issue.category for issue in output.issues}


def doc(submission: dict, doc_id: str) -> dict:
    return next(d for d in submission["documents"] if d["doc_id"] == doc_id)


# --- mutations, reused by the single tests and by the safety sweep --------------


def make_high_risk(s, f, cbc_days=5, cmp_days=5):
    s["procedure"]["procedure_risk"] = "HIGH"
    s["labs"] = [{"id": "lab-cbc-1", "code": "CBC", "effective_at": ago(cbc_days, "08:00"), "status": "final"}]
    if cmp_days is not None:
        s["labs"].append({"id": "lab-cmp-1", "code": "LAB-CMP", "effective_at": ago(cmp_days, "08:00"), "status": "final"})


def note_bp(s, f, systolic, diastolic, time="08:00", written=True, reported=True):
    line = f"{time or ''} BP {systolic}/{diastolic} mmHg".strip()
    if written:
        doc(s, "note1")["text"] += "\n" + line
    if reported:
        f["note_blood_pressures"].append(
            {"doc_id": "note1", "systolic": systolic, "diastolic": diastolic, "time": time, "quote": line}
        )


def note_temp(s, f, value, unit, time="08:00"):
    line = f"{time} Temp {value} {unit}"
    doc(s, "note1")["text"] += "\n" + line
    f["note_temperatures"].append({"doc_id": "note1", "value": value, "unit": unit, "time": time, "quote": line})


def on_drug(s, f, name="warfarin 5 mg PO daily", active=True):
    s["medications"].append({"name": name, "active": active})


def plan(s, f, drug="warfarin", preop="concrete", postop="concrete", mention="current", verified=True):
    takes = f"Patient takes {drug} daily."
    hold = f"Hold {drug} 5 days before surgery."
    resume = f"Resume {drug} 24 hours after surgery." if postop == "concrete" else "Post-op dosing to be determined."
    s["documents"].append({"doc_id": "plan1", "type": "Anticoagulation Plan", "date": ago(4),
                           "text": "\n".join(["PERIOPERATIVE PLAN", takes, hold, resume])})
    f["anticoagulant_mentions"].append(
        {"doc_id": "plan1", "drug_as_written": drug, "generic_name": drug, "status": mention, "quote": takes}
    )
    f["anticoagulation_plans"].append({
        "doc_id": "plan1", "generic_name": drug,
        "preop": preop, "preop_quote": hold if verified else "Hold it for a while.",
        "postop": postop, "postop_quote": resume,
    })


# --- the starting point ---------------------------------------------------------


def test_the_starting_case_is_ready_with_seven_rows() -> None:
    output = run(*ready_case())

    assert output.decision == "READY"
    assert output.issues == []
    assert tuple(check.id for check in output.checks) == CHECK_ORDER
    assert statuses(output) == {
        "hp": "PASS", "consent": "PASS", "cbc": "PASS", "cmp": "NOT_APPLICABLE",
        "anticoagulation": "NOT_APPLICABLE", "blood_pressure": "PASS", "temperature": "PASS",
    }


def test_the_same_inputs_give_the_same_output() -> None:
    assert run(*ready_case()).model_dump_json() == run(*ready_case()).model_dump_json()


def test_output_is_accepted_by_the_scoring_harness() -> None:
    from run_evals import ScoredOutput

    s, f = ready_case()
    s["labs"] = []
    scored = ScoredOutput.model_validate(run(s, f).model_dump())
    assert scored.decision == "NEEDS_FOLLOW_UP"
    assert [issue.category for issue in scored.issues] == ["REQUIRED_TESTING"]


# --- Rule 1: documentation ------------------------------------------------------


@pytest.mark.parametrize(("days", "expected"), [(0, "PASS"), (30, "PASS"), (31, "FAIL")])
def test_hp_window_is_thirty_calendar_days_inclusive(days: int, expected: str) -> None:
    s, f = ready_case()
    doc(s, "hp1")["date"] = ago(days)
    output = run(s, f)

    assert statuses(output)["hp"] == expected
    if expected == "FAIL":
        assert output.decision == "NEEDS_FOLLOW_UP"
        assert categories(output) == {"REQUIRED_DOCUMENTATION"}


def test_hp_at_the_limit_says_so() -> None:
    s, f = ready_case()
    doc(s, "hp1")["date"] = ago(30)
    assert "at the limit" in run(s, f).checks[0].finding


def test_no_hp_fails_and_a_wellness_visit_does_not_count() -> None:
    s, f = ready_case()
    f["hp_candidates"][0]["is_preop_evaluation"] = False
    output = run(s, f)

    assert statuses(output)["hp"] == "FAIL"
    assert output.checks[0].rejected[0].reason == "not the pre-op evaluation"
    assert categories(output) == {"REQUIRED_DOCUMENTATION"}


def test_unverified_hp_reading_is_unknown_not_missing() -> None:
    s, f = ready_case()
    f["hp_candidates"][0]["quote"] = "Complete pre-operative evaluation performed"
    output = run(s, f)

    assert statuses(output)["hp"] == "UNKNOWN"
    assert categories(output) == {"MISSING_REQUIRED_DATA"}
    assert "could not verify" in output.checks[0].finding


def unsigned(s, f, status="no"):
    """The consent's signature line reads "pending", and the reader quotes it."""

    doc(s, "consent1")["text"] = (
        "Procedure: knee arthroscopy\nPatient signature: pending\n"
        "Physician attestation: Electronically signed by Dr. Example"
    )
    f["consents"][0].update(patient_signed=status, signature_quote="Patient signature: pending")


@pytest.mark.parametrize("status", ["no", "unclear"])
def test_a_consent_quoted_as_unsigned_fails(status: str) -> None:
    s, f = ready_case()
    unsigned(s, f, status)
    output = run(s, f)

    assert statuses(output)["consent"] == "FAIL"
    assert categories(output) == {"REQUIRED_DOCUMENTATION"}
    assert output.checks[1].evidence[0].quote == "Patient signature: pending"


@pytest.mark.parametrize(
    ("change", "expected", "category"),
    [
        ({"matches_scheduled_procedure": "no"}, "FAIL", "REQUIRED_DOCUMENTATION"),
        ({"signature_quote": "Patient signed in clinic"}, "UNKNOWN", "MISSING_REQUIRED_DATA"),
        ({"signature_quote": None}, "UNKNOWN", "MISSING_REQUIRED_DATA"),
        # Setting a consent aside is a claim too: with nothing quoted it is not established.
        ({"patient_signed": "no", "signature_quote": None}, "UNKNOWN", "MISSING_REQUIRED_DATA"),
        ({"patient_signed": "unclear", "signature_quote": None}, "UNKNOWN", "MISSING_REQUIRED_DATA"),
        ({"matches_scheduled_procedure": "no", "procedure_quote": None}, "UNKNOWN", "MISSING_REQUIRED_DATA"),
    ],
)
def test_consent(change: dict, expected: str, category: str) -> None:
    s, f = ready_case()
    f["consents"][0].update(change)
    output = run(s, f)

    assert statuses(output)["consent"] == expected
    assert categories(output) == {category}


def test_an_hp_set_aside_without_a_verified_quote_is_unknown_not_missing() -> None:
    """Otherwise "no pre-op H&P on file" would rest on a reading nobody can check."""

    s, f = ready_case()
    f["hp_candidates"][0].update(is_preop_evaluation=False, quote="Annual wellness visit")  # not in the document
    output = run(s, f)

    assert statuses(output)["hp"] == "UNKNOWN"
    assert categories(output) == {"MISSING_REQUIRED_DATA"}
    assert output.checks[0].rejected == []


def test_no_consent_document_fails() -> None:
    s, f = ready_case()
    s["documents"] = [d for d in s["documents"] if d["doc_id"] != "consent1"]
    f["consents"] = []
    output = run(s, f)

    assert statuses(output)["consent"] == "FAIL"
    assert categories(output) == {"REQUIRED_DOCUMENTATION"}


@pytest.mark.parametrize(
    ("doc_id", "fact_list", "check_id", "title"),
    [("consent1", "consents", "consent", "Surgical Consent"), ("hp1", "hp_candidates", "hp", "H&P")],
)
def test_a_titled_document_the_reader_skipped_is_unknown_not_missing(doc_id, fact_list, check_id, title) -> None:
    """Without this the nurse would read "no consent on file" about a chart that has one."""

    s, f = ready_case()
    f[fact_list] = []
    output = run(s, f)

    assert statuses(output)[check_id] == "UNKNOWN"
    assert categories(output) == {"MISSING_REQUIRED_DATA"}
    finding = next(k.finding for k in output.checks if k.id == check_id)
    assert title in finding and "the reader reported nothing from it" in finding


@pytest.mark.parametrize(
    "title",
    ["History and Pyhsical", "Pre-op H and P", "Preoperative History & Physical (H&P)", "Clinic H&P (external)"],
)
def test_hp_titles_are_recognised_whatever_the_spelling(title: str) -> None:
    s, f = ready_case()
    doc(s, "hp1")["type"] = title
    f["hp_candidates"] = []
    assert statuses(run(s, f))["hp"] == "UNKNOWN"


def test_no_hp_document_at_all_fails() -> None:
    s, f = ready_case()
    s["documents"] = [d for d in s["documents"] if d["doc_id"] != "hp1"]
    f["hp_candidates"] = []
    output = run(s, f)

    assert statuses(output)["hp"] == "FAIL"
    assert categories(output) == {"REQUIRED_DOCUMENTATION"}


def test_a_signed_consent_stands_alongside_an_unsigned_one() -> None:
    s, f = ready_case()
    s["documents"].append({"doc_id": "consent0", "type": "Consent", "date": ago(9),
                           "text": "Procedure: knee arthroscopy\nPatient signature: pending"})
    f["consents"].insert(0, {
        "doc_id": "consent0", "patient_signed": "no", "signature_quote": "Patient signature: pending",
        "matches_scheduled_procedure": "yes", "procedure_quote": "Procedure: knee arthroscopy",
    })
    output = run(s, f)

    assert output.decision == "READY"
    assert "not signed by the patient" in output.checks[1].rejected[0].reason


# --- Rule 2: testing ------------------------------------------------------------


@pytest.mark.parametrize(
    ("risk", "days", "expected"),
    [("LOW", 30, "PASS"), ("LOW", 31, "FAIL"), ("MODERATE", 30, "PASS"), ("MODERATE", 31, "FAIL"),
     ("HIGH", 14, "PASS"), ("HIGH", 15, "FAIL")],
)
def test_cbc_window_by_risk(risk: str, days: int, expected: str) -> None:
    s, f = ready_case()
    if risk == "HIGH":
        make_high_risk(s, f, cbc_days=days)
    else:
        s["procedure"]["procedure_risk"] = risk
        s["labs"][0]["effective_at"] = ago(days, "08:00")
    output = run(s, f)

    assert statuses(output)["cbc"] == expected
    if expected == "FAIL":
        assert categories(output) == {"REQUIRED_TESTING"}


@pytest.mark.parametrize(("days", "expected"), [(14, "PASS"), (15, "FAIL"), (None, "FAIL")])
def test_cmp_is_required_within_fourteen_days_at_high_risk(days: int | None, expected: str) -> None:
    s, f = ready_case()
    make_high_risk(s, f, cmp_days=days)
    output = run(s, f)

    assert statuses(output)["cmp"] == expected
    if expected == "FAIL":
        assert categories(output) == {"REQUIRED_TESTING"}


def test_windows_count_calendar_days_not_elapsed_hours() -> None:
    s, f = ready_case()
    s["labs"][0]["effective_at"] = ago(30, "00:01")  # more than 30 x 24 hours before any time on the day
    assert statuses(run(s, f))["cbc"] == "PASS"

    s, f = ready_case()
    make_high_risk(s, f, cbc_days=15)
    s["labs"][0]["effective_at"] = ago(15, "23:59")  # under 15 x 24 hours before the day starts
    s["procedure"]["procedure_date"] = PROCEDURE.isoformat() + "T00:00:00Z"
    assert statuses(run(s, f))["cbc"] == "FAIL"


def test_a_bmp_is_not_a_cmp() -> None:
    s, f = ready_case()
    make_high_risk(s, f, cmp_days=None)
    s["labs"].append({"id": "lab-bmp-1", "code": "BMP", "effective_at": ago(3, "08:00"), "status": "final"})
    output = run(s, f)

    assert statuses(output)["cmp"] == "FAIL"
    assert output.checks[3].rejected[0].reason == "a BMP is not a CMP"


def test_an_absence_issue_cites_what_was_considered_and_rejected() -> None:
    """Bug: the issue said "nothing on file to cite" while the BMP sat only in the explanation."""

    s, f = ready_case()
    make_high_risk(s, f, cmp_days=None)
    s["labs"].append({"id": "lab-bmp-1", "code": "BMP", "effective_at": ago(3, "08:00"), "status": "final"})
    issue = run(s, f).issues[0]

    assert issue.description == "CMP: no CMP on file"
    assert "BMP" in issue.evidence and "a BMP is not a CMP" in issue.evidence
    assert "nothing on file to cite" not in issue.evidence


@pytest.mark.parametrize(("older_days", "expected"), [(20, "PASS"), (44, "FAIL")])
def test_entered_in_error_falls_back_to_the_latest_real_result(older_days: int, expected: str) -> None:
    s, f = ready_case()
    s["labs"] = [
        {"id": "lab-new", "code": "CBC", "effective_at": ago(3, "08:00"), "status": "entered-in-error"},
        {"id": "lab-old", "code": "LAB-CBC", "effective_at": ago(older_days, "08:00"), "status": "final"},
    ]
    output = run(s, f)

    assert statuses(output)["cbc"] == expected
    assert any(r.reason == "entered in error" for r in output.checks[2].rejected)


def test_issue_evidence_names_unusual_rejections_but_not_merely_older_results() -> None:
    """Review finding: an entered-in-error CBC case cited only the old CBC. The nurse
    needs to see that a newer result exists and why it was not used."""

    s, f = ready_case()
    s["labs"] = [
        {"id": "lab-new", "code": "CBC", "effective_at": ago(3, "08:00"), "status": "entered-in-error"},
        {"id": "lab-mid", "code": "CBC", "effective_at": ago(44, "08:00"), "status": "final"},
        {"id": "lab-old", "code": "CBC", "effective_at": ago(90, "08:00"), "status": "final"},
    ]
    issue = run(s, f).issues[0]

    assert issue.category == "REQUIRED_TESTING"
    assert "lab #lab-mid" in issue.evidence  # the result the finding rests on
    assert "lab #lab-new (entered in error)" in issue.evidence
    assert "lab-old" not in issue.evidence and "older result" not in issue.evidence


def test_written_readings_are_found_by_one_shared_helper() -> None:
    from rules import blood_pressures_written, temperature_reported, temperatures_written

    text = "Visit 03/19/2026. Pain 6/10. 08:06 NIBP 189/112, T 37.1 C. Later 137/78 mmHg, Temp 99.1 °F. Goal INR 2-3."
    assert blood_pressures_written(text) == [(189, 112), (137, 78)]
    assert temperatures_written(text) == [(37.1, "C"), (99.1, "F")]
    assert temperature_reported(38.6, "C", [101.5]) and not temperature_reported(100.9, "F", [100.0])


def test_the_latest_result_decides_whatever_the_list_order() -> None:
    s, f = ready_case()
    s["labs"] = [
        {"id": "lab-b", "code": "CBC", "effective_at": ago(5, "08:00"), "status": "final"},
        {"id": "lab-a", "code": "CBC", "effective_at": ago(60, "08:00"), "status": "final"},
    ]
    assert statuses(run(s, f))["cbc"] == "PASS"


def test_missing_procedure_date_is_missing_data_only() -> None:
    s, f = ready_case()
    s["procedure"]["procedure_date"] = None
    output = run(s, f)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert statuses(output)["hp"] == statuses(output)["cbc"] == "UNKNOWN"
    assert categories(output) == {"MISSING_REQUIRED_DATA"}


@pytest.mark.parametrize("risk", [None, "INTERMEDIATE"])
def test_missing_or_unfamiliar_risk_is_missing_data_even_when_labs_would_pass(risk: str | None) -> None:
    s, f = ready_case()
    make_high_risk(s, f, cbc_days=3, cmp_days=3)  # would pass every regime
    s["procedure"]["procedure_risk"] = risk
    output = run(s, f)

    assert statuses(output)["cbc"] == statuses(output)["cmp"] == "UNKNOWN"
    assert output.decision == "NEEDS_FOLLOW_UP"
    assert categories(output) == {"MISSING_REQUIRED_DATA"}


def test_unknown_risk_still_fails_a_cbc_too_old_for_any_risk_level() -> None:
    s, f = ready_case()
    s["procedure"]["procedure_risk"] = None
    s["labs"][0]["effective_at"] = ago(40, "08:00")
    output = run(s, f)

    assert statuses(output)["cbc"] == "FAIL"
    assert categories(output) == {"REQUIRED_TESTING", "MISSING_REQUIRED_DATA"}


def test_unrecognised_lab_code_blocks_a_no_cbc_finding_but_not_a_pass() -> None:
    s, f = ready_case()
    s["labs"] = [{"id": "lab-x", "code": "HEMOGRAM", "effective_at": ago(3, "08:00"), "status": "final"}]
    output = run(s, f)
    assert statuses(output)["cbc"] == "UNKNOWN"
    assert "HEMOGRAM" in output.checks[2].finding
    assert categories(output) == {"MISSING_REQUIRED_DATA"}

    s, f = ready_case()
    s["labs"].append({"id": "lab-x", "code": "HEMOGRAM", "effective_at": ago(3, "08:00"), "status": "final"})
    assert run(s, f).decision == "READY"


def test_unrecognised_lab_status_is_unknown() -> None:
    s, f = ready_case()
    s["labs"][0]["status"] = "preliminary"
    output = run(s, f)

    assert statuses(output)["cbc"] == "UNKNOWN"
    assert "preliminary" in output.checks[2].finding


# --- Rule 3: anticoagulation ----------------------------------------------------


def test_anticoagulated_with_a_complete_plan_passes() -> None:
    s, f = ready_case()
    on_drug(s, f)
    plan(s, f)
    output = run(s, f)

    assert statuses(output)["anticoagulation"] == "PASS"
    assert output.decision == "READY"


@pytest.mark.parametrize(
    "setup",
    [
        lambda s, f: on_drug(s, f),  # no plan at all
        lambda s, f: (on_drug(s, f), plan(s, f, postop="deferred")),  # before only
        lambda s, f: (on_drug(s, f, "Eliquis 5 mg PO BID"), plan(s, f, drug="warfarin", mention="stopped")),  # wrong drug
    ],
    ids=["no-plan", "post-op-deferred", "plan-for-previous-drug"],
)
def test_anticoagulated_without_a_complete_plan_for_the_current_drug_fails(setup) -> None:
    s, f = ready_case()
    setup(s, f)
    output = run(s, f)

    assert statuses(output)["anticoagulation"] == "FAIL"
    assert output.decision == "NEEDS_FOLLOW_UP"
    assert categories(output) == {"ANTICOAGULATION_MANAGEMENT"}


def test_plan_for_the_previous_drug_is_named_in_the_finding() -> None:
    s, f = ready_case()
    on_drug(s, f, "Eliquis 5 mg PO BID")
    plan(s, f, drug="warfarin", mention="stopped")
    assert "the plan on file is for warfarin" in run(s, f).checks[4].finding


def test_drug_reported_only_in_a_note_counts_as_taken() -> None:
    s, f = ready_case()
    doc(s, "note1")["text"] += "\nPatient also reports taking Eliquis 5 mg twice daily."
    f["anticoagulant_mentions"].append({
        "doc_id": "note1", "drug_as_written": "Eliquis", "generic_name": "apixaban", "status": "current",
        "quote": "Patient also reports taking Eliquis 5 mg twice daily.",
    })
    output = run(s, f)
    assert statuses(output)["anticoagulation"] == "FAIL"
    assert categories(output) == {"ANTICOAGULATION_MANAGEMENT"}

    plan(s, f, drug="apixaban")
    output = run(s, f)
    assert statuses(output)["anticoagulation"] == "PASS"
    assert "found in a note" in output.checks[4].finding


@pytest.mark.parametrize(
    "setup",
    [
        lambda s, f: on_drug(s, f, active=None),
        lambda s, f: f["medication_list_anticoagulants"].append("betrixaban 80 mg PO daily"),
        lambda s, f: doc(s, "note1").update(text="Pt states she takes Xarelto every evening."),
        lambda s, f: f["anticoagulant_mentions"].append({
            "doc_id": "note1", "drug_as_written": "Xarelto", "generic_name": "rivaroxaban",
            "status": "current", "quote": "takes Xarelto nightly"}),
        lambda s, f: (on_drug(s, f), plan(s, f, verified=False)),
    ],
    ids=["status-not-given", "drug-outside-the-table", "named-but-not-reported", "mention-unverified", "plan-unverified"],
)
def test_anticoagulation_doubt_is_unknown(setup) -> None:
    s, f = ready_case()
    setup(s, f)
    output = run(s, f)

    assert statuses(output)["anticoagulation"] == "UNKNOWN"
    assert output.decision == "NEEDS_FOLLOW_UP"
    assert categories(output) == {"MISSING_REQUIRED_DATA"}


@pytest.mark.parametrize(
    "setup",
    [
        lambda s, f: on_drug(s, f, active=False),
        lambda s, f: on_drug(s, f, "ASA 81 mg PO daily"),
        lambda s, f: on_drug(s, f, "clopidogrel 75 mg PO daily"),
        lambda s, f: (on_drug(s, f, "Aspirin EC 81 mg PO daily"),
                      f["medication_list_anticoagulants"].append("Aspirin EC 81 mg PO daily")),
    ],
    ids=["inactive", "aspirin", "clopidogrel", "reader-calls-aspirin-an-anticoagulant"],
)
def test_not_anticoagulated(setup) -> None:
    s, f = ready_case()
    setup(s, f)
    output = run(s, f)

    assert statuses(output)["anticoagulation"] == "NOT_APPLICABLE"
    assert output.decision == "READY"


def says(s, f, doc_id, drug, status, sentence, date=...):
    """Add a document statement about a drug, creating the document if needed."""

    existing = [d for d in s["documents"] if d["doc_id"] == doc_id]
    if existing:
        existing[0]["text"] += "\n" + sentence
    else:
        s["documents"].append({"doc_id": doc_id, "type": "Clinic Note", "date": date, "text": sentence})
    f["anticoagulant_mentions"].append(
        {"doc_id": doc_id, "drug_as_written": drug, "generic_name": drug, "status": status, "quote": sentence}
    )


def test_an_undated_stopped_note_cannot_clear_a_dated_current_one() -> None:
    """Bug: the undated document was treated as the newest, giving NOT_APPLICABLE and READY."""

    s, f = ready_case()
    says(s, f, "note1", "Xarelto", "current", "Pt takes Xarelto nightly.")
    says(s, f, "undated1", "Xarelto", "stopped", "Xarelto was stopped.", date=None)
    output = run(s, f)

    assert statuses(output)["anticoagulation"] == "UNKNOWN"
    assert output.decision == "NEEDS_FOLLOW_UP"
    assert categories(output) == {"MISSING_REQUIRED_DATA"}
    assert "whether rivaroxaban is being taken is not established" in output.checks[4].finding


@pytest.mark.parametrize(
    ("stopped_days_ago", "expected", "decision"),
    [(0, "NOT_APPLICABLE", "READY"),  # a provably later document says stopped
     (1, "UNKNOWN", "NEEDS_FOLLOW_UP"),  # same day as the note that says taken
     (3, "FAIL", "NEEDS_FOLLOW_UP")],  # the note that says taken is the later one
)
def test_document_dates_decide_between_conflicting_drug_statements(stopped_days_ago, expected, decision) -> None:
    s, f = ready_case()
    says(s, f, "note1", "Xarelto", "current", "Pt takes Xarelto nightly.")  # note1 is dated one day before
    says(s, f, "clinic1", "Xarelto", "stopped", "Xarelto was stopped.", date=ago(stopped_days_ago))
    output = run(s, f)

    assert statuses(output)["anticoagulation"] == expected
    assert output.decision == decision


def test_a_note_cannot_silently_override_the_medication_list() -> None:
    """Bug: the list said active, a note said stopped, and the row read "taking warfarin"
    beside "rejected: warfarin in note: stopped". The two sources disagree and the list
    carries no date, so which is right is not established."""

    s, f = ready_case()
    on_drug(s, f)  # warfarin, active on the list
    says(s, f, "note1", "warfarin", "stopped", "Warfarin discontinued last month.")
    output = run(s, f)

    assert statuses(output)["anticoagulation"] == "UNKNOWN"
    assert categories(output) == {"MISSING_REQUIRED_DATA"}
    finding = output.checks[4].finding
    assert "the medication list has warfarin 5 mg PO daily as active" in finding
    assert "says warfarin was stopped" in finding
    assert all("stopped" not in r.reason for r in output.checks[4].rejected)


def test_a_complete_plan_settles_a_drug_whose_status_is_disputed() -> None:
    s, f = ready_case()
    on_drug(s, f)
    plan(s, f)
    says(s, f, "note1", "warfarin", "stopped", "Warfarin discontinued last month.", date=None)
    doc(s, "note1")["date"] = ago(0)  # later than the plan document
    output = run(s, f)

    assert statuses(output)["anticoagulation"] == "PASS"
    assert "whether or not it is still taken" in output.checks[4].finding
    assert output.decision == "READY"


@pytest.mark.parametrize(
    ("active", "status", "expected"),
    [(False, "current", "UNKNOWN"),  # list says inactive, note says taken
     (None, "current", "FAIL"),  # list does not say; the note does, and there is no plan
     (None, "stopped", "NOT_APPLICABLE"),
     (False, "stopped", "NOT_APPLICABLE")],
)
def test_medication_list_and_note_together(active, status, expected) -> None:
    s, f = ready_case()
    on_drug(s, f, active=active)
    says(s, f, "note1", "warfarin", status, "Patient discussed warfarin at intake.")
    assert statuses(run(s, f))["anticoagulation"] == expected


def second_plan(s, f, days_ago, postop):
    resume = "Resume warfarin the evening after surgery." if postop == "concrete" else "Post-op anticoagulation to be addressed after surgery."
    hold = "Stop warfarin 5 days prior."
    s["documents"].append({"doc_id": "plan2", "type": "Hematology Consult", "date": ago(days_ago),
                           "text": "\n".join(["Continues on warfarin.", hold, resume])})
    f["anticoagulant_mentions"].append(
        {"doc_id": "plan2", "drug_as_written": "warfarin", "generic_name": "warfarin", "status": "current", "quote": "Continues on warfarin."}
    )
    f["anticoagulation_plans"].append({
        "doc_id": "plan2", "generic_name": "warfarin",
        "preop": "concrete", "preop_quote": hold, "postop": postop, "postop_quote": resume,
    })


@pytest.mark.parametrize(
    ("first_postop", "second_days_ago", "second_postop", "expected"),
    [("concrete", 2, "deferred", "FAIL"),  # a later plan defers what the earlier one settled
     ("deferred", 2, "concrete", "PASS"),  # a later plan completes it
     ("concrete", 4, "deferred", "FAIL")],  # same day, and they disagree: ambiguous
)
def test_the_newest_plan_decides(first_postop, second_days_ago, second_postop, expected) -> None:
    s, f = ready_case()
    on_drug(s, f)
    plan(s, f, postop=first_postop)  # dated four days before the procedure
    second_plan(s, f, second_days_ago, second_postop)
    output = run(s, f)

    assert statuses(output)["anticoagulation"] == expected
    if expected == "FAIL":
        assert categories(output) == {"ANTICOAGULATION_MANAGEMENT"}


def reported_plan(f, doc_id, postop="deferred"):
    """A plan the reader reports whose quotes are not in the document it cites."""

    f["anticoagulation_plans"].append({
        "doc_id": doc_id, "generic_name": "warfarin",
        "preop": "concrete", "preop_quote": "Hold warfarin for a week.",
        "postop": postop, "postop_quote": "Restart to be decided later.",
    })


@pytest.mark.parametrize("doc_id", ["no-such-document", "note1"], ids=["document-does-not-exist", "quote-not-in-a-later-note"])
def test_an_unverified_plan_never_fails_the_row_or_outranks_a_verified_plan(doc_id: str) -> None:
    """Bug: incompleteness was checked before the quotes, so a plan that could not be
    verified overrode a verified complete plan and failed the row."""

    s, f = ready_case()
    on_drug(s, f)
    plan(s, f)  # verified and complete, dated four days before the procedure
    reported_plan(f, doc_id)  # note1 is dated one day before, so it could be the newest
    output = run(s, f)

    assert statuses(output)["anticoagulation"] == "UNKNOWN"
    assert categories(output) == {"MISSING_REQUIRED_DATA"}
    assert "could not verify the plan" in output.checks[4].finding


def test_an_unverified_plan_older_than_a_verified_one_is_ignored() -> None:
    s, f = ready_case()
    on_drug(s, f)
    plan(s, f)
    reported_plan(f, "hp1")  # dated ten days before: provably older than the verified plan
    output = run(s, f)

    assert statuses(output)["anticoagulation"] == "PASS"
    assert output.decision == "READY"


def test_a_plan_with_nothing_quoted_is_not_established() -> None:
    s, f = ready_case()
    on_drug(s, f)
    plan(s, f)
    f["anticoagulation_plans"][0].update(preop="absent", preop_quote=None, postop="absent", postop_quote=None)
    output = run(s, f)

    assert statuses(output)["anticoagulation"] == "UNKNOWN"


def test_a_planned_bridge_does_not_need_its_own_plan() -> None:
    s, f = ready_case()
    on_drug(s, f)
    plan(s, f)
    doc(s, "plan1")["text"] += "\nBridge with enoxaparin once INR is below 2."
    f["anticoagulant_mentions"].append({
        "doc_id": "plan1", "drug_as_written": "enoxaparin", "generic_name": "enoxaparin",
        "status": "planned", "quote": "Bridge with enoxaparin once INR is below 2.",
    })
    assert run(s, f).decision == "READY"


# --- Rule 4: acute safety -------------------------------------------------------


@pytest.mark.parametrize(
    ("systolic", "diastolic", "expected"),
    [(179, 109, "PASS"), (180, 80, "FAIL"), (120, 110, "FAIL"), (200, 120, "FAIL")],
)
def test_blood_pressure_thresholds_are_inclusive(systolic: int, diastolic: int, expected: str) -> None:
    s, f = ready_case()
    s["vitals"][0].update(systolic=systolic, diastolic=diastolic)
    output = run(s, f)

    assert statuses(output)["blood_pressure"] == expected
    if expected == "FAIL":
        assert output.decision == "NOT_CLEARED"
        assert categories(output) == {"ACUTE_SAFETY_EXCLUSION"}


@pytest.mark.parametrize(("value_f", "expected"), [(100.4, "PASS"), (100.5, "FAIL")])
def test_temperature_threshold_is_exclusive(value_f: float, expected: str) -> None:
    s, f = ready_case()
    s["vitals"][1]["value_f"] = value_f
    output = run(s, f)

    assert statuses(output)["temperature"] == expected
    assert output.decision == ("NOT_CLEARED" if expected == "FAIL" else "READY")


@pytest.mark.parametrize(("celsius", "expected"), [(37.9, "PASS"), (38.0, "PASS"), (38.1, "FAIL")])
def test_celsius_in_a_note_is_converted(celsius: float, expected: str) -> None:
    s, f = ready_case()
    note_temp(s, f, celsius, "C")
    output = run(s, f)

    assert statuses(output)["temperature"] == expected
    assert f"{celsius:g} C = " in output.checks[6].finding


def test_missing_temperature_is_missing_data_even_with_a_normal_blood_pressure() -> None:
    s, f = ready_case()
    s["vitals"] = s["vitals"][:1]
    output = run(s, f)

    assert statuses(output)["blood_pressure"] == "PASS"
    assert statuses(output)["temperature"] == "UNKNOWN"
    assert categories(output) == {"MISSING_REQUIRED_DATA"}


def test_structured_vitals_are_ordered_by_time_not_list_position() -> None:
    s, f = ready_case()
    s["vitals"] = [
        {"type": "blood_pressure", "systolic": 122, "diastolic": 78, "date": ago(1, "10:00"), "source": "clinic"},
        {"type": "blood_pressure", "systolic": 190, "diastolic": 115, "date": ago(6, "10:00"), "source": "clinic"},
        s["vitals"][1],
    ]
    assert run(s, f).decision == "READY"

    s["vitals"][0], s["vitals"][1] = s["vitals"][1], s["vitals"][0]
    s["vitals"][0]["date"], s["vitals"][1]["date"] = ago(1, "10:00"), ago(6, "10:00")
    assert run(s, f).decision == "NOT_CLEARED"


def test_a_later_reading_in_a_note_outranks_structured_vitals() -> None:
    s, f = ready_case()
    note_bp(s, f, 157, 117, time="09:05")
    output = run(s, f)

    assert output.decision == "NOT_CLEARED"
    assert output.checks[5].evidence[0].quote == "09:05 BP 157/117 mmHg"


def test_a_recheck_in_range_clears_an_earlier_breach() -> None:
    s, f = ready_case()
    note_bp(s, f, 189, 112, time="08:06")
    note_bp(s, f, 137, 78, time="08:30")
    output = run(s, f)

    assert output.decision == "READY"
    assert "earlier breach superseded" in output.checks[5].finding
    assert output.checks[5].rejected[0].reason == "breach superseded by a later reading"


def test_a_recheck_that_still_breaches_does_not_clear() -> None:
    s, f = ready_case()
    note_bp(s, f, 189, 112, time="08:06")
    note_bp(s, f, 181, 98, time="08:30")
    assert run(s, f).decision == "NOT_CLEARED"


def test_a_breach_and_an_in_range_reading_that_cannot_be_ordered_is_follow_up() -> None:
    """NOT_CLEARED claims the breach is the most recent reading. With no time on it, that
    is not established, and neither is the opposite."""

    s, f = ready_case()
    note_bp(s, f, 190, 115, time=None)  # untimed
    note_bp(s, f, 130, 80, time="08:30")
    output = run(s, f)

    assert statuses(output)["blood_pressure"] == "UNKNOWN"
    assert output.decision == "NEEDS_FOLLOW_UP"
    assert categories(output) == {"MISSING_REQUIRED_DATA"}
    assert "cannot be put in order" in output.checks[5].finding
    assert "190/115" in output.checks[5].finding and "130/80" in output.checks[5].finding


@pytest.mark.parametrize(
    ("first", "second", "expected", "decision"),
    [((190, 115), (185, 112), "FAIL", "NOT_CLEARED"), ((130, 80), (125, 78), "PASS", "READY")],
)
def test_unordered_readings_that_agree_still_decide(first, second, expected, decision) -> None:
    """Whichever came last, the answer is the same, so the order does not matter."""

    s, f = ready_case()
    note_bp(s, f, *first, time=None)
    note_bp(s, f, *second, time="08:30")
    output = run(s, f)

    assert statuses(output)["blood_pressure"] == expected
    assert output.decision == decision


def test_an_unverified_breach_is_follow_up_not_a_hold() -> None:
    s, f = ready_case()
    note_bp(s, f, 190, 115, written=False)
    output = run(s, f)

    assert statuses(output)["blood_pressure"] == "UNKNOWN"
    assert output.decision == "NEEDS_FOLLOW_UP"
    assert categories(output) == {"MISSING_REQUIRED_DATA"}


def test_an_unverified_recheck_neither_clears_nor_confirms_a_breach() -> None:
    s, f = ready_case()
    note_bp(s, f, 189, 112, time="08:06")
    note_bp(s, f, 137, 78, time="08:30", written=False)
    output = run(s, f)

    assert statuses(output)["blood_pressure"] == "UNKNOWN"
    assert output.decision == "NEEDS_FOLLOW_UP"
    assert "breaches the limit" in output.checks[5].finding
    assert "could not verify 137/78" in output.checks[5].finding


@pytest.mark.parametrize("with_structured_vitals", [True, False])
def test_an_unverified_reading_from_an_older_document_cannot_unsettle_a_breach(with_structured_vitals: bool) -> None:
    """Bug: an unverified reading from any document made the row UNKNOWN, so a verified
    190/115 from the latest note plus an unverified 128/80 from a 10-day-old H&P came
    out NEEDS_FOLLOW_UP. A reading that old cannot be the most recent one."""

    s, f = ready_case()
    if not with_structured_vitals:
        s["vitals"] = s["vitals"][1:]  # keep the temperature only
    note_bp(s, f, 190, 115)  # verified, from the note dated one day before the procedure
    f["note_blood_pressures"].append(
        {"doc_id": "hp1", "systolic": 128, "diastolic": 80, "time": "10:00", "quote": "BP 128/80 at intake"}
    )
    output = run(s, f)

    assert statuses(output)["blood_pressure"] == "FAIL"
    assert output.decision == "NOT_CLEARED"
    assert categories(output) == {"ACUTE_SAFETY_EXCLUSION"}


def test_an_unverified_fever_from_an_older_document_cannot_unsettle_a_normal_temperature() -> None:
    s, f = ready_case()
    f["note_temperatures"].append({"doc_id": "hp1", "value": 101.2, "unit": "F", "time": None, "quote": "febrile at 101.2"})
    assert run(s, f).decision == "READY"


def test_a_claimed_time_on_an_unverified_reading_is_not_trusted() -> None:
    """Same document and day as the verified reading: it could be the latest, whatever time it claims."""

    s, f = ready_case()
    note_bp(s, f, 130, 80, time="08:30")
    f["note_blood_pressures"].append(
        {"doc_id": "note1", "systolic": 190, "diastolic": 115, "time": "06:00", "quote": "06:00 BP 190/115"}
    )
    assert statuses(run(s, f))["blood_pressure"] == "UNKNOWN"


def test_a_reported_temperature_must_be_the_number_written() -> None:
    """Bug: 100 was accepted for "Temp 100.9 F" because "100" is a substring of "100.9"."""

    s, f = ready_case()
    note_temp(s, f, 100.9, "F")
    f["note_temperatures"][0]["value"] = 100  # the reader misreads a fever as in range
    output = run(s, f)

    assert statuses(output)["temperature"] == "UNKNOWN"
    assert "could not verify 100 F" in output.checks[6].finding
    assert "(100 F" not in output.checks[6].finding  # never presented as a verified reading
    assert output.decision == "NEEDS_FOLLOW_UP"


@pytest.mark.parametrize(
    ("quote", "value", "expected"),
    [("Temp 100.9 F", 100, False), ("Temp 100.9 F", 100.9, True), ("T 99.0 F", 99, True),
     ("T 37.1 C at 08:06", 37.1, True), ("T 37.1 C at 08:06", 7.1, False), ("afebrile", 98.6, False)],
)
def test_writes_number(quote: str, value: float, expected: bool) -> None:
    from rules import _writes_number

    assert _writes_number(quote, value) is expected


def test_a_reading_written_in_a_note_but_not_reported_blocks_ready() -> None:
    s, f = ready_case()
    note_bp(s, f, 190, 115, reported=False)
    output = run(s, f)

    assert statuses(output)["blood_pressure"] == "UNKNOWN"
    assert "190/115 is written in" in output.checks[5].finding


def test_a_missed_reading_older_than_the_structured_vitals_does_not_matter() -> None:
    s, f = ready_case()
    doc(s, "hp1")["text"] += "\nVitals 10:05 132/76 mmHg, Temperature 99.1 °F"
    assert run(s, f).decision == "READY"


def test_the_reported_number_must_appear_in_its_quote() -> None:
    s, f = ready_case()
    note_bp(s, f, 130, 80)
    f["note_blood_pressures"][0]["systolic"] = 120  # quote says 130
    assert statuses(run(s, f))["blood_pressure"] == "UNKNOWN"


def test_dates_and_pain_scores_are_not_mistaken_for_blood_pressures() -> None:
    s, f = ready_case()
    doc(s, "note1")["text"] += "\nVisit date: 03/19/2026. Pain 6/10. CBC drawn 03/10. Seen 12/25."
    assert run(s, f).decision == "READY"


def test_unrecognised_vital_type_blocks_ready() -> None:
    s, f = ready_case()
    s["vitals"].append({"type": "bp_standing", "systolic": 190, "diastolic": 120, "date": ago(0, "07:00")})
    output = run(s, f)

    assert statuses(output)["blood_pressure"] == statuses(output)["temperature"] == "UNKNOWN"
    assert "bp_standing" in output.checks[5].finding


def test_not_cleared_still_lists_every_other_failing_category() -> None:
    s, f = ready_case()
    s["vitals"][0].update(systolic=150, diastolic=110)
    unsigned(s, f)
    s["labs"] = []
    output = run(s, f)

    assert output.decision == "NOT_CLEARED"
    assert categories(output) == {"ACUTE_SAFETY_EXCLUSION", "REQUIRED_DOCUMENTATION", "REQUIRED_TESTING"}


# --- the safety property --------------------------------------------------------

ADVERSE = {
    "hp stale": lambda s, f: doc(s, "hp1").update(date=ago(31)),
    "hp missing": lambda s, f: f.update(hp_candidates=[]),
    "hp unverified": lambda s, f: f["hp_candidates"][0].update(quote="not in the document"),
    "consent unsigned": lambda s, f: f["consents"][0].update(patient_signed="no", signature_quote=None),
    "consent unclear": lambda s, f: f["consents"][0].update(patient_signed="unclear"),
    "consent unverified": lambda s, f: f["consents"][0].update(signature_quote="not in the document"),
    "consent wrong procedure": lambda s, f: f["consents"][0].update(matches_scheduled_procedure="no"),
    "cbc stale": lambda s, f: s["labs"][0].update(effective_at=ago(31, "08:00")),
    "cbc missing": lambda s, f: s.update(labs=[]),
    "cbc entered in error": lambda s, f: s["labs"][0].update(status="entered-in-error"),
    "cbc status unfamiliar": lambda s, f: s["labs"][0].update(status="preliminary"),
    "high risk without cmp": lambda s, f: make_high_risk(s, f, cmp_days=None),
    "procedure date missing": lambda s, f: s["procedure"].update(procedure_date=None),
    "risk missing": lambda s, f: s["procedure"].update(procedure_risk=None),
    "risk unfamiliar": lambda s, f: s["procedure"].update(procedure_risk="INTERMEDIATE"),
    "anticoagulant without plan": lambda s, f: on_drug(s, f),
    "anticoagulant status unknown": lambda s, f: on_drug(s, f, active=None),
    "anticoagulant plan half done": lambda s, f: (on_drug(s, f), plan(s, f, postop="deferred")),
    "anticoagulant plan unverified": lambda s, f: (on_drug(s, f), plan(s, f, verified=False)),
    "anticoagulant named but unreported": lambda s, f: doc(s, "note1").update(text="Takes Coumadin."),
    "structured bp breach": lambda s, f: s["vitals"][0].update(systolic=180),
    "structured fever": lambda s, f: s["vitals"][1].update(value_f=100.5),
    "note bp breach": lambda s, f: note_bp(s, f, 185, 90),
    "note bp breach unverified": lambda s, f: note_bp(s, f, 185, 90, written=False),
    "note bp breach unreported": lambda s, f: note_bp(s, f, 185, 90, reported=False),
    "note fever": lambda s, f: note_temp(s, f, 38.6, "C"),
    "no temperature": lambda s, f: s.update(vitals=s["vitals"][:1]),
    "no vitals": lambda s, f: s.update(vitals=[]),
    "vital type unfamiliar": lambda s, f: s["vitals"].append({"type": "bp_standing", "date": ago(0)}),
}


@pytest.mark.parametrize("name", sorted(ADVERSE))
def test_no_single_problem_can_produce_ready(name: str) -> None:
    s, f = ready_case()
    ADVERSE[name](s, f)
    assert run(s, f).decision != "READY"


def test_decision_issues_and_quotes_stay_consistent_under_random_combinations() -> None:
    rng = random.Random(20260320)
    names = sorted(ADVERSE)
    for _ in range(400):
        s, f = ready_case()
        for name in rng.sample(names, rng.randint(0, 4)):
            try:
                ADVERSE[name](s, f)
            except (IndexError, KeyError, StopIteration):
                pass  # an earlier change removed what this one edits
        output = run(s, f)
        by_id = statuses(output)

        assert tuple(by_id) == CHECK_ORDER
        all_clear = all(status in ("PASS", "NOT_APPLICABLE") for status in by_id.values())
        hold = "FAIL" in (by_id["blood_pressure"], by_id["temperature"])
        assert (output.decision == "READY") == all_clear
        assert (output.decision == "NOT_CLEARED") == hold
        assert len(output.issues) == sum(status in ("FAIL", "UNKNOWN") for status in by_id.values())
        assert ("MISSING_REQUIRED_DATA" in categories(output)) == ("UNKNOWN" in by_id.values())

        texts = {d["doc_id"]: _norm(d["text"]) for d in s["documents"]}
        for check in output.checks:
            for evidence in check.evidence:
                if evidence.source == "document":
                    assert _norm(evidence.quote) in texts[evidence.ref], (check.id, evidence.quote)
