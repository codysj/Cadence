"""Stage 2 check: what a nurse reads for a real case.

case_00016 is READY, but its nursing note opens with a breaching 189/112 that a recheck
24 minutes later cleared. The facts below are what a perfect reader would return for it.
"""

from __future__ import annotations

import json
from pathlib import Path

from models import PatientSubmission
from facts import Facts
from rules import evaluate

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "patients_sample_50.jsonl"

WELLNESS_HP = "89fbc4a7-5a5e-5b71-85eb-d5b254e0ba55"
NURSING = "1f5be5f7-1a56-5e66-9c45-2e94e55b0271"
PREOP_HP = "81e2f72a-f558-5c73-853d-5383ea9f59b1"
CONSENT = "08378923-717b-50f8-9bb1-0d78e19d1f66"

FACTS_00016 = {
    "hp_candidates": [
        {"doc_id": WELLNESS_HP, "is_preop_evaluation": False, "quote": "Encounter: annual wellness / establish care"},
        {"doc_id": PREOP_HP, "is_preop_evaluation": True, "quote": "PREOPERATIVE HISTORY AND PHYSICAL"},
    ],
    "consents": [{
        "doc_id": CONSENT,
        "patient_signed": "yes",
        "signature_quote": "Patient signature on file - signed in clinic 03/13/2026 12:06 (scanned copy attached).",
        "matches_scheduled_procedure": "yes",
        "procedure_quote": "Procedure: inguinal hernia repair",
    }],
    "note_blood_pressures": [
        {"doc_id": PREOP_HP, "systolic": 132, "diastolic": 76, "time": "10:05",
         "quote": "Vitals 10:05  132/76 mmHg, HR 78, Temperature 99.1 °F (oral), SpO2 95% RA"},
        {"doc_id": NURSING, "systolic": 189, "diastolic": 112, "time": "08:06",
         "quote": "08:06  NIBP 189/112, T 37.1 C, HR 74, SpO2 96% RA. See addendum."},
        {"doc_id": NURSING, "systolic": 137, "diastolic": 78, "time": "08:30",
         "quote": "ADDENDUM 08:30: Re-measured 137/78 mmHg following 24 minutes in a quiet room."},
    ],
    "note_temperatures": [
        {"doc_id": PREOP_HP, "value": 99.1, "unit": "F", "time": "10:05",
         "quote": "Vitals 10:05  132/76 mmHg, HR 78, Temperature 99.1 °F (oral), SpO2 95% RA"},
        {"doc_id": NURSING, "value": 37.1, "unit": "C", "time": "08:06",
         "quote": "08:06  NIBP 189/112, T 37.1 C, HR 74, SpO2 96% RA. See addendum."},
    ],
    "anticoagulant_mentions": [],
    "medication_list_anticoagulants": [],
    "anticoagulation_plans": [],
}

EXPECTED = """\
READY: all applicable checks pass
case_00016 | Elective inguinal hernia repair | procedure date 2026-03-17 | risk LOW

PASS    H&P: dated 2026-02-21, 24 days before the procedure, limit 30
    source: H&P [PDF] 2026-02-21 #81e2f72a
      "PREOPERATIVE HISTORY AND PHYSICAL"
    rejected: History and Physical 2025-06-22 #89fbc4a7: not the pre-op evaluation

PASS    Consent: signed by the patient for the scheduled procedure
    source: Procedure Consent Form 2026-03-13 #08378923
      "Patient signature on file - signed in clinic 03/13/2026 12:06 (scanned copy attached)."
    source: Procedure Consent Form 2026-03-13 #08378923
      "Procedure: inguinal hernia repair"

PASS    CBC: dated 2026-03-02, 15 days before the procedure, limit 30
    source: CBC 2026-03-02 lab #95160d11
    rejected: CBC 2026-02-07 lab #9c9239da: older result

N/A     CMP: not required at LOW risk

N/A     Anticoagulation: no anticoagulant currently taken (4 medications and 4 documents checked)
    rejected: medication list: ASA 81 mg PO daily: antiplatelet, not an anticoagulant

PASS    Blood pressure: 137/78 on 2026-03-15 08:30, limits 180/110 (earlier breach superseded by this reading)
    source: Nursing Pre-procedure Assessment 2026-03-15 #1f5be5f7
      "ADDENDUM 08:30: Re-measured 137/78 mmHg following 24 minutes in a quiet room."
    rejected: 189/112 on 2026-03-15 08:06 (Nursing Pre-procedure Assessment 2026-03-15 #1f5be5f7): breach superseded by a later reading
    rejected: 123/75 on 2026-03-12 10:12 (structured vital, Cardiology clinic): older reading
    rejected: 117/72 on 2026-03-05 09:05 (structured vital, Pre-op clinic): older reading
    rejected: 132/76 on 2026-02-21 10:05 (H&P [PDF] 2026-02-21 #81e2f72a): older reading

PASS    Temperature: 37.1 C = 98.8 F on 2026-03-15 08:06, limit 100.4 F
    source: Nursing Pre-procedure Assessment 2026-03-15 #1f5be5f7
      "08:06  NIBP 189/112, T 37.1 C, HR 74, SpO2 96% RA. See addendum."
    rejected: 98.8 F on 2026-03-12 10:15 (structured vital, Nursing triage): older reading
    rejected: 99.1 F on 2026-02-21 10:05 (H&P [PDF] 2026-02-21 #81e2f72a): older reading
"""


def case_00016():
    with SAMPLE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["case_id"] == "case_00016":
                return row
    raise AssertionError("case_00016 not found")


def test_case_00016_renders_as_a_checklist_a_nurse_can_verify() -> None:
    row = case_00016()
    output = evaluate(PatientSubmission.model_validate(row["submission"]), Facts.model_validate(FACTS_00016))

    assert output.decision == row["label"]["decision"] == "READY"
    assert output.issues == []
    assert output.explanation == EXPECTED.rstrip("\n")
