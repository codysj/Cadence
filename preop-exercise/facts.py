"""The reader's contract: what the one model call is shown, and what it returns.

It returns atomic facts read from the free-text documents, each tied to a document id
and a verbatim quote that code checks against the source. The reader never returns a
decision, a category, a day count or a unit conversion: those belong to rules.py.

Every field is required and no extra fields are allowed, which is what OpenAI's
strict structured-output mode needs. "Not stated" is an explicit null or "unclear".
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from models import PatientSubmission


def reader_input(submission: PatientSubmission) -> dict[str, Any]:
    """What the reader is shown: the documents, the medication list and the scheduled
    procedure. Nothing it could decide with: no labs, no structured vitals, no risk
    level, no procedure date field and no thresholds."""

    return {
        "scheduled_procedure": submission.procedure.procedure_type if submission.procedure else None,
        "medication_list": [{"name": m.name, "active": m.active} for m in submission.medications],
        "documents": [
            {"doc_id": d.doc_id, "type": d.type, "date": d.date, "author": d.author, "text": d.text}
            for d in submission.documents
        ],
    }


def submission_hash(submission: dict[str, Any]) -> str:
    """Identity of a submission as received, for keying saved facts."""

    canonical = json.dumps(submission, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


YesNoUnclear = Literal["yes", "no", "unclear"]
Instruction = Literal["concrete", "deferred", "absent"]


class _Fact(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HpCandidate(_Fact):
    """A document that presents itself as a History and Physical."""

    doc_id: str
    is_preop_evaluation: bool  # False for wellness, establish-care or anesthesia notes
    quote: str


class Consent(_Fact):
    doc_id: str
    patient_signed: YesNoUnclear  # the patient's signature, not the physician's attestation
    signature_quote: str | None
    matches_scheduled_procedure: YesNoUnclear
    procedure_quote: str | None


class NoteBloodPressure(_Fact):
    doc_id: str
    systolic: int
    diastolic: int
    time: str | None  # clock time as written, "HH:MM"; null if the note gives none
    quote: str


class NoteTemperature(_Fact):
    doc_id: str
    value: float  # as written; code converts Celsius
    unit: Literal["F", "C"]
    time: str | None
    quote: str


class AnticoagulantMention(_Fact):
    """An anticoagulant named in a document, whatever its status."""

    doc_id: str
    drug_as_written: str
    generic_name: str  # lower case, e.g. "rivaroxaban" for Xarelto
    status: Literal["current", "stopped", "planned", "unknown"]  # planned: e.g. a bridge
    quote: str


class AnticoagulationPlan(_Fact):
    doc_id: str
    generic_name: str  # the drug the plan is written for
    preop: Instruction
    preop_quote: str | None
    postop: Instruction
    postop_quote: str | None


class Facts(_Fact):
    hp_candidates: list[HpCandidate]
    consents: list[Consent]
    note_blood_pressures: list[NoteBloodPressure]
    note_temperatures: list[NoteTemperature]
    anticoagulant_mentions: list[AnticoagulantMention]
    # Entries of the structured medication list the reader considers anticoagulants,
    # copied verbatim. A second net under the code's own drug table.
    medication_list_anticoagulants: list[str]
    anticoagulation_plans: list[AnticoagulationPlan]
