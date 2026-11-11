"""What the reader (the one model call) returns.

Atomic facts read from the free-text documents, each tied to a document id and a
verbatim quote that code checks against the source. The reader never returns a
decision, a category, a day count or a unit conversion: those belong to rules.py.

Every field is required and no extra fields are allowed, which is what OpenAI's
strict structured-output mode needs. "Not stated" is an explicit null or "unclear".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

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
