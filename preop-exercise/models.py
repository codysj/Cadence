"""Shared data models: the input package, the checklist rows and the output.

Nothing here calls a model or applies a rule. Dependencies run one way:
models -> facts -> rules -> core.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

Decision = Literal["READY", "NEEDS_FOLLOW_UP", "NOT_CLEARED"]
ProcedureRisk = Literal["LOW", "MODERATE", "HIGH"]
IssueCategory = Literal[
    "REQUIRED_DOCUMENTATION",
    "REQUIRED_TESTING",
    "ANTICOAGULATION_MANAGEMENT",
    "ACUTE_SAFETY_EXCLUSION",
    "MISSING_REQUIRED_DATA",
]

# -------------------------
# Schemas
# -------------------------

class _Strict(BaseModel):
    """Input models reject unknown fields.

    Every field is optional with an empty default, so a renamed field would otherwise
    read as "this patient has no labs". A change in the feed's shape must be an error.
    Unfamiliar *values* (a new lab code, status or risk level) are not errors here:
    the rules classify them and mark the affected check UNKNOWN.
    """

    model_config = ConfigDict(extra="forbid")


class PatientName(_Strict):

    given: str | None = None
    family: str | None = None

class PatientInfo(_Strict):

    id: str | None = None
    mrn: str | None = None
    name: PatientName | None = None
    dob: str | None = None
    sex: str | None = None

class ProcedureInfo(_Strict):

    case_id: str | None = None
    procedure_type: str | None = None
    procedure_risk: str | None = None  # checked against ProcedureRisk by the rules
    procedure_date: str | None = None
    is_elective: bool | None = None
    location: str | None = None

class Vital(_Strict):
    """One structured reading; `type` says which value fields apply."""

    type: str | None = None
    systolic: float | int | None = None
    diastolic: float | int | None = None
    value_f: float | int | None = None
    date: str | None = None
    source: str | None = None

class LabResult(_Strict):

    id: str | None = None
    code: str | None = None
    display: str | None = None
    effective_at: str | None = None
    status: str | None = None
    source: str | None = None

class Medication(_Strict):

    name: str | None = None
    active: bool | None = None

class Condition(_Strict):

    name: str | None = None
    active: bool | None = None

class Document(_Strict):

    doc_id: str | None = None
    type: str | None = None
    date: str | None = None
    author: str | None = None
    text: str | None = None

class SubmissionMetadata(_Strict):

    submission_received_at: str | None = None
    source_system: str | None = None

class PatientSubmission(_Strict):
    """Single submission package shape from the take-home prompt."""

    patient: PatientInfo | None = None
    procedure: ProcedureInfo | None = None
    vitals: list[Vital] = Field(default_factory=list)
    labs: list[LabResult] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    metadata: SubmissionMetadata | None = None

# -------------------------
# Evidence packet rows
# -------------------------

CheckId = Literal[
    "hp", "consent", "cbc", "cmp", "anticoagulation", "blood_pressure", "temperature"
]
CheckStatus = Literal["PASS", "FAIL", "UNKNOWN", "NOT_APPLICABLE"]
CHECK_ORDER: tuple[str, ...] = get_args(CheckId)


class Evidence(BaseModel):
    """One item a nurse can find in the chart."""

    source: Literal["structured", "document"]
    ref: str  # lab id, doc_id, or a position such as "vitals[3]"
    label: str  # what the nurse reads, e.g. "H&P [PDF] 02/21 · 81e2f72a"
    quote: str | None = None  # verbatim and verified by code; None for structured items


class Rejected(BaseModel):
    """Something considered and not used, with the reason."""

    label: str
    reason: str


class Check(BaseModel):
    """One row of the checklist. Every case carries all seven rows, READY included."""

    id: CheckId
    status: CheckStatus
    finding: str  # written by code: the value against its limit
    evidence: list[Evidence] = Field(default_factory=list)
    rejected: list[Rejected] = Field(default_factory=list)


class TriageIssue(BaseModel):
    """One check that did not pass. `description` and `evidence` are the two issue
    fields the stock report viewer prints, so both are plain text written by code."""

    category: IssueCategory
    evidence: str
    description: str = ""
    check_ids: list[CheckId] = Field(default_factory=list)


class TriageOutput(BaseModel):
    """Structured output contract for triage responses.

    `checks` is the source of truth. `issues` and `explanation` are derived from it by
    code, so the text a nurse reads cannot disagree with the decision.
    """

    decision: Decision
    issues: list[TriageIssue] = Field(validation_alias=AliasChoices("issues"))
    explanation: str
    checks: list[Check] = Field(default_factory=list)
