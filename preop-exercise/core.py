"""Pre-op triage entry point: validate the package, read the documents, apply the rules.

    submission  ->  models.PatientSubmission      strict validation; unknown fields fail closed
                ->  reader.read                   one model call (one retry): quoted facts, no decisions
                ->  rules.evaluate                pure code: seven checks, decision, issues, explanation

The model reads and the code decides. Whatever goes wrong, the result is never READY:
a submission that cannot be validated, or a model call that fails for good, returns
NEEDS_FOLLOW_UP with `system_failure` set, so the case still reaches a nurse.

Modules depend one way: models -> facts -> rules -> reader -> core.
"""

from __future__ import annotations

import os

from pydantic import ValidationError

import reader

# Re-exported: the harness scripts and tests import these names from core.
from models import (  # noqa: F401
    CHECK_ORDER,
    Check,
    CheckId,
    CheckStatus,
    Condition,
    Decision,
    Document,
    Evidence,
    IssueCategory,
    LabResult,
    Medication,
    PatientInfo,
    PatientName,
    PatientSubmission,
    ProcedureInfo,
    ProcedureRisk,
    Rejected,
    SubmissionMetadata,
    TriageIssue,
    TriageOutput,
    Vital,
)
from reader import OPENROUTER_BASE_URL, ReaderError, make_client  # noqa: F401
from rules import evaluate, reader_failed, unreadable


def validation_summary(exc: ValidationError, limit: int = 5) -> str:
    """Where a submission breaks the input contract: field paths and reasons only,
    never the offending values, which may be patient data."""

    errors = exc.errors()
    shown = [f"{'.'.join(str(part) for part in e['loc'])}: {e['msg']}" for e in errors[:limit]]
    more = f"; and {len(errors) - limit} more" if len(errors) > limit else ""
    return "; ".join(shown) + more


def triage_submission(
    submission: dict[str, object] | PatientSubmission,
    *,
    model: str,
) -> TriageOutput:
    """Triage one submission package. Always returns an output; never raises for a bad
    record or a failed model call."""

    try:
        parsed = (
            submission
            if isinstance(submission, PatientSubmission)
            else PatientSubmission.model_validate(submission)
        )
    except ValidationError as exc:
        return unreadable(validation_summary(exc))

    raw = submission if isinstance(submission, dict) else parsed.model_dump()
    # Optional: keep what the reader returned, for later re-scoring. Saved facts are
    # never read back here, so every call makes a fresh reading.
    facts_dir = os.environ.get("TRIAGE_FACTS_DIR")
    try:
        reading = reader.read(parsed, model=model)
    except ReaderError as exc:
        if facts_dir:
            reader.save_reading(facts_dir, None, raw, error=str(exc), model=model)
        return reader_failed(str(exc))
    if facts_dir:
        reader.save_reading(facts_dir, reading, raw)
    return evaluate(parsed, reading.facts)
