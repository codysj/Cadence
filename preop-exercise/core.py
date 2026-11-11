"""Shared policy, schema, and prompt helpers for the pre-op triage scripts."""

from __future__ import annotations

import json
import os
from pydantic import ValidationError

# -------------------------
# Policy + system prompt
# -------------------------

BASELINE_SYSTEM_PROMPT = """
You are a clinical operations assistant for pre-op scheduling triage.
Base decisions only on the policy below. General clinical knowledge may be used to interpret the record (e.g., drug brand names, unit conversions), not to add criteria.

Cadence Surgical Center Pre-Operative Scheduling Policy (effective Jan 1, 2026)

Output exactly one status:
- READY
- NEEDS_FOLLOW_UP
- NOT_CLEARED

Rule 1: Required documentation
- History and Physical (H&P) must exist and be completed within 30 days of procedure date.
- Signed Surgical Consent must exist.
If documentation is missing/outdated -> NEEDS_FOLLOW_UP.

Rule 2: Required testing by procedure risk
- LOW or MODERATE risk: CBC within 30 days of procedure date.
- HIGH risk: CBC within 14 days and CMP within 14 days.
Use only the most recent result for each required test.
If a required test is missing or outside window -> NEEDS_FOLLOW_UP.

Rule 3: Anticoagulation management
If the patient is currently taking an anticoagulant, a perioperative anticoagulation plan must be documented and clear.
If no clear plan is documented -> NEEDS_FOLLOW_UP.

Rule 4: Acute safety exclusions
If any of the following are present at review time -> NOT_CLEARED:
- Systolic BP >= 180 mmHg
- Diastolic BP >= 110 mmHg
- Temperature > 100.4 F
Use the most recent relevant vital.

Final determination
- READY only if all required criteria are satisfied and no exclusions are present.
- If a required field needed to evaluate a rule is missing/unknown -> NEEDS_FOLLOW_UP.

Output requirements
- Return exactly one JSON object.
""".strip()

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
from rules import unreadable

def triage_output_json_schema() -> dict[str, object]:
    """Return the JSON schema used for structured model outputs."""

    schema = TriageOutput.model_json_schema()
    return schema


# -------------------------
# Prompt helpers
# -------------------------


def build_user_prompt(submission: dict[str, object]) -> str:
    """Format a single submission package as user input."""

    sections = [
        "Evaluate this patient package for pre-op scheduling readiness using the provided policy.",
        "Return your response as JSON.",
        "Submission JSON:",
        json.dumps(submission, sort_keys=True),
    ]
    return "\n".join(sections)


def validation_summary(exc: ValidationError, limit: int = 5) -> str:
    """Where a submission breaks the input contract: field paths and reasons only,
    never the offending values, which may be patient data."""

    errors = exc.errors()
    shown = [f"{'.'.join(str(part) for part in e['loc'])}: {e['msg']}" for e in errors[:limit]]
    more = f"; and {len(errors) - limit} more" if len(errors) > limit else ""
    return "; ".join(shown) + more


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def make_client():
    """OpenAI client; OpenRouter's OpenAI-compatible endpoint if OPENROUTER_API_KEY is set.

    With OpenRouter, pass its model ids as `model` (e.g. "openai/gpt-6-luna").
    """

    # Import lazily so core utilities remain usable without OpenAI installed.
    from openai import OpenAI

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=openrouter_key)
    return OpenAI()


def triage_submission(
    submission: dict[str, object] | PatientSubmission,
    *,
    model: str,
) -> TriageOutput:
    """Naive baseline implementation: single LLM call with JSON response output."""

    try:
        parsed = (
            submission
            if isinstance(submission, PatientSubmission)
            else PatientSubmission.model_validate(submission)
        )
    except ValidationError as exc:
        # Fail closed for this one case: no model call, and the case goes to a nurse.
        return unreadable(validation_summary(exc))
    submission_payload = parsed.model_dump()

    client = make_client()
    request_kwargs: dict[str, object] = {
        "model": model,
        "instructions": BASELINE_SYSTEM_PROMPT,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_user_prompt(submission_payload),
                    }
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "preop_triage_output",
                "schema": triage_output_json_schema(),
                "strict": False,
            }
        },
    }

    response = client.responses.create(**request_kwargs)
    return TriageOutput.model_validate_json(response.output_text)
