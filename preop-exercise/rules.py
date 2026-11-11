"""Policy rules. Pure code: no network, no clock, no model.

evaluate(submission, facts) turns one patient package plus the reader's facts into the
seven-row checklist, and derives the decision, the issues and the explanation from those
rows. The same inputs always give the same output.

One principle decides every row:

    READY and NOT_CLEARED are both claims, and a claim needs established facts.
    A row gets PASS or FAIL only when that outcome holds under every way the open
    questions could resolve. Otherwise it is UNKNOWN, which means NEEDS_FOLLOW_UP.

"Open questions" are: a reader fact whose quote is not found in the document it cites,
something written in a document that the reader did not report, a value outside the
closed vocabularies below, and sources that disagree and cannot be put in order
(an undated document, two readings on one day with no time).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import get_args

from models import (
    CHECK_ORDER,
    Check,
    Evidence,
    PatientSubmission,
    ProcedureRisk,
    Rejected,
    TriageIssue,
    TriageOutput,
)
from facts import AnticoagulationPlan, Facts

# ---------------------------------------------------------------------------
# Policy numbers (Cadence Pre-Operative Scheduling Policy, effective 2026-01-01)
# ---------------------------------------------------------------------------

HP_LIMIT_DAYS = 30
LAB_LIMIT_DAYS = {"LOW": 30, "MODERATE": 30, "HIGH": 14}
CMP_REQUIRED_AT = {"HIGH"}
SYSTOLIC_LIMIT = 180  # at or above
DIASTOLIC_LIMIT = 110  # at or above
TEMPERATURE_LIMIT_F = 100.4  # strictly above

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

# ponytail: closed tables built from the 50 sample cases plus standard drug names.
# Grow them when the feed does; an unknown value surfaces as UNKNOWN for a nurse.

RISK_LEVELS: tuple[str, ...] = get_args(ProcedureRisk)

VITAL_TYPES = {"blood_pressure", "temperature"}

# Lab status -> does it count as a result? entered-in-error is skipped (labels 04, 06, 41).
LAB_STATUSES = {"final": True, "entered-in-error": False}

# Lab code -> the test it is. OTHER is a known test the policy does not require.
# BMP is a different panel from CMP and does not satisfy it (label 17).
_LAB_TESTS = {
    "CBC": "CBC", "LAB-CBC": "CBC",
    "CMP": "CMP", "LAB-CMP": "CMP",
    "BMP": "OTHER", "LAB-BMP": "OTHER",
    "PT-INR": "OTHER", "UA": "OTHER", "TSH": "OTHER", "T&S": "OTHER", "HBA1C": "OTHER",
}


def lab_test(code: str | None) -> str | None:
    """'CBC', 'CMP' or 'OTHER'; None for a code this system has never seen."""

    return _LAB_TESTS.get((code or "").strip().upper())


# Generic name -> names it is written under.
_ANTICOAGULANTS = {
    "warfarin": ("warfarin", "coumadin", "jantoven"),
    "apixaban": ("apixaban", "eliquis"),
    "rivaroxaban": ("rivaroxaban", "xarelto"),
    "dabigatran": ("dabigatran", "pradaxa"),
    "edoxaban": ("edoxaban", "savaysa"),
    "enoxaparin": ("enoxaparin", "lovenox"),
    "dalteparin": ("dalteparin", "fragmin"),
    "fondaparinux": ("fondaparinux", "arixtra"),
    "heparin": ("heparin",),
}
_ANTICOAGULANT_PATTERN = re.compile(
    r"\b(" + "|".join(n for names in _ANTICOAGULANTS.values() for n in names) + r")\b",
    re.IGNORECASE,
)
_GENERIC_OF = {n: generic for generic, names in _ANTICOAGULANTS.items() for n in names}

# Antiplatelets are not anticoagulants: the policy covers anticoagulants only, and the
# labels never ask for a plan for aspirin or clopidogrel (labels 24, 25).
_ANTIPLATELET_PATTERN = re.compile(
    r"\b(aspirin|asa|clopidogrel|plavix|ticagrelor|brilinta|prasugrel|effient|dipyridamole|cilostazol)\b",
    re.IGNORECASE,
)

# Document titles that announce an H&P or a consent. Used only to notice a document the
# reader skipped; the reader, not the title, decides what a document is.
_HP_TITLE = re.compile(r"\bh\s*(?:&|and)\s*p\b|\bhistory\s*(?:&|and)\s*p", re.IGNORECASE)
_CONSENT_TITLE = re.compile(r"consent", re.IGNORECASE)


def anticoagulants_named(text: str | None) -> set[str]:
    """Generic names of every anticoagulant written in `text`."""

    return {_GENERIC_OF[m.lower()] for m in _ANTICOAGULANT_PATTERN.findall(text or "")}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _day(value: str | None) -> date | None:
    """The calendar date as written. Windows count calendar days and ignore the time of day.

    Label 09 fails a CBC taken 15 calendar days before the procedure date, although it
    was taken less than 15 x 24 hours before that date begins, so elapsed-time arithmetic
    would pass it. Label 12 passes one taken exactly 30 calendar days before.
    """

    try:
        return date.fromisoformat((value or "")[:10])
    except ValueError:
        return None


def _clock(value: str | None) -> str | None:
    """'HH:MM' from an ISO timestamp or a clock time written in a note."""

    match = re.search(r"(?:T|^)\s*(\d{1,2}):(\d{2})", value or "")
    return f"{int(match.group(1)):02d}:{match.group(2)}" if match else None


def _norm(text: str | None) -> str:
    """Text as compared for quote checks: Unicode-normalised, case-folded, single-spaced."""

    return " ".join(unicodedata.normalize("NFKC", text or "").casefold().split())


def _days(n: int) -> str:
    return f"{n} day" if n == 1 else f"{n} days"


def _num(value: float) -> str:
    return f"{value:g}"


def _join(items: list[str]) -> str:
    return "; ".join(dict.fromkeys(items))  # de-duplicated, order kept


def _later(a: date | None, b: date | None) -> bool:
    """True when `a` is provably after `b`. An undated item is never earlier or later."""

    return a is not None and b is not None and a > b


def _writes_number(quote: str | None, value: float) -> bool:
    """True when `value` is one of the numbers written in `quote`, as a whole number:
    100 is not written in "Temp 100.9 F"."""

    return any(abs(float(n) - value) < 1e-9 for n in re.findall(r"\d+(?:\.\d+)?", quote or ""))


class _Case:
    """One submission with the lookups every rule needs."""

    def __init__(self, submission: PatientSubmission, facts: Facts) -> None:
        self.s = submission
        self.facts = facts
        self.docs = {d.doc_id: d for d in submission.documents if d.doc_id}
        procedure = submission.procedure
        self.procedure_day = _day(procedure.procedure_date) if procedure else None
        self.raw_risk = procedure.procedure_risk if procedure else None
        self.risk = self.raw_risk if self.raw_risk in RISK_LEVELS else None

    def quoted(self, doc_id: str, quote: str | None) -> bool:
        """True when `quote` really appears in the document it cites."""

        doc = self.docs.get(doc_id)
        wanted = _norm(quote)
        return bool(doc and wanted and wanted in _norm(doc.text))

    def label(self, doc_id: str | None) -> str:
        doc = self.docs.get(doc_id or "")
        if doc is None:
            return f"unknown document #{(doc_id or '')[:8]}"
        return f"{doc.type or 'untitled document'} {doc.date or 'undated'} #{doc.doc_id[:8]}"

    def doc_day(self, doc_id: str) -> date | None:
        doc = self.docs.get(doc_id)
        return _day(doc.date) if doc else None

    def cite(self, doc_id: str, quote: str | None) -> Evidence:
        return Evidence(source="document", ref=doc_id, label=self.label(doc_id), quote=quote)

    def risk_problem(self) -> str:
        if self.raw_risk is None:
            return "procedure risk level is missing"
        return f"unrecognised procedure risk level {self.raw_risk!r}"

    def unreported(self, title: re.Pattern, reported: set[str], kind: str) -> list[str]:
        """Documents whose title announces `kind` but from which the reader reported nothing."""

        return [
            f"{self.label(d.doc_id)} looks like {kind} but the reader reported nothing from it; check that document"
            for d in self.s.documents
            if title.search(d.type or "") and d.doc_id not in reported
        ]


def _window(c: _Case, day: date, limit: int) -> tuple[str, str]:
    """Status and wording for 'within `limit` days of the procedure date'."""

    if c.procedure_day is None:
        return "UNKNOWN", f"dated {day}; procedure date is missing, so the {limit}-day window cannot be checked"
    days = (c.procedure_day - day).days
    if days < 0:
        return "UNKNOWN", f"dated {day}, which is after the procedure date {c.procedure_day}"
    wording = f"dated {day}, {_days(days)} before the procedure, limit {limit}"
    if days > limit:
        return "FAIL", wording
    return "PASS", wording + (" (at the limit)" if days == limit else "")


# ---------------------------------------------------------------------------
# Rule 1: required documentation
# ---------------------------------------------------------------------------


def _hp(c: _Case) -> Check:
    usable, doubt, rejected = [], [], []
    for fact in c.facts.hp_candidates:
        label = c.label(fact.doc_id)
        # Setting a document aside is a claim too, so it needs a verified quote like any other.
        if not c.quoted(fact.doc_id, fact.quote):
            doubt.append(f"could not verify the automated reading of {label}; check that document")
        elif not fact.is_preop_evaluation:
            rejected.append(Rejected(label=label, reason="not the pre-op evaluation"))
        elif c.doc_day(fact.doc_id) is None:
            doubt.append(f"{label} has no usable date")
        else:
            usable.append((c.doc_day(fact.doc_id), fact))
    doubt += c.unreported(_HP_TITLE, {f.doc_id for f in c.facts.hp_candidates}, "an H&P")

    # A pass needs one verified H&P inside the window; nothing unread can undo that.
    # A fail claims there is none, which an unread or unverified document could overturn.
    if not usable:
        if doubt:
            return Check(id="hp", status="UNKNOWN", finding=_join(doubt), rejected=rejected)
        return Check(id="hp", status="FAIL", finding="no pre-op H&P on file", rejected=rejected)

    usable.sort(key=lambda item: item[0])
    day, fact = usable[-1]
    rejected += [Rejected(label=c.label(f.doc_id), reason="older H&P") for _, f in usable[:-1]]
    status, finding = _window(c, day, HP_LIMIT_DAYS)
    if status == "FAIL" and doubt:
        status, finding = "UNKNOWN", _join([finding, *doubt])
    return Check(id="hp", status=status, finding=finding, evidence=[c.cite(fact.doc_id, fact.quote)], rejected=rejected)


def _consent(c: _Case) -> Check:
    valid, doubt, problems = [], [], []
    for fact in c.facts.consents:
        label = c.label(fact.doc_id)
        quotes = [q for q in (fact.signature_quote, fact.procedure_quote) if q is not None]
        # Every claim the row relies on needs a verified quote, whether it accepts the
        # consent or sets it aside: "unsigned" with nothing quoted is not established.
        signature_claimed = fact.signature_quote is not None
        both_claimed = signature_claimed and fact.procedure_quote is not None
        if any(not c.quoted(fact.doc_id, q) for q in quotes):
            doubt.append(f"could not verify the automated reading of {label}; check that document")
        elif fact.patient_signed != "yes":
            reason = "not signed by the patient" if fact.patient_signed == "no" else "patient signature unclear"
            if signature_claimed:
                problems.append((fact, reason, fact.signature_quote))
            else:
                doubt.append(f"the reading of {label} says it is {reason} but quotes nothing; check that document")
        elif not both_claimed:
            doubt.append(f"the reading of {label} gave no quote for the signature or the procedure; check that document")
        elif fact.matches_scheduled_procedure == "yes":
            valid.append(fact)
        else:
            reason = (
                "names a different procedure"
                if fact.matches_scheduled_procedure == "no"
                else "does not clearly name the scheduled procedure"
            )
            problems.append((fact, reason, fact.procedure_quote))
    doubt += c.unreported(_CONSENT_TITLE, {f.doc_id for f in c.facts.consents}, "a consent")

    by_date = lambda fact: c.doc_day(fact.doc_id) or date.min  # noqa: E731
    if valid:
        fact = max(valid, key=by_date)
        return Check(
            id="consent",
            status="PASS",
            finding="signed by the patient for the scheduled procedure",
            evidence=[c.cite(fact.doc_id, fact.signature_quote), c.cite(fact.doc_id, fact.procedure_quote)],
            rejected=[Rejected(label=c.label(f.doc_id), reason=f"{reason}; a signed consent is on file") for f, reason, _ in problems],
        )
    if doubt:
        return Check(id="consent", status="UNKNOWN", finding=_join(doubt))
    if problems:
        fact, reason, quote = max(problems, key=lambda item: by_date(item[0]))
        return Check(
            id="consent",
            status="FAIL",
            finding=f"{reason}: {c.label(fact.doc_id)}",
            evidence=[c.cite(fact.doc_id, quote)] if quote else [],
        )
    return Check(id="consent", status="FAIL", finding="no surgical consent on file")


# ---------------------------------------------------------------------------
# Rule 2: required testing
# ---------------------------------------------------------------------------


def _lab(c: _Case, check_id: str, test: str) -> Check:
    if test == "CMP" and c.risk is not None and c.risk not in CMP_REQUIRED_AT:
        return Check(id=check_id, status="NOT_APPLICABLE", finding=f"not required at {c.risk} risk")

    valid, doubt, rejected = [], [], []
    for lab in c.s.labs:
        kind = lab_test(lab.code)
        day = _day(lab.effective_at)
        label = f"{lab.code} {day or 'undated'} lab #{(lab.id or 'no id')[:8]}"
        if kind is None:
            doubt.append(f"unrecognised lab code {lab.code!r}; check it manually")
        elif kind != test:
            if test == "CMP" and "BMP" in (lab.code or "").upper():
                rejected.append(Rejected(label=label, reason="a BMP is not a CMP"))
        elif LAB_STATUSES.get(lab.status) is None:
            doubt.append(f"{label} has unrecognised status {lab.status!r}")
        elif not LAB_STATUSES[lab.status]:
            rejected.append(Rejected(label=label, reason="entered in error"))
        elif day is None:
            doubt.append(f"{label} has no usable date")
        else:
            valid.append((day, lab, label))

    def row(status: str, finding: str, evidence: list[Evidence] | None = None) -> Check:
        # A pass rests on one valid result inside the window; an unrecognised extra result
        # cannot undo it. A fail claims there is no such result, which it could.
        if status == "FAIL" and doubt:
            status, finding = "UNKNOWN", _join([finding, *doubt])
        return Check(id=check_id, status=status, finding=finding, evidence=evidence or [], rejected=rejected)

    if test == "CMP" and c.risk is None:
        return row("UNKNOWN", f"{c.risk_problem()}, so it is not known whether a CMP is required")
    if not valid:
        return row("FAIL", f"no {test} on file")

    valid.sort(key=lambda item: item[0])
    day, lab, label = valid[-1]
    rejected += [Rejected(label=older, reason="older result") for _, _, older in valid[:-1]]
    evidence = [Evidence(source="structured", ref=lab.id or label, label=label)]

    if c.risk is None:
        # The limit is 30 or 14 days. Older than 30 fails either way; otherwise it is unknown.
        status, finding = _window(c, day, max(LAB_LIMIT_DAYS.values()))
        if status == "PASS":
            status, finding = "UNKNOWN", f"dated {day}; {c.risk_problem()}, so the limit (30 or 14 days) is not known"
        return row(status, finding, evidence)

    status, finding = _window(c, day, LAB_LIMIT_DAYS[c.risk])
    return row(status, finding, evidence)


# ---------------------------------------------------------------------------
# Rule 3: anticoagulation management
# ---------------------------------------------------------------------------

_SAYS = {
    "current": "is being taken",
    "stopped": "was stopped",
    "planned": "is planned, not yet taken",
    "unknown": "may or may not still be taken",
}
_STATE = {"current": "taken", "stopped": "not taken", "planned": "not taken", "unknown": "open"}


@dataclass(frozen=True)
class _Said:
    """What one document says about one drug. state: 'taken', 'not taken' or 'open'."""

    day: date | None
    state: str
    sentence: str
    evidence: Evidence | None = None
    rejected: Rejected | None = None


def _plan_for(c: _Case, generic: str) -> tuple[str, list[Evidence], str]:
    """('complete' | 'open' | 'gap', evidence, wording) for the plan covering `generic`.

    A reported plan counts only once it is verified: it cites a real document, every
    quote it gives is found there, it gives at least one, and each instruction it calls
    concrete is quoted. A plan that is not verified is an open question. It never fails
    the row and never outranks a verified plan; if it could be the newest, the row is
    UNKNOWN.

    Among verified plans the newest decides. A later plan that defers or drops an
    instruction makes the management unclear even if an earlier plan was complete, and
    Rule 3 treats an incomplete or ambiguous plan the same as no plan.

    Known exception to "every claim is quoted": an instruction reported as absent has
    nothing to quote. It is accepted on the strength of the plan's other, verified quote.
    """

    def day(plan: AnticoagulationPlan) -> date | None:
        return c.doc_day(plan.doc_id)

    def is_verified(plan: AnticoagulationPlan) -> bool:
        quotes = [q for q in (plan.preop_quote, plan.postop_quote) if q is not None]
        concrete_unquoted = (plan.preop == "concrete" and plan.preop_quote is None) or (
            plan.postop == "concrete" and plan.postop_quote is None
        )
        return bool(quotes) and not concrete_unquoted and all(c.quoted(plan.doc_id, q) for q in quotes)

    def newest_first(group: list[AnticoagulationPlan]) -> list[AnticoagulationPlan]:
        return sorted(group, key=lambda p: day(p) or date.max, reverse=True)

    plans = [p for p in c.facts.anticoagulation_plans if generic in anticoagulants_named(p.generic_name)]
    verified = [p for p in plans if is_verified(p)]
    could_be_newest = [p for p in plans if not is_verified(p) and not any(_later(day(v), day(p)) for v in verified)]
    if could_be_newest:
        return "open", [], f"could not verify the plan reported from {c.label(could_be_newest[0].doc_id)}; check that document"
    if not verified:
        others = sorted({g for p in c.facts.anticoagulation_plans for g in anticoagulants_named(p.generic_name)} - {generic})
        return "gap", [], "no documented plan" + (f" (the plan on file is for {', '.join(others)})" if others else "")

    standing = [p for p in verified if not any(_later(day(o), day(p)) for o in verified)]
    incomplete = newest_first([p for p in standing if p.preop != "concrete" or p.postop != "concrete"])
    if incomplete:
        plan = incomplete[0]
        gaps = [f"{when} instruction {state}" for when, state in (("pre-op", plan.preop), ("post-op", plan.postop)) if state != "concrete"]
        wording = f"the plan in {c.label(plan.doc_id)} is incomplete: {' and '.join(gaps)}"
        if len(verified) > 1:
            wording += "; it is the newest plan on file, or cannot be shown to be older than the others"
        quotes = [q for q in (plan.preop_quote, plan.postop_quote) if q is not None]
        return "gap", [c.cite(plan.doc_id, q) for q in quotes], wording

    plan = newest_first(standing)[0]
    return "complete", [c.cite(plan.doc_id, plan.preop_quote), c.cite(plan.doc_id, plan.postop_quote)], ""


def _anticoagulation(c: _Case) -> Check:
    doubt: list[str] = []
    rejected: list[Rejected] = []

    # 1. What the structured medication list says.
    listed: dict[str, list[tuple[bool | None, Evidence]]] = {}
    for index, med in enumerate(c.s.medications):
        entry = Evidence(source="structured", ref=f"medications[{index}]", label=f"medication list: {med.name}")
        generics = anticoagulants_named(med.name)
        if not generics and _ANTIPLATELET_PATTERN.search(med.name or ""):
            rejected.append(Rejected(label=entry.label, reason="antiplatelet, not an anticoagulant"))
        for generic in generics:
            listed.setdefault(generic, []).append((med.active, entry))

    # 2. The reader's view of the same list: a second net under the drug table above.
    for entry_text in c.facts.medication_list_anticoagulants:
        if not anticoagulants_named(entry_text) and not _ANTIPLATELET_PATTERN.search(entry_text):
            doubt.append(f"the reader considers {entry_text!r} an anticoagulant, but it is not in this system's drug table")

    # 3. What each document says. An unverified mention, and a drug named in a document
    # the reader reported nothing about, are open statements: they could say either thing.
    said: dict[str, list[_Said]] = {}
    for fact in c.facts.anticoagulant_mentions:
        label = c.label(fact.doc_id)
        generics = anticoagulants_named(fact.drug_as_written) | anticoagulants_named(fact.generic_name)
        if not generics:
            if not _ANTIPLATELET_PATTERN.search(f"{fact.drug_as_written} {fact.generic_name}"):
                doubt.append(f"{fact.drug_as_written!r} in {label} is not in this system's drug table")
            continue
        day = c.doc_day(fact.doc_id)
        if c.quoted(fact.doc_id, fact.quote):
            statement = _Said(
                day,
                _STATE[fact.status],
                f"{label} says {fact.drug_as_written} {_SAYS[fact.status]}",
                c.cite(fact.doc_id, fact.quote),
                Rejected(label=f"{fact.drug_as_written} in {label}", reason=fact.status),
            )
        else:
            statement = _Said(day, "open", f"could not verify a mention of {fact.drug_as_written} in {label}; check that document")
        for generic in generics:
            said.setdefault(generic, []).append(statement)
    for doc in c.s.documents:
        reported: set[str] = set()
        for fact in c.facts.anticoagulant_mentions:
            if fact.doc_id == doc.doc_id:
                reported |= anticoagulants_named(fact.drug_as_written) | anticoagulants_named(fact.generic_name)
        for generic in anticoagulants_named(doc.text) - reported:
            sentence = f"{generic} is named in {c.label(doc.doc_id)} but the reader did not report it; check that document"
            said.setdefault(generic, []).append(_Said(_day(doc.date), "open", sentence))

    # 4. Is each drug being taken? Later documents supersede earlier ones. Statements that
    # are left must agree with each other and with the list; if not, it is not established.
    taken: dict[str, Evidence] = {}
    from_note_only: set[str] = set()
    unsettled: dict[str, str] = {}
    for generic in sorted(set(listed) | set(said)):
        statements = said.get(generic, [])
        standing = [s for s in statements if not any(_later(o.day, s.day) for o in statements)]
        entries = listed.get(generic, [])
        flags = [active for active, _ in entries]
        on_list = None if not entries else "taken" if True in flags else "open" if None in flags else "not taken"
        votes = {s.state for s in standing}
        if on_list == "open" and votes and "open" not in votes:
            on_list = None  # the list does not say; the documents do
        if on_list:
            votes.add(on_list)

        if votes == {"taken"}:
            listed_active = next((entry for active, entry in entries if active), None)
            taken[generic] = listed_active or max(standing, key=lambda s: s.day or date.max).evidence
            if listed_active is None:
                from_note_only.add(generic)
        elif votes == {"not taken"}:
            rejected += [Rejected(label=entry.label, reason="inactive on the medication list") for _, entry in entries]
            rejected += [s.rejected for s in standing if s.rejected]
        else:
            parts = []
            if entries:
                names = ", ".join(entry.label.removeprefix("medication list: ") for _, entry in entries)
                parts.append(
                    f"the medication list does not say whether {names} is active"
                    if None in flags and True not in flags
                    else f"the medication list has {names} as {'active' if True in flags else 'inactive'}"
                )
            parts += [s.sentence for s in standing]
            unsettled[generic] = f"whether {generic} is being taken is not established: {_join(parts)}"

    # 5. Every drug that is, or may be, taken needs a plan covering before and after.
    evidence: list[Evidence] = []
    failures: list[str] = []
    covered: list[str] = []
    covered_either_way: list[str] = []
    for generic in sorted(taken):
        evidence.append(taken[generic])
        state, cited, wording = _plan_for(c, generic)
        evidence += cited
        if state == "complete":
            covered.append(generic)
        elif state == "open":
            doubt.append(wording)
        else:
            failures.append(f"taking {generic}: {wording}")
    for generic in sorted(unsettled):
        state, cited, wording = _plan_for(c, generic)
        if state == "complete":
            covered_either_way.append(generic)  # the row holds whether or not it is taken
            evidence += cited
        else:
            doubt.append(f"{unsettled[generic]}; {wording}")

    if failures:  # true however the open questions about other drugs resolve
        return Check(id="anticoagulation", status="FAIL", finding=_join(failures), evidence=evidence, rejected=rejected)
    if doubt:
        return Check(id="anticoagulation", status="UNKNOWN", finding=_join(doubt), evidence=evidence, rejected=rejected)
    if not covered and not covered_either_way:
        finding = f"no anticoagulant currently taken ({len(c.s.medications)} medications and {len(c.s.documents)} documents checked)"
        return Check(id="anticoagulation", status="NOT_APPLICABLE", finding=finding, rejected=rejected)

    parts = []
    if covered:
        parts.append(f"taking {', '.join(covered)}; plan covers before and after the procedure")
    if from_note_only:
        parts.append(f"{', '.join(sorted(from_note_only))} found in a note, not on the medication list")
    if covered_either_way:
        parts.append(f"a complete plan covers {', '.join(covered_either_way)}, whether or not it is still taken")
    return Check(id="anticoagulation", status="PASS", finding=_join(parts), evidence=evidence, rejected=rejected)


# ---------------------------------------------------------------------------
# Rule 4: acute safety exclusions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Reading:
    day: date
    clock: str | None
    breach: bool
    text: str
    evidence: Evidence

    def later_than(self, other: "_Reading") -> bool:
        """Provably later. Two readings on one day compare only if both carry a time.

        Times are compared as written, with no time-zone conversion: structured vitals are
        UTC and note times are local. In the sample, cases 08, 09, 20 and 38 have a
        structured vital and a note reading on the same day, and the order does not change
        any of their decisions.
        """

        if self.day != other.day:
            return self.day > other.day
        return self.clock is not None and other.clock is not None and self.clock > other.clock

    @property
    def when(self) -> str:
        return f"{self.day}" + (f" {self.clock}" if self.clock else " (time not stated)")


# Blood-pressure-shaped and temperature-shaped text, used only to check that the reader
# reported what is written. They never supply a reading themselves.
# ponytail: plausibility-filtered patterns; a "92 F" meaning a 92-year-old woman would
# false-alarm into UNKNOWN. Tighten with context words if stage 5 shows false alarms.
_BP_SHAPE = re.compile(r"(?<![\d/.])(\d{2,3})\s*/\s*(\d{2,3})(?![\d/.])")
_TEMP_SHAPE = re.compile(r"(?<![\d.])(\d{2,3}(?:\.\d+)?)\s*(?:°|deg(?:rees)?\.?)?\s*([CF])\b")


def _fahrenheit(value: float, unit: str) -> float:
    return round(value * 9 / 5 + 32 if unit == "C" else value, 2)


def blood_pressures_written(text: str | None) -> list[tuple[int, int]]:
    """Every plausible blood pressure written in `text`, as (systolic, diastolic).
    Dates such as 03/15/2026 and scores such as 6/10 are not blood pressures."""

    found = []
    for match in _BP_SHAPE.finditer(unicodedata.normalize("NFKC", text or "")):
        systolic, diastolic = int(match.group(1)), int(match.group(2))
        if 60 <= systolic <= 260 and 30 <= diastolic <= 160 and systolic > diastolic:
            found.append((systolic, diastolic))
    return found


def temperatures_written(text: str | None) -> list[tuple[float, str]]:
    """Every plausible body temperature written in `text`, as (value, 'C' or 'F')."""

    found = []
    for match in _TEMP_SHAPE.finditer(unicodedata.normalize("NFKC", text or "")):
        value, unit = float(match.group(1)), match.group(2)
        if (30 <= value <= 45) if unit == "C" else (90 <= value <= 110):
            found.append((value, unit))
    return found


def temperature_reported(value: float, unit: str, reported_fahrenheit: list[float]) -> bool:
    """True when a written temperature matches a reported one, in either unit:
    "38.6 C (101.5 F)" is one reading written twice."""

    return any(abs(_fahrenheit(value, unit) - reported) <= 0.15 for reported in reported_fahrenheit)


def _could_be_latest(open_items: list[tuple[date | None, str]], readings: list[_Reading]) -> list[str]:
    """The open items (an unverified reading, or one written but not reported) that could
    be the most recent reading. One from a document dated before a verified reading
    cannot be, so it cannot change the row. Only the document's date is trusted here:
    a clock time claimed by an unverified reading is not."""

    newest = max((r.day for r in readings), default=None)
    return [sentence for day, sentence in open_items if newest is None or day is None or day >= newest]


def _structured_vitals(c: _Case, kind: str, doubt: list[str]):
    for index, vital in enumerate(c.s.vitals):
        if vital.type not in VITAL_TYPES:
            doubt.append(f"unrecognised vital type {vital.type!r} in vitals[{index}]; check it manually")
        elif vital.type == kind:
            yield index, vital


def _vital_row(check_id: str, name: str, limits: str, readings: list[_Reading], doubt: list[str]) -> Check:
    if not readings:
        finding = _join(doubt) if doubt else f"no {name} reading in structured vitals or notes"
        return Check(id=check_id, status="UNKNOWN", finding=finding)

    # Only the most recent reading counts. `standing` holds every reading that no other
    # reading is provably later than; usually that is exactly one.
    standing = sorted(
        (r for r in readings if not any(o.later_than(r) for o in readings)),
        key=lambda r: (r.day, r.clock or ""),
        reverse=True,
    )
    superseded = sorted((r for r in readings if r not in standing), key=lambda r: (r.day, r.clock or ""), reverse=True)
    rejected = [
        Rejected(label=f"{r.text} on {r.when} ({r.evidence.label})", reason="breach superseded by a later reading" if r.breach else "older reading")
        for r in superseded
    ]
    evidence = [r.evidence for r in standing]

    breaching = [r for r in standing if r.breach]
    if breaching and len(breaching) < len(standing):
        # A breach and an in-range reading, and no way to tell which came last.
        described = " and ".join(f"{r.text} on {r.when}" for r in standing)
        finding = f"the most recent reading is not established: {described} cannot be put in order, {limits}"
        return Check(id=check_id, status="UNKNOWN", finding=_join([finding, *doubt]), evidence=evidence, rejected=rejected)

    reading = standing[0]
    finding = f"{reading.text} on {reading.when}, {limits}"
    if not breaching and any(r.breach for r in superseded):
        finding += " (earlier breach superseded by this reading)"
    if doubt:
        # Whether it clears or holds the patient, the claim needs this to be the latest reading.
        verdict = "breaches the limit" if breaching else "is in range"
        finding = _join([f"the latest verified reading {verdict} ({finding}), but it may not be the latest", *doubt])
        return Check(id=check_id, status="UNKNOWN", finding=finding, evidence=evidence, rejected=rejected)
    return Check(id=check_id, status="FAIL" if breaching else "PASS", finding=finding, evidence=evidence, rejected=rejected)


def _blood_pressure(c: _Case) -> Check:
    doubt: list[str] = []
    readings: list[_Reading] = []

    def reading(day: date, clock: str | None, systolic: float, diastolic: float, evidence: Evidence) -> _Reading:
        breach = systolic >= SYSTOLIC_LIMIT or diastolic >= DIASTOLIC_LIMIT
        return _Reading(day, clock, breach, f"{_num(systolic)}/{_num(diastolic)}", evidence)

    for index, vital in _structured_vitals(c, "blood_pressure", doubt):
        day = _day(vital.date)
        if day is None or vital.systolic is None or vital.diastolic is None:
            doubt.append(f"structured blood pressure vitals[{index}] is incomplete")
            continue
        source = Evidence(source="structured", ref=f"vitals[{index}]", label=f"structured vital, {vital.source or 'source not given'}")
        readings.append(reading(day, _clock(vital.date), vital.systolic, vital.diastolic, source))

    open_items: list[tuple[date | None, str]] = []
    for fact in c.facts.note_blood_pressures:
        label = c.label(fact.doc_id)
        day = c.doc_day(fact.doc_id)
        written = re.search(rf"(?<!\d){fact.systolic}\s*/\s*{fact.diastolic}(?!\d)", _norm(fact.quote))
        if not (c.quoted(fact.doc_id, fact.quote) and written):
            open_items.append((day, f"could not verify {fact.systolic}/{fact.diastolic} reported from {label}; check that document"))
        elif day is None:
            doubt.append(f"{label} has no usable date for {fact.systolic}/{fact.diastolic}")
        else:
            readings.append(reading(day, _clock(fact.time), fact.systolic, fact.diastolic, c.cite(fact.doc_id, fact.quote)))

    for doc in c.s.documents:
        reported = {(f.systolic, f.diastolic) for f in c.facts.note_blood_pressures if f.doc_id == doc.doc_id}
        for pair in blood_pressures_written(doc.text):
            if pair not in reported:
                open_items.append((_day(doc.date), f"{pair[0]}/{pair[1]} is written in {c.label(doc.doc_id)} but the reader did not report it; check that document"))
    doubt += _could_be_latest(open_items, readings)

    limits = f"limits {SYSTOLIC_LIMIT}/{DIASTOLIC_LIMIT}"
    return _vital_row("blood_pressure", "blood pressure", limits, readings, doubt)


def _temperature(c: _Case) -> Check:
    doubt: list[str] = []
    readings: list[_Reading] = []

    def reading(day: date, clock: str | None, value: float, unit: str, evidence: Evidence) -> _Reading:
        fahrenheit = _fahrenheit(value, unit)
        text = f"{_num(value)} C = {fahrenheit:.1f} F" if unit == "C" else f"{_num(value)} F"
        return _Reading(day, clock, fahrenheit > TEMPERATURE_LIMIT_F, text, evidence)

    for index, vital in _structured_vitals(c, "temperature", doubt):
        day = _day(vital.date)
        if day is None or vital.value_f is None:
            doubt.append(f"structured temperature vitals[{index}] is incomplete")
            continue
        source = Evidence(source="structured", ref=f"vitals[{index}]", label=f"structured vital, {vital.source or 'source not given'}")
        readings.append(reading(day, _clock(vital.date), vital.value_f, "F", source))

    open_items: list[tuple[date | None, str]] = []
    for fact in c.facts.note_temperatures:
        label = c.label(fact.doc_id)
        day = c.doc_day(fact.doc_id)
        if not (c.quoted(fact.doc_id, fact.quote) and _writes_number(fact.quote, fact.value)):
            open_items.append((day, f"could not verify {_num(fact.value)} {fact.unit} reported from {label}; check that document"))
        elif day is None:
            doubt.append(f"{label} has no usable date for {_num(fact.value)} {fact.unit}")
        else:
            readings.append(reading(day, _clock(fact.time), fact.value, fact.unit, c.cite(fact.doc_id, fact.quote)))

    for doc in c.s.documents:
        reported = [_fahrenheit(f.value, f.unit) for f in c.facts.note_temperatures if f.doc_id == doc.doc_id]
        for value, unit in temperatures_written(doc.text):
            if not temperature_reported(value, unit, reported):
                open_items.append((_day(doc.date), f"{_num(value)} {unit} is written in {c.label(doc.doc_id)} but the reader did not report it; check that document"))
    doubt += _could_be_latest(open_items, readings)

    limits = f"limit {TEMPERATURE_LIMIT_F} F"
    return _vital_row("temperature", "temperature", limits, readings, doubt)


# ---------------------------------------------------------------------------
# What a reading leaves open
# ---------------------------------------------------------------------------


def open_items(submission: PatientSubmission, facts: Facts) -> list[str]:
    """Everything a complete, verifiable reading must not leave open, across ALL documents.

    Stricter than the rules need: the rules ignore an unreported reading in a document
    too old to hold the latest value, but a reference reading should report it anyway.
    """

    c = _Case(submission, facts)
    problems: list[str] = []

    def quote(doc_id: str, text: str | None, what: str) -> None:
        if doc_id not in c.docs:
            problems.append(f"{what}: document {doc_id!r} does not exist")
        elif text is not None and not c.quoted(doc_id, text):
            problems.append(f"{what}: quote not found in {c.label(doc_id)}: {text!r}")

    for f in facts.hp_candidates:
        quote(f.doc_id, f.quote, "hp_candidate")
    for f in facts.consents:
        quote(f.doc_id, f.signature_quote, "consent signature")
        quote(f.doc_id, f.procedure_quote, "consent procedure")
        if f.signature_quote is None:
            problems.append(f"consent in {c.label(f.doc_id)}: signature_quote is required whatever the status")
        if f.patient_signed == "yes" and f.procedure_quote is None:
            problems.append(f"consent in {c.label(f.doc_id)}: procedure_quote is required")
    for f in facts.note_blood_pressures:
        quote(f.doc_id, f.quote, f"blood pressure {f.systolic}/{f.diastolic}")
        if not re.search(rf"(?<!\d){f.systolic}\s*/\s*{f.diastolic}(?!\d)", _norm(f.quote)):
            problems.append(f"blood pressure {f.systolic}/{f.diastolic}: not written in its quote {f.quote!r}")
    for f in facts.note_temperatures:
        quote(f.doc_id, f.quote, f"temperature {f.value} {f.unit}")
        if not _writes_number(f.quote, f.value):
            problems.append(f"temperature {f.value} {f.unit}: not written in its quote {f.quote!r}")
    for f in facts.anticoagulant_mentions:
        quote(f.doc_id, f.quote, f"mention of {f.drug_as_written}")
        if not anticoagulants_named(f"{f.drug_as_written} {f.generic_name}"):
            problems.append(f"mention of {f.drug_as_written!r}: not in the drug table (antiplatelets are not reported)")
    for f in facts.anticoagulation_plans:
        quote(f.doc_id, f.preop_quote, f"{f.generic_name} plan, pre-op")
        quote(f.doc_id, f.postop_quote, f"{f.generic_name} plan, post-op")
        for when, state, text in (("pre-op", f.preop, f.preop_quote), ("post-op", f.postop, f.postop_quote)):
            if state != "absent" and text is None:
                problems.append(f"{f.generic_name} plan in {c.label(f.doc_id)}: {when} is {state} but has no quote")

    listed = {m.name for m in submission.medications if anticoagulants_named(m.name)}
    reported_listed = set(facts.medication_list_anticoagulants)
    for name in sorted(listed - reported_listed):
        problems.append(f"medication list entry {name!r} is an anticoagulant but is not in medication_list_anticoagulants")
    for name in sorted(reported_listed - {m.name for m in submission.medications}):
        problems.append(f"medication_list_anticoagulants has {name!r}, which is not an entry of the medication list (copy it verbatim)")

    problems += c.unreported(_HP_TITLE, {f.doc_id for f in facts.hp_candidates}, "an H&P")
    problems += c.unreported(_CONSENT_TITLE, {f.doc_id for f in facts.consents}, "a consent")

    for doc in submission.documents:
        label = c.label(doc.doc_id)
        text = unicodedata.normalize("NFKC", doc.text or "")
        named = anticoagulants_named(text)
        reported: set[str] = set()
        for f in facts.anticoagulant_mentions:
            if f.doc_id == doc.doc_id:
                reported |= anticoagulants_named(f"{f.drug_as_written} {f.generic_name}")
        for generic in sorted(named - reported):
            problems.append(f"{generic} is named in {label} but there is no mention for it from that document")

        pairs = {(f.systolic, f.diastolic) for f in facts.note_blood_pressures if f.doc_id == doc.doc_id}
        for pair in blood_pressures_written(doc.text):
            if pair not in pairs:
                problems.append(f"{pair[0]}/{pair[1]} is written in {label} but is not reported")
        temps = [_fahrenheit(f.value, f.unit) for f in facts.note_temperatures if f.doc_id == doc.doc_id]
        for value, unit in temperatures_written(doc.text):
            if not temperature_reported(value, unit, temps):
                problems.append(f"{value:g} {unit} is written in {label} but is not reported")
    return problems


# ---------------------------------------------------------------------------
# Decision, issues and the text a nurse reads
# ---------------------------------------------------------------------------

_TITLE = {
    "hp": "H&P",
    "consent": "Consent",
    "cbc": "CBC",
    "cmp": "CMP",
    "anticoagulation": "Anticoagulation",
    "blood_pressure": "Blood pressure",
    "temperature": "Temperature",
}
_FAIL_CATEGORY = {
    "hp": "REQUIRED_DOCUMENTATION",
    "consent": "REQUIRED_DOCUMENTATION",
    "cbc": "REQUIRED_TESTING",
    "cmp": "REQUIRED_TESTING",
    "anticoagulation": "ANTICOAGULATION_MANAGEMENT",
    "blood_pressure": "ACUTE_SAFETY_EXCLUSION",
    "temperature": "ACUTE_SAFETY_EXCLUSION",
}
_STATUS_WORD = {"PASS": "PASS", "FAIL": "FAIL", "UNKNOWN": "UNKNOWN", "NOT_APPLICABLE": "N/A"}
# Rejection reasons that only mean "a newer one exists". Routine, so left out of issue evidence.
_SUPERSEDED_BY_AGE = {"older H&P", "older result", "older reading"}


def _issue_evidence(check: Check) -> str:
    """One line for the issue: what was found, then anything set aside for a reason a
    nurse would want to see (entered in error, a BMP offered for a CMP, a wellness visit).
    Items passed over only because something newer exists stay in the explanation."""

    found = [e.label + (f' "{e.quote}"' if e.quote else "") for e in check.evidence]
    set_aside = [f"rejected: {r.label} ({r.reason})" for r in check.rejected if r.reason not in _SUPERSEDED_BY_AGE]
    return "; ".join(found + set_aside) or "nothing on file to cite"


def render(decision: str, checks: list[Check], submission: PatientSubmission) -> str:
    """The checklist as stacked short lines, for the report viewer's narrow text pane."""

    procedure = submission.procedure
    attention = [_TITLE[k.id] for k in checks if k.status in ("FAIL", "UNKNOWN")]
    summary = "all applicable checks pass" if not attention else f"needs attention: {', '.join(attention)}"
    header = " | ".join(
        [
            (procedure.case_id if procedure else None) or "case id missing",
            (procedure.procedure_type if procedure else None) or "procedure missing",
            f"procedure date {(procedure.procedure_date if procedure else None) or 'missing'}",
            f"risk {(procedure.procedure_risk if procedure else None) or 'missing'}",
        ]
    )
    lines = [f"{decision}: {summary}", header]
    for check in checks:
        lines.append("")  # the viewer wraps long lines to column 0; a gap keeps each row distinct
        lines.append(f"{_STATUS_WORD[check.status]:<8}{_TITLE[check.id]}: {check.finding}")
        for evidence in check.evidence:
            lines.append(f"    source: {evidence.label}")
            if evidence.quote:
                lines.append(f'      "{evidence.quote}"')
        lines += [f"    rejected: {r.label}: {r.reason}" for r in check.rejected]
    return "\n".join(lines)


def evaluate(submission: PatientSubmission, facts: Facts) -> TriageOutput:
    """Apply the policy. READY is returned only when every row is PASS or NOT_APPLICABLE."""

    c = _Case(submission, facts)
    checks = [
        _hp(c),
        _consent(c),
        _lab(c, "cbc", "CBC"),
        _lab(c, "cmp", "CMP"),
        _anticoagulation(c),
        _blood_pressure(c),
        _temperature(c),
    ]
    assert tuple(k.id for k in checks) == CHECK_ORDER

    failed = [k for k in checks if k.status == "FAIL"]
    unknown = [k for k in checks if k.status == "UNKNOWN"]
    if any(_FAIL_CATEGORY[k.id] == "ACUTE_SAFETY_EXCLUSION" for k in failed):
        decision = "NOT_CLEARED"
    elif failed or unknown:
        decision = "NEEDS_FOLLOW_UP"
    else:
        decision = "READY"

    issues = [
        TriageIssue(
            category=_FAIL_CATEGORY[k.id] if k.status == "FAIL" else "MISSING_REQUIRED_DATA",
            description=f"{_TITLE[k.id]}: {k.finding}",
            evidence=_issue_evidence(k),
            check_ids=[k.id],
        )
        for k in checks
        if k.status in ("FAIL", "UNKNOWN")
    ]
    return TriageOutput(decision=decision, issues=issues, explanation=render(decision, checks, submission), checks=checks)


def _not_evaluated(kind: str, what_failed: str, problem: str) -> TriageOutput:
    """Fail closed when the system, not the chart, is the reason no check could run.

    Every row is UNKNOWN and the case goes to a nurse. The wording names the system, so
    nobody goes looking for a missing chart item, and `system_failure` lets a caller count
    these apart from real triage results. The category is MISSING_REQUIRED_DATA only
    because the policy offers nothing closer.
    """

    checks = [Check(id=check_id, status="UNKNOWN", finding=f"not evaluated: {what_failed}") for check_id in CHECK_ORDER]
    issue = TriageIssue(
        category="MISSING_REQUIRED_DATA",
        description=f"SYSTEM: {what_failed}, so no check was run. Review this case manually.",
        evidence=problem,
        check_ids=list(CHECK_ORDER),
    )
    explanation = f"NEEDS_FOLLOW_UP: SYSTEM: {what_failed}, so no check was run. Review manually.\n{problem}"
    return TriageOutput(decision="NEEDS_FOLLOW_UP", issues=[issue], explanation=explanation, checks=checks, system_failure=kind)


def unreadable(problem: str) -> TriageOutput:
    """A submission that does not match the input contract."""

    return _not_evaluated("invalid_submission", "the submission could not be read", f"The record does not match the expected format: {problem}")


def reader_failed(problem: str) -> TriageOutput:
    """The model call failed for good: an API error, a refusal, a cut-off or an unusable answer."""

    return _not_evaluated("reader_failed", "the automated reading of the documents failed", problem)
