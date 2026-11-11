"""Evaluate without a model call and without the OpenAI Evals API.

The rules are pure code over (submission, facts), so saved facts can be re-scored as
often as needed. Facts live one file per case: {"case_id", "submission_sha256", "facts"}.

  view  CASE            what the reader is shown for one case (never the label)
  check CASE [CASE...]  check a facts file against its documents (never the label)
  score [--facts DIR]   run the rules on saved facts for every case and compare with labels

`score` applies the same three metrics as run_evals.py and writes a report that
view_report.py can open:  uv run view_report.py --report data/offline_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rules
from facts import Facts, reader_input, submission_hash
from models import PatientSubmission, TriageOutput

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "data" / "patients_sample_50.jsonl"
GOLD = ROOT / "data" / "gold_facts"


def load_rows(path: Path = SAMPLE) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_facts(facts_dir: Path, row: dict[str, Any]) -> Facts:
    """Facts saved for this case. A file written for a different version of the
    submission is refused, so stale facts cannot be scored by mistake."""

    saved = json.loads((facts_dir / f"{row['case_id']}.json").read_text(encoding="utf-8"))
    expected = saved.get("submission_sha256")
    if expected and expected != submission_hash(row["submission"]):
        raise ValueError(f"{row['case_id']}: saved facts were written for a different submission")
    return Facts.model_validate(saved["facts"])


def open_items(submission: PatientSubmission, facts: Facts) -> list[str]:
    """Everything a complete, verifiable reading must not leave open, across ALL documents.

    Stricter than the rules need: the rules ignore an unreported reading in a document
    too old to hold the latest value, but a reference reading should report it anyway.
    """

    c = rules._Case(submission, facts)
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
        if not re.search(rf"(?<!\d){f.systolic}\s*/\s*{f.diastolic}(?!\d)", rules._norm(f.quote)):
            problems.append(f"blood pressure {f.systolic}/{f.diastolic}: not written in its quote {f.quote!r}")
    for f in facts.note_temperatures:
        quote(f.doc_id, f.quote, f"temperature {f.value} {f.unit}")
        if not rules._writes_number(f.quote, f.value):
            problems.append(f"temperature {f.value} {f.unit}: not written in its quote {f.quote!r}")
    for f in facts.anticoagulant_mentions:
        quote(f.doc_id, f.quote, f"mention of {f.drug_as_written}")
        if not rules.anticoagulants_named(f"{f.drug_as_written} {f.generic_name}"):
            problems.append(f"mention of {f.drug_as_written!r}: not in the drug table (antiplatelets are not reported)")
    for f in facts.anticoagulation_plans:
        quote(f.doc_id, f.preop_quote, f"{f.generic_name} plan, pre-op")
        quote(f.doc_id, f.postop_quote, f"{f.generic_name} plan, post-op")
        for when, state, text in (("pre-op", f.preop, f.preop_quote), ("post-op", f.postop, f.postop_quote)):
            if state != "absent" and text is None:
                problems.append(f"{f.generic_name} plan in {c.label(f.doc_id)}: {when} is {state} but has no quote")

    listed = {m.name for m in submission.medications if rules.anticoagulants_named(m.name)}
    reported_listed = set(facts.medication_list_anticoagulants)
    for name in sorted(listed - reported_listed):
        problems.append(f"medication list entry {name!r} is an anticoagulant but is not in medication_list_anticoagulants")
    for name in sorted(reported_listed - {m.name for m in submission.medications}):
        problems.append(f"medication_list_anticoagulants has {name!r}, which is not an entry of the medication list (copy it verbatim)")

    problems += c.unreported(rules._HP_TITLE, {f.doc_id for f in facts.hp_candidates}, "an H&P")
    problems += c.unreported(rules._CONSENT_TITLE, {f.doc_id for f in facts.consents}, "a consent")

    for doc in submission.documents:
        label = c.label(doc.doc_id)
        text = unicodedata.normalize("NFKC", doc.text or "")
        named = rules.anticoagulants_named(text)
        reported: set[str] = set()
        for f in facts.anticoagulant_mentions:
            if f.doc_id == doc.doc_id:
                reported |= rules.anticoagulants_named(f"{f.drug_as_written} {f.generic_name}")
        for generic in sorted(named - reported):
            problems.append(f"{generic} is named in {label} but there is no mention for it from that document")

        pairs = {(f.systolic, f.diastolic) for f in facts.note_blood_pressures if f.doc_id == doc.doc_id}
        for match in rules._BP_SHAPE.finditer(text):
            pair = (int(match.group(1)), int(match.group(2)))
            if 60 <= pair[0] <= 260 and 30 <= pair[1] <= 160 and pair[0] > pair[1] and pair not in pairs:
                problems.append(f"{pair[0]}/{pair[1]} is written in {label} but is not reported")
        temps = [rules._fahrenheit(f.value, f.unit) for f in facts.note_temperatures if f.doc_id == doc.doc_id]
        for match in rules._TEMP_SHAPE.finditer(text):
            value, unit = float(match.group(1)), match.group(2)
            plausible = 30 <= value <= 45 if unit == "C" else 90 <= value <= 110
            if plausible and not any(abs(rules._fahrenheit(value, unit) - t) <= 0.15 for t in temps):
                problems.append(f"{value:g} {unit} is written in {label} but is not reported")
    return problems


# --- commands ---------------------------------------------------------------


def _row(case_id: str) -> dict[str, Any]:
    for row in load_rows():
        if row["case_id"] == case_id:
            return row
    raise SystemExit(f"unknown case {case_id!r}")


def cmd_view(args: argparse.Namespace) -> int:
    shown = reader_input(PatientSubmission.model_validate(_row(args.case)["submission"]))
    print(f"CASE {args.case}")
    print(f"SCHEDULED PROCEDURE: {shown['scheduled_procedure']}")
    print("MEDICATION LIST (structured):")
    for med in shown["medication_list"]:
        print(f"  - {med['name']!r}  active={med['active']}")
    for doc in shown["documents"]:
        print(f"\n===== doc_id={doc['doc_id']} | type={doc['type']} | date={doc['date']} | author={doc['author']}")
        print(doc["text"])
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    failed = 0
    for case_id in args.cases:
        row = _row(case_id)
        submission = PatientSubmission.model_validate(row["submission"])
        try:
            facts = load_facts(args.facts, row)
        except Exception as exc:  # a malformed file is something to report, not a crash
            print(f"{case_id}: CANNOT LOAD: {exc}")
            failed += 1
            continue
        problems = open_items(submission, facts)
        output = rules.evaluate(submission, facts)
        print(f"\n{case_id}: {'OK' if not problems else f'{len(problems)} OPEN ITEM(S)'}")
        for problem in problems:
            print(f"  ! {problem}")
        print("  what the rules make of these facts:")
        for line in output.explanation.splitlines():
            print(f"    {line}")
        failed += bool(problems)
    return 1 if failed else 0


def score_rows(rows: list[dict[str, Any]], facts_dir: Path) -> list[dict[str, Any]]:
    """One record per case, in the shape run_evals.py writes and view_report.py reads."""

    records = []
    for index, row in enumerate(rows):
        label = row["label"]
        output: TriageOutput | None = None
        error = None
        try:
            output = rules.evaluate(PatientSubmission.model_validate(row["submission"]), load_facts(facts_dir, row))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        metrics = {
            "json_schema_valid": output is not None,
            "decision_match_oracle": output is not None and output.decision == label["decision"],
            "issue_categories_match_oracle": output is not None
            and sorted({i.category for i in output.issues}) == sorted(set(label["categories"])),
        }
        records.append(
            {
                "record_index": index,
                "case_id": row["case_id"],
                "submission": row["submission"],
                "baseline_error": error,
                "label": label,
                "parsed_output": output.model_dump() if output else None,
                "parse_error": None,
                "metrics": metrics,
                "aggregate_local_score": 100.0 * sum(metrics.values()) / len(metrics),
            }
        )
    return records


def cmd_score(args: argparse.Namespace) -> int:
    records = score_rows(load_rows(), args.facts)
    total = len(records)
    score = round(sum(r["aggregate_local_score"] for r in records) / total, 2)
    report = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "mode": f"offline ({args.facts.name})",
        "primary_score": {"name": "aggregate_local_score_pct", "value_pct": score, "goal": "maximize"},
        "facts_source": str(args.facts),
        "records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"facts: {args.facts}   cases: {total}   score: {score}%")
    for metric in ("json_schema_valid", "decision_match_oracle", "issue_categories_match_oracle"):
        print(f"  {metric}: {sum(r['metrics'][metric] for r in records)}/{total}")
    for record in records:
        if all(record["metrics"].values()):
            continue
        label, output = record["label"], record["parsed_output"]
        print(f"\n{record['case_id']}  label {label['decision']} {sorted(label['categories'])}")
        if output is None:
            print(f"  no output: {record['baseline_error']}")
            continue
        print(f"  got   {output['decision']} {sorted({i['category'] for i in output['issues']})}")
        for check in output["checks"]:
            if check["status"] in ("FAIL", "UNKNOWN"):
                print(f"    {check['status']:<8}{check['id']}: {check['finding']}")
    print(f"\nreport -> {args.report}")
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # document text contains non-ASCII characters
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    view = commands.add_parser("view")
    view.add_argument("case")
    view.set_defaults(run=cmd_view)
    check = commands.add_parser("check")
    check.add_argument("cases", nargs="+")
    check.add_argument("--facts", type=Path, default=GOLD)
    check.set_defaults(run=cmd_check)
    score = commands.add_parser("score")
    score.add_argument("--facts", type=Path, default=GOLD)
    score.add_argument("--report", type=Path, default=ROOT / "data" / "offline_report.json")
    score.set_defaults(run=cmd_score)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
