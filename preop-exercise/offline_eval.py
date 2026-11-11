#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai>=2.0.0",
#   "pydantic>=2.8.0",
# ]
# ///

"""Score saved facts without a model call and without the OpenAI Evals API.

The rules are pure code over (submission, facts), so saved facts can be re-scored as
often as needed. Facts live one file per case: {"case_id", "submission_sha256", "facts", ...}.
run_baseline.py saves what the reader returned for each case when TRIAGE_FACTS_DIR is set.

  view  CASE              what the reader is shown for one case (never the label)
  check CASE [CASE...]    check a facts file against its documents (never the label)
  score [--facts DIR]     run the rules on saved facts, score against the labels, and
                          compare the facts one by one with the reference facts

`score` calls run_evals.py's own scoring functions, so its three metrics are the
harness's, and writes a report that view_report.py can open:
  uv run view_report.py --report data/offline_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rules
import run_evals
from facts import Facts, reader_input, submission_hash
from models import PatientSubmission, TriageOutput
from rules import open_items  # noqa: F401  (re-exported for the tests)

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "data" / "patients_sample_50.jsonl"
GOLD = ROOT / "data" / "gold_facts"

def load_rows(path: Path = SAMPLE) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_saved(facts_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    """The file saved for this case. One written for a different version of the
    submission is refused, so stale facts cannot be scored by mistake."""

    saved = json.loads((facts_dir / f"{row['case_id']}.json").read_text(encoding="utf-8"))
    expected = saved.get("submission_sha256")
    if expected and expected != submission_hash(row["submission"]):
        raise ValueError(f"{row['case_id']}: saved facts were written for a different submission")
    return saved


def load_facts(facts_dir: Path, row: dict[str, Any]) -> Facts:
    return Facts.model_validate(load_saved(facts_dir, row)["facts"])


def output_for(row: dict[str, Any], facts_dir: Path) -> TriageOutput:
    """What the pipeline returns for this case given what was saved for it. A saved
    reader failure gives the fail-closed result, as triage_submission would."""

    saved = load_saved(facts_dir, row)
    if saved.get("error"):
        return rules.reader_failed(saved["error"])
    return rules.evaluate(PatientSubmission.model_validate(row["submission"]), Facts.model_validate(saved["facts"]))


# --- scoring ----------------------------------------------------------------


def score_outputs(rows: list[dict[str, Any]], outputs: list[tuple[TriageOutput | None, str | None]]) -> list[dict[str, Any]]:
    """One record per case, scored by the harness's own code (run_evals), so these
    numbers are the harness's numbers. `outputs` holds (output, error) per row."""

    by_index = {
        index: {"record_index": index, "output": output.model_dump() if output else None, "error": error}
        for index, (output, error) in enumerate(outputs)
    }
    cases = [run_evals.EvalCase.model_validate(row) for row in rows]
    _, records = run_evals._build_eval_items(cases, by_index)
    return records


def score_rows(rows: list[dict[str, Any]], facts_dir: Path) -> list[dict[str, Any]]:
    """Run the rules on the facts saved in `facts_dir` and score every case."""

    outputs: list[tuple[TriageOutput | None, str | None]] = []
    for row in rows:
        try:
            outputs.append((output_for(row, facts_dir), None))
        except Exception as exc:
            outputs.append((None, f"{type(exc).__name__}: {exc}"))
    return score_outputs(rows, outputs)


def _facts_by_identity(facts: Facts) -> dict[str, dict[tuple, tuple]]:
    """Each kind of fact as {what it is about: what was judged about it}."""

    def generic(*text: str) -> str:
        names = rules.anticoagulants_named(" ".join(text))
        return sorted(names)[0] if names else text[-1].lower()

    return {
        "blood pressure readings": {(f.doc_id, f.systolic, f.diastolic): (rules._clock(f.time),) for f in facts.note_blood_pressures},
        "temperature readings": {(f.doc_id, round(rules._fahrenheit(f.value, f.unit), 1)): (rules._clock(f.time),) for f in facts.note_temperatures},
        "H&P documents": {(f.doc_id,): (f.is_preop_evaluation,) for f in facts.hp_candidates},
        "consents": {(f.doc_id,): (f.patient_signed, f.matches_scheduled_procedure) for f in facts.consents},
        "drug mentions": {(f.doc_id, generic(f.drug_as_written, f.generic_name)): (f.status,) for f in facts.anticoagulant_mentions},
        "medication-list drugs": {(name,): () for name in facts.medication_list_anticoagulants},
        "plans": {(f.doc_id, generic(f.generic_name)): (f.preop, f.postop) for f in facts.anticoagulation_plans},
    }


def compare_facts(reference: Facts, got: Facts) -> dict[str, dict[str, Any]]:
    """Fact by fact: which reference facts were found or missed, what was reported that
    the reference lacks, and where a found fact was judged differently (a reading's
    time, a status, a signature, a plan's instructions)."""

    want, have = _facts_by_identity(reference), _facts_by_identity(got)
    result = {}
    for kind in want:
        both = want[kind].keys() & have[kind].keys()
        result[kind] = {
            "reference": len(want[kind]),
            "found": len(both),
            "missed": sorted(want[kind].keys() - have[kind].keys(), key=str),
            "extra": sorted(have[kind].keys() - want[kind].keys(), key=str),
            "differ": sorted(((key, want[kind][key], have[kind][key]) for key in both if want[kind][key] != have[kind][key]), key=str),
        }
    return result


def print_scores(rows: list[dict[str, Any]], facts_dir: Path, report_path: Path | None, reference_dir: Path | None) -> list[dict[str, Any]]:
    records = score_rows(rows, facts_dir)
    total = len(records)
    summary = run_evals._summarize_local_rows(records)
    print(f"\nfacts: {facts_dir}   cases: {total}   harness score: {summary['aggregate_local_score_pct']}%")
    for metric in run_evals.METRIC_WEIGHTS:
        print(f"  {metric}: {sum(r['metrics'][metric] for r in records)}/{total}")
    outputs = [r["parsed_output"] or {} for r in records]
    labels = [r["label"]["decision"] for r in records]
    print(f"  system failures (counted apart): {sum(1 for o in outputs if o.get('system_failure'))}")
    print(f"  READY given to a case labelled otherwise: {sum(1 for o, l in zip(outputs, labels) if o.get('decision') == 'READY' and l != 'READY')}")
    print(f"  NOT_CLEARED label that came out NEEDS_FOLLOW_UP: {sum(1 for o, l in zip(outputs, labels) if l == 'NOT_CLEARED' and o.get('decision') == 'NEEDS_FOLLOW_UP')}")

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
                print(f"    {check['status']:<8}{check['id']}: {check['finding'][:300]}")

    if reference_dir and reference_dir.resolve() != facts_dir.resolve():
        totals: dict[str, dict[str, int]] = {}
        details: list[str] = []
        unverified = 0
        for row in rows:
            try:
                saved = load_saved(facts_dir, row)
                got, reference = Facts.model_validate(saved["facts"]), load_facts(reference_dir, row)
            except Exception:
                continue  # no usable reading for this case; it is already counted above
            unverified += sum("quote not found" in item for item in open_items(PatientSubmission.model_validate(row["submission"]), got))
            for kind, counts in compare_facts(reference, got).items():
                slot = totals.setdefault(kind, {"reference": 0, "found": 0, "missed": 0, "extra": 0, "differ": 0})
                slot["reference"] += counts["reference"]
                slot["found"] += counts["found"]
                for what in ("missed", "extra", "differ"):
                    slot[what] += len(counts[what])
                    details += [f"  {row['case_id'][-2:]} {kind}: {what} {item}" for item in counts[what]]
        print(f"\nfact by fact, against {reference_dir.name}:")
        print(f"  {'kind':<26}{'reference':>10}{'found':>8}{'missed':>8}{'extra':>8}{'differ':>8}")
        for kind, slot in totals.items():
            print(f"  {kind:<26}{slot['reference']:>10}{slot['found']:>8}{slot['missed']:>8}{slot['extra']:>8}{slot['differ']:>8}")
        print(f"  quotes not found in their document: {unverified}")
        for line in details[:60]:
            print(line)
        if len(details) > 60:
            print(f"  ... and {len(details) - 60} more")

    if report_path:
        report = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "mode": f"offline ({facts_dir.name})",
            "primary_score": {"name": run_evals.PRIMARY_SCORE_NAME, "value_pct": summary["aggregate_local_score_pct"], "goal": "maximize"},
            "facts_source": str(facts_dir),
            "local_metrics_summary": summary,
            "records": records,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"\nreport -> {report_path}")
    return records


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


def cmd_score(args: argparse.Namespace) -> int:
    print_scores(load_rows(), args.facts, args.report, args.against)
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # document text contains non-ASCII characters
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    view = commands.add_parser("view")
    view.add_argument("case")
    view.set_defaults(handler=cmd_view)
    check = commands.add_parser("check")
    check.add_argument("cases", nargs="+")
    check.add_argument("--facts", type=Path, default=GOLD)
    check.set_defaults(handler=cmd_check)
    score = commands.add_parser("score")
    score.add_argument("--facts", type=Path, default=GOLD)
    score.add_argument("--against", type=Path, default=GOLD, help="reference facts for the fact-by-fact comparison")
    score.add_argument("--report", type=Path, default=ROOT / "data" / "offline_report.json")
    score.set_defaults(handler=cmd_score)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
