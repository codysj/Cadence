# Approach and results

## Architecture: the model reads, the code decides

1. **Validate.** The package is checked against a strict schema. An unknown field is an
   error, because every field is optional and a renamed `labs` would otherwise read as
   "no labs".
2. **Read.** One model call sees the documents, the medication list and the scheduled
   procedure. It returns atomic facts, each with a document id and a verbatim quote. It
   never sees labs, structured vitals, the risk level or a threshold, and it returns no
   decision.
3. **Verify and decide.** Code checks that every quote is in the document it cites, then
   applies the four policy rules as plain functions. The output is the same seven checks
   on every case, `READY` included, and the decision, issues and explanation are derived
   from those rows, so the text cannot disagree with the decision.

Failures are handled by kind. A malformed record or a failed model call returns
`NEEDS_FOLLOW_UP` for that case with `system_failure` set. A wrong key or model id stops
the run, since it would fail every case alike.

## Principle: doubt never passes

`READY` and `NOT_CLEARED` are both claims, and a claim needs established facts. A check
passes or fails only when that outcome holds however the open questions resolve.
Otherwise it is `UNKNOWN` and the case is `NEEDS_FOLLOW_UP`. Open questions are a quote
that is not in its document, a reading or drug written in a document but not reported,
an unrecognised value such as a new lab code, and sources that cannot be put in order.

## One evidence packet

case_00005 from the run below. Its structured vitals are normal; the breach is only in a
nursing note. The issue, then two of the seven rows as the report viewer prints them:

```text
ACUTE_SAFETY_EXCLUSION  Blood pressure: 157/117 on 2026-03-03 09:05, limits 180/110
  evidence: Nursing Pre-procedure Assessment 2026-03-03 #7aefc9ae
            "  09:05  BP 157/117, Temp 36.7 °C, HR 69, SpO2 99% RA"

NOT_CLEARED: needs attention: Blood pressure
case_00005 | Elective colonic resection | procedure date 2026-03-06 | risk HIGH
...
PASS    CBC: dated 2026-02-22, 12 days before the procedure, limit 14
    source: CBC 2026-02-22 lab #af188ba3
    rejected: LAB-CBC 2026-01-28 lab #c7b5d08a: older result
...
FAIL    Blood pressure: 157/117 on 2026-03-03 09:05, limits 180/110
    source: Nursing Pre-procedure Assessment 2026-03-03 #7aefc9ae
      "  09:05  BP 157/117, Temp 36.7 °C, HR 69, SpO2 99% RA"
    rejected: 119/78 on 2026-03-01 10:12 (structured vital, Pre-op clinic): older reading
```

## Assumptions

| Assumption | Basis |
|---|---|
| Windows count calendar days, and the limit day is inside | Label 12 is READY at exactly 30 days; label 09 fails at 15 against a limit of 14 |
| Only anticoagulants need a plan; aspirin and clopidogrel do not | Rule 3 says "anticoagulant medication"; labels 24, 25 |
| A drug the patient reports in a note counts, even if it is not on the medication list | Labels 00, 27, 38 |
| The plan must be for the drug now taken and cover before and after | Rule 3 "before and after the procedure"; labels 18, 20 |
| The most recent reading wins across structured vitals and notes, and a later in-range recheck clears a breach | Rule 4 "most recent", "a rechecked vital within range is sufficient"; labels 05, 16 |
| A missing procedure date, risk level or temperature gives `MISSING_REQUIRED_DATA` only; nothing is inferred | The policy's last line; labels 11, 29, 09 |
| `NOT_CLEARED` still lists every other failing category | Labels 28, 32 |
| A consent counts only if the patient signed it and it names the scheduled procedure | Rule 1 "Signed Surgical Consent"; labels 28, 24 |
| Entered-in-error labs are skipped; a BMP is not a CMP | Labels 06, 17 |
| Doubt blocks `NOT_CLEARED` as well as `READY` | The policy's last line (unknown means `NEEDS_FOLLOW_UP`); no label tests it |
| Note clock times and structured timestamps are compared as written, with no time-zone conversion | Cases 08, 09, 20, 38 have both on one day; the order changes none of them |

## Results

One run of all 50 sample cases on 2026-11-10, model `openai/gpt-6-luna` through
OpenRouter, driven by the stock `run_baseline.py`: 51 model calls (one case needed the
retry) and $0.04.

| Measure | Result | What it means |
|---|---|---|
| Harness metrics | 50/50 valid, 50/50 decisions, 50/50 exact category sets | The three metrics `run_evals.py` computes, applied to `baseline_outputs.jsonl` with its own functions. The Evals API step was not run |
| Fact by fact | 328 of 328 reference facts found; none missed, added or judged differently; every quote found in its document | Agreement with reference facts written to the same reading guide as the prompt, so it shows consistency, not independent accuracy |
| Determinism | Harness check on case 0, ten fresh calls: decision 100%, exact output 60% | Decisions are code given the facts. The text varies because the model picks different quotes for the same fact. Re-running the rules on the saved readings reproduced all 50 outputs exactly |

## Limits

- **Fit to the sample.** The rules, the reader prompt and the reading guide behind the
  reference facts were all written after studying these 50 labels. There are no held-out
  cases. 50 of 50 is a consistency check and not an accuracy estimate.
- **Reference facts.** 49 were written by subagents that never saw labels, following a
  reading guide I wrote after studying the labels; case 16 I wrote knowing its label.
- **Untested by real data.** With complete facts no check goes `UNKNOWN` from doubt, so
  the doubt paths, and rules such as "the newest plan decides", rest on synthetic tests.
- **Size.** `rules.py` is about 1,000 lines for a four-rule policy, and much of it
  handles situations no sample case contains. That is more than the exercise needed.
- **One model, one run.** Stability was measured on one case, and no second model or
  repeat run is reported.
- **Not run live.** The direct OpenAI path, the `make` targets and Evals API scoring.
- **What a quote check cannot do.** A real quote can be attached to the wrong claim, and
  the omission checks rely on text patterns. The nurse sees the quote either way.

Built with Claude Code; the exported session shows where a design review and review
feedback changed the design.
