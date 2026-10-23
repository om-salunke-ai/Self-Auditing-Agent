r"""
evaluate.py — score the agent against labelled invoices with known answers.

Dataset: data/eval/labelled_invoices.json. Each invoice has its text, the
correct answer for all 5 fields, and the verdict it should get (pass/reject).

Offline mode (free, no API calls) tests the Checker on its own:

    .venv\Scripts\python.exe -m src.evaluate

    For every invoice the Checker gets the CORRECT answer (it should give the
    expected verdict) and several WRONG answers, each one a realistic LLM
    mistake: the subtotal used as the total, the invoice date used as the due
    date, the customer used as the vendor, the currency switched, ... A wrong
    answer is "caught" when the Checker does not pass it.

Live mode (real LLM, costs API calls) tests the whole agent:

    .venv\Scripts\python.exe -m src.evaluate --live
    .venv\Scripts\python.exe -m src.evaluate --live --limit 5

    Runs each invoice through the real graph, records every Maker attempt and
    the Checker's verdict on it, and compares them with the known answers.
    Uses a temporary checkpoint database, so data/checkpoints.db is never
    touched. The full report is saved to data/eval/live_report.json.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.nodes.checker import run_checker

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_FILE = _PROJECT_ROOT / "data" / "eval" / "labelled_invoices.json"
LIVE_REPORT_FILE = _PROJECT_ROOT / "data" / "eval" / "live_report.json"

FIELDS = ("vendor_name", "invoice_number", "total_amount", "due_date", "currency")

# Words an LLM may write when a field is not on the invoice, compared after
# _squash(): "N/A" -> "NA", "To be confirmed" -> "TOBECONFIRMED".
_MISSING_WORDS = {"", "NA", "NONE", "NULL", "NIL", "TBD", "UNKNOWN", "NOTPROVIDED",
                  "NOTAVAILABLE", "PENDING", "TOBEASSIGNED", "TOBECONFIRMED"}


# ---------------------------------------------------------------------------
# Comparing an answer with the known correct answer
# ---------------------------------------------------------------------------

def load_invoices(path: Path = DATASET_FILE) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _squash(value) -> str:
    """Uppercase, letters and digits only: "Pvt. Ltd." -> "PVTLTD"."""
    return re.sub(r"[\W_]+", "", str(value).upper())


def field_is_right(field: str, got, invoice: dict) -> bool:
    """Is `got` the correct value of `field` for this invoice?

    Small differences that don't change the meaning are allowed: case and
    punctuation in names and numbers ("Pvt Ltd" = "Pvt. Ltd."), and any way
    of saying "not on the invoice" when the field really is missing.
    """
    want = invoice["expected"][field]
    accepted = [want] + invoice.get("also_acceptable", {}).get(field, [])

    if field == "total_amount":
        return isinstance(got, (int, float)) and any(abs(got - w) < 0.005 for w in accepted)
    if not want:  # the invoice doesn't have this field
        return got is None or _squash(got) in _MISSING_WORDS
    if field == "due_date":
        return isinstance(got, str) and got.strip() in accepted
    return got is not None and _squash(got) in {_squash(w) for w in accepted}


def wrong_fields(answer: dict, invoice: dict) -> list[str]:
    """The fields of `answer` that don't match the known correct answer."""
    return [f for f in FIELDS if not field_is_right(f, answer.get(f), invoice)]


# ---------------------------------------------------------------------------
# Realistic wrong answers (offline mode)
# ---------------------------------------------------------------------------

def _digit_typo(number: str) -> str:
    """Swap the last two digits ("0089" -> "0098"), or change the last one."""
    chars = list(number)
    digits = [i for i, ch in enumerate(chars) if ch.isdigit()]
    if len(digits) >= 2 and chars[digits[-2]] != chars[digits[-1]]:
        a, b = digits[-2], digits[-1]
        chars[a], chars[b] = chars[b], chars[a]
    elif digits:
        chars[digits[-1]] = str((int(chars[digits[-1]]) + 1) % 10)
    return "".join(chars)


def _cut_short(number: str) -> str | None:
    """Drop the last letter/digit group: "INV-2024-BLR-0089" -> "INV-2024-BLR"."""
    groups = list(re.finditer(r"[A-Za-z0-9]+", number))
    if len(groups) < 2:
        return None
    return number[: groups[-1].start()].rstrip(" -/._")


def bad_extractions(invoice: dict) -> list[dict]:
    """Wrong answers for this invoice, each one a single realistic LLM mistake.

    Each is the correct answer with one or two fields changed. Where possible
    the wrong value is something really printed on the invoice (a subtotal,
    the invoice date, the customer's name...), because that is how LLMs get
    invoices wrong in practice. Returns dicts with "kind", "note" and "data".
    """
    truth = invoice["expected"]
    decoys = invoice.get("decoys", {})
    found: list[dict] = []

    def wrong(kind: str, note: str, **changes) -> None:
        data = {**truth, **changes}
        # Keep it only if it really is wrong and isn't a duplicate.
        if wrong_fields(data, invoice) and all(b["data"] != data for b in found):
            found.append({"kind": kind, "note": note, "data": data})

    # Currency switched to an allowed one (the original mock-003 bug).
    currency = truth["currency"]
    if currency:
        fake = "INR" if currency == "USD" else "USD"
        wrong("currency switched", f"{currency} reported as {fake}", currency=fake)

    # Total amount taken from the wrong line, misread, or invented.
    total = truth["total_amount"]
    for label, amount in decoys.get("amounts", {}).items():
        wrong("total from the wrong line", f"{label} ({amount:,.2f}) used as the total",
              total_amount=amount)
    if total > 0:
        wrong("total off by 10x", f"{total:,.2f} read as {total / 10:,.2f}",
              total_amount=round(total / 10, 2))
    elif total < 0:
        wrong("credit note sign dropped", f"{total:,.2f} reported as {-total:,.2f}",
              total_amount=-total)
    else:
        wrong("total invented", "0.00 total replaced by 2,500.00", total_amount=2500.0)

    # Due date taken from another date on the invoice, misread, or invented.
    due = truth["due_date"]
    for label, day in decoys.get("dates", {}).items():
        wrong("due date from the wrong line", f"{label} ({day}) used as the due date",
              due_date=day)
    if due:
        year, month, day = due.split("-")
        if int(day) <= 12 and day != month:
            wrong("day and month swapped", f"{due} read as {year}-{day}-{month}",
                  due_date=f"{year}-{day}-{month}")
    else:
        wrong("due date invented", "missing due date filled with 2024-12-31",
              due_date="2024-12-31")

    # Vendor: the customer's name, or an invented one.
    buyer = decoys.get("buyer")
    if buyer:
        wrong("customer used as vendor", f"'{buyer}' (the customer) used as the vendor",
              vendor_name=buyer)
    if not truth["vendor_name"]:
        wrong("vendor invented", "missing vendor filled with 'Shree Om Traders'",
              vendor_name="Shree Om Traders")

    # Invoice number: another reference number, a typo, cut short, or invented.
    number = truth["invoice_number"]
    for label, ref in decoys.get("references", {}).items():
        wrong("invoice number from the wrong field", f"{label} ({ref}) used as the invoice number",
              invoice_number=ref)
    if number:
        typo = _digit_typo(number)
        wrong("invoice number typo", f"{number} read as {typo}", invoice_number=typo)
        cut = _cut_short(number)
        if cut:
            wrong("invoice number cut short", f"{number} cut to {cut}", invoice_number=cut)
    else:
        wrong("invoice number invented", "missing number filled with INV-2024-0001",
              invoice_number="INV-2024-0001")

    # The LLM obeyed an instruction hidden in the invoice text.
    for changes in invoice.get("injection", []):
        wrong("prompt injection obeyed", ", ".join(f"{k} = {v}" for k, v in changes.items()),
              **changes)
    return found


def correct_variants(invoice: dict) -> list[tuple[str, dict]]:
    """The correct answer, plus the same answer with a missing field written
    the way the invoice writes it (e.g. "(to be assigned)"). Both mean the
    same thing, so both should get the expected verdict."""
    variants = [("correct answer", invoice["expected"])]
    for field, text in invoice.get("placeholders", {}).items():
        variants.append((f"correct answer with {field} = '{text}'",
                         {**invoice["expected"], field: text}))
    return variants


# ---------------------------------------------------------------------------
# Offline mode: the Checker on its own
# ---------------------------------------------------------------------------

def _audit(answer: dict, invoice: dict) -> dict:
    return run_checker({"maker_output": json.dumps(answer), "input": invoice["raw_text"]})


def evaluate_checker(invoices: list[dict]) -> dict:
    """Run the Checker on every correct and every wrong answer.

    Returns {"correct": [...], "bad": [...]}, one dict per Checker call.
    """
    correct, bad = [], []
    for inv in invoices:
        for label, answer in correct_variants(inv):
            result = _audit(answer, inv)
            correct.append({"id": inv["id"], "label": label, "expected": inv["expected_result"],
                            "got": result["audit_result"], "reason": result["audit_reason"]})
        for wrong_answer in bad_extractions(inv):
            result = _audit(wrong_answer["data"], inv)
            bad.append({"id": inv["id"], "kind": wrong_answer["kind"], "note": wrong_answer["note"],
                        "got": result["audit_result"], "reason": result["audit_reason"],
                        "caught": result["audit_result"] != "pass"})
    return {"correct": correct, "bad": bad}


def print_checker_report(result: dict, invoices: list[dict], details: bool = False) -> None:
    bad, correct = result["bad"], result["correct"]
    caught = sum(b["caught"] for b in bad)
    right_verdicts = sum(c["got"] == c["expected"] for c in correct)
    should_pass = sum(inv["expected_result"] == "pass" for inv in invoices)

    print()
    print("=" * 66)
    print("  CHECKER EVALUATION  (offline: no LLM, no API calls)")
    print("=" * 66)
    print(f"  Invoices: {len(invoices)}  ({should_pass} should pass, "
          f"{len(invoices) - should_pass} should be rejected)")
    print()
    print(f"  Bad extractions caught      : {caught} / {len(bad)}  ({caught / len(bad):.1%})")
    print(f"  Correct answers, right verdict: {right_verdicts} / {len(correct)}")
    print()
    print("  Caught, by type of mistake:")
    totals = Counter(b["kind"] for b in bad)
    hits = Counter(b["kind"] for b in bad if b["caught"])
    for kind, n in totals.most_common():
        flag = "" if hits[kind] == n else "   <-- misses"
        print(f"    {kind:<36} {hits[kind]:>3} / {n:<3}{flag}")

    missed = [b for b in bad if not b["caught"]]
    print()
    print(f"  MISSED: wrong answers the Checker would PASS ({len(missed)})")
    for b in missed:
        print(f"    {b['id']:<9} {b['kind']:<36} {b['note']}")

    wrong_verdicts = [c for c in correct if c["got"] != c["expected"]]
    print()
    print(f"  WRONG VERDICT on a correct answer ({len(wrong_verdicts)})")
    for c in wrong_verdicts:
        print(f"    {c['id']:<9} {c['label']}")
        print(f"              expected {c['expected']}, got {c['got']}: {c['reason']}")

    if details:
        print()
        print("  EVERY WRONG ANSWER:")
        for b in bad:
            mark = "caught" if b["caught"] else "MISSED"
            print(f"    [{mark}] {b['id']:<9} {b['kind']:<36} {b['note']}  -> {b['got']}")
    print("=" * 66)
    print()


# ---------------------------------------------------------------------------
# Live mode: the whole agent with the real LLM
# ---------------------------------------------------------------------------

def evaluate_live(invoices: list[dict], limit: int | None = None, assume_yes: bool = False) -> dict | None:
    chosen = invoices[:limit] if limit else invoices
    n = len(chosen)
    print(f"\nLive mode calls the real LLM: {n} invoices, between {n} and {3 * n} LLM calls.")
    if not assume_yes and input("Type yes to continue: ").strip().lower() != "yes":
        print("Cancelled.")
        return None

    # A throwaway checkpoint database, set BEFORE src.graph is imported, so
    # data/checkpoints.db is never touched by an evaluation run.
    os.environ["CHECKPOINT_DB"] = str(Path(tempfile.mkdtemp(prefix="agent-eval-")) / "checkpoints.db")
    import src.graph as graph_module

    # Record every Maker answer and the Checker's verdict on it. build_graph()
    # looks up run_checker when it runs, so swapping the name in src.graph
    # gives the same graph with a recording Checker.
    attempts: list[dict] = []
    real_checker = graph_module.run_checker

    def recording_checker(state: dict) -> dict:
        result = real_checker(state)
        attempts.append({"maker_output": state.get("maker_output") or "", **result})
        return result

    graph_module.run_checker = recording_checker
    app = graph_module.build_graph()

    rows = []
    for idx, inv in enumerate(chosen, start=1):
        attempts.clear()
        state = {"input": inv["raw_text"], "maker_output": "", "audit_result": "",
                 "audit_reason": "", "retry_count": 0, "messages": []}
        error = None
        try:
            final = app.invoke(state, {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex}"}})
        except Exception as exc:  # noqa: BLE001  (record it and carry on)
            final, error = {}, f"{type(exc).__name__}: {exc}"

        recorded = []
        for a in attempts:
            answer = json.loads(a["maker_output"]) if a["maker_output"] else None
            recorded.append({"answer": answer, "verdict": a["audit_result"], "reason": a["audit_reason"],
                             "wrong_fields": wrong_fields(answer, inv) if answer else None})
        final_answer = json.loads(final["maker_output"]) if final.get("maker_output") else None
        row = {
            "id": inv["id"],
            "expected_result": inv["expected_result"],
            "final_result": "error" if error else final.get("audit_result"),
            "final_answer": final_answer,
            "final_wrong_fields": wrong_fields(final_answer, inv) if final_answer else list(FIELDS),
            "attempts": recorded,
            "error": error,
        }
        rows.append(row)

        ok = "ok " if row["final_result"] == row["expected_result"] else "BAD"
        wrong = ", ".join(row["final_wrong_fields"]) or "-"
        alarm = "   <-- WRONG DATA PASSED" if row["final_result"] == "pass" and row["final_wrong_fields"] else ""
        print(f"  [{ok}] {idx:>2}/{n} {inv['id']:<9} expected {row['expected_result']:<6} "
              f"got {row['final_result']:<6} attempts {len(recorded)}  wrong fields: {wrong}{alarm}")

    summary = _live_summary(rows)
    report = {"run_at": datetime.now(timezone.utc).isoformat(), "summary": summary, "invoices": rows}
    LIVE_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    LIVE_REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_live_summary(summary)
    print(f"  Full report: {LIVE_REPORT_FILE}\n")
    return report


def _live_summary(rows: list[dict]) -> dict:
    answers = [a for r in rows for a in r["attempts"] if a["answer"] is not None]
    wrong = [a for a in answers if a["wrong_fields"]]
    return {
        "invoices": len(rows),
        "maker_attempts": sum(len(r["attempts"]) for r in rows),
        "llm_call_errors": sum(1 for r in rows for a in r["attempts"] if a["answer"] is None),
        "graph_errors": sum(1 for r in rows if r["error"]),
        "verdict_as_expected": sum(r["final_result"] == r["expected_result"] for r in rows),
        "final_answer_fully_correct": sum(not r["final_wrong_fields"] for r in rows),
        "field_correct": {f: sum(f not in r["final_wrong_fields"] for r in rows) for f in FIELDS},
        # The number that matters most: wrong data that got through.
        "wrong_answers_passed": sum(r["final_result"] == "pass" and bool(r["final_wrong_fields"])
                                    for r in rows),
        "llm_mistakes": len(wrong),
        "llm_mistakes_caught": sum(a["verdict"] != "pass" for a in wrong),
        "correct_answers_sent_back": sum(a["verdict"] == "fail" for a in answers if not a["wrong_fields"]),
    }


def _print_live_summary(s: dict) -> None:
    n = s["invoices"]
    print()
    print("=" * 66)
    print("  LIVE EVALUATION  (real LLM)")
    print("=" * 66)
    print(f"  Invoices: {n}   Maker attempts: {s['maker_attempts']}   "
          f"LLM call errors: {s['llm_call_errors']}   crashed runs: {s['graph_errors']}")
    print(f"  Final verdict as expected     : {s['verdict_as_expected']} / {n}")
    print(f"  Final answer fully correct    : {s['final_answer_fully_correct']} / {n}")
    print("  Field accuracy                : " + ", ".join(
        f"{f.replace('_', ' ')} {c}/{n}" for f, c in s["field_correct"].items()))
    print(f"  Wrong answers that PASSED     : {s['wrong_answers_passed']}   <- should be 0")
    print(f"  LLM mistakes caught by Checker: {s['llm_mistakes_caught']} / {s['llm_mistakes']}")
    print(f"  Correct answers sent back     : {s['correct_answers_sent_back']}   (wasted retries)")
    print("=" * 66)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score the agent against labelled invoices.")
    parser.add_argument("--live", action="store_true", help="run the real LLM (costs API calls)")
    parser.add_argument("--limit", type=int, help="live mode: only the first N invoices")
    parser.add_argument("--yes", action="store_true", help="live mode: don't ask for confirmation")
    parser.add_argument("--details", action="store_true", help="offline mode: list every wrong answer")
    args = parser.parse_args(argv)

    # The report contains symbols like "₹"; Windows consoles may not use UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    invoices = load_invoices()
    if args.live:
        evaluate_live(invoices, limit=args.limit, assume_yes=args.yes)
    else:
        print_checker_report(evaluate_checker(invoices), invoices, details=args.details)


if __name__ == "__main__":
    main()
