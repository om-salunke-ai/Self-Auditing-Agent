r"""

    .venv\Scripts\python.exe -m src.run_agent

"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.runnables import RunnableConfig

# ---------------------------------------------------------------------------
# Resolve project root regardless of cwd
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MOCKS_FILE   = _PROJECT_ROOT / "data" / "mocks" / "sample_invoice.json"
_AUDIT_FILE   = _PROJECT_ROOT / "data" / "audit_trail.json"
# Every result of every run is appended here, one JSON object per line, so
# older runs are kept when audit_trail.json is rewritten (BUG-11).
_HISTORY_FILE = _PROJECT_ROOT / "data" / "audit_history.jsonl"
# Longer texts are almost surely not one invoice; sending them to the LLM
# (up to 3 times) would be slow and costly (BUG-13).
_MAX_INVOICE_CHARS = 20_000

# ---------------------------------------------------------------------------
# Logging — INFO to stdout so it interleaves cleanly with print() output
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,             # show every Maker / Checker / Router step (BUG-15)
    format="%(levelname)-8s %(name)s — %(message)s",
    stream=sys.stdout,
)
# httpx logs every HTTP request at INFO; that is noise here, so hide it.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Import after logging config so the graph's start-up log uses this format.
from src.graph import app  # noqa: E402  (intentional late import)


# ---------------------------------------------------------------------------
# Terminal formatting helpers
# ---------------------------------------------------------------------------

_DIVIDER   = "-" * 60
_THICK_DIV = "=" * 60

PASS_ICON   = "[PASS]"
FAIL_ICON   = "[FAIL]"
REJECT_ICON = "[REJECT]"
WARN_ICON   = "[WARN]"


def _force_utf8_output() -> None:
    """On Windows, Python writes cp1252 when output goes to a file or a pipe,
    and crashes on text such as "₹" or Hindi names. Use UTF-8 (BUG-12)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _print_header() -> None:
    print()
    print(_THICK_DIV)
    print("  Self-Auditing Workflow Agent -- Batch Runner")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(_THICK_DIV)


def _print_mock_banner(index: int, total: int, mock_id: str) -> None:
    print()
    print(_DIVIDER)
    print(f"  Invoice {index}/{total}  |  ID: {mock_id}")
    print(_DIVIDER)


def _print_result(final_state: dict, mock_id: str) -> None:
    result      = final_state.get("audit_result", "unknown")
    reason      = final_state.get("audit_reason", "—")
    attempts    = final_state.get("retry_count", 0)  # counts Maker attempts, not retries
    icon        = {"pass": PASS_ICON, "reject": REJECT_ICON, "error": WARN_ICON}.get(result, FAIL_ICON)

    print(f"  {icon}  Result      : {result.upper()}")
    print(f"  Attempts    : {attempts}")
    print(f"  Reason      : {reason}")

    # Pretty-print the extracted fields if the Maker produced output
    raw_output = final_state.get("maker_output", "")
    if raw_output:
        try:
            extracted = json.loads(raw_output)
            print("  Extracted   :")
            for key, value in extracted.items():
                print(f"    {key:<20} {value}")
        except json.JSONDecodeError:
            print(f"  {WARN_ICON} maker_output could not be parsed for display.")


def _print_summary(results: list[dict]) -> None:
    # Count each outcome on its own: "reject" (document problem) and "error"
    # (crash / skipped) are not the same as "fail" (BUG-18).
    def count(result: str) -> int:
        return sum(1 for r in results if r["audit_result"] == result)

    print()
    print(_THICK_DIV)
    print("  BATCH SUMMARY")
    print(_THICK_DIV)
    print(f"  Total invoices : {len(results)}")
    print(f"  {PASS_ICON}    Passed   : {count('pass')}")
    print(f"  {FAIL_ICON}    Failed   : {count('fail')}")
    print(f"  {REJECT_ICON}  Rejected : {count('reject')}")
    print(f"  {WARN_ICON}    Errors   : {count('error')}")
    print(_THICK_DIV)
    print()


def _save_audit_trail(audit_trail: list[dict]) -> None:
    """Write audit_trail.json safely (BUG-10).

    Called after every invoice, so a crash halfway keeps the results so far
    instead of leaving the last run's file behind. Writing a temp file and
    then swapping it in means the file is never left half-written.
    """
    _AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = _AUDIT_FILE.with_name(_AUDIT_FILE.name + ".tmp")
    with tmp_file.open("w", encoding="utf-8") as fh:
        json.dump(audit_trail, fh, indent=2, ensure_ascii=False)
    os.replace(tmp_file, _AUDIT_FILE)  # swaps the new file in, in one step


def _input_problem(raw_text) -> str | None:
    """Why `raw_text` can't be sent to the LLM, or None if it looks fine (BUG-13)."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return "raw_text is missing or empty"
    if len(raw_text) > _MAX_INVOICE_CHARS:
        return f"raw_text is {len(raw_text):,} characters long (limit {_MAX_INVOICE_CHARS:,})"
    return None


def _append_history(entry: dict) -> None:
    """Add one result to audit_history.jsonl. The file is only ever appended
    to, so results from earlier runs are never lost (BUG-11)."""
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _HISTORY_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_all_mocks() -> None:
    """Load mock invoices, run each through the graph, export audit trail."""

    # -- Load mocks ----------------------------------------------------------
    if not _MOCKS_FILE.exists():
        sys.exit(f"[ERROR] Mock file not found: {_MOCKS_FILE}")

    # "utf-8-sig" also reads files saved with a BOM, which Windows
    # PowerShell 5.1 adds by default (BUG-13).
    with _MOCKS_FILE.open(encoding="utf-8-sig") as fh:
        mocks: list[dict] = json.load(fh)
    if not isinstance(mocks, list):
        sys.exit(f"[ERROR] {_MOCKS_FILE} must contain a JSON list of invoices.")

    _print_header()
    print(f"\n  Loaded {len(mocks)} mock invoice(s) from:\n  {_MOCKS_FILE}\n")

    audit_trail: list[dict] = []
    # One id per run, so checkpoints and history from different runs never
    # mix (BUG-11).
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    seen_ids: set = set()

    # -- Run each mock -------------------------------------------------------
    for idx, mock in enumerate(mocks, start=1):
        if not isinstance(mock, dict):
            mock = {}  # not an invoice object; reported as "no text" below (BUG-13)
        mock_id   = mock.get("id", f"mock-{idx:03d}")
        raw_text  = mock.get("raw_text", "")

        _print_mock_banner(idx, len(mocks), mock_id)
        if mock_id in seen_ids:
            print(f"  {WARN_ICON} Duplicate invoice id '{mock_id}': it still gets its "
                  "own thread, but the audit trail will show this id twice.")
        seen_ids.add(mock_id)

        # Initial state — all AgentState keys must be present
        initial_state = {
            "input":        raw_text,
            "maker_output": "",
            "audit_result": "",
            "audit_reason": "",
            "retry_count":  0,
            "messages":     [],
        }

        # The thread_id is unique per run AND per position, so a second run,
        # or two invoices with the same id, never share checkpoints (BUG-11).
        thread_id = f"{run_id}-{idx:03d}-{mock_id}"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        problem = _input_problem(raw_text)
        if problem:
            # Don't pay for LLM calls on input that can't be an invoice (BUG-13).
            print(f"  {WARN_ICON} Skipped '{mock_id}': {problem}")
            final_state = {
                **initial_state,
                "audit_result": "error",
                "audit_reason": f"Skipped, no LLM call: {problem}.",
            }
        else:
            print(f"  Running graph for '{mock_id}'...")
            try:
                final_state: dict = app.invoke(initial_state, config=config)
            except Exception as exc:  # noqa: BLE001
                print(f"  {WARN_ICON} Graph raised an exception for '{mock_id}': {exc}")
                logger.exception("Graph error for mock '%s'", mock_id)
                final_state = {
                    **initial_state,
                    "audit_result": "error",
                    "audit_reason": str(exc),
                }

        _print_result(final_state, mock_id)

        # Collect for audit trail export
        entry = {
            "mock_id":       mock_id,
            "run_id":        run_id,
            "thread_id":     thread_id,
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "audit_result":  final_state.get("audit_result"),
            "audit_reason":  final_state.get("audit_reason"),
            "retry_count":   final_state.get("retry_count"),
            "maker_output":  (
                json.loads(final_state["maker_output"])
                if final_state.get("maker_output")
                else None
            ),
        }
        audit_trail.append(entry)
        _append_history(entry)
        _save_audit_trail(audit_trail)

    # -- Summary & export ----------------------------------------------------
    _print_summary(audit_trail)

    _save_audit_trail(audit_trail)  # also writes "[]" when there were no invoices
    print(f"  Audit trail written to:\n      {_AUDIT_FILE}\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _force_utf8_output()
    run_all_mocks()
