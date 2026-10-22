"""

    .venv\Scripts\python.exe -m src.run_agent

"""

import json
import logging
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

# ---------------------------------------------------------------------------
# Logging — INFO to stdout so it interleaves cleanly with print() output
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,          # suppress DEBUG/INFO chatter from nodes
    format="%(levelname)-8s %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Import after logging config so module-level graph compilation messages are
# still suppressed.
from src.graph import app  # noqa: E402  (intentional late import)


# ---------------------------------------------------------------------------
# Terminal formatting helpers
# ---------------------------------------------------------------------------

_DIVIDER   = "-" * 60
_THICK_DIV = "=" * 60

PASS_ICON = "[PASS]"
FAIL_ICON = "[FAIL]"
WARN_ICON = "[WARN]"


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
    retries     = final_state.get("retry_count", 0)
    icon        = PASS_ICON if result == "pass" else FAIL_ICON

    print(f"  {icon}  Result      : {result.upper()}")
    print(f"  Retries     : {retries}")
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
    passed  = sum(1 for r in results if r["audit_result"] == "pass")
    failed  = len(results) - passed
    print()
    print(_THICK_DIV)
    print("  BATCH SUMMARY")
    print(_THICK_DIV)
    print(f"  Total invoices : {len(results)}")
    print(f"  {PASS_ICON}  Passed        : {passed}")
    print(f"  {FAIL_ICON}  Failed        : {failed}")
    print(_THICK_DIV)
    print()


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_all_mocks() -> None:
    """Load mock invoices, run each through the graph, export audit trail."""

    # -- Load mocks ----------------------------------------------------------
    if not _MOCKS_FILE.exists():
        sys.exit(f"[ERROR] Mock file not found: {_MOCKS_FILE}")

    with _MOCKS_FILE.open(encoding="utf-8") as fh:
        mocks: list[dict] = json.load(fh)

    _print_header()
    print(f"\n  Loaded {len(mocks)} mock invoice(s) from:\n  {_MOCKS_FILE}\n")

    audit_trail: list[dict] = []

    # -- Run each mock -------------------------------------------------------
    for idx, mock in enumerate(mocks, start=1):
        mock_id   = mock.get("id", f"mock-{idx:03d}")
        raw_text  = mock.get("raw_text", "")

        _print_mock_banner(idx, len(mocks), mock_id)

        # Initial state — all AgentState keys must be present
        initial_state = {
            "input":        raw_text,
            "maker_output": "",
            "audit_result": "",
            "audit_reason": "",
            "retry_count":  0,
            "messages":     [],
        }

        # Unique thread_id per mock ensures SQLite checkpoints don't bleed
        # across loop iterations.
        config: RunnableConfig = {
            "configurable": {
                "thread_id": mock_id,
            }
        }

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
        audit_trail.append({
            "mock_id":       mock_id,
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "audit_result":  final_state.get("audit_result"),
            "audit_reason":  final_state.get("audit_reason"),
            "retry_count":   final_state.get("retry_count"),
            "maker_output":  (
                json.loads(final_state["maker_output"])
                if final_state.get("maker_output")
                else None
            ),
        })

    # -- Summary & export ----------------------------------------------------
    _print_summary(audit_trail)

    _AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _AUDIT_FILE.open("w", encoding="utf-8") as fh:
        json.dump(audit_trail, fh, indent=2, ensure_ascii=False)

    print(f"  Audit trail written to:\n      {_AUDIT_FILE}\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all_mocks()
