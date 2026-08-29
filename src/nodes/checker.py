"""
checker.py — Checker node (Agent B) for the LangGraph Maker/Checker workflow.

Responsibilities
----------------
* Parse the Maker's JSON output from ``state["maker_output"]``.
* Run deterministic Python validation rules against the extracted fields.
* Return ``audit_result`` (``"pass"`` | ``"fail"``) and a human-readable
  ``audit_reason`` that the Maker can use to correct its output on retry.

Design note: The Checker is intentionally *not* an LLM call.  Validation
logic that can be expressed as deterministic Python should be — it is faster,
cheaper, and 100% reproducible.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

ALLOWED_CURRENCIES = {"USD", "INR"}
MAX_RETRY_COUNT = 3

# Loose ISO 8601 date pattern: YYYY-MM-DD
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Individual validators
# ---------------------------------------------------------------------------

def _check_vendor_name(vendor_name: str) -> str | None:
    """Return an error string or None if valid."""
    if not vendor_name or not vendor_name.strip():
        return "vendor_name is empty or missing."
    return None


def _check_invoice_number(invoice_number: str) -> str | None:
    if not invoice_number or not invoice_number.strip():
        return "invoice_number is empty or missing."
    return None


def _check_total_amount(total_amount: float) -> str | None:
    if not isinstance(total_amount, (int, float)):
        return f"total_amount is not a number (got {type(total_amount).__name__})."
    if total_amount <= 0:
        return f"total_amount must be greater than 0 (got {total_amount})."
    return None


def _check_currency(currency: str) -> str | None:
    if not currency or currency.strip().upper() not in ALLOWED_CURRENCIES:
        return (
            f"currency '{currency}' is not allowed. "
            f"Expected one of: {sorted(ALLOWED_CURRENCIES)}."
        )
    return None


def _check_due_date(due_date: str) -> str | None:
    if not due_date or not due_date.strip():
        return "due_date is empty or missing."
    if not _DATE_RE.match(due_date.strip()):
        return (
            f"due_date '{due_date}' is not in ISO 8601 format (YYYY-MM-DD). "
            "Example of a valid date: '2024-08-31'."
        )
    return None


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def run_checker(state: dict) -> dict:
    """Checker node: validate the Maker's extracted invoice fields.

    Parameters
    ----------
    state : dict
        The current ``AgentState``. Key fields consumed:
        - ``maker_output`` : JSON string produced by the Maker node.

    Returns
    -------
    dict
        Partial state update:
        - ``audit_result`` : ``"pass"`` or ``"fail"``
        - ``audit_reason`` : Plain-English explanation of the outcome.
    """
    logger.info("Checker node invoked.")

    # ------------------------------------------------------------------
    # 1. Parse the Maker's JSON output
    # ------------------------------------------------------------------
    raw = state.get("maker_output", "")
    if not raw:
        logger.warning("maker_output is empty — failing audit.")
        return {
            "audit_result": "fail",
            "audit_reason": "maker_output is empty; the Maker produced no extraction.",
        }

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse maker_output as JSON: %s", exc)
        return {
            "audit_result": "fail",
            "audit_reason": (
                f"maker_output could not be parsed as JSON: {exc}. "
                "The Maker must return a valid JSON object."
            ),
        }

    # ------------------------------------------------------------------
    # 2. Run all validators — collect every failure, not just the first
    # ------------------------------------------------------------------
    errors: list[str] = []

    for checker_fn, field_key in [
        (_check_vendor_name,    "vendor_name"),
        (_check_invoice_number, "invoice_number"),
        (_check_total_amount,   "total_amount"),
        (_check_currency,       "currency"),
        (_check_due_date,       "due_date"),
    ]:
        error = checker_fn(data.get(field_key))  # type: ignore[arg-type]
        if error:
            errors.append(error)
            logger.debug("Validation failure: %s", error)

    # ------------------------------------------------------------------
    # 3. Return structured audit result
    # ------------------------------------------------------------------
    if errors:
        reason = " | ".join(errors)
        logger.info("Audit FAILED: %s", reason)
        return {
            "audit_result": "fail",
            "audit_reason": reason,
        }

    logger.info("Audit PASSED.")
    return {
        "audit_result": "pass",
        "audit_reason": "All checks passed.",
    }
