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

# Other ways an allowed currency may be written on an invoice.
_CURRENCY_SYMBOLS = {"INR": ("INR", "₹", "Rs", "₹"), "USD": ("USD", "$")}



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
    """Format only: is it a three-letter code? (Policy is checked separately.)"""
    if not currency or not re.fullmatch(r"[A-Z]{3}", currency.strip().upper()):
        return f"currency '{currency}' is not a three-letter ISO 4217 code."
    return None


def _check_currency_policy(currency: str) -> str | None:
    """Business rule the *document* can break. Retrying cannot fix this
    without inventing data, so it ends the run as "reject"."""
    if currency.strip().upper() not in ALLOWED_CURRENCIES:
        return f"currency '{currency}' is not accepted by policy."
    return None


def _check_grounded(data: dict, source_text: str) -> list[str]:
    """Values the Maker reports must actually appear in the invoice text."""
    errors: list[str] = []
    text = source_text.upper()

    currency = (data.get("currency") or "").strip().upper()
    if currency:
        ways_to_write = _CURRENCY_SYMBOLS.get(currency, (currency,))
        if not any(w.upper() in text for w in ways_to_write):
            errors.append(
                f"currency '{currency}' does not appear in the invoice text. "
                "Report the currency exactly as the invoice states it."
            )

    invoice_number = (data.get("invoice_number") or "").strip()
    if invoice_number and invoice_number.upper() not in text:
        errors.append(
            f"invoice_number '{invoice_number}' does not appear in the invoice text."
        )
    return errors


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
    # 2a. Grounding — did the Maker invent a value? (retryable)
    # ------------------------------------------------------------------
    grounding_errors = _check_grounded(data, state.get("input", ""))
    if grounding_errors:
        reason = " | ".join(grounding_errors)
        logger.info("Audit FAILED (not grounded): %s", reason)
        return {"audit_result": "fail", "audit_reason": reason}

    # ------------------------------------------------------------------
    # 2b. Policy — the document itself breaks a business rule (final)
    # ------------------------------------------------------------------
    if not _check_currency(data.get("currency")):
        policy_error = _check_currency_policy(data["currency"])
        if policy_error:
            logger.info("Audit REJECTED: %s", policy_error)
            return {"audit_result": "reject", "audit_reason": policy_error}

    # ------------------------------------------------------------------
    # 2c. Format validators — collect every failure, not just the first
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
