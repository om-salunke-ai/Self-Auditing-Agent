import json
import logging
import math
import re
from datetime import date

logger = logging.getLogger(__name__)


ALLOWED_CURRENCIES = {"USD", "INR"}

# ISO 8601 date shape: YYYY-MM-DD. [0-9] instead of \d, because \d also
# matches other digits such as "२०२४".
_DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")


_CURRENCY_SYMBOLS = {
    "INR": ("INR", "₹", "RS", "RUPEE", "RUPEES"),
    "USD": ("USD", "US$", "$", "US DOLLAR", "US DOLLARS"),
    "EUR": ("EUR", "€", "EURO", "EUROS"),
    "GBP": ("GBP", "£"),
}

# When one of these appears, a bare "$" does not prove US dollars.
_OTHER_DOLLAR_CODES = ("CAD", "AUD", "NZD", "SGD", "HKD")
# A line that states the currency, e.g. "Currency: INR".
_CURRENCY_LINE_RE = re.compile(r"\bCURRENCY\s*:")

# Values that mean "the invoice doesn't have this", compared after _squash():
# "N/A" -> "NA", "[NOT PROVIDED]" -> "NOTPROVIDED", "-" -> "".
_PLACEHOLDERS = {"", "NA", "NONE", "NULL", "NIL", "TBD", "UNKNOWN",
                 "NOTPROVIDED", "NOTAVAILABLE", "PENDING"}

# Labelled lines used for grounding. \b stops "SUBTOTAL" counting as a total.
_TOTAL_LINE_RE = re.compile(r"\b(TOTAL|AMOUNT DUE|BALANCE DUE|AMOUNT PAYABLE)\b")
# Lines for what is still owed. When an invoice has one, the total must come
# from it: an "Invoice Total" line may be from before a deposit was paid.
_OWED_LINE_RE = re.compile(r"\b(AMOUNT DUE|BALANCE DUE|AMOUNT PAYABLE)\b")
_DUE_LINE_RE = re.compile(r"\b(DUE|PAY BY)\b")
# A line that labels the invoice number: "Invoice No", "Inv. No", "Invoice #",
# "Bill No", "Credit Note No". PO, challan or consumer numbers have other labels.
_INVOICE_NO_LINE_RE = re.compile(
    r"\b(INVOICE|INV|BILL|CREDIT NOTE)\b\.?\s*(NO|NUMBER)\b|\b(INVOICE|INV)\s*#"
)
# Scans can put spaces around a separator inside a code: "GST-T/ 2024 /  561".
# A "/" or dash with spaces still joins two parts of the code when the part
# across it has a digit, so "INV-0089 - Paid" still ends at "0089".
_SPACED_SEP = r"[ \t]*[-/\u2010-\u2015\u2212][ \t]*"
_CODE_GOES_ON_RE = re.compile(rf"{_SPACED_SEP}[^\W_]*[0-9]")
_CODE_CAME_BEFORE_RE = re.compile(rf"[0-9][^\W_]*{_SPACED_SEP}$")
# Lines that name the customer, not the vendor. "To:" and "Client:" count only
# at the start of a line and with a colon, so neither "due to" nor a vendor
# called "To-Do Cleaning" is taken for a buyer label.
_BUYER_LINE_RE = re.compile(
    r"\b(BILL(ED)? TO|SHIP(PED)? TO|SOLD TO|CUSTOMER|BUYER)\b|^\s*(TO|CLIENT( NAME)?)\s*:"
)

# A number as printed on an invoice: 472000, 4,72,000.00, 34,926.50, 1.200,00
_NUMBER_RE = re.compile(r"\d[\d.,]*\d|\d")

_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
_YMD_RE = re.compile(r"\b([0-9]{4})[-/.]([0-9]{1,2})[-/.]([0-9]{1,2})\b")        # 2024-08-14
_XXY_RE = re.compile(r"\b([0-9]{1,2})[-/.]([0-9]{1,2})[-/.]([0-9]{4})\b")        # 14/08/2024
_DAY_MON_RE = re.compile(
    r"\b([0-9]{1,2})(?:ST|ND|RD|TH)?[\s-]+([A-Z]{3})[A-Z]*\.?[\s,-]+([0-9]{4})\b"  # 14 Aug 2024
)
_MON_DAY_RE = re.compile(
    r"\b([A-Z]{3})[A-Z]*\.?\s+([0-9]{1,2})(?:ST|ND|RD|TH)?,?\s+([0-9]{4})\b"       # Aug 14, 2024
)


# Helpers for reading the invoice text

def _lines_with(text: str, label_re: re.Pattern) -> str:
    """Only the lines of `text` that match `label_re` (e.g. the "Total" lines)."""
    return "\n".join(line for line in text.splitlines() if label_re.search(line))


def _written_in(word: str, text: str) -> bool:
    """Whole-word search in upper-case text: "RS" doesn't match inside
    "ORDERS", and "$" doesn't match the "$" of "C$" or "A$"."""
    return re.search(rf"(?<![A-Z]){re.escape(word)}(?![A-Z])", text) is not None


def _currency_in_text(currency: str, text: str) -> bool:
    """Is `currency` what the invoice really states? """
    ways = _CURRENCY_SYMBOLS.get(currency, (currency,))
    if currency == "USD" and any(_written_in(code, text) for code in _OTHER_DOLLAR_CODES):
        ways = tuple(w for w in ways if w != "$")  # here "$" may be another dollar
    # If the invoice has a "Currency:" line, only that line counts, so a stray
    # "USD" in a note (e.g. an exchange rate) is not enough.
    where = _lines_with(text, _CURRENCY_LINE_RE) or text
    return any(_written_in(w, where) for w in ways)


def _squash(value: str) -> str:
    """Uppercase and keep only letters and digits: "Pvt. Ltd." -> "PVTLTD"."""
    return re.sub(r"[\W_]+", "", value.upper())


def _invoice_number_lines(text: str) -> list[tuple[int, int]]:
    """Start and end positions of the labelled invoice-number lines that hold
    a digit. A label with no code on it ("Invoice No: (to be assigned)", or a
    number printed on the next line) is ignored."""
    spans, start = [], 0
    for line in text.split("\n"):
        if _INVOICE_NO_LINE_RE.search(line) and re.search(r"[0-9]", line):
            spans.append((start, start + len(line)))
        start += len(line) + 1
    return spans


def _invoice_number_in_text(invoice_number: str, text: str) -> bool:
    """Is the invoice number printed in the text as a whole code?

    Separators may differ ("INV 2024 0089", "INV–2024–0089" or a line break
    all match "INV-2024-0089"), but a piece of a longer code ("INV-2024"
    inside "INV-2024-BLR-0089") does not count.
    """
    pieces = re.findall(r"[^\W_]+", invoice_number.upper())  # the letter/digit parts
    if not pieces:
        return False
    body = r"[\W_]*".join(re.escape(piece) for piece in pieces)
    # Nothing that continues the code may touch it on either side: no
    # letter/digit, and no "-X", "/X" or ".X" (any kind of dash).
    sep = r"[-/.\u2010-\u2015\u2212]"  # "-", "/", "." and Unicode dashes like "–"
    before = rf"(?<![^\W_])(?<![^\W_]{sep})"
    after = rf"(?![^\W_])(?!{sep}[^\W_])"
    # When the invoice has a labelled number line with a code on it, the
    # number must start on that line: a PO, challan or consumer number printed
    # elsewhere doesn't count. (It may run on to the next line if wrapped.)
    labelled = _invoice_number_lines(text)
    for match in re.finditer(before + body + after, text):
        if (_CODE_GOES_ON_RE.match(text, match.end())
                or _CODE_CAME_BEFORE_RE.search(text, 0, match.start())):
            continue  # only a piece of a longer code that a scan spread out
        if not labelled or any(start <= match.start() < end for start, end in labelled):
            return True
    return False


def _is_missing(value) -> bool:
    """True when the Maker found nothing: None, "", or a placeholder like "N/A"."""
    return value is None or (isinstance(value, str) and _squash(value) in _PLACEHOLDERS)


def _numbers_in(text: str) -> set[float]:
    """Every number in the text, read the English way (1,200.50) and the
    European way (1.200,50), so an amount matches in either style."""
    numbers = set()
    for raw in _NUMBER_RE.findall(text):
        for candidate in (raw.replace(",", ""), raw.replace(".", "").replace(",", ".")):
            try:
                numbers.add(round(float(candidate), 2))
            except ValueError:
                pass  # e.g. "4.72.00000" is not a number in this reading
    return numbers


def _printed_totals(text: str) -> set[float]:
    """Amounts that can be the invoice total, taken from the best lines the
    invoice has: what is still owed ("Balance Due"), else any "Total" line,
    else any number in the text."""
    for where in (_lines_with(text, _OWED_LINE_RE), _lines_with(text, _TOTAL_LINE_RE), text):
        numbers = _numbers_in(where)
        if numbers:
            return numbers
    return set()


def _make_date(year, month, day) -> date | None:
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None  # not a real day, e.g. 30 February


def _iso_date(value) -> date | None:
    """The date in a "YYYY-MM-DD" string, or None if it isn't a real one."""
    if not isinstance(value, str) or not _DATE_RE.match(value.strip()):
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _date_order(text: str) -> tuple[bool, bool]:
    """Clues for reading dates like 03/04/2024: 25/07/2024 can only be
    day-first, 07/25/2024 can only be month-first (BUG-04)."""
    pairs = _XXY_RE.findall(text)
    return any(int(a) > 12 for a, _, _ in pairs), any(int(b) > 12 for _, b, _ in pairs)


def _dates_in(text: str, day_first: bool = False, month_first: bool = False) -> list[set[date]]:
    """Every date written in the text, as the set of days it could mean:
    "2024-08-14" -> {14 Aug}; "03/04/2024" -> {3 Apr, 4 Mar} unless a clue
    says which order the invoice uses."""
    found = [{_make_date(y, m, d)} for y, m, d in _YMD_RE.findall(text)]
    for a, b, y in _XXY_RE.findall(text):
        days = set()
        if day_first or not month_first:
            days.add(_make_date(y, b, a))  # a is the day
        if month_first or not day_first:
            days.add(_make_date(y, a, b))  # a is the month
        found.append(days)
    for d, mon, y in _DAY_MON_RE.findall(text):
        if mon in _MONTHS:
            found.append({_make_date(y, _MONTHS.index(mon) + 1, d)})
    for mon, d, y in _MON_DAY_RE.findall(text):
        if mon in _MONTHS:
            found.append({_make_date(y, _MONTHS.index(mon) + 1, d)})
    return [days - {None} for days in found if days - {None}]


def _due_date_readings(text: str) -> list[set[date]]:
    """Dates on the "Due" line(s); if those lines hold no date, every date in the text."""
    order = _date_order(text)  # clues may come from any date on the invoice
    return _dates_in(_lines_with(text, _DUE_LINE_RE), *order) or _dates_in(text, *order)


def _due_date_alternatives(due_date, text: str) -> list[date]:
    """If the due date is only written like 03/04/2024 (3 April or 4 March?)
    and nothing on the invoice says which, return both possible days (BUG-04)."""
    due = _iso_date(due_date)
    matches = [days for days in _due_date_readings(text) if due in days]
    if matches and all(len(days) > 1 for days in matches):
        return sorted(matches[0])
    return []



def _check_vendor_name(vendor_name: str) -> str | None:
    """Return an error string or None if valid."""
    if not vendor_name or not vendor_name.strip():
        return "vendor_name is empty or missing."
    return None


def _check_invoice_number(invoice_number: str) -> str | None:
    if not invoice_number or not invoice_number.strip():
        return "invoice_number is empty or missing."
    # "1" or "INVOICE" would match almost any text, so a real invoice number
    # needs a digit and at least 3 letters/digits (BUG-05).
    if len(_squash(invoice_number)) < 3 or not re.search(r"[0-9]", invoice_number):
        return (
            f"invoice_number '{invoice_number}' is too short or has no digit; "
            "copy the full invoice number from the text."
        )
    return None


def _check_total_amount(total_amount: float) -> str | None:
    if not isinstance(total_amount, (int, float)):
        return f"total_amount is not a number (got {type(total_amount).__name__})."
    # NaN and Infinity are floats too, and "nan <= 0" is False (BUG-09).
    if not math.isfinite(total_amount):
        return f"total_amount must be a real number (got {total_amount})."
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

    # Missing values (empty / "N/A") are not grounded here; step 2b rejects them.
    currency = (data.get("currency") or "").strip().upper()
    if not _is_missing(currency):
        if not _currency_in_text(currency, text):
            errors.append(
                f"currency '{currency}' does not appear in the invoice text. "
                "Report the currency exactly as the invoice states it."
            )

    invoice_number = (data.get("invoice_number") or "").strip()
    if not _is_missing(invoice_number) and not _invoice_number_in_text(invoice_number, text):
        errors.append(
            f"invoice_number '{invoice_number}' does not appear in the invoice text."
        )

    # total_amount, vendor_name and due_date must come from the text too (BUG-01).
    amount = data.get("total_amount")
    if isinstance(amount, (int, float)) and amount > 0:
        # Only the labelled total lines count, so a Subtotal or a line-item
        # price can't pass as the total.
        if round(amount, 2) not in _printed_totals(text):
            errors.append(
                f"total_amount {amount} is not the total printed on the invoice "
                "(read the Total / Amount Due line)."
            )

    vendor = (data.get("vendor_name") or "").strip()
    if not _is_missing(vendor):
        name = _squash(vendor)
        vendor_lines = [line for line in text.splitlines() if name in _squash(line)]
        if name not in _squash(text):
            errors.append(f"vendor_name '{vendor}' does not appear in the invoice text.")
        elif vendor_lines and all(_BUYER_LINE_RE.search(line) for line in vendor_lines):
            errors.append(
                f"vendor_name '{vendor}' is the customer (Bill To / Ship To), "
                "not the vendor who issued the invoice."
            )

    due = _iso_date(data.get("due_date"))
    if due and not any(due in days for days in _due_date_readings(text)):
        errors.append(
            f"due_date '{data['due_date']}' is not the due date written on the invoice."
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
    if _iso_date(due_date) is None:
        # Right shape, but the day doesn't exist, e.g. 2024-02-30 (BUG-04).
        return f"due_date '{due_date}' is not a real calendar date."
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
        - ``audit_result`` : ``"pass"``, ``"fail"`` (retry) or ``"reject"`` (no retry)
        - ``audit_reason`` : Plain-English explanation of the outcome.
    """
    logger.info("Checker node invoked.")

    # ------------------------------------------------------------------
    # 1. Parse the Maker's JSON output
    # ------------------------------------------------------------------
    raw = state.get("maker_output", "")
    if not raw:
        logger.warning("maker_output is empty — failing audit.")
        # When the LLM call failed, the Maker left the error in audit_reason
        # (BUG-07); keep it so the audit trail says what went wrong.
        return {
            "audit_result": "fail",
            "audit_reason": state.get("audit_reason")
            or "maker_output is empty; the Maker produced no extraction.",
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

    # A field the invoice doesn't have, or a zero/negative total, is a problem
    # with the document, not a Maker mistake. Retrying would only push the
    # Maker to invent a value, so a human must review it (BUG-03).
    problems = [
        f"{field} is missing from the invoice"
        for field in ("vendor_name", "invoice_number", "due_date", "currency")
        if _is_missing(data.get(field))
    ]
    amount = data.get("total_amount")
    if isinstance(amount, (int, float)) and amount <= 0:
        problems.append(
            f"total_amount is {amount}; a zero or negative total "
            "(e.g. a credit note) cannot be paid automatically"
        )
    alternatives = _due_date_alternatives(data.get("due_date"), state.get("input", "").upper())
    if alternatives:
        problems.append(
            "due_date is ambiguous on the invoice: it could be "
            + " or ".join(day.isoformat() for day in alternatives)
        )
    if problems:
        reason = " | ".join(problems) + ". Needs human review."
        logger.info("Audit REJECTED: %s", reason)
        return {"audit_result": "reject", "audit_reason": reason}

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
