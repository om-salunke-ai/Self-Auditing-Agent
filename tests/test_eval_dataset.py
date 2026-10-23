"""The labelled evaluation set in data/eval/labelled_invoices.json.

1. Are the "known correct answers" really correct? These checks read the
   invoice text directly and never use the Checker, so a wrong label can't
   hide behind a Checker bug.
2. Does the Checker's score stay at least as good as it is today?
"""
import json
import re
from datetime import date

import pytest

from src.evaluate import DATASET_FILE, FIELDS, bad_extractions, evaluate_checker

INVOICES = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
IDS = [inv["id"] for inv in INVOICES]

ALLOWED = {"INR", "USD"}
CURRENCY_MARKS = {"INR": ("INR", "₹", "Rs"), "USD": ("USD", "$"), "EUR": ("EUR", "€"),
                  "GBP": ("GBP", "£"), "CAD": ("CAD", "C$"), "AED": ("AED",)}


def _squash(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.upper())


def _lakh(x: float) -> str:
    """Indian grouping: 125000000 -> 12,50,00,000.00"""
    whole, frac = f"{x:.2f}".split(".")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return ",".join(([head] if head else []) + groups + [tail]) + "." + frac


def _ways_to_print_amount(x: float) -> set[str]:
    a = abs(x)
    english = f"{a:,.2f}"                                                    # 1,18,000 / 118,000.00
    european = english.replace(",", "X").replace(".", ",").replace("X", ".")  # 4.860,00
    return {english, european, _lakh(a)}


def _ways_to_print_date(iso: str) -> set[str]:
    d = date.fromisoformat(iso)
    mon, month = d.strftime("%b"), d.strftime("%B")
    return {iso, d.strftime("%d/%m/%Y"), d.strftime("%d-%m-%Y"),
            f"{d.day:02d} {mon} {d.year}", f"{d.day} {mon} {d.year}", f"{month} {d.day}, {d.year}"}


def test_dataset_size_and_unique_ids():
    assert len(INVOICES) >= 25
    assert len(set(IDS)) == len(IDS)


@pytest.mark.parametrize("inv", INVOICES, ids=IDS)
def test_labels_are_well_formed(inv):
    assert inv["expected_result"] in ("pass", "reject")
    assert set(inv["expected"]) == set(FIELDS)
    assert isinstance(inv["expected"]["total_amount"], (int, float))
    if inv["expected"]["due_date"]:
        date.fromisoformat(inv["expected"]["due_date"])  # raises if not a real date


# The kind of line each answer must be printed on. Being printed somewhere
# is not enough: a subtotal or an invoice date is printed too.
TOTAL_LINE = re.compile(r"\b(total|amount|balance|payable)\b", re.IGNORECASE)
DUE_LINE = re.compile(r"\b(due|pay by)\b", re.IGNORECASE)
NUMBER_LINE = re.compile(r"\b(invoice|inv|bill no|credit note no)\b|#", re.IGNORECASE)
BUYER_LINE = re.compile(r"^\s*(bill to|ship to|sold to|customer|client|to)\b", re.IGNORECASE)


def _lines(text: str, label: re.Pattern) -> list[str]:
    return [line for line in text.splitlines() if label.search(line)]


def _amount_on(line: str, printed: str) -> bool:
    """Whole-number match, so "412.50" doesn't count inside "5,412.50"."""
    return re.search(rf"(?<![\d.,]){re.escape(printed)}(?!\d)", line) is not None


@pytest.mark.parametrize("inv", INVOICES, ids=IDS)
def test_known_answer_is_printed_on_the_right_line(inv):
    text, want = inv["raw_text"], inv["expected"]
    if want["vendor_name"]:
        seller_lines = [l for l in text.splitlines() if not BUYER_LINE.search(l)]
        assert any(_squash(want["vendor_name"]) in _squash(l) for l in seller_lines)
    if want["invoice_number"]:
        assert any(_squash(want["invoice_number"]) in _squash(l) for l in _lines(text, NUMBER_LINE))
    assert any(_amount_on(line, p) for line in _lines(text, TOTAL_LINE)
               for p in _ways_to_print_amount(want["total_amount"]))
    if want["due_date"]:
        assert any(p in line for line in _lines(text, DUE_LINE)
                   for p in _ways_to_print_date(want["due_date"]))
    assert any(mark in text for mark in CURRENCY_MARKS[want["currency"]])


@pytest.mark.parametrize("inv", INVOICES, ids=IDS)
def test_expected_verdict_follows_the_business_rules(inv):
    """Reject when the document itself is the problem: a currency we don't
    accept, a missing field, a zero/negative total, or a due date that can
    be read two ways. Otherwise the invoice should pass."""
    want = inv["expected"]
    document_problem = (
        want["currency"] not in ALLOWED
        or not all(want[f] for f in ("vendor_name", "invoice_number", "due_date", "currency"))
        or want["total_amount"] <= 0
        or "due_date" in inv.get("also_acceptable", {})
    )
    assert (inv["expected_result"] == "reject") == document_problem


@pytest.mark.parametrize("inv", INVOICES, ids=IDS)
def test_every_generated_bad_answer_is_really_wrong(inv):
    for bad in bad_extractions(inv):
        assert bad["data"] != inv["expected"], bad["kind"]


# Measured on 22 Oct 2026: 221 of 231 bad answers caught (10 missed), and
# 30 of 32 correct answers given the right verdict. When you improve the
# Checker, lower these numbers so the improvement can't be lost later.
MAX_MISSED = 10
MAX_WRONG_VERDICTS = 2


def test_checker_score_does_not_get_worse():
    result = evaluate_checker(INVOICES)
    missed = [f"{b['id']} {b['kind']}" for b in result["bad"] if not b["caught"]]
    wrong_verdicts = [f"{c['id']} {c['label']}" for c in result["correct"] if c["got"] != c["expected"]]
    assert len(missed) <= MAX_MISSED, missed
    assert len(wrong_verdicts) <= MAX_WRONG_VERDICTS, wrong_verdicts
