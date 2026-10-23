"""Checker rules, tested directly with hand-made Maker outputs (no LLM at all)."""
import json
import math

import pytest

from src.nodes.checker import run_checker
from tests.conftest import EUROPA, INV1, INV3, good


def audit(data: dict, text: str = INV1) -> str:
    return run_checker({"maker_output": json.dumps(data), "input": text})["audit_result"]


def test_correct_extraction_passes():
    assert audit(good()) == "pass"


# --- BUG-01: every field must come from the invoice text --------------------

@pytest.mark.parametrize("wrong", [
    {"total_amount": 999.0},             # invented total
    {"total_amount": 400000.0},          # the Subtotal, not the Total
    {"vendor_name": "Totally Fake Corp"},
    {"vendor_name": "Infosys Limited"},  # the customer on the "Bill To" line
    {"due_date": "2030-01-01"},          # invented due date
    {"due_date": "2024-07-15"},          # the invoice date, not the due date
])
def test_bug01_values_not_on_the_invoice_fail(wrong):
    assert audit(good(**wrong)) == "fail"


# --- BUG-02: currency must really be what the invoice states ----------------

MAPLE = {"vendor_name": "Maple Leaf Supplies Inc.", "invoice_number": "ML-7781",
         "total_amount": 1200.0, "due_date": "2024-09-30", "currency": "USD"}
MAPLE_TEXT = ("INVOICE\nVendor: Maple Leaf Supplies Inc.\nInvoice Number: ML-7781\n"
              "Due Date: 2024-09-30\nTotal Due: C$ 1,200.00\n")


def test_bug02_canadian_dollar_is_not_usd():
    assert audit(MAPLE, MAPLE_TEXT) == "fail"


def test_bug02_plain_dollar_invoice_is_usd():
    assert audit(MAPLE, MAPLE_TEXT.replace("C$", "$")) == "pass"


def test_bug02_stray_usd_in_a_note_does_not_beat_the_currency_line():
    text = INV3 + "\nReference rate: 1 EUR = 1.08 USD"
    assert audit({**EUROPA, "currency": "USD"}, text) == "fail"


def test_bug02_rupees_written_as_rs_are_inr():
    text = INV1.replace("Currency: INR\n", "").replace("(INR)", "(Rs.)")
    assert audit(good(), text) == "pass"


def test_bug02_rs_inside_a_word_is_not_inr():
    text = MAPLE_TEXT.replace("C$", "$") + "Ref: Purchase Orders 4411\n"
    assert audit({**MAPLE, "currency": "INR"}, text) == "fail"


# --- BUG-03: document problems are rejected, not retried ---------------------

def test_bug03_missing_field_is_rejected():
    no_due_date = INV1.replace("Due Date:       2024-08-14\n", "")
    assert audit(good(due_date=""), no_due_date) == "reject"


def test_bug03_placeholder_counts_as_missing():
    text = INV1.replace("INV-2024-BLR-0089", "N/A")
    assert audit(good(invoice_number="N/A"), text) == "reject"


def test_bug03_negative_total_is_rejected():
    assert audit(good(total_amount=-472000.0)) == "reject"


# --- BUG-04: dates must be real and unambiguous -----------------------------

@pytest.mark.parametrize("bad_date", ["2024-02-30", "2024-13-45", "२०२४-०८-१४"])
def test_bug04_impossible_or_non_ascii_dates_fail(bad_date):
    assert audit(good(due_date=bad_date)) == "fail"


AMBIGUOUS = INV1.replace("Due Date:       2024-08-14", "Due Date:       03/04/2024")


@pytest.mark.parametrize("reading", ["2024-03-04", "2024-04-03"])
def test_bug04_ambiguous_day_month_is_rejected(reading):
    assert audit(good(due_date=reading), AMBIGUOUS) == "reject"


def test_bug04_another_date_on_the_invoice_decides_the_order():
    # 25/03/2024 can only be day-first, so 03/04/2024 means 3 April.
    text = AMBIGUOUS.replace("Invoice Date:   2024-07-15", "Invoice Date:   25/03/2024")
    assert audit(good(due_date="2024-04-03"), text) == "pass"
    assert audit(good(due_date="2024-03-04"), text) == "fail"


# --- BUG-05: invoice number must match as a whole code ----------------------

@pytest.mark.parametrize("junk", ["INV-2024", "2024", "1"])
def test_bug05_partial_or_junk_invoice_numbers_fail(junk):
    assert audit(good(invoice_number=junk)) == "fail"


@pytest.mark.parametrize("printed", ["INV 2024 BLR 0089", "INV–2024–BLR–0089", "INV-2024-\nBLR-0089"])
def test_bug05_same_number_with_other_separators_passes(printed):
    assert audit(good(), INV1.replace("INV-2024-BLR-0089", printed)) == "pass"


# --- BUG-09: NaN / Infinity are not amounts ---------------------------------

@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_bug09_nan_and_infinity_fail(value):
    assert audit(good(total_amount=value)) == "fail"


# --- BUG-17: a single retry limit -------------------------------------------

def test_bug17_checker_has_no_second_retry_limit():
    import src.nodes.checker as checker
    assert not hasattr(checker, "MAX_RETRY_COUNT")


# --- Each rule below is tested on new invoice text written for the test -----

# Part-paid: only the "Amount Due" line is still owed.
PART_PAID = ("Harbor Lane Printing\n12 Wharf Road, Halifax\n\nInvoice No: HLP-5521\n"
             "Due Date: 2025-03-14\nBill To: Oakmont Library\n\n"
             "Invoice Total        $4,200.00\nPayment received    -$1,200.00\n"
             "Amount Due           $3,000.00\n\nCurrency: USD\n")
HARBOR = {"vendor_name": "Harbor Lane Printing", "invoice_number": "HLP-5521",
          "total_amount": 3000.0, "due_date": "2025-03-14", "currency": "USD"}


def test_total_before_a_payment_is_not_the_amount_due():
    assert audit({**HARBOR, "total_amount": 4200.0}, PART_PAID) == "fail"


def test_amount_due_line_gives_the_total():
    assert audit(HARBOR, PART_PAID) == "pass"


# Buyer on a "Client:" line; the vendor is only named after "due to".
PHOTO = ("TAX INVOICE\nInvoice No: MIP-0042\nDue Date: 2025-02-28\n"
         "Client: Bluefin Hotels Pvt. Ltd.\n\nWedding shoot (2 days)    Rs. 18,000.00\n"
         "Total: Rs. 18,000.00\nCurrency: INR\n\n"
         "Payment due to Meera Iyer Photography (UPI: meera@okbank)\n")
MEERA = {"vendor_name": "Meera Iyer Photography", "invoice_number": "MIP-0042",
         "total_amount": 18000.0, "due_date": "2025-02-28", "currency": "INR"}


def test_customer_on_a_client_line_is_not_the_vendor():
    assert audit({**MEERA, "vendor_name": "Bluefin Hotels Pvt. Ltd."}, PHOTO) == "fail"


def test_to_in_a_sentence_or_a_name_is_not_a_buyer_label():
    assert audit(MEERA, PHOTO) == "pass"
    # A vendor whose name starts with the word "To" is still the vendor.
    todo = "To-Do Cleaning Services\n" + PHOTO.replace(
        "Payment due to Meera Iyer Photography (UPI: meera@okbank)", "Thank you!")
    assert audit({**MEERA, "vendor_name": "To-Do Cleaning Services"}, todo) == "pass"
