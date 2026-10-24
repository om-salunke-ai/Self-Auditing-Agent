from pydantic import BaseModel, Field, field_validator

# Currency symbols the LLM may copy from an invoice, and their ISO codes.
_SYMBOL_TO_CODE = {"₹": "INR", "RS": "INR", "RS.": "INR",
                   "$": "USD", "US$": "USD", "€": "EUR", "£": "GBP"}


class InvoiceExtraction(BaseModel):
    """Structured output expected from the Maker (Agent A) node.

    All fields are extracted from raw invoice text. The Checker (Agent B)
    validates these fields against business rules defined separately.
    """

    vendor_name: str = Field(
        description="Full legal name of the vendor or supplier issuing the invoice."
    )
    invoice_number: str = Field(
        description="Unique identifier for the invoice (e.g., 'INV-2024-00042')."
    )
    total_amount: float = Field(
        description="Total amount due (e.g., 5934.49). Negative for a credit note (e.g., -12500.0).",
        allow_inf_nan=False,  # "NaN"/"Infinity" would slip past "<= 0" checks 
    )
    due_date: str = Field(
        description="Payment due date in ISO 8601 format: YYYY-MM-DD (e.g., '2024-08-31')."
    )
    currency: str = Field(
        description="Three-letter ISO 4217 currency code (e.g., 'INR', 'USD')."
    )

    @field_validator("currency")
    @classmethod
    def _clean_currency(cls, value: str) -> str:
        """Store the currency as an upper-case ISO code, so " usd " becomes
        "USD" and "€" becomes "EUR" before the Checker sees it."""
        value = value.strip().upper()
        return _SYMBOL_TO_CODE.get(value, value)
