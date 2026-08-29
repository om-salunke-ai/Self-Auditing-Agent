"""
schema.py — Pydantic models for structured LLM output.

This module defines the expected extraction schema for the Maker node.
The `InvoiceExtraction` model is passed to `llm.with_structured_output()`
so LangChain enforces type-safe, validated JSON from the LLM response.
"""

from pydantic import BaseModel, Field


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
        description="Total amount due, as a positive numeric value (e.g., 5934.49)."
    )
    due_date: str = Field(
        description="Payment due date in ISO 8601 format: YYYY-MM-DD (e.g., '2024-08-31')."
    )
    currency: str = Field(
        description="Three-letter ISO 4217 currency code (e.g., 'INR', 'USD')."
    )
