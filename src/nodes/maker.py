"""
maker.py — Maker node (Agent A) for the LangGraph Maker/Checker workflow.

Responsibilities
----------------
* Extract structured invoice data from raw text using an LLM.
* On retry (when ``state["audit_result"] == "fail"``), inject the Checker's
  ``audit_reason`` into the prompt so the model knows exactly what to fix.
* Return a partial state dict that LangGraph merges into the running state.
"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import llm
from src.schema import InvoiceExtraction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a precise invoice data-extraction assistant.
Your sole job is to read the raw invoice text provided by the user and extract
the following fields with exact accuracy:

  • vendor_name    — Full legal name of the vendor/supplier
  • invoice_number — Unique invoice identifier (e.g. INV-2024-00042)
  • total_amount   — Final amount due as a positive float (e.g. 5934.49)
  • due_date       — Payment due date strictly in ISO 8601 format: YYYY-MM-DD
  • currency       — Three-letter ISO 4217 currency code (e.g. INR, USD)

Rules:
  - If a field is genuinely absent from the text, use an empty string "" for
    string fields and 0.0 for total_amount.
  - Do NOT invent or infer data that is not present.
  - Return ONLY the structured fields — no prose, no extra keys."""

_RETRY_ADDENDUM = """
⚠️  IMPORTANT — PREVIOUS ATTEMPT FAILED AUDIT ⚠️
The last extraction was rejected for the following reason:

  {audit_reason}

Carefully re-read the invoice text and correct exactly the issues described
above. Do not change fields that were already correct."""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def run_maker(state: dict) -> dict:
    """Maker node: extract ``InvoiceExtraction`` fields from raw invoice text.

    Parameters
    ----------
    state : dict
        The current ``AgentState``. Key fields consumed:
        - ``input``        : raw invoice text to parse
        - ``audit_result`` : ``"fail"`` triggers retry-aware prompt injection
        - ``audit_reason`` : Checker's explanation, injected on retry
        - ``retry_count``  : incremented and returned

    Returns
    -------
    dict
        Partial state update:
        - ``maker_output`` : JSON string of the extracted ``InvoiceExtraction``
        - ``retry_count``  : previous count + 1
    """
    current_retry = state.get("retry_count", 0)
    is_retry = state.get("audit_result") == "fail"

    logger.info(
        "Maker node invoked | retry=%s | attempt=%d",
        is_retry,
        current_retry + 1,
    )

    # ------------------------------------------------------------------
    # Build the system message, optionally appending retry context
    # ------------------------------------------------------------------
    system_content = _SYSTEM_PROMPT
    if is_retry:
        audit_reason = state.get("audit_reason", "No reason provided.")
        system_content += _RETRY_ADDENDUM.format(audit_reason=audit_reason)
        logger.debug("Retry context injected: %s", audit_reason)

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=f"Extract invoice data from the following text:\n\n{state['input']}"),
    ]

    # ------------------------------------------------------------------
    # Bind structured output schema and invoke the LLM
    # ------------------------------------------------------------------
    structured_llm = llm.with_structured_output(InvoiceExtraction)

    logger.debug("Calling LLM with %d message(s)...", len(messages))
    extraction: InvoiceExtraction = structured_llm.invoke(messages)

    # Serialise to JSON string for storage in AgentState
    maker_output_json = json.dumps(extraction.model_dump())
    logger.info("Maker output: %s", maker_output_json)

    return {
        "maker_output": maker_output_json,
        "retry_count": current_retry + 1,
    }
