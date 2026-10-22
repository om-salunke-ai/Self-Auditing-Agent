from typing import TypedDict


class AgentState(TypedDict):
    """The single source of truth passed between all graph nodes.

    Fields
    ------
    input : str
        The raw invoice text that the Maker must parse.
    maker_output : str
        JSON string of the last ``InvoiceExtraction`` produced by the Maker.
        Empty string on the first invocation.
    audit_result : str
        ``"pass"``, ``"fail"`` or ``"reject"`` — set by the Checker node each
        cycle. ``"fail"`` is retried; ``"reject"`` means the document itself is
        the problem (e.g. not an allowed currency), so the run ends.
        Empty string before the first audit.
    audit_reason : str
        Human-readable explanation from the Checker describing why the
        extraction passed, failed or was rejected (or the Maker's error if
        the LLM call failed). Injected into the Maker prompt on retry.
    retry_count : int
        Number of Maker → Checker cycles completed so far.
        Starts at 0 and is incremented by the Maker at the top of each call.
    messages : list
        Full conversation history (list of BaseMessage objects).
        Reserved for future multi-turn context; currently unused by nodes.
    """

    input: str
    maker_output: str
    audit_result: str
    audit_reason: str
    retry_count: int
    messages: list
