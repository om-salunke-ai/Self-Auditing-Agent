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
        ``"pass"`` or ``"fail"`` — set by the Checker node each cycle.
        Empty string before the first audit.
    audit_reason : str
        Human-readable explanation from the Checker describing why the
        extraction passed or failed. Injected into the Maker prompt on retry.
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
