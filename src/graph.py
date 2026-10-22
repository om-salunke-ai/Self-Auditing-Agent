"""
graph.py — LangGraph StateGraph definition for the Maker/Checker workflow.

Topology
--------
    START
      │
      ▼
  [maker] ──────────────────────────────────────────────────►
      │                                                      │
      ▼                                                      │  (retry loop)
  [checker]                                                  │
      │                                                      │
      ├─ audit_result == "pass"  ─────────────────────► END │
      │                                                      │
      ├─ audit_result == "reject"  ───────────────────► END │
      │     (the document itself is the problem — no retry)  │
      │                                                      │
      ├─ audit_result == "fail" AND retry_count < 3  ───────┘
      │
      └─ audit_result == "fail" AND retry_count >= 3  ──► END
            (max retries exhausted — exit with failure state)

Persistence
-----------
The graph is compiled with a ``SqliteSaver`` checkpointer so every node
transition is persisted to ``data/checkpoints.db``.  Pass a unique
``thread_id`` in ``RunnableConfig`` to isolate each document's run history.
Set the ``CHECKPOINT_DB`` environment variable to use another file path.
"""

import logging
import os
import sqlite3
from pathlib import Path

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from src.nodes.checker import run_checker
from src.nodes.maker import run_maker
from src.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RETRIES = 3

# SQLite file lives in data/ relative to the project root.
# __file__ → src/graph.py → parent = src/ → parent = project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# CHECKPOINT_DB lets you keep the database outside OneDrive, whose syncing can
# lock the file mid-write (BUG-21). Tests also use it to get a throwaway DB.
_CHECKPOINT_DB = Path(os.environ.get("CHECKPOINT_DB") or _PROJECT_ROOT / "data" / "checkpoints.db")


# ---------------------------------------------------------------------------
# Conditional routing function
# ---------------------------------------------------------------------------

def _route_after_checker(state: dict) -> str:
    """Decide the next node after the Checker runs.

    Returns
    -------
    str
        ``"maker"`` to retry, or ``"__end__"`` to terminate the graph.
    """
    audit_result = state.get("audit_result", "fail")
    retry_count = state.get("retry_count", 0)

    if audit_result == "pass":
        logger.info("Router: audit passed → END")
        return END

    if audit_result == "reject":
        logger.info("Router: document breaks policy → END (no retry)")
        return END

    if retry_count < MAX_RETRIES:
        logger.info(
            "Router: audit failed, retry %d/%d → maker",
            retry_count,
            MAX_RETRIES,
        )
        return "maker"

    logger.warning(
        "Router: audit failed after %d/%d retries -> END (exhausted)",
        retry_count,
        MAX_RETRIES,
    )
    return END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the Maker/Checker StateGraph.

    Returns
    -------
    CompiledStateGraph
        The compiled graph with SQLite checkpointing, ready to ``.invoke()``.
    """
    # Ensure the data/ directory exists before SQLite tries to create the file
    _CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)

    graph = StateGraph(AgentState)

    # Nodes
    #  use graph.add_node(...) to plug in your Maker and Checker.
    graph.add_node("maker", run_maker)
    graph.add_node("checker", run_checker)

    # Entry point
    graph.set_entry_point("maker")

    # Edges
    # use graph.add_conditional_edges(...) to create the retry loop. If the Checker fails the invoice, LangGraph knows how to route it back to the Maker.
    graph.add_edge("maker", "checker")

    graph.add_conditional_edges(
        "checker",
        _route_after_checker,
        {
            "maker": "maker",   # retry path
            END: END,           # pass or exhausted path
        },
    )

    # Compile with SQLite checkpointer for state persistence.
    # SqliteSaver.from_conn_string() returns a context manager in v3.x,
    # NOT a SqliteSaver instance — so we open the connection directly.
    conn = sqlite3.connect(str(_CHECKPOINT_DB), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    compiled = graph.compile(checkpointer=checkpointer)

    logger.info(
        "Graph compiled. Checkpoint DB: %s",
        _CHECKPOINT_DB,
    )
    return compiled


# ---------------------------------------------------------------------------
# Module-level compiled app — import this in your entrypoint / tests
# ---------------------------------------------------------------------------

app = build_graph()
