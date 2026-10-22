# Self-Auditing Workflow Automation Agent — Project Context

## Overview

A LangGraph-powered automation agent that processes structured documents (e.g., invoices) through a **Maker/Checker loop**. Agent A (Maker) executes the primary task; Agent B (Checker/Auditor) validates the output; the graph conditionally retries on failure, creating a self-correcting pipeline.

---

## Tech Stack

| Concern              | Choice                                                   |
|----------------------|----------------------------------------------------------|
| **Framework**        | LangGraph & LangChain                                    |
| **Language**         | Python 3.14                                              |
| **State/Memory**     | `langgraph-checkpoint-sqlite` (SQLite-backed checkpointing) |
| **LLM Client**       | `ChatOpenAI` (from `langchain-openai`)                   |
| **LLM Provider**     | Sarvam (`sarvam-105b-conversations`), Groq (`qwen/qwen3.6-27b`) as automatic backup |
| **Base URLs**        | `https://api.sarvam.ai/v1`, `https://api.groq.com/openai/v1` |
| **Testing**          | `pytest`                                                 |
| **Config**           | `python-dotenv` (`.env` file)                            |
| **Observability**    | LangSmith (optional, via `LANGCHAIN_API_KEY`)            |

---

## Architecture

### Core Pattern: Maker → Checker → Conditional Retry

```
           ┌────────────────────────────────────────────┐
           │              LangGraph StateGraph           │
           │                                             │
  START ──►│  [Maker Node]  ──►  [Checker Node]         │
           │       ▲                    │                │
           │       │         ┌──────────┴──────────┐    │
           │       │         │  Conditional Edge    │    │
           │       │         │  passed? ──────────► END  │
           │       └─────────┤  failed? ──► retry        │
           │                 └─────────────────────-┘    │
           └────────────────────────────────────────────┘
```

### Nodes

- **Maker Node** (`src/nodes/maker.py`): Asks the LLM to extract 5 invoice fields into the `InvoiceExtraction` schema. The invoice is passed inside `<invoice>` tags as untrusted data. LLM errors count as a failed attempt.
- **Checker Node** (`src/nodes/checker.py`): Plain Python rules, no LLM. Every value must appear in the invoice text (grounding), then policy and format rules run. Result: `pass`, `fail` (Maker mistake → retry) or `reject` (document problem → human review, no retry), plus a `reason`.
- **Conditional Edge**: Routes back to the Maker for a retry (with checker feedback and the previous answer) or forwards to `END` on pass, reject, or after 3 attempts.

### State Schema (`AgentState`)

```python
class AgentState(TypedDict):
    input: str              # Raw input document / task description
    maker_output: str       # Maker's most recent output
    audit_result: str       # "pass" | "fail" | "reject"
    audit_reason: str       # Checker's explanation (or the Maker's LLM error)
    retry_count: int        # Number of Maker attempts so far
    messages: list          # Full message history (for LLM context)
```

### Persistence

- SQLite checkpoint file `data/checkpoints.db` (or the path in `CHECKPOINT_DB`) via `SqliteSaver` from `langgraph-checkpoint-sqlite`.
- Each invoice in each run gets its own `thread_id` (`<run_id>-<position>-<invoice id>`), so run histories never mix.
- `data/audit_trail.json` holds the latest run (rewritten after every invoice); `data/audit_history.jsonl` keeps every result of every run.

### LLM Configuration

See `src/config.py`: Sarvam is used when `SARVAM_API_KEY` is set, with Groq as an automatic fallback (`with_fallbacks`) when `GROQ_API_KEY` is set too. Either key alone also works.

---

## Folder Structure

```
Self-Auditing Agent/
├── project-context.md        # This file
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── .env                      # Local secrets (git-ignored)
├── src/
│   ├── __init__.py
│   ├── graph.py              # LangGraph StateGraph definition
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── maker.py          # Maker (Agent A) node logic
│   │   └── checker.py        # Checker (Agent B) node logic
│   ├── state.py              # AgentState TypedDict definition
│   ├── schema.py             # InvoiceExtraction Pydantic model (LLM output)
│   ├── run_agent.py          # Batch runner: python -m src.run_agent
│   └── config.py             # LLM client + env config
├── tests/                    # pytest suite (fake LLM, no API calls)
│   ├── conftest.py           # test setup: dummy keys, temp DB, FakeLLM
│   ├── test_checker.py
│   ├── test_maker.py
│   ├── test_graph.py
│   └── test_run_agent.py
└── data/
    ├── mocks/
    │   └── sample_invoice.json  # Mock invoices for local testing
    ├── audit_trail.json         # Results of the latest run
    ├── audit_history.jsonl      # Results of every run (git-ignored)
    └── checkpoints.db           # SQLite checkpoint store (runtime, git-ignored)
```

---

## Key Decisions & Constraints

- **Retry limit**: The conditional edge caps attempts (`retry_count >= 3 → END` with failure state) to prevent infinite loops.
- **Structured output**: The Maker uses `with_structured_output(InvoiceExtraction)`; the Checker is plain Python, so its result is always machine-parseable.
- **Fail vs reject**: `fail` = the Maker can fix it (retry). `reject` = the document is the problem (missing field, credit note, ambiguous date, currency not allowed), so retrying would only push the LLM to invent data.
- **Thread safety**: Each invoice in each run uses a unique `thread_id` in `RunnableConfig` so checkpoint state is isolated.
- **No streaming** (v1): Invoke graphs synchronously; streaming can be added later.

---

## Environment Variables

| Variable              | Required | Purpose                                         |
|-----------------------|----------|-------------------------------------------------|
| `SARVAM_API_KEY`      | ✅ One of these two | Sarvam LLM (used first)          |
| `GROQ_API_KEY`        | ✅ One of these two | Groq LLM (backup, or main if no Sarvam key) |
| `CHECKPOINT_DB`       | ⬜ No    | Other path for the SQLite checkpoint file       |
| `NVIDIA_API_KEY`      | ⬜ No    | Not used yet (kept for future use)              |
| `LANGCHAIN_API_KEY`   | ⬜ No    | Enables LangSmith tracing / observability       |
| `LANGCHAIN_TRACING_V2`| ⬜ No   | Set to `"true"` to activate LangSmith tracing   |
| `LANGCHAIN_PROJECT`   | ⬜ No    | LangSmith project name for grouping traces      |
