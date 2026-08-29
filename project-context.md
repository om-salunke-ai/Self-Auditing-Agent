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
| **LLM Provider**     | DeepSeek via NVIDIA NIM                                  |
| **NIM Base URL**     | `https://integrate.api.nvidia.com/v1`                    |
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

- **Maker Node** (`agent_a`): Executes the primary workflow task (e.g., parse & extract invoice fields). Writes result to shared `AgentState`.
- **Checker Node** (`agent_b`): Audits the Maker's output against defined acceptance criteria. Produces a structured audit report with `pass`/`fail` and a `reason`.
- **Conditional Edge**: Routes back to the Maker for a retry (with checker feedback injected into context) or forwards to `END` on success.

### State Schema (`AgentState`)

```python
class AgentState(TypedDict):
    input: str              # Raw input document / task description
    maker_output: str       # Maker's most recent output
    audit_result: str       # "pass" | "fail"
    audit_reason: str       # Checker's explanation
    retry_count: int        # Number of retries attempted
    messages: list          # Full message history (for LLM context)
```

### Persistence

- SQLite checkpoint file (e.g., `data/checkpoints.db`) via `SqliteSaver` from `langgraph-checkpoint-sqlite`.
- Enables mid-graph resumption and full run history.

### LLM Configuration

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-ai/deepseek-r1",          # or deepseek-chat
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)
```

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
│   └── config.py             # LLM client + env config
├── tests/
│   ├── __init__.py
│   └── test_graph.py         # pytest test suite
└── data/
    ├── mocks/
    │   └── sample_invoice.json  # Mock invoice for local testing
    └── checkpoints.db           # SQLite checkpoint store (runtime)
```

---

## Key Decisions & Constraints

- **Retry limit**: The conditional edge should cap retries (e.g., `retry_count >= 3 → END` with failure state) to prevent infinite loops.
- **Structured output**: The Checker node should use `with_structured_output()` to ensure the audit result is machine-parseable.
- **Thread safety**: Each run uses a unique `thread_id` in `RunnableConfig` so checkpoint state is isolated per document.
- **No streaming** (v1): Invoke graphs synchronously; streaming can be added later.

---

## Environment Variables

| Variable              | Required | Purpose                                         |
|-----------------------|----------|-------------------------------------------------|
| `NVIDIA_API_KEY`      | ✅ Yes   | Authenticates requests to NVIDIA NIM endpoint   |
| `LANGCHAIN_API_KEY`   | ⬜ No    | Enables LangSmith tracing / observability       |
| `LANGCHAIN_TRACING_V2`| ⬜ No   | Set to `"true"` to activate LangSmith tracing   |
| `LANGCHAIN_PROJECT`   | ⬜ No    | LangSmith project name for grouping traces      |
