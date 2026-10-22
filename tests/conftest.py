"""Shared pytest setup. pytest loads this file before any test imports src.*

Safety first, so tests can never cost money or touch real data:
* dummy API keys, and PYTHON_DOTENV_DISABLED=1 so the real .env is never read;
* CHECKPOINT_DB points at a temp file, so data/checkpoints.db is never used;
* the runner's output files are redirected to a temp folder in every test;
* FakeLLM replaces the real LLM inside the Maker (no network calls).
"""
import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path

import pytest

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["SARVAM_API_KEY"] = "test-dummy-sarvam-key"
os.environ["GROQ_API_KEY"] = "test-dummy-groq-key"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["CHECKPOINT_DB"] = str(Path(tempfile.mkdtemp(prefix="agent-tests-")) / "checkpoints.db")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOCKS = json.loads(
    (PROJECT_ROOT / "data" / "mocks" / "sample_invoice.json").read_text(encoding="utf-8")
)
INV1, INV2, INV3 = (mock["raw_text"] for mock in MOCKS)

# What the real LLM returned for the 3 mock invoices (from data/audit_trail.json).
GOOD = {
    "vendor_name": "Bharat Tech Solutions Pvt. Ltd.",
    "invoice_number": "INV-2024-BLR-0089",
    "total_amount": 472000.0,
    "due_date": "2024-08-14",
    "currency": "INR",
}
SPEEDY = {
    "vendor_name": "Speedy Courier Services",
    "invoice_number": "",
    "total_amount": 0.0,
    "due_date": "2024-08-19",
    "currency": "USD",
}
EUROPA = {
    "vendor_name": "Europa Maschinenbau GmbH",
    "invoice_number": "EUR-INV-2024-3301",
    "total_amount": 34926.5,
    "due_date": "2024-08-10",
    "currency": "EUR",
}


def good(**changes) -> dict:
    """The correct extraction of mock-001, with some fields changed."""
    return {**GOOD, **changes}


class FakeLLM:
    """Stands in for the real LLM. Each Maker call takes the next reply from
    `script` (the last one repeats):
      dict      -> returned as an InvoiceExtraction (validated like real output)
      Exception -> raised, like a timeout or a rate limit
      callable  -> called with the messages; its return value is used
      None      -> returned as-is
    Every call's messages are kept in `.calls`.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def with_structured_output(self, schema):
        fake = self

        class _Structured:
            def invoke(self, messages):
                fake.calls.append(messages)
                reply = fake.script.pop(0) if len(fake.script) > 1 else fake.script[0]
                if isinstance(reply, BaseException):
                    raise reply
                if callable(reply):
                    return reply(messages)
                if reply is None:
                    return None
                return schema(**reply)

        return _Structured()


@pytest.fixture
def fake_llm(monkeypatch):
    """Call fake_llm([reply, ...]) to put a FakeLLM into the Maker."""
    import src.nodes.maker as maker

    def install(script):
        fake = FakeLLM(script)
        monkeypatch.setattr(maker, "llm", fake)
        return fake

    return install


def run_graph(text: str) -> dict:
    """Run the real compiled graph on one invoice text."""
    from src.graph import app

    state = {"input": text, "maker_output": "", "audit_result": "",
             "audit_reason": "", "retry_count": 0, "messages": []}
    return app.invoke(state, {"configurable": {"thread_id": uuid.uuid4().hex}})


@pytest.fixture(autouse=True)
def _runner_writes_to_tmp(tmp_path, monkeypatch):
    """In every test, the batch runner reads and writes only temp files."""
    import src.run_agent as ra

    monkeypatch.setattr(ra, "_MOCKS_FILE", tmp_path / "mocks.json")
    monkeypatch.setattr(ra, "_AUDIT_FILE", tmp_path / "audit_trail.json")
    monkeypatch.setattr(ra, "_HISTORY_FILE", tmp_path / "audit_history.jsonl")


@pytest.fixture(autouse=True, scope="session")
def _real_data_files_unchanged():
    """Fail the run if any test changed the real data files."""
    def fingerprint():
        files = ("audit_trail.json", "audit_history.jsonl", "checkpoints.db")
        return {
            name: hashlib.sha256((PROJECT_ROOT / "data" / name).read_bytes()).hexdigest()
            for name in files
            if (PROJECT_ROOT / "data" / name).exists()
        }

    before = fingerprint()
    yield
    assert fingerprint() == before, "a test changed a file in data/"
