"""Batch runner (src/run_agent.py). conftest.py points all its files at tmp_path."""
import io
import json
import os
import subprocess
import sys

import pytest

import src.run_agent as ra
from tests.conftest import EUROPA, INV1, MOCKS, PROJECT_ROOT, SPEEDY, good


def write_mocks(mocks, encoding="utf-8"):
    ra._MOCKS_FILE.write_text(json.dumps(mocks), encoding=encoding)


def audit_trail() -> list:
    return json.loads(ra._AUDIT_FILE.read_text(encoding="utf-8"))


def real_answers(messages):
    """What the real LLM answered for each of the 3 mock invoices."""
    from src.schema import InvoiceExtraction
    text = messages[1].content
    answer = SPEEDY if "Speedy" in text else EUROPA if "Europa" in text else good()
    return InvoiceExtraction(**answer)


# --- BUG-10: results are saved after every invoice --------------------------

def test_bug10_crash_halfway_keeps_finished_results(fake_llm):
    def ctrl_c_on_second_invoice(messages):
        if "Speedy" in messages[1].content:
            raise KeyboardInterrupt
        return real_answers(messages)

    ra._AUDIT_FILE.write_text('"results of an old run"', encoding="utf-8")
    write_mocks(MOCKS)
    fake_llm([ctrl_c_on_second_invoice])
    with pytest.raises(KeyboardInterrupt):
        ra.run_all_mocks()
    assert [entry["mock_id"] for entry in audit_trail()] == ["mock-001"]


# --- BUG-11: one thread per invoice per run; history is kept ----------------

def test_bug11_runs_and_duplicate_ids_never_share_a_thread(fake_llm, capsys):
    write_mocks([{"id": "dup", "raw_text": INV1}, {"id": "dup", "raw_text": INV1}])
    fake_llm([good()])
    ra.run_all_mocks()
    ra.run_all_mocks()
    history = [json.loads(line) for line in ra._HISTORY_FILE.read_text(encoding="utf-8").splitlines()]
    assert len(history) == 4
    assert len({entry["thread_id"] for entry in history}) == 4
    assert "Duplicate invoice id 'dup'" in capsys.readouterr().out


# --- BUG-12: output can't crash on a cp1252 console -------------------------

def test_bug12_non_english_text_prints_after_forcing_utf8(monkeypatch):
    cp1252_stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", cp1252_stdout)
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    ra._force_utf8_output()
    print("Vendor: भारत टेक | Total: ₹4,72,000")  # used to raise UnicodeEncodeError
    assert sys.stdout.encoding == "utf-8"


# --- BUG-13: bad input is skipped before any LLM call -----------------------

def test_bug13_bad_entries_are_skipped_without_llm_calls(fake_llm):
    write_mocks([{"id": "null", "raw_text": None}, {"id": "blank", "raw_text": "   "},
                 {"id": "huge", "raw_text": "x" * 50_000}, "not an invoice object"])
    fake = fake_llm([good()])
    ra.run_all_mocks()
    assert [entry["audit_result"] for entry in audit_trail()] == ["error"] * 4
    assert fake.calls == []


def test_bug13_file_saved_with_a_bom_loads(fake_llm):
    write_mocks(MOCKS[:1], encoding="utf-8-sig")
    fake_llm([good()])
    ra.run_all_mocks()
    assert audit_trail()[0]["audit_result"] == "pass"


def test_bug13_file_that_is_not_a_list_exits_cleanly():
    write_mocks(MOCKS[0])
    with pytest.raises(SystemExit):
        ra.run_all_mocks()


# --- BUG-15: attempt logs are visible ---------------------------------------

def test_bug15_log_level_is_info():
    # A fresh process, because pytest installs its own logging handlers.
    code = "import logging, src.run_agent; print(logging.getLogger().getEffectiveLevel())"
    out = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=os.environ.copy(),
                         capture_output=True, text=True, check=True).stdout
    assert out.strip().splitlines()[-1] == "20"  # logging.INFO


# --- BUG-18: the summary counts each outcome --------------------------------

def test_bug18_output_labels(fake_llm, capsys):
    write_mocks(MOCKS)
    fake_llm([real_answers])
    ra.run_all_mocks()
    out = capsys.readouterr().out
    assert "Attempts    : 1" in out and "Retries" not in out
    assert "Passed   : 1" in out and "Rejected : 2" in out and "Errors   : 0" in out
