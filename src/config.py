"""
config.py — LLM client configuration for NVIDIA NIM.

Loads credentials from the environment and exports a single ``llm``
instance that all nodes import. This is the only place in the codebase
where the LLM is instantiated, keeping model/provider changes to one file.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env (no-op if env vars are already set, e.g. in CI)
load_dotenv()

_api_key = os.environ.get("GROQ_API_KEY")

if not _api_key:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. "
        "Get a free key at https://console.groq.com, then add it to your .env file."
    )

# ---------------------------------------------------------------------------
# Primary LLM — Llama 3.1 70B via Groq (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
# Groq is free-tier, sub-2s latency, and uses the same ChatOpenAI client.
# Swap model here to try other Groq-hosted models e.g.:
#   "llama-3.1-8b-instant"    (faster, lighter)
#   "mixtral-8x7b-32768"      (larger context)
# ---------------------------------------------------------------------------
llm = ChatOpenAI(
    model="qwen/qwen3.6-27b",
    base_url="https://api.groq.com/openai/v1",
    api_key=_api_key,
    temperature=0,          # Deterministic extraction — no creative variation
    max_tokens=4096,        # Qwen emits CoT tokens before JSON; needs headroom
    max_retries=2,          # LangChain-level HTTP retries (≠ graph retries)
)
