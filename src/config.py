"""
config.py — LLM client configuration: Sarvam first, Groq as the backup.

Loads credentials from the environment and exports a single ``llm``
instance that all nodes import. This is the only place in the codebase
where the LLM is instantiated, keeping model/provider changes to one file.
"""

import os
import warnings

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env (no-op if env vars are already set, e.g. in CI)
load_dotenv()

groq_key = os.environ.get("GROQ_API_KEY")
sarvam_key = os.environ.get("SARVAM_API_KEY")


if not groq_key:
    warnings.warn(
        "GROQ_API_KEY is not set — Groq LLM will be unavailable. "
        "Get a free key at https://console.groq.com if you need it."
    )

    

if not sarvam_key:
    warnings.warn(
        "SARVAM_API_KEY is not set — Sarvam LLM will be unavailable. "
        "Get a key from your Sarvam AI dashboard if you need it."
    )




groq_llm = None
sarvam_llm = None

if groq_key:
    groq_llm = ChatOpenAI(  # langchain_openai package. This is a LangChain tool that creates the connection to the Groq AI servers.
        model="qwen/qwen3.6-27b",
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_key,
        temperature=0,          # Deterministic extraction — no creative variation
        max_tokens=4096,        # Qwen emits CoT tokens before JSON; needs headroom
        max_retries=2,          # LangChain-level HTTP retries (≠ graph retries)
    )

if sarvam_key:
    sarvam_llm = ChatOpenAI(
        model="sarvam-105b-conversations",                          # The name of the Sarvam model
        base_url="https://api.sarvam.ai/v1",       # Sarvam's API web address
        api_key=sarvam_key,                          # (Leave this as is)
        temperature=0,                             # (Leave this as is)
        max_tokens=4096,                           # (Leave this as is)
        max_retries=2,                             # (Leave this as is)
    )


# Pick the active LLM — prefer Sarvam. When both keys are set, a Sarvam call
# that fails (bad key, no credits, outage, bad output) is automatically
# repeated on Groq instead of failing the invoice (BUG-14).
if sarvam_llm and groq_llm:
    llm = sarvam_llm.with_fallbacks([groq_llm])
else:
    llm = sarvam_llm or groq_llm

if llm is None:
    raise EnvironmentError(
        "No LLM available. Set at least one of GROQ_API_KEY or SARVAM_API_KEY in your .env file."
    )
