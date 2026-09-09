"""Open-weights LLM client factory.

Primary reasoning model: Groq's free-tier inference of an open-weight model
(default Llama 3.3 70B). Provider/model are read from Settings so they stay
configurable via environment variables without code changes.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_groq import ChatGroq

from config.settings import settings


class LLMNotConfiguredError(RuntimeError):
    """Raised when a call needs the LLM but no API key is configured."""


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0) -> ChatGroq:
    """Return a cached chat model client for the configured provider.

    Only Groq is implemented — it's the only provider in this project that
    is both genuinely free and serves open-weight models directly (no
    closed-weight fallback), per the project's cost and licensing rules.
    """
    if settings.model_provider != "groq":
        raise LLMNotConfiguredError(
            f"Unsupported MODEL_PROVIDER '{settings.model_provider}'. Only 'groq' is implemented."
        )
    if not settings.groq_configured:
        raise LLMNotConfiguredError(
            "GROQ_API_KEY is not set. Add it to .env (local) or Streamlit secrets (deployed). "
            "Get a free key at https://console.groq.com/keys."
        )
    return ChatGroq(
        model=settings.model_name,
        api_key=settings.groq_api_key,
        temperature=temperature,
        timeout=30,
        max_retries=2,
    )
