"""LLM client tests.

The real-call test only runs when GROQ_API_KEY is present (local .env or CI
secret) so the suite stays runnable without credentials, per the project's
"fail honestly, never fake a result" rule.
"""

import dataclasses

import pytest

import agent.llm as llm_module
from agent.llm import LLMNotConfiguredError, get_llm
from config.settings import settings


def test_raises_clear_error_without_api_key(monkeypatch):
    monkeypatch.setattr(llm_module, "settings", dataclasses.replace(settings, groq_api_key=""))
    get_llm.cache_clear()
    with pytest.raises(LLMNotConfiguredError, match="GROQ_API_KEY"):
        get_llm()


def test_rejects_unsupported_provider(monkeypatch):
    monkeypatch.setattr(llm_module, "settings", dataclasses.replace(settings, model_provider="openai"))
    get_llm.cache_clear()
    with pytest.raises(LLMNotConfiguredError, match="Unsupported MODEL_PROVIDER"):
        get_llm()


@pytest.mark.skipif(not settings.groq_configured, reason="GROQ_API_KEY not set")
def test_live_groq_call_returns_text():
    llm = get_llm()
    response = llm.invoke("Reply with exactly one word: OK")
    assert response.content.strip()
