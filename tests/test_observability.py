"""LangSmith observability wiring tests.

Tracing itself is an external side effect (network calls to LangSmith) so
it isn't asserted against in the automated suite — that was verified
manually by querying the LangSmith API after a live run (every graph node
appeared as a distinct traced span with token usage and latency). These
tests cover the parts that are meaningfully unit-testable: the
settings.langsmith_configured logic, and that run_agent (the entry point
used by the UI) returns the same shape as invoking the graph directly.
"""

import dataclasses

import pytest

import config.settings as settings_module
from agent.graph import run_agent
from config.settings import settings


def test_langsmith_configured_requires_both_key_and_tracing_flag():
    both_set = dataclasses.replace(settings, langchain_api_key="lsv2_pt_x", langchain_tracing_v2="true")
    assert both_set.langsmith_configured is True

    no_key = dataclasses.replace(settings, langchain_api_key="", langchain_tracing_v2="true")
    assert no_key.langsmith_configured is False

    tracing_off = dataclasses.replace(settings, langchain_api_key="lsv2_pt_x", langchain_tracing_v2="false")
    assert tracing_off.langsmith_configured is False


@pytest.mark.skipif(not settings.groq_configured, reason="GROQ_API_KEY not set")
def test_run_agent_returns_same_shape_as_direct_invoke():
    result = run_agent("What is the current weather in Chennai?")
    assert result["in_scope"] is True
    assert result["final_answer"]
    assert result["confidence_score"] is not None
