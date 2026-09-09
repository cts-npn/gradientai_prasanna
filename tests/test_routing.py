"""Routing tests — run the compiled graph against the project's required
test cases. These call the live Groq model and live retrieval APIs (all
free), matching the project's rule against faking tool/LLM behavior in
tests. Skipped as a whole when GROQ_API_KEY isn't configured.
"""

import pytest

from agent.graph import get_graph
from config.settings import settings

pytestmark = pytest.mark.skipif(not settings.groq_configured, reason="GROQ_API_KEY not set")


def _run(question: str) -> dict:
    return get_graph().invoke({"question": question, "status_log": []})


def test_weather_only_question_routes_to_weather_alone():
    result = _run("What is the current weather in Chennai?")
    assert result["in_scope"] is True
    plan = result["plan"]
    assert plan.needs_weather is True
    assert plan.needs_hackernews is False
    assert plan.needs_stackexchange is False
    assert result["weather_result"] is not None
    assert not result.get("refusal")


def test_social_opinion_question_routes_to_hackernews():
    result = _run("What do people think about electric vehicles?")
    assert result["in_scope"] is True
    plan = result["plan"]
    assert plan.needs_hackernews is True
    assert plan.needs_weather is False
    assert len(result["hn_results"]) > 0
    assert not result.get("refusal")


def test_combined_question_routes_to_weather_and_social():
    result = _run("What is the current weather in Chennai and what are people saying about it online?")
    assert result["in_scope"] is True
    plan = result["plan"]
    assert plan.needs_weather is True
    assert plan.needs_hackernews is True
    assert result["weather_result"] is not None
    assert not result.get("refusal")


def test_out_of_scope_question_is_refused_without_calling_tools():
    result = _run("Write a romantic poem for my girlfriend")
    assert result["in_scope"] is False
    assert result.get("refusal")
    assert not result.get("hn_results")
    assert not result.get("se_results")
    assert result.get("weather_result") is None
    assert "plan" not in result


def test_in_scope_question_with_no_matching_evidence_is_refused_by_grounding_gate():
    # A fictional, specific enough product that HN/SE genuinely return nothing.
    result = _run("What do people think about the Zeltrix Q9 folding phone hinge durability in humid climates")
    assert result["in_scope"] is True
    assert result["grounded"] is False
    assert result.get("refusal")
    assert result["evidence_count"] == 0
