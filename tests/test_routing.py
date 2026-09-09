"""Routing tests — run the compiled graph against the project's required
test cases. These call the live Groq model and live retrieval APIs (all
free), matching the project's rule against faking tool/LLM behavior in
tests. Skipped as a whole when GROQ_API_KEY isn't configured.

Two tests mock only the retrieval layer (agent.nodes.search_hackernews)
to force a specific evidence shape (genuinely conflicting sentiment; a
raw unexpected exception) while every downstream step — the LLM, the
confidence calculation, answer generation — still runs live and real.
That's a test fixture standing in for what a search API could plausibly
return, not fabricated output shown to a user.
"""

from unittest.mock import patch

import pytest

from agent.graph import get_graph
from config.settings import settings
from tools.tool_utils import EvidenceItem

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
    assert result["citation_validation_passed"] is True
    assert result["final_answer"]
    assert result["cited_source_ids"] == ["E1"]  # only source available is E1


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


def test_conflicting_evidence_produces_a_hedged_answer_not_a_forced_verdict():
    """TEST 7 from the project spec, end-to-end: with genuinely conflicting
    evidence, the confidence module must flag the conflict AND the actual
    LLM-generated answer (live call) must reflect disagreement rather than
    picking one side, per the Phase 8 answer prompt's rule 4.
    """
    conflicting_evidence = [
        EvidenceItem(
            source_type="hackernews", site="news.ycombinator.com",
            title="This laptop's battery life is excellent",
            url="https://news.ycombinator.com/item?id=1001",
            author="userA", score=80, num_comments=20, created_at="2026-06-01T00:00:00Z",
            text="Battery life is excellent, very reliable, impressed with how long it lasts on a full charge.",
        ),
        EvidenceItem(
            source_type="hackernews", site="news.ycombinator.com",
            title="This laptop's battery life is terrible",
            url="https://news.ycombinator.com/item?id=1002",
            author="userB", score=75, num_comments=18, created_at="2026-06-02T00:00:00Z",
            text="Battery life is terrible, broken after a month, very disappointing purchase.",
        ),
    ]
    with patch("agent.nodes.search_hackernews", return_value=conflicting_evidence):
        result = _run("What do people think about this laptop's battery life?")

    assert result["conflict_detected"] is True
    assert result["confidence_breakdown"]["agreement"] < 100.0
    if result["grounded"]:
        answer_lower = result["final_answer"].lower()
        hedging_signals = [
            "mixed", "however", "some", "others", "disagree", "differ", "vary", "while",
            "divided", "contrast", "conflict", "sharply", "on the other hand", "in contrast",
        ]
        assert any(word in answer_lower for word in hedging_signals), (
            f"Expected the answer to hedge given conflicting evidence, got: {result['final_answer']!r}"
        )


def test_unexpected_tool_failure_does_not_produce_an_ungrounded_answer():
    """TEST 8 from the project spec: a genuinely unexpected failure (not
    the handled ToolRequestError path) must not crash into an uncaught
    state or, worse, silently fall back to an ungrounded model answer.
    invoking the graph directly should raise (nodes don't swallow bugs);
    the UI layer (tests/test_app.py) verifies the user-facing message.
    """
    with patch("agent.nodes.search_hackernews", side_effect=RuntimeError("simulated unexpected failure")):
        with pytest.raises(RuntimeError):
            _run("What do people think about electric vehicles?")
