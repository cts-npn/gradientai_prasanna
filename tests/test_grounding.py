"""Grounding evaluation and gate tests.

Pure-function tests (grounding/evidence.py) use synthetic inputs — that's
standard unit testing of our own decision logic, not the "never fake
retrieved evidence" rule, which is about what's shown to a user as if it
were real. Gate/graph-level tests that need retrieved evidence hit the
live model and APIs, matching the rest of the suite.
"""

import pytest

from agent import nodes
from grounding.evidence import evaluate_evidence, item_relevance, tokenize


def test_tokenize_drops_stopwords_and_short_tokens():
    tokens = tokenize("What is the current weather in Chennai?")
    assert "chennai" in tokens
    assert "weather" in tokens
    assert "the" not in tokens
    assert "is" not in tokens


def test_item_relevance_full_overlap_scores_one():
    query_tokens = tokenize("electric vehicle battery")
    assert item_relevance(query_tokens, "electric vehicle battery problems") == 1.0


def test_item_relevance_no_overlap_scores_zero():
    query_tokens = tokenize("electric vehicle battery")
    assert item_relevance(query_tokens, "recipe for chocolate cake") == 0.0


def test_item_relevance_empty_query_scores_zero():
    assert item_relevance(set(), "anything at all") == 0.0


def test_evaluate_evidence_empty_sources():
    result = evaluate_evidence("question", "query", [])
    assert result["evidence_count"] == 0
    assert result["relevance_score"] == 0.0


def test_evaluate_evidence_weather_source_scores_full_relevance():
    weather_source = {"id": "E1", "resolved_name": "Chennai", "condition": "Clear sky"}
    result = evaluate_evidence("weather in Chennai", "Chennai weather", [weather_source])
    assert result["per_source_relevance"]["E1"] == 100.0


def test_evaluate_evidence_mixes_relevant_and_irrelevant_social_sources():
    sources = [
        {"id": "E1", "source_type": "hackernews", "title": "Electric vehicle battery costs falling", "text": ""},
        {"id": "E2", "source_type": "hackernews", "title": "Best chocolate cake recipe", "text": ""},
    ]
    result = evaluate_evidence("electric vehicle battery costs", "electric vehicle battery", sources)
    assert result["per_source_relevance"]["E1"] > result["per_source_relevance"]["E2"]


def test_grounding_gate_fails_with_zero_evidence():
    state = {"evidence_count": 0, "confidence_score": 0.0}
    result = nodes.grounding_gate(state)
    assert result["grounded"] is False
    assert "sufficient grounded evidence" in result["refusal"]


def test_grounding_gate_fails_with_low_confidence():
    state = {"evidence_count": 3, "confidence_score": 20.0}
    result = nodes.grounding_gate(state)
    assert result["grounded"] is False
    assert "limited relevant evidence" in result["refusal"]


def test_grounding_gate_passes_with_sufficient_confidence():
    state = {"evidence_count": 3, "confidence_score": 80.0}
    result = nodes.grounding_gate(state)
    assert result["grounded"] is True
    assert "refusal" not in result
