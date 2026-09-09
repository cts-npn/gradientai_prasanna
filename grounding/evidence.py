"""Deterministic evidence evaluation — no LLM involved.

Relevance is scored by keyword overlap between the question/search query
and each retrieved item's title+text, not by asking the model whether it
"feels" relevant. This keeps the grounding gate's core sufficiency check
auditable and immune to the model rationalizing a weak match.

Phase 10 builds the full weighted Evidence Confidence Score (relevance +
source diversity + freshness + agreement + coverage) on top of this
module's relevance numbers. `evaluate_evidence` here is the interim,
documented sufficiency check the grounding gate uses until then.
"""

from __future__ import annotations

import re
from typing import List

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "so", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "to", "from", "in", "on", "off", "over", "under", "again", "do",
    "does", "did", "doing", "have", "has", "had", "having", "i", "you",
    "he", "she", "it", "we", "they", "them", "their", "its", "current",
    "currently", "people", "think", "online", "saying",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set:
    """Lowercase, strip punctuation, drop stopwords and 1-2 char tokens."""
    words = _WORD_RE.findall((text or "").lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def item_relevance(query_tokens: set, item_text: str) -> float:
    """Fraction of the query's meaningful terms that also appear in this
    item's text. 0.0 (no overlap) to 1.0 (every query term present).
    """
    if not query_tokens:
        return 0.0
    item_tokens = tokenize(item_text)
    if not item_tokens:
        return 0.0
    overlap = query_tokens & item_tokens
    return len(overlap) / len(query_tokens)


def evaluate_evidence(question: str, search_query: str, sources: List[dict]) -> dict:
    """Score retrieved evidence against the question. Returns:

    - evidence_count: total sources retrieved
    - relevance_score: 0-100, average per-source relevance
    - per_source_relevance: {source_id: 0-100} for transparency/debugging

    A live-weather source (source_type == "weather", identified by the
    presence of `resolved_name`/`source_url` instead of `title`/`url`)
    scores 100 by construction: it's structured data that answers a
    weather question directly, not text to keyword-match against.
    """
    if not sources:
        return {"evidence_count": 0, "relevance_score": 0.0, "per_source_relevance": {}}

    # Combine the question and the planner's search query so relevance
    # isn't solely dependent on the LLM's query-phrasing choice.
    query_tokens = tokenize(question) | tokenize(search_query)

    per_source: dict = {}
    total = 0.0
    for source in sources:
        source_id = source.get("id", "?")
        if source.get("source_type") is None and "resolved_name" in source:
            # Weather sources (see tools/weather_tool.WeatherResult.to_dict) carry
            # no source_type field; identify by their distinctive keys instead.
            score = 100.0
        else:
            text = f"{source.get('title', '')} {source.get('text', '')}"
            score = item_relevance(query_tokens, text) * 100
        per_source[source_id] = round(score, 1)
        total += score

    return {
        "evidence_count": len(sources),
        "relevance_score": round(total / len(sources), 1),
        "per_source_relevance": per_source,
    }
