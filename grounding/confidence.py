"""Evidence Confidence Score — the project's core differentiator.

This is a deterministic, documented evidence-quality HEURISTIC, not a
statistically calibrated probability. It never asks the model how
confident it feels; every input is computed from data already sitting in
state, using fixed rules defined below.

    confidence = 0.25*relevance + 0.15*source_coverage + 0.15*source_diversity
               + 0.15*freshness + 0.20*agreement + 0.10*claim_coverage

All six inputs are available BEFORE an answer is generated, matching the
architecture's evaluate -> confidence -> grounding_gate -> generate
ordering: the score gates whether generation is even attempted, so it
can't depend on the answer that generation would produce. That is a
deliberate definition choice, documented per-function below:

- relevance: from grounding/evidence.py (Phase 7) — average keyword-overlap
  relevance across all sources.
- source_coverage: how much of the evidence the plan actually asked for
  (5 results per social tool requested, 1 for weather) was retrieved.
- source_diversity: rewards distinct, non-duplicate items and cross-source
  corroboration (e.g. Hacker News AND Stack Exchange agreeing beats five
  results from one site).
- freshness: exponential recency decay per source (weather is always
  live/fresh; social posts decay with a 1-year half-life, floored so old
  evidence is discounted, not zeroed).
- agreement: lightweight sentiment-polarity check across social sources —
  do they cluster on one side or visibly conflict? Also produces the
  conflict_detected flag used for the "mixed evidence" UI warning.
- claim_coverage: fraction of retrieved sources that individually clear a
  "substantive" per-source relevance bar, i.e. how much of what was
  gathered is actually usable rather than noise (distinct from the
  average-based `relevance` score above).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

WEIGHTS = {
    "relevance": 0.25,
    "source_coverage": 0.15,
    "source_diversity": 0.15,
    "freshness": 0.15,
    "agreement": 0.20,
    "claim_coverage": 0.10,
}

FRESHNESS_HALF_LIFE_DAYS = 365.0
FRESHNESS_FLOOR = 15.0
SUBSTANTIVE_RELEVANCE_THRESHOLD = 40.0
EXPECTED_PER_SOCIAL_TOOL = 5
EXPECTED_FOR_WEATHER = 1

_POSITIVE_WORDS = {
    "great", "good", "excellent", "love", "loved", "solid", "reliable",
    "impressive", "satisfied", "recommend", "best", "amazing", "happy",
    "works well", "fast", "smooth", "worth it", "impressed",
}
_NEGATIVE_WORDS = {
    "bad", "terrible", "poor", "hate", "hated", "issue", "issues",
    "problem", "problems", "complaint", "complaints", "disappointing",
    "disappointed", "fail", "fails", "failed", "broken", "worst",
    "annoying", "frustrating", "regret", "avoid", "worse",
}


def _is_weather_source(source: dict) -> bool:
    # Weather sources (WeatherResult.to_dict) carry no source_type field;
    # identify by their distinctive key instead.
    return "resolved_name" in source


def _source_coverage(sources: List[dict], plan) -> float:
    expected = 0
    if plan is not None:
        if getattr(plan, "needs_hackernews", False):
            expected += EXPECTED_PER_SOCIAL_TOOL
        if getattr(plan, "needs_stackexchange", False):
            expected += EXPECTED_PER_SOCIAL_TOOL
        if getattr(plan, "needs_weather", False):
            expected += EXPECTED_FOR_WEATHER
    if expected == 0:
        return 100.0 if sources else 0.0
    return round(min(100.0, len(sources) / expected * 100), 1)


def _source_diversity(sources: List[dict]) -> float:
    if not sources:
        return 0.0
    urls = [s.get("url") or s.get("source_url") for s in sources]
    unique_ratio = len(set(urls)) / len(urls) * 100
    types = {"weather" if _is_weather_source(s) else s.get("source_type") for s in sources}
    type_score = {1: 60.0, 2: 85.0}.get(len(types), 100.0)
    return round((unique_ratio + type_score) / 2, 1)


def _age_in_days(timestamp: Optional[str]) -> Optional[float]:
    if not timestamp:
        return None
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() / 86400


def _freshness(sources: List[dict]) -> float:
    if not sources:
        return 0.0
    scores = []
    for s in sources:
        if _is_weather_source(s):
            scores.append(100.0)  # live data, retrieved this run
            continue
        age = _age_in_days(s.get("created_at"))
        if age is None:
            scores.append(50.0)  # unknown age: neutral, neither rewarded nor penalized
            continue
        scores.append(max(FRESHNESS_FLOOR, 100 * (0.5 ** (age / FRESHNESS_HALF_LIFE_DAYS))))
    return round(sum(scores) / len(scores), 1)


def _polarity(text: str) -> int:
    text_l = (text or "").lower()
    pos = sum(1 for w in _POSITIVE_WORDS if w in text_l)
    neg = sum(1 for w in _NEGATIVE_WORDS if w in text_l)
    if pos > neg:
        return 1
    if neg > pos:
        return -1
    return 0


def _agreement(sources: List[dict]) -> Tuple[float, bool]:
    social = [s for s in sources if not _is_weather_source(s)]
    polarities = [_polarity(f"{s.get('title', '')} {s.get('text', '')}") for s in social]
    non_neutral = [p for p in polarities if p != 0]
    if not non_neutral:
        return 100.0, False  # nothing opinionated enough to conflict
    pos_count = non_neutral.count(1)
    neg_count = non_neutral.count(-1)
    majority = max(pos_count, neg_count)
    conflict = pos_count > 0 and neg_count > 0
    return round(majority / len(non_neutral) * 100, 1), conflict


def _claim_coverage(per_source_relevance: Dict[str, float]) -> float:
    if not per_source_relevance:
        return 0.0
    substantive = sum(1 for v in per_source_relevance.values() if v >= SUBSTANTIVE_RELEVANCE_THRESHOLD)
    return round(substantive / len(per_source_relevance) * 100, 1)


def band_for(score: float) -> str:
    if score >= 90:
        return "strongly_grounded"
    if score >= 75:
        return "well_grounded"
    if score >= 50:
        return "partially_grounded"
    return "insufficient"


def calculate_confidence(
    sources: List[dict],
    plan,
    relevance_score: float,
    per_source_relevance: Dict[str, float],
) -> dict:
    """Compute the Evidence Confidence Score and its breakdown."""
    agreement_score, conflict_detected = _agreement(sources)
    breakdown = {
        "relevance": relevance_score,
        "source_coverage": _source_coverage(sources, plan),
        "source_diversity": _source_diversity(sources),
        "freshness": _freshness(sources),
        "agreement": agreement_score,
        "claim_coverage": _claim_coverage(per_source_relevance),
    }
    total = sum(WEIGHTS[k] * breakdown[k] for k in WEIGHTS)
    return {
        "confidence_score": round(total, 1),
        "breakdown": breakdown,
        "band": band_for(total),
        "conflict_detected": conflict_detected,
    }
