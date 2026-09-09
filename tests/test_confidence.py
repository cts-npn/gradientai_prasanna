"""Evidence Confidence Score tests (grounding/confidence.py)."""

from types import SimpleNamespace

from grounding.confidence import (
    WEIGHTS,
    band_for,
    calculate_confidence,
)


def _plan(needs_hackernews=False, needs_stackexchange=False, needs_weather=False):
    return SimpleNamespace(
        needs_hackernews=needs_hackernews, needs_stackexchange=needs_stackexchange, needs_weather=needs_weather
    )


def test_weights_sum_to_one():
    assert round(sum(WEIGHTS.values()), 6) == 1.0


def test_band_thresholds_match_documented_ranges():
    assert band_for(95) == "strongly_grounded"
    assert band_for(90) == "strongly_grounded"
    assert band_for(89.9) == "well_grounded"
    assert band_for(75) == "well_grounded"
    assert band_for(74.9) == "partially_grounded"
    assert band_for(50) == "partially_grounded"
    assert band_for(49.9) == "insufficient"
    assert band_for(0) == "insufficient"


def test_weather_only_source_scores_near_perfect():
    sources = [{"id": "E1", "resolved_name": "Chennai", "condition": "Clear sky", "source_url": "https://api.open-meteo.com/x"}]
    result = calculate_confidence(sources, _plan(needs_weather=True), relevance_score=100.0, per_source_relevance={"E1": 100.0})
    assert result["confidence_score"] > 90
    assert result["band"] == "strongly_grounded"
    assert result["conflict_detected"] is False


def test_empty_evidence_scores_low_and_is_insufficient():
    result = calculate_confidence([], _plan(needs_hackernews=True), relevance_score=0.0, per_source_relevance={})
    assert result["confidence_score"] < 50
    assert result["band"] == "insufficient"


def test_conflicting_sentiment_sources_flagged_as_conflict():
    """TEST 7 from the project spec: conflicting evidence."""
    sources = [
        {"id": "E1", "source_type": "hackernews", "title": "Great battery life", "text": "This laptop is excellent, love it, very reliable", "created_at": "2026-01-01T00:00:00Z"},
        {"id": "E2", "source_type": "hackernews", "title": "Terrible battery life", "text": "This laptop is terrible, broken after a week, disappointing", "created_at": "2026-01-01T00:00:00Z"},
    ]
    per_source = {"E1": 60.0, "E2": 60.0}
    result = calculate_confidence(sources, _plan(needs_hackernews=True), relevance_score=60.0, per_source_relevance=per_source)
    assert result["conflict_detected"] is True
    assert result["breakdown"]["agreement"] < 100.0


def test_agreeing_sentiment_sources_no_conflict():
    sources = [
        {"id": "E1", "source_type": "hackernews", "title": "Great battery", "text": "Excellent, love it, very reliable", "created_at": "2026-01-01T00:00:00Z"},
        {"id": "E2", "source_type": "hackernews", "title": "Also great", "text": "Solid, impressed, works well", "created_at": "2026-01-01T00:00:00Z"},
    ]
    per_source = {"E1": 60.0, "E2": 60.0}
    result = calculate_confidence(sources, _plan(needs_hackernews=True), relevance_score=60.0, per_source_relevance=per_source)
    assert result["conflict_detected"] is False
    assert result["breakdown"]["agreement"] == 100.0


def test_duplicate_urls_reduce_source_diversity():
    same_url = "https://news.ycombinator.com/item?id=1"
    dup_sources = [
        {"id": "E1", "source_type": "hackernews", "url": same_url, "title": "a", "text": "", "created_at": None},
        {"id": "E2", "source_type": "hackernews", "url": same_url, "title": "b", "text": "", "created_at": None},
    ]
    unique_sources = [
        {"id": "E1", "source_type": "hackernews", "url": "https://news.ycombinator.com/item?id=1", "title": "a", "text": "", "created_at": None},
        {"id": "E2", "source_type": "hackernews", "url": "https://news.ycombinator.com/item?id=2", "title": "b", "text": "", "created_at": None},
    ]
    per_source = {"E1": 50.0, "E2": 50.0}
    dup_result = calculate_confidence(dup_sources, _plan(needs_hackernews=True), 50.0, per_source)
    unique_result = calculate_confidence(unique_sources, _plan(needs_hackernews=True), 50.0, per_source)
    assert dup_result["breakdown"]["source_diversity"] < unique_result["breakdown"]["source_diversity"]


def test_cross_source_type_increases_diversity_over_single_type():
    hn_only = [
        {"id": "E1", "source_type": "hackernews", "url": "https://x/1", "title": "a", "text": "", "created_at": None},
        {"id": "E2", "source_type": "hackernews", "url": "https://x/2", "title": "b", "text": "", "created_at": None},
    ]
    hn_plus_se = [
        {"id": "E1", "source_type": "hackernews", "url": "https://x/1", "title": "a", "text": "", "created_at": None},
        {"id": "E2", "source_type": "stackexchange", "url": "https://y/2", "title": "b", "text": "", "created_at": None},
    ]
    per_source = {"E1": 50.0, "E2": 50.0}
    single_type = calculate_confidence(hn_only, _plan(needs_hackernews=True), 50.0, per_source)
    mixed_type = calculate_confidence(hn_plus_se, _plan(needs_hackernews=True, needs_stackexchange=True), 50.0, per_source)
    assert mixed_type["breakdown"]["source_diversity"] > single_type["breakdown"]["source_diversity"]


def test_old_social_evidence_scores_lower_freshness_than_recent():
    recent = [{"id": "E1", "source_type": "hackernews", "url": "https://x/1", "title": "a", "text": "", "created_at": "2026-09-01T00:00:00Z"}]
    old = [{"id": "E1", "source_type": "hackernews", "url": "https://x/1", "title": "a", "text": "", "created_at": "2016-01-01T00:00:00Z"}]
    per_source = {"E1": 50.0}
    recent_result = calculate_confidence(recent, _plan(needs_hackernews=True), 50.0, per_source)
    old_result = calculate_confidence(old, _plan(needs_hackernews=True), 50.0, per_source)
    assert old_result["breakdown"]["freshness"] < recent_result["breakdown"]["freshness"]
