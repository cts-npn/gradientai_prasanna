"""Tool tests. Grows across phases as reddit/stackexchange/weather land."""

import pytest

from tools import hackernews_tool, reddit_tool, stackexchange_tool, weather_tool
from tools.tool_utils import ToolRequestError, ttl_cache


def test_ttl_cache_does_not_collide_across_different_functions():
    calls = {"a": 0, "b": 0}

    @ttl_cache
    def tool_a(x):
        calls["a"] += 1
        return f"a-result-{x}"

    @ttl_cache
    def tool_b(x):
        calls["b"] += 1
        return f"b-result-{x}"

    assert tool_a("same-arg") == "a-result-same-arg"
    assert tool_b("same-arg") == "b-result-same-arg"
    assert calls == {"a": 1, "b": 1}


def test_hackernews_empty_query_returns_empty_list():
    assert hackernews_tool.search_hackernews("") == []
    assert hackernews_tool.search_hackernews("   ") == []


def test_hackernews_fails_gracefully_on_api_error(monkeypatch):
    def boom(*args, **kwargs):
        raise ToolRequestError("simulated network failure")

    monkeypatch.setattr(hackernews_tool, "http_get", boom)
    hackernews_tool.search_hackernews.cache_clear()
    results = hackernews_tool.search_hackernews("anything unique to avoid cache hit xyz123")
    assert results == []


def test_hackernews_live_search_returns_real_items():
    results = hackernews_tool.search_hackernews("electric vehicle battery", max_results=3)
    assert 1 <= len(results) <= 3
    for item in results:
        assert item.url.startswith("https://news.ycombinator.com/item?id=")
        assert item.source_type == "hackernews"
        assert item.title


def test_age_in_days_handles_missing_and_valid_timestamps():
    assert hackernews_tool.age_in_days(None) is None
    assert hackernews_tool.age_in_days("not-a-date") is None
    assert hackernews_tool.age_in_days("2020-01-01T00:00:00Z") > 1000


def test_stackexchange_empty_query_returns_empty_list():
    assert stackexchange_tool.search_stackexchange("") == []
    assert stackexchange_tool.search_stackexchange("   ") == []


def test_stackexchange_fails_gracefully_on_api_error(monkeypatch):
    def boom(*args, **kwargs):
        raise ToolRequestError("simulated network failure")

    monkeypatch.setattr(stackexchange_tool, "http_get", boom)
    stackexchange_tool.search_stackexchange.cache_clear()
    results = stackexchange_tool.search_stackexchange("anything unique xyz456 to dodge cache")
    assert results == []


def test_stackexchange_live_search_returns_real_items():
    results = stackexchange_tool.search_stackexchange("python asyncio timeout", max_results=3)
    assert 1 <= len(results) <= 3
    for item in results:
        assert item.url.startswith("https://stackoverflow.com/questions/")
        assert item.source_type == "stackexchange"
        assert item.title


def test_stackexchange_builds_correct_urls_for_custom_and_standard_domains():
    assert stackexchange_tool._question_url("stackoverflow", 123) == "https://stackoverflow.com/questions/123"
    assert stackexchange_tool._question_url("superuser", 5) == "https://superuser.com/questions/5"
    assert (
        stackexchange_tool._question_url("sustainability", 9)
        == "https://sustainability.stackexchange.com/questions/9"
    )


def test_weather_empty_city_returns_none():
    assert weather_tool.get_weather("") is None
    assert weather_tool.get_weather("   ") is None


def test_weather_unresolvable_city_returns_none():
    weather_tool.get_weather.cache_clear()
    weather_tool._geocode.cache_clear()
    result = weather_tool.get_weather("zzzznotarealplaceqxqxqx123")
    assert result is None


def test_weather_geocode_failure_returns_none(monkeypatch):
    def boom(*args, **kwargs):
        raise ToolRequestError("simulated network failure")

    monkeypatch.setattr(weather_tool, "http_get", boom)
    weather_tool.get_weather.cache_clear()
    weather_tool._geocode.cache_clear()
    assert weather_tool.get_weather("Chennai") is None


def test_weather_live_lookup_returns_real_data():
    weather_tool.get_weather.cache_clear()
    result = weather_tool.get_weather("Chennai")
    assert result is not None
    assert result.resolved_name
    assert -90 <= result.latitude <= 90
    assert -180 <= result.longitude <= 180
    assert result.temperature_c is not None
    assert result.condition and "Unknown" not in result.condition
    assert result.source_url.startswith("https://api.open-meteo.com/v1/forecast?")


def test_reddit_returns_empty_list_when_not_configured(monkeypatch):
    from config.settings import settings as live_settings

    assert live_settings.reddit_enabled is False  # true by default in this project's env
    reddit_tool.search_reddit.cache_clear()
    assert reddit_tool.search_reddit("electric vehicles") == []


def test_reddit_empty_query_returns_empty_list():
    assert reddit_tool.search_reddit("") == []
    assert reddit_tool.search_reddit("   ") == []


def test_reddit_get_client_returns_none_when_disabled():
    assert reddit_tool._get_client() is None
