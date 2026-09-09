"""Tool tests. Grows across phases as reddit/stackexchange/weather land."""

import pytest

from tools import hackernews_tool, stackexchange_tool
from tools.tool_utils import ToolRequestError


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
