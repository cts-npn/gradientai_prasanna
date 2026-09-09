"""Shared helpers for retrieval tools: HTTP with timeouts, TTL caching, and
a common evidence-item shape so downstream grounding code doesn't need to
special-case each source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests
from cachetools import TTLCache, cached

DEFAULT_TIMEOUT = 10  # seconds
_http_cache: TTLCache = TTLCache(maxsize=256, ttl=300)  # 5-minute TTL


class ToolRequestError(RuntimeError):
    """Raised when an external API call fails after retries/timeout."""


def http_get(url: str, *, params: dict | None = None, headers: dict | None = None) -> dict:
    """GET a JSON endpoint with a hard timeout, raising ToolRequestError on failure."""
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout as exc:
        raise ToolRequestError(f"Request to {url} timed out after {DEFAULT_TIMEOUT}s") from exc
    except requests.exceptions.RequestException as exc:
        raise ToolRequestError(f"Request to {url} failed: {exc}") from exc
    except ValueError as exc:  # JSON decode error
        raise ToolRequestError(f"Request to {url} returned invalid JSON: {exc}") from exc


def cache_key(*args: Any, **kwargs: Any) -> str:
    return str(args) + str(sorted(kwargs.items()))


def ttl_cache(func):
    """Decorator: cache a tool function's results for 5 minutes.

    Keeps repeated identical queries within one session (or across a few
    users hitting the same demo question) from re-hitting rate-limited
    free APIs.
    """
    return cached(cache=_http_cache, key=cache_key)(func)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EvidenceItem:
    """Common shape for one retrieved social/discussion item.

    Every tool (Hacker News, Stack Exchange, Reddit-when-enabled) returns a
    list of these so the grounding layer can treat sources uniformly.
    """

    source_type: str  # "hackernews" | "stackexchange" | "reddit"
    site: str  # e.g. "news.ycombinator.com", "stackoverflow", "r/electricvehicles"
    title: str
    url: str
    author: str | None
    score: int | None
    num_comments: int | None
    created_at: str | None  # ISO 8601, from the source; None if unavailable
    text: str  # body / selftext / top-comment snippet, truncated
    retrieved_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict:
        return {
            "source_type": self.source_type,
            "site": self.site,
            "title": self.title,
            "url": self.url,
            "author": self.author,
            "score": self.score,
            "num_comments": self.num_comments,
            "created_at": self.created_at,
            "text": self.text,
            "retrieved_at": self.retrieved_at,
        }
