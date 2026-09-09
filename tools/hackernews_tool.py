"""Hacker News discussion retrieval via the Algolia HN Search API.

Free, no API key, no auth, no rate-limit registration — used as the
primary social-discussion source while Reddit access is pending review
(see config/settings.py). Functionally comparable to Reddit for this
project's purpose: threaded public discussion with scores and comment
counts on real, dated, linkable items.

API docs: https://hn.algolia.com/api
"""

from __future__ import annotations

from datetime import datetime, timezone

from tools.tool_utils import EvidenceItem, ToolRequestError, http_get, now_iso, ttl_cache

SEARCH_URL = "https://hn.algolia.com/api/v1/search"
ITEM_URL = "https://news.ycombinator.com/item?id={id}"
MAX_TEXT_CHARS = 600


def _truncate(text: str | None, limit: int = MAX_TEXT_CHARS) -> str:
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _strip_html(text: str) -> str:
    # HN story/comment text is HTML-escaped fragments (e.g. <p>, &quot;).
    import html
    import re

    text = re.sub(r"<[^>]+>", " ", text or "")
    return html.unescape(text)


@ttl_cache
def search_hackernews(query: str, max_results: int = 5) -> list[EvidenceItem]:
    """Search HN stories matching `query`, return up to `max_results` items.

    Each item corresponds to one real HN story with a real, verifiable URL
    (`news.ycombinator.com/item?id=...`). Never fabricates results — an
    empty or failed search returns an empty list; callers must treat that
    as "no evidence found," not fall back to model memory.
    """
    if not query or not query.strip():
        return []

    try:
        data = http_get(
            SEARCH_URL,
            params={
                "query": query,
                "tags": "story",
                "hitsPerPage": max_results,
            },
        )
    except ToolRequestError:
        # Fail honestly: no results rather than a crash or invented data.
        return []

    items: list[EvidenceItem] = []
    for hit in data.get("hits", [])[:max_results]:
        object_id = hit.get("objectID")
        if not object_id:
            continue
        title = hit.get("title") or hit.get("story_title") or ""
        story_text = _strip_html(hit.get("story_text") or "")
        author = hit.get("author")
        points = hit.get("points")
        num_comments = hit.get("num_comments")
        created_at = hit.get("created_at")  # already ISO 8601 from Algolia

        items.append(
            EvidenceItem(
                source_type="hackernews",
                site="news.ycombinator.com",
                title=title,
                url=ITEM_URL.format(id=object_id),
                author=author,
                score=points,
                num_comments=num_comments,
                created_at=created_at,
                text=_truncate(story_text) or title,
            )
        )
    return items


def age_in_days(created_at: str | None) -> float | None:
    """Days since an HN item's created_at timestamp; None if unavailable."""
    if not created_at:
        return None
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - created).total_seconds() / 86400
