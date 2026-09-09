"""Reddit discussion retrieval via PRAW (official Reddit API, OAuth).

Feature-flagged off by default (see config/settings.Settings.reddit_enabled):
Reddit closed self-service OAuth app registration in late 2025, and new
apps now require manual review with a multi-week timeline — incompatible
with this project's delivery window. This module is real and functional,
not a stub, but it is NOT wired into agent/graph.py's router in this
build: activating it would need a new ResearchPlan field, planner-prompt
changes, a graph node, and edges, none of which can be end-to-end tested
without a reviewed app's credentials. Once REDDIT_CLIENT_ID/SECRET/
USER_AGENT are set, search_reddit() is ready to call directly; wiring it
into the graph is a mechanical follow-up (mirrors hackernews_tool.py's
pattern exactly), not a redesign.

API docs: https://praw.readthedocs.io/
"""

from __future__ import annotations

from datetime import datetime, timezone

import praw
import prawcore

from config.settings import settings
from tools.tool_utils import EvidenceItem, ttl_cache

MAX_TEXT_CHARS = 600


def _truncate(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _get_client():
    if not settings.reddit_enabled:
        return None
    return praw.Reddit(
        client_id=settings.reddit_client_id,
        client_secret=settings.reddit_client_secret,
        user_agent=settings.reddit_user_agent,
    )


@ttl_cache
def search_reddit(query: str, max_results: int = 5) -> list:
    """Search Reddit (all subreddits) for `query`, return up to
    `max_results` real, linkable posts as EvidenceItem. Returns an empty
    list — never fabricated data — if Reddit isn't configured, the query
    is empty, or the search finds nothing. Genuine PRAW/network errors
    (auth failure, rate limit, connectivity) are caught narrowly so
    unrelated bugs still surface rather than being silently hidden.
    """
    if not query or not query.strip():
        return []

    client = _get_client()
    if client is None:
        return []

    try:
        submissions = client.subreddit("all").search(query, limit=max_results, sort="relevance")
        items = []
        for submission in submissions:
            author = str(submission.author) if submission.author else None
            created_at = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat()
            items.append(
                EvidenceItem(
                    source_type="reddit",
                    site=f"r/{submission.subreddit.display_name}",
                    title=submission.title,
                    url=f"https://www.reddit.com{submission.permalink}",
                    author=author,
                    score=submission.score,
                    num_comments=submission.num_comments,
                    created_at=created_at,
                    text=_truncate(submission.selftext) or submission.title,
                )
            )
        return items
    except (praw.exceptions.PRAWException, prawcore.exceptions.PrawcoreException):
        return []
