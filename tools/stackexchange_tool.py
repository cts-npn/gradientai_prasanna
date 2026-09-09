"""Stack Exchange discussion retrieval — the second community-evidence
source (Quora has no public API suitable for this project; see README).

Uses the free Stack Exchange API v2.3 `/search/excerpts` endpoint: no key
required for light use (300 req/day, shared per IP), or pass
STACKEXCHANGE_KEY for a registered app's 10,000 req/day quota.

API docs: https://api.stackexchange.com/docs/excerpt-search
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone

from config.settings import settings
from tools.tool_utils import EvidenceItem, ToolRequestError, http_get, ttl_cache

SEARCH_URL = "https://api.stackexchange.com/2.3/search/excerpts"
DEFAULT_SITE = "stackoverflow"
MAX_TEXT_CHARS = 600

# Stack Exchange's API `site` parameter is normally the {site}.stackexchange.com
# subdomain slug, except for these historical sites that predate the network's
# unified domain scheme. This is Stack Exchange's own documented mapping, not
# a guess — used to build a real, verifiable question URL from question_id,
# since /search/excerpts (needed for the body excerpt) omits the `link` field.
_CUSTOM_DOMAINS = {
    "stackoverflow": "stackoverflow.com",
    "serverfault": "serverfault.com",
    "superuser": "superuser.com",
    "askubuntu": "askubuntu.com",
    "mathoverflow.net": "mathoverflow.net",
    "stackapps": "stackapps.com",
}


def _question_url(site: str, question_id: int) -> str:
    domain = _CUSTOM_DOMAINS.get(site, f"{site}.stackexchange.com")
    return f"https://{domain}/questions/{question_id}"


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def _truncate(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


@ttl_cache
def search_stackexchange(
    query: str, site: str = DEFAULT_SITE, max_results: int = 5
) -> list[EvidenceItem]:
    """Search Stack Exchange questions matching `query` on `site`.

    Returns real, linkable Q&A items only. `num_comments` on the returned
    EvidenceItem holds the question's answer_count (the closest available
    engagement signal — this endpoint doesn't expose comment counts).
    Never fabricates results: API errors or no matches both yield [].
    """
    if not query or not query.strip():
        return []

    params = {
        "order": "desc",
        "sort": "relevance",
        "q": query,
        "site": site,
        "pagesize": max_results,
    }
    if settings.stackexchange_key:
        params["key"] = settings.stackexchange_key

    try:
        data = http_get(SEARCH_URL, params=params)
    except ToolRequestError:
        return []

    items: list[EvidenceItem] = []
    for hit in data.get("items", [])[:max_results]:
        if hit.get("item_type") != "question":
            continue
        question_id = hit.get("question_id")
        if not question_id:
            continue
        link = _question_url(site, question_id)
        created_ts = hit.get("creation_date")
        created_at = (
            datetime.fromtimestamp(created_ts, tz=timezone.utc).isoformat()
            if created_ts
            else None
        )
        items.append(
            EvidenceItem(
                source_type="stackexchange",
                site=site,
                title=_strip_html(hit.get("title", "")),
                url=link,
                author=None,  # /search/excerpts doesn't return the question owner
                score=hit.get("score"),
                num_comments=hit.get("answer_count"),
                created_at=created_at,
                text=_truncate(_strip_html(hit.get("excerpt", ""))),
            )
        )
    return items
