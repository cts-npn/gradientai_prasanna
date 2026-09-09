"""Lightweight, deterministic safety filtering for retrieved evidence text.

This is a pattern-based filter appropriate to this project's scale, not a
production content-moderation system. It's also worth being explicit that
Hacker News and Stack Overflow are both actively, heavily moderated
communities, so the realistic residual risk from these specific sources
is low — this layer exists because the assignment requires retrieved
content to never be treated as automatically safe to reproduce, not
because these sources are expected to surface much of it in practice.
"""

from __future__ import annotations

import re
from typing import List, Tuple

_DANGEROUS_PATTERNS = [
    re.compile(r"how\s+to\s+(make|build|synthesi[sz]e)\s+(a\s+)?(bomb|explosive|nerve\s+agent|chemical\s+weapon)", re.I),
    re.compile(r"instructions?\s+for\s+(making|building)\s+(a\s+)?(bomb|explosive|weapon)", re.I),
    re.compile(r"how\s+to\s+kill\s+(yourself|myself)", re.I),
    re.compile(r"\bsuicide\s+method\b", re.I),
    re.compile(r"how\s+to\s+make\s+a\s+gun\s+at\s+home", re.I),
]

_HARASSMENT_PATTERNS = [
    re.compile(r"\bf+u+c+k+(ing|er|ers)?\b", re.I),
    re.compile(r"\bshit+(ty)?\b", re.I),
    re.compile(r"\bbitch(es)?\b", re.I),
    re.compile(r"\basshole?s?\b", re.I),
    re.compile(r"\bi\s+will\s+kill\s+you\b", re.I),
    re.compile(r"\bkys\b", re.I),
]

_SEXUAL_PATTERNS = [
    re.compile(r"\bnsfw\b", re.I),
    re.compile(r"\bporn(ography)?\b", re.I),
    re.compile(r"\bexplicit\s+sexual\s+content\b", re.I),
]

_CATEGORIES = (
    ("dangerous_instructions", _DANGEROUS_PATTERNS),
    ("harassment_language", _HARASSMENT_PATTERNS),
    ("sexual_content", _SEXUAL_PATTERNS),
)

REDACTION_MARKER = "[REDACTED: unsafe content filtered]"


def scan_safety(text: str) -> List[str]:
    """Return the list of flagged category names found in `text` (empty if clean)."""
    if not text:
        return []
    return [name for name, patterns in _CATEGORIES if any(p.search(text) for p in patterns)]


def sanitize_safety(text: str) -> Tuple[str, List[str]]:
    """Mask flagged spans. Returns (sanitized_text, flagged_categories)."""
    flags = scan_safety(text)
    if not flags:
        return text, flags
    clean = text
    for _, patterns in _CATEGORIES:
        for pattern in patterns:
            clean = pattern.sub(REDACTION_MARKER, clean)
    return clean, flags
