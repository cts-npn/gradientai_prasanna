"""Deterministic prompt-injection detection for retrieved evidence text.

Runs BEFORE evidence reaches the LLM prompt. This is a second, independent
layer on top of the answer prompt's own untrusted-data framing (see
agent/prompts.py) — the project explicitly requires not relying on the
model alone to resist injected instructions, since "the model behaved
well in testing" is not a security guarantee.

Pattern-based detection is inherently incomplete (a determined attacker
can phrase around any fixed pattern list); it's documented here as a
best-effort layer, not a claim of full injection immunity.
"""

from __future__ import annotations

import re
from typing import List, Tuple

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+instructions?", re.I),
    re.compile(r"disregard\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+instructions?", re.I),
    re.compile(r"forget\s+(everything|all)\s+(you\s+(were\s+told|know|learned))?", re.I),
    re.compile(r"reveal\s+(your\s+|the\s+)?(system\s+prompt|instructions|hidden\s+prompt)", re.I),
    re.compile(r"print\s+(your\s+|the\s+)?(system\s+prompt|instructions)", re.I),
    re.compile(r"show\s+me\s+(your\s+|the\s+)?(system\s+prompt|instructions)", re.I),
    re.compile(r"you\s+are\s+now\s+in\s+(developer|debug|dan|admin)\s+mode", re.I),
    re.compile(r"new\s+instructions?\s*:", re.I),
    re.compile(r"^\s*system\s*:\s*", re.I | re.M),
    re.compile(r"\bDAN\b"),
    re.compile(r"do\s+anything\s+now", re.I),
    re.compile(r"i\s+have\s+been\s+pwned", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"override\s+(your\s+|the\s+)?(instructions|programming|rules|guidelines)", re.I),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+(not\s+)?an?\s+ai", re.I),
    re.compile(r"this\s+is\s+(a|an)\s+(system|admin)\s+(message|override)", re.I),
]

REDACTION_MARKER = "[REDACTED: possible prompt injection]"


def detect_injection(text: str) -> bool:
    """True if any known injection pattern is found in `text`."""
    if not text:
        return False
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def sanitize_injection(text: str) -> Tuple[str, bool]:
    """Replace matched spans with a redaction marker.

    Returns (sanitized_text, was_detected). The rest of the pipeline only
    ever sees the sanitized text — matched spans never reach the LLM
    prompt as live text.
    """
    if not text:
        return text, False
    detected = False
    clean = text
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(clean):
            detected = True
            clean = pattern.sub(REDACTION_MARKER, clean)
    return clean, detected
