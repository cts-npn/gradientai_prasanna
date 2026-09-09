"""Deterministic citation validation. No LLM: whether a citation is
fabricated is a plain membership check against the source registry that
was actually populated during this run — not a judgment call.
"""

from __future__ import annotations

import re
from typing import List

CITATION_PATTERN = re.compile(r"\[E(\d+)\]")


def extract_citation_ids(text: str) -> set:
    return {f"E{n}" for n in CITATION_PATTERN.findall(text or "")}


def validate_citations(answer_text: str, sources: List[dict]) -> dict:
    """Check every citation id embedded in `answer_text` against the ids
    actually present in `sources` (the real, retrieved-this-run registry).

    Returns:
    - valid: True only if no fabricated ids were found
    - cited_ids: every [E#] id found in the text
    - fabricated_ids: cited ids that do NOT exist in the source registry
    - used_source_ids: cited ids that DO exist (for display filtering)
    """
    valid_ids = {s["id"] for s in sources}
    cited_ids = extract_citation_ids(answer_text)
    fabricated = cited_ids - valid_ids
    used = cited_ids & valid_ids
    return {
        "valid": len(fabricated) == 0,
        "cited_ids": sorted(cited_ids),
        "fabricated_ids": sorted(fabricated),
        "used_source_ids": sorted(used),
    }
