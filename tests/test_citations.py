"""Citation validation tests (grounding/citations.py) and the
validate_citations graph node's behavior on a fabricated citation."""

from agent import nodes
from grounding.citations import extract_citation_ids, validate_citations


def test_extract_citation_ids_finds_all_markers():
    text = "Battery life is mixed [E1][E3]. Some users disagree [E2]."
    assert extract_citation_ids(text) == {"E1", "E2", "E3"}


def test_extract_citation_ids_empty_text():
    assert extract_citation_ids("") == set()
    assert extract_citation_ids(None) == set()


def test_validate_citations_all_real_passes():
    sources = [{"id": "E1"}, {"id": "E2"}]
    result = validate_citations("Some claim [E1] and another [E2].", sources)
    assert result["valid"] is True
    assert result["fabricated_ids"] == []
    assert result["used_source_ids"] == ["E1", "E2"]


def test_validate_citations_fabricated_id_fails():
    """TEST 9 from the project spec: fake citation attempt is rejected."""
    sources = [{"id": "E1"}]
    result = validate_citations("A claim supported by a source that was never retrieved [E7].", sources)
    assert result["valid"] is False
    assert "E7" in result["fabricated_ids"]


def test_validate_citations_mixed_real_and_fake():
    sources = [{"id": "E1"}]
    result = validate_citations("Real claim [E1] and a made-up one [E9].", sources)
    assert result["valid"] is False
    assert result["fabricated_ids"] == ["E9"]
    assert result["used_source_ids"] == ["E1"]


def test_validate_citations_node_withholds_answer_on_fabricated_citation():
    state = {
        "draft_answer": "This is backed by a source that doesn't exist [E99].",
        "sources": [{"id": "E1"}],
    }
    result = nodes.validate_citations(state)
    assert result["citation_validation_passed"] is False
    assert result["final_answer"] is None
    assert result.get("refusal")


def test_validate_citations_node_passes_through_valid_answer():
    state = {
        "draft_answer": "Battery life complaints are common [E1].",
        "sources": [{"id": "E1"}],
    }
    result = nodes.validate_citations(state)
    assert result["citation_validation_passed"] is True
    assert result["final_answer"] == state["draft_answer"]
    assert result["cited_source_ids"] == ["E1"]
