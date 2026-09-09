"""Builds and compiles the LangGraph state graph.

Full flow (Phase 10 — this is the complete pipeline):

    START -> validate_scope --[in scope]--> plan_research --> route_tools --> {retrieve_*} -> collect_evidence
                             --[out of scope]--> refuse_out_of_scope -> END

    collect_evidence -> sanitize_content -> evaluate_evidence -> calculate_confidence -> grounding_gate
        --[grounded]--> generate_answer -> validate_citations -> END
        --[not grounded]--> END (refusal set by the gate)

sanitize_content scans every source for prompt-injection and unsafe-content
patterns and redacts matches BEFORE relevance scoring or the answer LLM
ever see the text. calculate_confidence computes the Evidence Confidence
Score from evaluate_evidence's output; grounding_gate thresholds on that
score (>=50, matching the documented "at least partially grounded" band)
instead of raw relevance alone.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent import nodes, router
from agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("validate_scope", nodes.validate_scope)
    graph.add_node("refuse_out_of_scope", nodes.refuse_out_of_scope)
    graph.add_node("plan_research", nodes.plan_research)
    graph.add_node("retrieve_hackernews", nodes.retrieve_hackernews)
    graph.add_node("retrieve_stackexchange", nodes.retrieve_stackexchange)
    graph.add_node("retrieve_weather", nodes.retrieve_weather)
    graph.add_node("collect_evidence", nodes.collect_evidence)
    graph.add_node("sanitize_content", nodes.sanitize_content)
    graph.add_node("evaluate_evidence", nodes.evaluate_evidence)
    graph.add_node("calculate_confidence", nodes.calculate_confidence)
    graph.add_node("grounding_gate", nodes.grounding_gate)
    graph.add_node("generate_answer", nodes.generate_answer)
    graph.add_node("validate_citations", nodes.validate_citations)

    graph.add_edge(START, "validate_scope")
    graph.add_conditional_edges("validate_scope", router.scope_router)
    graph.add_edge("refuse_out_of_scope", END)

    graph.add_conditional_edges("plan_research", router.tool_router)
    graph.add_edge("retrieve_hackernews", "collect_evidence")
    graph.add_edge("retrieve_stackexchange", "collect_evidence")
    graph.add_edge("retrieve_weather", "collect_evidence")

    graph.add_edge("collect_evidence", "sanitize_content")
    graph.add_edge("sanitize_content", "evaluate_evidence")
    graph.add_edge("evaluate_evidence", "calculate_confidence")
    graph.add_edge("calculate_confidence", "grounding_gate")
    graph.add_conditional_edges("grounding_gate", router.gate_router)
    graph.add_edge("generate_answer", "validate_citations")
    graph.add_edge("validate_citations", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
