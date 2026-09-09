"""Routing decisions for conditional graph edges. Kept deterministic (pure
functions over already-decided state) — the LLM's job is over by the time
these run; it already produced `in_scope` and `plan` upstream.
"""

from __future__ import annotations

from typing import List

from langgraph.graph import END

from agent.state import AgentState


def scope_router(state: AgentState) -> str:
    return "plan_research" if state.get("in_scope") else "refuse_out_of_scope"


def tool_router(state: AgentState) -> List[str]:
    """Fan out to every retrieval node the plan actually requires. Never
    calls a tool "just in case" — if the plan needs nothing (e.g. a
    factual_other question with no matching tool), evidence collection
    still runs on an empty set so the grounding gate downstream can refuse
    for insufficient evidence rather than the router silently guessing.
    """
    plan = state["plan"]
    targets: List[str] = []
    if plan.needs_hackernews:
        targets.append("retrieve_hackernews")
    if plan.needs_stackexchange:
        targets.append("retrieve_stackexchange")
    if plan.needs_weather:
        targets.append("retrieve_weather")
    return targets or ["collect_evidence"]


def gate_router(state: AgentState) -> str:
    """The grounding gate node already writes the refusal message itself
    when it fails, so the failing path here goes straight to END rather
    than a separate refusal node.
    """
    return "generate_answer" if state.get("grounded") else END
