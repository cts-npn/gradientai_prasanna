"""Graph node functions. Each takes the current AgentState and returns a
partial dict that LangGraph merges into it — this is the standard
LangGraph node contract (a node never mutates state in place).
"""

from __future__ import annotations

from agent.llm import get_llm
from agent.prompts import PLAN_SYSTEM_PROMPT, SCOPE_SYSTEM_PROMPT
from agent.state import AgentState, ResearchPlan, ScopeDecision
from tools.hackernews_tool import search_hackernews
from tools.stackexchange_tool import search_stackexchange
from tools.weather_tool import get_weather


def validate_scope(state: AgentState) -> dict:
    """First graph node: is this question inside the agent's defined domain?
    Runs before any retrieval so out-of-scope questions never trigger tool
    calls or an LLM answer attempt.
    """
    question = state["question"]
    llm = get_llm()
    decision: ScopeDecision = llm.with_structured_output(ScopeDecision).invoke(
        [("system", SCOPE_SYSTEM_PROMPT), ("human", question)]
    )
    verdict = "in scope" if decision.in_scope else "out of scope"
    return {
        "in_scope": decision.in_scope,
        "scope_reason": decision.reason,
        "status_log": [f"Scope check: {verdict} — {decision.reason}"],
    }


def refuse_out_of_scope(state: AgentState) -> dict:
    reason = state.get("scope_reason", "This question is outside the agent's defined scope.")
    return {
        "refusal": (
            "This agent is designed for grounded technology/product research and supported "
            f"live-data questions. I don't have grounding for this request. ({reason})"
        ),
        "status_log": ["Refused: question out of scope."],
    }


def plan_research(state: AgentState) -> dict:
    """Decide which retrieval tools are actually needed for this question."""
    question = state["question"]
    llm = get_llm()
    plan: ResearchPlan = llm.with_structured_output(ResearchPlan).invoke(
        [("system", PLAN_SYSTEM_PROMPT), ("human", question)]
    )
    return {"plan": plan, "status_log": [f"Research plan ({plan.category}): {plan.reasoning}"]}


def retrieve_hackernews(state: AgentState) -> dict:
    plan = state["plan"]
    results = search_hackernews(plan.search_query)
    return {
        "hn_results": results,
        "status_log": [f"Hacker News: {len(results)} result(s) for '{plan.search_query}'."],
    }


def retrieve_stackexchange(state: AgentState) -> dict:
    plan = state["plan"]
    results = search_stackexchange(plan.search_query)
    return {
        "se_results": results,
        "status_log": [f"Stack Exchange: {len(results)} result(s) for '{plan.search_query}'."],
    }


def retrieve_weather(state: AgentState) -> dict:
    plan = state["plan"]
    city = plan.weather_city
    if not city:
        return {
            "weather_result": None,
            "status_log": ["Weather requested but no city could be identified in the question."],
        }
    result = get_weather(city)
    msg = f"Weather: retrieved current conditions for {city}." if result else f"Weather: could not resolve '{city}'."
    return {"weather_result": result, "status_log": [msg]}


def collect_evidence(state: AgentState) -> dict:
    """Convergence point after retrieval: assigns stable citation IDs to
    every retrieved item, in a fixed order, regardless of which order the
    parallel retrieval branches happened to finish in.
    """
    sources: list[dict] = []
    counter = 1
    for item in state.get("hn_results") or []:
        sources.append({"id": f"E{counter}", **item.to_dict()})
        counter += 1
    for item in state.get("se_results") or []:
        sources.append({"id": f"E{counter}", **item.to_dict()})
        counter += 1
    weather = state.get("weather_result")
    if weather:
        sources.append({"id": f"E{counter}", **weather.to_dict()})
        counter += 1
    return {"sources": sources, "status_log": [f"Collected {len(sources)} source(s) for grounding."]}
