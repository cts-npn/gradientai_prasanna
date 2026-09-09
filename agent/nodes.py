"""Graph node functions. Each takes the current AgentState and returns a
partial dict that LangGraph merges into it — this is the standard
LangGraph node contract (a node never mutates state in place).
"""

from __future__ import annotations

from agent.llm import get_llm
from agent.prompts import (
    ANSWER_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    SCOPE_SYSTEM_PROMPT,
    format_evidence_block,
)
from agent.state import AgentState, GeneratedAnswer, ResearchPlan, ScopeDecision
from grounding.citations import validate_citations as check_citations
from grounding.evidence import evaluate_evidence as compute_evidence_evaluation
from tools.hackernews_tool import search_hackernews
from tools.stackexchange_tool import search_stackexchange
from tools.weather_tool import get_weather

# Interim grounding-gate thresholds. Phase 10 replaces relevance_score here
# with the full weighted Evidence Confidence Score and re-thresholds using
# the documented 0-100 bands (>=50 = at least "partially grounded"); this
# gate uses the same cutoff in the meantime rather than inventing a
# different, undocumented number.
MIN_EVIDENCE_COUNT = 1
MIN_RELEVANCE_SCORE = 30.0


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


def evaluate_evidence(state: AgentState) -> dict:
    """Score retrieved evidence for relevance before any answer is attempted.
    Pure deterministic computation — see grounding/evidence.py.
    """
    plan = state.get("plan")
    search_query = plan.search_query if plan else ""
    evaluation = compute_evidence_evaluation(state["question"], search_query, state.get("sources") or [])
    return {
        "evidence_count": evaluation["evidence_count"],
        "relevance_score": evaluation["relevance_score"],
        "per_source_relevance": evaluation["per_source_relevance"],
        "status_log": [
            f"Evidence evaluated: {evaluation['evidence_count']} source(s), "
            f"relevance {evaluation['relevance_score']:.0f}/100."
        ],
    }


def grounding_gate(state: AgentState) -> dict:
    """Decide whether the agent may proceed toward an answer. Never lets a
    generation step run on insufficient evidence — this is the one place
    "I don't have enough evidence" gets enforced before any LLM synthesis.
    """
    count = state.get("evidence_count", 0)
    relevance = state.get("relevance_score", 0.0)

    if count < MIN_EVIDENCE_COUNT:
        return {
            "grounded": False,
            "refusal": (
                "I don't have sufficient grounded evidence from my available sources to "
                "answer this reliably."
            ),
            "status_log": ["Grounding gate: FAILED (no evidence retrieved)."],
        }

    if relevance < MIN_RELEVANCE_SCORE:
        return {
            "grounded": False,
            "refusal": (
                "I found limited relevant evidence, so I don't have enough grounding to "
                "provide a reliable answer."
            ),
            "status_log": [f"Grounding gate: FAILED (relevance {relevance:.0f} < {MIN_RELEVANCE_SCORE:.0f})."],
        }

    return {
        "grounded": True,
        "status_log": [f"Grounding gate: PASSED (relevance {relevance:.0f} >= {MIN_RELEVANCE_SCORE:.0f})."],
    }


def generate_answer(state: AgentState) -> dict:
    """Synthesize an answer strictly from the collected evidence. Only
    reached after the grounding gate has passed, so `sources` is guaranteed
    non-empty here.
    """
    sources = state["sources"]
    evidence_block = format_evidence_block(sources)
    llm = get_llm()
    result: GeneratedAnswer = llm.with_structured_output(GeneratedAnswer).invoke(
        [
            ("system", ANSWER_SYSTEM_PROMPT),
            ("human", f"Question: {state['question']}\n\n{evidence_block}"),
        ]
    )
    return {
        "draft_answer": result.answer,
        "status_log": [f"Answer drafted, citing {len(result.citations_used)} source(s)."],
    }


def validate_citations(state: AgentState) -> dict:
    """Deterministic check (grounding/citations.py): every [E#] the model
    cited must exist in the source registry actually populated this run.
    A fabricated citation fails the run rather than being silently shown.
    """
    validation = check_citations(state["draft_answer"], state["sources"])
    if not validation["valid"]:
        return {
            "citation_validation_passed": False,
            "final_answer": None,
            "refusal": (
                "I drafted an answer but it referenced a source that wasn't actually "
                "retrieved in this run, so I'm withholding it rather than show an "
                "unverifiable citation."
            ),
            "status_log": [f"Citation validation FAILED — fabricated id(s): {validation['fabricated_ids']}"],
        }
    return {
        "citation_validation_passed": True,
        "cited_source_ids": validation["used_source_ids"],
        "final_answer": state["draft_answer"],
        "status_log": [f"Citation validation passed — {len(validation['used_source_ids'])} source(s) cited."],
    }
