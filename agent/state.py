"""Shared state that flows through the LangGraph graph, plus the
structured (Pydantic) models the LLM fills in at the scope-validation and
research-planning steps.

Using `typing.Optional` rather than the `X | None` syntax throughout this
file deliberately: Pydantic resolves annotations at class-definition time,
and this project targets Python 3.9, where `X | None` as a runtime type
expression isn't supported.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

from tools.tool_utils import EvidenceItem
from tools.weather_tool import WeatherResult


class ScopeDecision(BaseModel):
    """Output of the scope-validation LLM call."""

    in_scope: bool = Field(description="True if the question fits the agent's defined domain scope")
    reason: str = Field(description="One short sentence explaining the decision")


class ResearchPlan(BaseModel):
    """Output of the research-planning LLM call — which evidence to retrieve."""

    category: Literal["social_opinion", "live_weather", "hybrid", "factual_other"] = Field(
        description="The kind of evidence this question needs"
    )
    needs_hackernews: bool = Field(description="Whether Hacker News discussion evidence is needed")
    needs_stackexchange: bool = Field(description="Whether Stack Exchange technical Q&A evidence is needed")
    needs_weather: bool = Field(description="Whether live weather data is needed")
    weather_city: Optional[str] = Field(
        default=None, description="City name to look up weather for, if needs_weather is true"
    )
    search_query: str = Field(description="A concise search query to use against Hacker News / Stack Exchange")
    reasoning: str = Field(description="One or two sentences explaining the plan")


class GeneratedAnswer(BaseModel):
    """Output of the answer-synthesis LLM call."""

    answer: str = Field(description="The answer text, with inline [E#] citations for every factual claim")
    citations_used: List[str] = Field(
        default_factory=list, description="Every citation id (e.g. 'E1') actually used in the answer"
    )


class AgentState(TypedDict, total=False):
    """The graph's shared state. Nodes read and write slices of this dict;
    LangGraph merges each node's returned partial dict into the running
    state between steps.
    """

    question: str
    # Multiple retrieval nodes can run in the same superstep (parallel tool
    # calls), each appending its own message — operator.add merges those
    # lists instead of LangGraph raising a concurrent-update conflict on a
    # plain overwrite key.
    status_log: Annotated[List[str], operator.add]

    in_scope: bool
    scope_reason: str

    plan: ResearchPlan

    hn_results: List[EvidenceItem]
    se_results: List[EvidenceItem]
    weather_result: Optional[WeatherResult]

    sources: List[dict]

    evidence_count: int
    relevance_score: float
    per_source_relevance: dict
    grounded: bool

    draft_answer: str
    citation_validation_passed: bool
    cited_source_ids: List[str]
    final_answer: Optional[str]

    refusal: Optional[str]
