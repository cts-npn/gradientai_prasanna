"""Grounded Research Agent — Streamlit entrypoint.

Streams the LangGraph run node-by-node so the UI shows live progress
instead of a blank screen, then renders the grounded answer, Evidence
Confidence Score, cited sources, and a "Research Details" panel with
observable pipeline metadata (never hidden chain-of-thought).
"""

from __future__ import annotations

import streamlit as st

from agent.graph import get_graph
from config.settings import settings

MAX_QUESTION_LENGTH = 500

NODE_MESSAGES = {
    "validate_scope": "Checking the question is in scope...",
    "refuse_out_of_scope": "Question is out of scope.",
    "plan_research": "Planning what evidence is needed...",
    "retrieve_hackernews": "Searching Hacker News...",
    "retrieve_stackexchange": "Searching Stack Exchange...",
    "retrieve_weather": "Calling the live weather API...",
    "collect_evidence": "Collecting retrieved evidence...",
    "sanitize_content": "Screening content for prompt injection / unsafe material...",
    "evaluate_evidence": "Evaluating evidence relevance...",
    "calculate_confidence": "Calculating the Evidence Confidence Score...",
    "grounding_gate": "Deciding whether there's enough evidence to answer...",
    "generate_answer": "Generating a cited answer from the evidence...",
    "validate_citations": "Validating citations against retrieved sources...",
}

BAND_DISPLAY = {
    "strongly_grounded": ("🟢", "Strongly grounded"),
    "well_grounded": ("🟢", "Well grounded"),
    "partially_grounded": ("🟡", "Partially grounded"),
    "insufficient": ("🔴", "Insufficient grounding"),
}

DEMO_QUESTIONS = [
    "What is the current weather in Chennai?",
    "What are the most common complaints people have about electric vehicles?",
    "What is the current weather in Chennai and what are people saying about it online?",
    "Write a romantic poem for my girlfriend",
]


def source_link_and_snippet(source: dict) -> tuple:
    """Return (title_or_name, url, snippet) for either a social or weather source."""
    if "resolved_name" in source:  # weather source
        title = f"Live weather — {source.get('resolved_name')}, {source.get('country')}"
        url = source.get("source_url")
        snippet = f"{source.get('temperature_c')}°C, {source.get('condition')}"
        return title, url, snippet
    return source.get("title", ""), source.get("url", ""), (source.get("text") or "")[:200]


def render_result(state: dict) -> None:
    if state.get("refusal"):
        st.warning(state["refusal"])
        return

    st.markdown("### Answer")
    st.write(state.get("final_answer") or "_No answer produced._")

    score = state.get("confidence_score")
    band = state.get("confidence_band")
    if score is not None and band in BAND_DISPLAY:
        emoji, label = BAND_DISPLAY[band]
        st.markdown(f"**Grounding Confidence: {score:.0f}%**  {emoji} {label}")
        if state.get("conflict_detected"):
            st.markdown("⚠️ **Conflicting evidence detected** — sources disagree; the answer above should reflect that.")

    cited_ids = set(state.get("cited_source_ids") or [])
    cited_sources = [s for s in state.get("sources", []) if s["id"] in cited_ids]
    if cited_sources:
        st.markdown("### Sources")
        for s in cited_sources:
            title, url, snippet = source_link_and_snippet(s)
            if url:
                st.markdown(f"**[{s['id']}]** [{title}]({url})")
            else:
                st.markdown(f"**[{s['id']}]** {title}")
            if snippet:
                st.caption(snippet)

    with st.expander("Research Details"):
        plan = state.get("plan")
        if plan is not None:
            st.markdown(f"**Intent:** {plan.category}")
            st.markdown(f"**Reasoning:** {plan.reasoning}")
            st.markdown("**Tools used:**")
            st.markdown(f"- {'✓' if plan.needs_hackernews else '✗'} Hacker News")
            st.markdown(f"- {'✓' if plan.needs_stackexchange else '✗'} Stack Exchange")
            st.markdown(f"- {'✓' if plan.needs_weather else '✗'} Weather (Open-Meteo)")

        st.markdown(f"**Evidence retrieved:** {state.get('evidence_count', 0)}")
        st.markdown(f"**Evidence cited in answer:** {len(cited_ids)}")

        injection = state.get("injection_detected")
        st.markdown(f"**Prompt injection scan:** {'🛡️ Detected and redacted' if injection else '✓ Clean'}")
        safety_flags = state.get("safety_flags") or {}
        st.markdown(f"**Safety scan:** {'⚠️ Filtered ' + str(len(safety_flags)) + ' source(s)' if safety_flags else '✓ Clean'}")
        st.markdown(
            f"**Citation validation:** {'✓ Passed' if state.get('citation_validation_passed') else '✗ Failed'}"
        )

        breakdown = state.get("confidence_breakdown")
        if breakdown:
            st.markdown("**Confidence breakdown:**")
            st.table(
                {
                    "Component": list(breakdown.keys()),
                    "Score": [f"{v:.0f}/100" for v in breakdown.values()],
                }
            )

        st.markdown("**Full run log:**")
        for line in state.get("status_log", []):
            st.text(line)


st.set_page_config(page_title="Grounded Research Agent", page_icon="🔎")

st.title("🔎 Grounded Research Agent")
st.caption("Evidence-first research with live sources and transparent confidence.")

with st.sidebar:
    st.markdown("### Scope")
    st.markdown(
        "Technology, consumer products, public web discussions, and selected "
        "live factual data (currently: weather)."
    )
    st.markdown("### Live sources")
    st.markdown("- Hacker News (social discussion)\n- Stack Exchange (technical Q&A)\n- Open-Meteo (live weather)")
    st.markdown("### Try a demo question")
    for q in DEMO_QUESTIONS:
        if st.button(q, key=f"demo_{q}", use_container_width=True):
            st.session_state["question_input"] = q
    if not settings.groq_configured:
        st.error("GROQ_API_KEY is not configured — the agent cannot run.")

question = st.text_input(
    "Ask a research question:",
    key="question_input",
    placeholder="What do people think about electric vehicles?",
    max_chars=MAX_QUESTION_LENGTH,
)
ask = st.button("Ask Agent", type="primary", disabled=not settings.groq_configured)

if ask:
    question = (question or "").strip()
    if not question:
        st.warning("Please enter a question.")
    else:
        status_box = st.status("Starting...", expanded=True)
        final_state: dict = {}
        seen_log_lines = 0
        try:
            graph = get_graph()
            run_name = f"query: {question[:60]}"
            for state_snapshot in graph.stream(
                {"question": question, "status_log": []},
                config={"run_name": run_name},
                stream_mode="values",
            ):
                final_state = state_snapshot
                log = state_snapshot.get("status_log", [])
                for line in log[seen_log_lines:]:
                    status_box.write(line)
                seen_log_lines = len(log)
            status_box.update(label="Done", state="complete")
        except Exception as exc:  # noqa: BLE001 — surface any failure to the user, never crash silently
            status_box.update(label="Error", state="error")
            st.error(
                "Something went wrong while running the agent, so no answer is shown "
                f"rather than risk an ungrounded one. ({exc})"
            )
            final_state = {}

        if final_state:
            render_result(final_state)
