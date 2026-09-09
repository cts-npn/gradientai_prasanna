"""Grounded Research Agent — Streamlit entrypoint.

Phase 1 scaffold: confirms the project boots and settings load. The real
LangGraph agent and UI are wired in during later phases.
"""

import streamlit as st

from config.settings import settings

st.set_page_config(page_title="Grounded Research Agent", page_icon="🔎")

st.title("🔎 Grounded Research Agent")
st.caption("Evidence-first research with live sources and transparent confidence.")

st.info(
    "Project scaffold is live. Agent graph, tools, and guardrails are added "
    "in the phases that follow."
)

with st.expander("Environment check"):
    st.write(
        {
            "model_provider": settings.model_provider,
            "model_name": settings.model_name,
            "groq_configured": settings.groq_configured,
            "reddit_enabled": settings.reddit_enabled,
            "langsmith_configured": settings.langsmith_configured,
        }
    )
