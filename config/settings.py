"""Central configuration loaded from environment variables / .env.

Streamlit Community Cloud injects secrets as environment variables at
runtime, and python-dotenv loads a local .env for local development, so
this module works unmodified in both places.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    # LLM
    model_provider: str = field(default_factory=lambda: _env("MODEL_PROVIDER", "groq"))
    model_name: str = field(default_factory=lambda: _env("MODEL_NAME", "qwen/qwen3.8-27b"))
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))

    # Reddit — only enabled when all three values are present. Reddit closed
    # self-service OAuth app registration in late 2025; new apps require a
    # manually reviewed application. Until a reviewed app exists, the agent
    # runs on Hacker News + Stack Exchange as its social-evidence sources.
    reddit_client_id: str = field(default_factory=lambda: _env("REDDIT_CLIENT_ID"))
    reddit_client_secret: str = field(default_factory=lambda: _env("REDDIT_CLIENT_SECRET"))
    reddit_user_agent: str = field(default_factory=lambda: _env("REDDIT_USER_AGENT"))

    # Stack Exchange
    stackexchange_key: str = field(default_factory=lambda: _env("STACKEXCHANGE_KEY"))

    # LangSmith
    langchain_tracing_v2: str = field(default_factory=lambda: _env("LANGCHAIN_TRACING_V2", "false"))
    langchain_api_key: str = field(default_factory=lambda: _env("LANGCHAIN_API_KEY"))
    langchain_project: str = field(default_factory=lambda: _env("LANGCHAIN_PROJECT", "grounded-research-agent"))

    @property
    def reddit_enabled(self) -> bool:
        return bool(self.reddit_client_id and self.reddit_client_secret and self.reddit_user_agent)

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def langsmith_configured(self) -> bool:
        return bool(self.langchain_api_key) and self.langchain_tracing_v2.lower() == "true"


settings = Settings()
