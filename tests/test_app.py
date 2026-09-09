"""Streamlit UI tests using streamlit.testing.v1.AppTest — actually runs
the app script and simulates user interaction (typing, clicking), rather
than just checking the static page loads. Live: hits the real model and
tools, so skipped without GROQ_API_KEY like the rest of the live suite.
"""

import pytest
from streamlit.testing.v1 import AppTest

from config.settings import settings

pytestmark = pytest.mark.skipif(not settings.groq_configured, reason="GROQ_API_KEY not set")


def _ask(at: AppTest, question: str) -> AppTest:
    at.text_input(key="question_input").input(question)
    next(b for b in at.button if b.label == "Ask Agent").click()
    at.run()
    return at


def test_app_loads_without_exceptions():
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    assert not at.exception


def test_grounded_question_renders_answer_and_confidence():
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    at = _ask(at, "What is the current weather in Chennai?")
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Answer" in text
    assert "Grounding Confidence" in text
    assert "Sources" in text


def test_out_of_scope_question_renders_refusal_warning():
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    at = _ask(at, "Write a romantic poem for my girlfriend")
    assert not at.exception
    assert len(at.warning) >= 1
    assert "outside the agent's scope" in at.warning[-1].value or "don't have grounding" in at.warning[-1].value


def test_demo_button_prefills_question_input():
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    demo_btn = next(b for b in at.button if "electric vehicles" in b.label)
    demo_btn.click()
    at.run()
    assert not at.exception
    assert at.text_input(key="question_input").value == (
        "What are the most common complaints people have about electric vehicles?"
    )


def test_empty_question_shows_warning_not_crash():
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    next(b for b in at.button if b.label == "Ask Agent").click()
    at.run()
    assert not at.exception
    assert any("enter a question" in w.value for w in at.warning)
