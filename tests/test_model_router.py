import os
import re

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

os.environ["NEO4J_URI"] = ""

import tools.pandas_tool as pt
from agent.graph import route_after_check
from agent.model_router import pick_tier, render_template
from tools.graph_tool import graph_query
from tools.pandas_tool import GENERATED_CODE_PREFIX
from tools.rag_tool import semantic_search


@pytest.fixture(autouse=True)
def router_on(monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER", "on")


# ── tier rules ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("route, output, history, tier", [
    ("pandas_query",        "There are 48 recognitions (4.8%) in the category 'X'.", False, "template"),
    ("pandas_query",        "There are 48 recognitions.",                           True,  "template"),  # exact either way
    ("pandas_query",        GENERATED_CODE_PREFIX + "21.0%",                         False, "light"),
    ("pandas_query",        GENERATED_CODE_PREFIX + "21.0%",                         True,  "strong"),    # follow-up needs history
    ("graph_query",         "=== Top 3 Flows ===\n  A → B: 3",                       False, "template"),
    ("semantic_search",     "SEARCH COMPLETE: Found 2 relevant recognitions ...",    False, "template"),
    ("semantic_search",     "x" * 20000,                                             False, "strong"),    # too long to show raw
    ("profile_and_replace", "Profile for Marcus ...",                                False, "strong"),
    ("llm_synthesis",       "The organization values ...",                           False, "strong"),
    (None,                  "",                                                      False, "strong"),
])
def test_pick_tier(route, output, history, tier):
    assert pick_tier(route, output, history) == tier


def test_router_off_sends_everything_to_strong(monkeypatch):
    monkeypatch.setenv("MODEL_ROUTER", "off")
    assert pick_tier("pandas_query", "There are 48.", False) == "strong"


# ── template rendering: formatting only, never new facts ──────────────────────

def _numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def test_pandas_breakdown_template_keeps_every_number():
    out = pt.pandas_query.invoke({"question": "How many awards are in each category?"})
    rendered = render_template("pandas_query", out)
    assert _numbers(rendered) == _numbers(out)
    assert rendered.count("\n- ") == 4


def test_graph_template_turns_headings_bold_and_adds_no_numbers():
    out = graph_query.invoke({"question": "Which departments recognize each other most?"})
    rendered = render_template("graph_query", out)
    assert "===" not in rendered and "**Top" in rendered
    assert _numbers(rendered) <= _numbers(out)


def test_search_template_shows_every_result_and_drops_the_instruction():
    out = semantic_search.invoke({"query": "Find recognitions about mentorship"})
    rendered = render_template("semantic_search", out)
    n = int(re.search(r"Found (\d+) relevant", out).group(1))
    assert len(re.findall(r"^\*\*\d+\. ", rendered, re.MULTILINE)) == n
    assert "INSTRUCTION" not in rendered and "Present ALL" not in rendered
    # every quoted message is text from the tool output
    for quote in re.findall(r"^> (.*)$", rendered, re.MULTILINE):
        assert quote[:80] in out


# ── escalation ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("state, decision", [
    ({"model_tier": "light",  "hallucination_score": 0.3, "escalated": False, "grader_error": False}, "escalate"),
    ({"model_tier": "light",  "hallucination_score": 0.9, "escalated": False, "grader_error": False}, "done"),
    ({"model_tier": "light",  "hallucination_score": 0.3, "escalated": True,  "grader_error": False}, "done"),  # one retry only
    ({"model_tier": "light",  "hallucination_score": 1.0, "escalated": False, "grader_error": True},  "done"),
    ({"model_tier": "strong", "hallucination_score": 0.3, "escalated": False, "grader_error": False}, "done"),
])
def test_route_after_check(state, decision):
    assert route_after_check(state) == decision


# ── generated-code marker ─────────────────────────────────────────────────────

def test_generated_code_results_are_marked(monkeypatch):
    monkeypatch.setattr(pt, "get_llm", lambda *a, **k: FakeListChatModel(responses=["result = '21.0%'"]))
    out = pt._llm_pandas_fallback("What share of titles mention engineer?", pt.load_dataframe())
    assert out == GENERATED_CODE_PREFIX + "21.0%"
    assert pick_tier("pandas_query", out, False) == "light"
