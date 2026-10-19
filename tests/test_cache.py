import time

import pytest
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration

import agent.cache as cache_mod
import agent.graph as graph
from agent.cache import SQLiteLLMCache, AnswerCache, normalize_question, is_cacheable, ensure_llm_cache


# ── LLM cache ─────────────────────────────────────────────────────────────────

def test_identical_calls_are_served_from_the_cache_and_persist(tmp_path):
    db = tmp_path / "llm.sqlite"
    model = FakeListChatModel(responses=["first", "second"], cache=SQLiteLLMCache(db))
    assert model.invoke("hi").content == "first"
    assert model.invoke("hi").content == "first"     # cached — uncached it would answer "second"
    assert model.invoke("other").content == "second"  # different prompt is a miss

    reopened = FakeListChatModel(responses=["first", "second"], cache=SQLiteLLMCache(db))
    assert reopened.invoke("other").content == "second"  # survives a restart


def test_tool_calls_survive_the_round_trip(tmp_path):
    c = SQLiteLLMCache(tmp_path / "llm.sqlite")
    msg = AIMessage(content="", tool_calls=[{"name": "pandas_query", "args": {"question": "x"}, "id": "call_1"}])
    c.update("prompt", "llm", [ChatGeneration(message=msg)])
    got = c.lookup("prompt", "llm")[0].message
    assert got.tool_calls[0]["name"] == "pandas_query" and got.tool_calls[0]["args"] == {"question": "x"}
    assert c.lookup("prompt", "other-model") is None


def test_ensure_llm_cache_follows_the_env_switch(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "LLM_DB", tmp_path / "llm.sqlite")
    previous = get_llm_cache()
    try:
        monkeypatch.setenv("LLM_CACHE", "on");  ensure_llm_cache()
        assert isinstance(get_llm_cache(), SQLiteLLMCache)
        monkeypatch.setenv("LLM_CACHE", "off"); ensure_llm_cache()
        assert get_llm_cache() is None
    finally:
        set_llm_cache(previous)


# ── Answer cache ──────────────────────────────────────────────────────────────

MODELS = {"provider": "ollama", "cheap": "a", "strong": "b", "grader": "b"}


def test_normalization_is_exact_match_only():
    assert normalize_question("  How many   categories?? ") == normalize_question("how many categories")
    assert normalize_question("How many in Creative & Brand Impact?") != normalize_question("How many in Strategic Business Outcomes?")


def test_answer_cache_hit_and_key_invalidation(tmp_path, monkeypatch):
    ac = AnswerCache(tmp_path / "answers.sqlite")
    ac.put("How many categories?", MODELS, {"answer": "4"})
    assert ac.get("how many categories", MODELS) == {"answer": "4"}
    assert ac.get("How many categories?", {**MODELS, "strong": "other"}) is None   # model change
    monkeypatch.setattr(cache_mod, "code_version", lambda: "edited-code")
    assert ac.get("How many categories?", MODELS) is None                          # code change


def test_answer_cache_ttl(tmp_path, monkeypatch):
    ac = AnswerCache(tmp_path / "answers.sqlite")
    ac.put("q", MODELS, {"answer": "x"})
    monkeypatch.setenv("ANSWER_CACHE_TTL_HOURS", "1")
    monkeypatch.setattr(cache_mod.time, "time", lambda: time.monotonic() + 10**10)
    assert ac.get("q", MODELS) is None


@pytest.mark.parametrize("result, ok", [
    ({"answer": "4", "route_used": "pandas_query", "was_refused": False, "grader_error": False}, True),
    ({"answer": "x", "route_used": "pandas_query", "was_refused": True,  "grader_error": False}, False),
    ({"answer": "x", "route_used": "pandas_query", "was_refused": False, "grader_error": True},  False),
    ({"answer": "x", "route_used": None,           "was_refused": False, "grader_error": False}, False),
])
def test_only_confident_answers_are_cacheable(result, ok):
    assert is_cacheable(result) is ok


# ── run_query integration (fake agent, no models) ─────────────────────────────

class FakeAgent:
    def __init__(self, score=1.0):
        self.calls, self.score = 0, score

    def invoke(self, state, config=None):
        self.calls += 1
        return {
            "messages": state["messages"] + [AIMessage(content=f"There are 4 categories. (call {self.calls})")],
            "route_used": "pandas_query", "last_tool_output": "There are 4 categories: A, B, C, D.",
            "hallucination_score": self.score, "hallucination_flag": False,
            "hallucinated_claims": [], "grader_error": False,
        }


@pytest.fixture
def fake_pipeline(tmp_path, monkeypatch):
    def _make(score=1.0):
        fake = FakeAgent(score)
        monkeypatch.setattr(graph, "get_agent", lambda: fake)
        monkeypatch.setattr(graph, "get_answer_cache", lambda: AnswerCache(tmp_path / "answers.sqlite"))
        return fake
    yield _make
    graph._conversation_history.clear()


def test_repeat_question_is_served_from_cache(fake_pipeline):
    fake = fake_pipeline()
    first  = graph.run_query("How many categories?", session_id="s1")
    second = graph.run_query("how many categories", session_id="s2")
    assert fake.calls == 1 and not first["cached"] and second["cached"]
    assert second["answer"] == first["answer"] and second["trace_id"] == first["trace_id"]


def test_follow_ups_bypass_the_cache_but_history_still_updates(fake_pipeline):
    fake = fake_pipeline()
    graph.run_query("How many categories?", session_id="s1")
    graph.run_query("How many categories?", session_id="s2")       # cache hit, still recorded in s2's history
    assert len(graph._conversation_history["s2"]) == 2
    graph.run_query("And which is largest?", session_id="s2")      # follow-up: pipeline runs
    assert fake.calls == 2


def test_refused_answers_are_not_cached(fake_pipeline):
    fake = fake_pipeline(score=0.2)
    graph.run_query("How many categories?", session_id="s1")
    out = graph.run_query("How many categories?", session_id="s2")
    assert fake.calls == 2 and not out["cached"]


def test_cache_key_ignores_message_bookkeeping():
    from langchain_core.load import dumps
    q = HumanMessage(content="How many categories?")
    a = AIMessage(content="", id="run-1", tool_calls=[{"name": "pandas_query", "args": {}, "id": "c1"}],
                  usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                  response_metadata={"total_duration": 123})
    b = AIMessage(content="", id="run-2", tool_calls=[{"name": "pandas_query", "args": {}, "id": "c1"}],
                  usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12, "total_cost": 0},
                  response_metadata={"total_duration": 999})
    different = AIMessage(content="", tool_calls=[{"name": "graph_query", "args": {}, "id": "c1"}])
    key = SQLiteLLMCache._key
    assert key(dumps([q, a]), "m") == key(dumps([q, b]), "m")
    assert key(dumps([q, a]), "m") != key(dumps([q, different]), "m")
