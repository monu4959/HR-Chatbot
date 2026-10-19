import importlib

import pytest

import agent.llm as llm


@pytest.fixture
def provider(monkeypatch):
    """Reload agent.llm under a given provider, then restore it."""
    def _set(name, **env):
        monkeypatch.setenv("LLM_PROVIDER", name)
        for var in ["LLM_MODEL_CHEAP", "LLM_MODEL_STRONG", "LLM_MODEL_GRADER"]:
            monkeypatch.delenv(var, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return importlib.reload(llm)
    yield _set
    monkeypatch.undo()
    importlib.reload(llm)


def test_anthropic_tiers_and_request_settings(provider):
    m = provider("anthropic", ANTHROPIC_API_KEY="test-key",
                 ANTHROPIC_MODEL_CHEAP="claude-haiku-5-5", ANTHROPIC_MODEL_STRONG="claude-opus-5-5")
    assert (m.model_name("cheap"), m.model_name("strong"), m.model_name("grader")) == \
           ("claude-haiku-5-5", "claude-opus-5-5", "claude-opus-5-5")

    cheap, strong = m.get_llm("cheap"), m.get_llm("strong", json_mode=True)
    assert cheap.effort == "low" and strong.effort == "medium"
    assert cheap.temperature is None and strong.temperature is None   # sampling params are rejected
    assert not cheap.betas                                           # Haiku has no refusal fallbacks
    assert strong.betas == ["server-side-fallback-2026-07-01"]
    assert strong.model_kwargs["fallbacks"] == "default"
    assert strong.max_tokens == 16000


def test_provider_scoped_models_and_global_override(provider):
    m = provider("ollama", OLLAMA_MODEL_CHEAP="qwen2.5:7b", ANTHROPIC_MODEL_CHEAP="claude-haiku-5-5")
    assert m.model_name("cheap") == "qwen2.5:7b"
    m = provider("ollama", OLLAMA_MODEL_CHEAP="qwen2.5:7b", LLM_MODEL_CHEAP="llama3.2:3b")
    assert m.model_name("cheap") == "llama3.2:3b"


@pytest.mark.parametrize("reply", ['{"score": 0.9}', '```json\n{"score": 0.9}\n```',
                                   'Here is the result: {"score": 0.9} — done.'])
def test_parse_json_reply(reply):
    assert llm.parse_json_reply(reply) == {"score": 0.9}


def test_judge_tier_can_use_another_provider(provider):
    m = provider("ollama", OLLAMA_MODEL_STRONG="qwen2.5:7b", JUDGE_PROVIDER="anthropic",
                 JUDGE_MODEL="claude-haiku-5-5", ANTHROPIC_API_KEY="test-key")
    assert (m.provider_for("strong"), m.model_name("strong")) == ("ollama", "qwen2.5:7b")
    assert (m.provider_for("grader"), m.model_name("grader")) == ("ollama", "qwen2.5:7b")   # the app's grader stays local
    judge = m.get_llm("judge")
    assert type(judge).__name__ == "ChatAnthropic" and judge.model == "claude-haiku-5-5"
