"""
agent/llm.py

Single place that builds chat models, so the provider can be swapped by env var.

  LLM_PROVIDER              anthropic | openai | ollama   (default: openai)

Models per tier — set per provider so switching LLM_PROVIDER is a one-line change:
  <PROVIDER>_MODEL_CHEAP    router, dynamic prompt, tool-internal calls
  <PROVIDER>_MODEL_STRONG   final answer synthesis
  <PROVIDER>_MODEL_GRADER   hallucination grader (default: same as STRONG)
  <PROVIDER>_MODEL_LIGHT    answers the model router marks as easy (default: STRONG on ollama, CHEAP elsewhere)

Eval judge (scores eval runs and grader validation; never used by the app itself):
  JUDGE_PROVIDER            default: LLM_PROVIDER — e.g. anthropic while the app runs on ollama
  JUDGE_MODEL               default: that provider's GRADER, else STRONG model
  e.g. ANTHROPIC_MODEL_CHEAP, OLLAMA_MODEL_STRONG. LLM_MODEL_CHEAP / _STRONG / _GRADER override
  any provider — handy for one-off eval runs.

Defaults:
  anthropic  cheap claude-haiku-5-5   strong claude-opus-5-5
  openai     cheap gpt-4o-mini        strong gpt-4o
  ollama     cheap llama3.2:3b        strong llama3.2:3b

Anthropic:  ANTHROPIC_API_KEY; ANTHROPIC_EFFORT_CHEAP / _STRONG / _GRADER (low|medium|high|xhigh|max)
Ollama:     OLLAMA_BASE_URL (default http://localhost:11434); OLLAMA_NUM_CTX (default 16384 — Ollama's own
            default is much smaller and silently truncates long prompts); OLLAMA_KEEP_ALIVE (default 30m)

Every model built here goes through the LLM cache in agent/cache.py (LLM_CACHE=on|off).
"""

import os
import json
import re
from dotenv import load_dotenv

from agent.cache import ensure_llm_cache

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

_DEFAULT_MODELS = {
    "anthropic": {"cheap": "claude-haiku-5-5", "strong": "claude-opus-5-5"},
    "openai":    {"cheap": "gpt-4o-mini",      "strong": "gpt-4o"},
    "ollama":    {"cheap": "llama3.2:3b",      "strong": "llama3.2:3b"},
}

# Claude Opus 5.5 and Haiku 5.5 default to "medium" effort, so it is set explicitly per tier:
# routing and tool-internal calls are simple; synthesis and grading need more care.
_ANTHROPIC_EFFORT = {"cheap": "low", "light": "low", "strong": "medium", "grader": "medium", "judge": "medium"}

# Models that accept server-side refusal fallbacks (Haiku has none)
_ANTHROPIC_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}

OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))


def provider_for(tier: str) -> str:
    """The "judge" tier (eval scoring only) can use a different provider from the app itself."""
    if tier == "judge":
        return os.getenv("JUDGE_PROVIDER", PROVIDER).strip().lower()
    return PROVIDER


def model_name(tier: str = "cheap") -> str:
    if tier == "judge":
        p = provider_for("judge").upper()
        return (os.getenv("JUDGE_MODEL") or os.getenv(f"{p}_MODEL_GRADER") or os.getenv(f"{p}_MODEL_STRONG")
                or _DEFAULT_MODELS[provider_for("judge")]["strong"])
    if tier == "grader":
        return (os.getenv("LLM_MODEL_GRADER") or os.getenv(f"{PROVIDER.upper()}_MODEL_GRADER")
                or model_name("strong"))
    if tier == "light":
        # Answers the model router marks as easy. On Ollama the default is the strong model itself: a second
        # model can't stay loaded beside it on an 8 GB GPU, and swapping costs more than it saves.
        return (os.getenv("LLM_MODEL_LIGHT") or os.getenv(f"{PROVIDER.upper()}_MODEL_LIGHT")
                or model_name("strong" if PROVIDER == "ollama" else "cheap"))
    key = "STRONG" if tier == "strong" else "CHEAP"
    return (os.getenv(f"LLM_MODEL_{key}") or os.getenv(f"{PROVIDER.upper()}_MODEL_{key}")
            or _DEFAULT_MODELS[PROVIDER][tier])


def max_prompt_chars() -> int | None:
    """Rough character budget for data stuffed into a prompt (~4 chars/token, 60% of context)."""
    if PROVIDER == "ollama":
        return int(OLLAMA_NUM_CTX * 4 * 0.6)
    return None


def parse_json_reply(text: str) -> dict:
    """Parse a JSON object from a model reply, tolerating code fences or a sentence around it."""
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1] if start >= 0 and end > start else text)


def get_llm(tier: str = "cheap", temperature: float = 0, json_mode: bool = False):
    """Return a chat model for the given tier ("cheap", "light", "strong", "grader" or "judge")."""
    ensure_llm_cache()
    name     = model_name(tier)
    provider = provider_for(tier)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        # No temperature: current Claude models reject sampling parameters. JSON is requested in the
        # prompt and read with parse_json_reply. Thinking is adaptive by default on these models.
        kwargs = {
            "model":      name,
            "max_tokens": 16000,
            "effort":     os.getenv(f"ANTHROPIC_EFFORT_{tier.upper()}") or _ANTHROPIC_EFFORT.get(tier, "medium"),
            # Prompt caching: the stable prefix (tools + system prompt) is billed at ~10% on repeat calls
            "model_kwargs": {"cache_control": {"type": "ephemeral"}},
        }
        if name in _ANTHROPIC_FALLBACK_MODELS:
            # If a safety classifier declines, the API retries on a fallback model inside the same call
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["model_kwargs"]["fallbacks"] = "default"
        return ChatAnthropic(**kwargs)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=name,
            temperature=temperature,
            num_ctx=OLLAMA_NUM_CTX,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            # Keep the model loaded between questions (no reload delay, and Ollama's prompt-prefix cache stays warm)
            keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
            format="json" if json_mode else None,
        )

    from langchain_openai import ChatOpenAI
    kwargs = {"model_kwargs": {"response_format": {"type": "json_object"}}} if json_mode else {}
    return ChatOpenAI(model=name, temperature=temperature, **kwargs)
