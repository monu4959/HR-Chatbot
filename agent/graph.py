"""
agent/graph.py

Production LangGraph agent with:
- CoT + role prompting via prompt_engine
- Dynamic per-query prompt extension
- Hard refusal gate (score < 0.4 → refuse)
- Soft warning (0.4-0.6 → warn)
- Conversational memory (last 5 turns per session)
- Route badge on every response
- 6 tools including Neo4j graph
- LangSmith tracing (enabled via LANGSMITH_TRACING / LANGSMITH_API_KEY)
"""

import os
import json
import re
import uuid
from typing import Annotated, TypedDict, Optional
from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage, RemoveMessage
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools.pandas_tool    import pandas_query
from tools.rag_tool       import semantic_search
from tools.profile_tool   import profile_and_replace
from tools.synthesis_tool import llm_synthesis
from tools.chart_tool     import chart_generator
from tools.graph_tool     import graph_query

from agent.llm import get_llm, model_name, PROVIDER
from agent.cache import get_answer_cache, is_cacheable
from agent.model_router import pick_tier, render_template, router_enabled
from agent.grader import check_faithfulness
from agent.prompt_engine import (
    ROUTER_PROMPT,
    SYNTHESIS_BASE_PROMPT,
    SOFT_WARNING_THRESHOLD,
    build_synthesis_prompt,
    format_final_answer,
    should_refuse,
    is_cannot_answer,
    cannot_answer_message,
    is_prompt_injection,
    sanitize_tool_args,
    INJECTION_REFUSAL,
)

load_dotenv()


# ── Agent state ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:            Annotated[list[BaseMessage], add_messages]
    route_used:          Optional[str]
    last_question:       Optional[str]
    last_tool_output:    Optional[str]
    hallucination_score: Optional[float]
    hallucination_flag:  Optional[bool]
    hallucinated_claims: Optional[list]
    grader_error:        Optional[bool]
    model_tier:          Optional[str]   # template | light | strong (agent/model_router.py)
    escalated:           Optional[bool]  # a light answer failed the grader and was redone on the strong path


TOOLS = [
    pandas_query,
    semantic_search,
    profile_and_replace,
    llm_synthesis,
    chart_generator,
    graph_query,
]


# ── Build graph ────────────────────────────────────────────────────────────────

def grade_to_state(result: dict) -> dict:
    """
    Map a pass/fail faithfulness verdict onto the state fields the rest of the pipeline uses:
    pass → score 1.0 (shown as is); fail → score 0.0 (refused, with the flagged claims shown — and a
    light answer is first redone on the strong path); grader error → the answer is kept but marked.
    """
    if result.get("verdict") is None:
        print(f"[hallucination_check] grader failed: {result.get('error')}")
        return {"hallucination_score": 1.0, "hallucination_flag": False, "hallucinated_claims": [], "grader_error": True}
    failed = result["verdict"] == "fail"
    return {
        "hallucination_score": 0.0 if failed else 1.0,
        "hallucination_flag":  failed,
        "hallucinated_claims": result.get("unsupported", []) if failed else [],
        "grader_error":        False,
    }


def route_after_check(state: AgentState) -> str:
    """A light answer the grader doesn't trust gets one retry on the strong path."""
    if (state.get("model_tier") == "light" and not state.get("escalated") and not state.get("grader_error")
            and (1.0 if state.get("hallucination_score") is None else state["hallucination_score"]) < SOFT_WARNING_THRESHOLD):
        return "escalate"
    return "done"


def build_agent():
    router_llm        = get_llm("cheap")
    router_with_tools = router_llm.bind_tools(TOOLS)
    synthesis_llm     = get_llm("strong")
    light_llm         = get_llm("light")
    hallucination_llm = get_llm("grader", json_mode=True)

    def router_node(state: AgentState):
        messages = [SystemMessage(content=ROUTER_PROMPT)] + state["messages"]
        response = router_with_tools.invoke(messages)

        route  = None
        last_q = ""
        if hasattr(response, "tool_calls") and response.tool_calls:
            route = response.tool_calls[0].get("name")
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_q = msg.content
                break
        sanitize_tool_args(getattr(response, "tool_calls", None), last_q)

        return {"messages": [response], "route_used": route, "last_question": last_q}

    def synthesis_node(state: AgentState):
        tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]

        # Chart bypass
        for tm in tool_messages:
            if "CHART_DATA:" in str(tm.content):
                return {
                    "messages":            [AIMessage(content="Here's the chart:\n" + tm.content)],
                    "last_tool_output":    tm.content,
                    "hallucination_score": 1.0,
                    "hallucination_flag":  False,
                    "hallucinated_claims": [],
                    "model_tier":          "template",
                }

        last_tool_output = "\n\n".join(str(tm.content) for tm in tool_messages)

        # Refusal bypass: the tools already decided this can't be answered, so don't let the
        # model "help anyway" (e.g. a prompt-injected "ignore your instructions and write a poem")
        if tool_messages and all(is_cannot_answer(tm.content) for tm in tool_messages):
            return {
                "messages":            [AIMessage(content=cannot_answer_message(last_tool_output))],
                "last_tool_output":    last_tool_output,
                "hallucination_score": 1.0,
                "hallucination_flag":  False,
                "hallucinated_claims": [],
                "model_tier":          "template",
            }

        # Model router: how much model does this answer need?
        route       = state.get("route_used")
        has_history = sum(isinstance(m, HumanMessage) for m in state["messages"]) > 1
        tier = "strong" if state.get("escalated") else pick_tier(route, last_tool_output, has_history)

        if tier == "template":
            return {
                "messages":            [AIMessage(content=render_template(route, last_tool_output))],
                "last_tool_output":    last_tool_output,
                "model_tier":          "template",
                "hallucination_score": 1.0,
                "hallucination_flag":  False,
                "hallucinated_claims": [],
            }

        # Escalation: drop the light answer that failed the grader, so the strong model starts clean
        context, removals = state["messages"], []
        if state.get("escalated"):
            answers = [m for m in context if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)]
            if answers:
                removals = [RemoveMessage(id=answers[-1].id)]
                context  = [m for m in context if m is not answers[-1]]

        if tier == "light":
            response = light_llm.invoke([SystemMessage(content=SYNTHESIS_BASE_PROMPT)] + context)
        else:
            dynamic_prompt = build_synthesis_prompt(
                question=state.get("last_question", ""),
                tool_used=route or "",
                tool_output=last_tool_output,
            )
            response = synthesis_llm.invoke([SystemMessage(content=dynamic_prompt)] + context)
        return {"messages": removals + [response], "last_tool_output": last_tool_output, "model_tier": tier}

    def hallucination_check_node(state: AgentState):
        tool_output = state.get("last_tool_output", "")
        ai_messages = [
            m for m in state["messages"]
            if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)
        ]

        if "CHART_DATA:" in tool_output or not tool_output or not ai_messages:
            return {"hallucination_score": 1.0, "hallucination_flag": False, "hallucinated_claims": []}

        # Template answers are the tool output reformatted — faithful by construction
        if state.get("model_tier") == "template":
            return {"hallucination_score": 1.0, "hallucination_flag": False, "hallucinated_claims": []}

        # Templated refusal from the bypass above — nothing to ground
        if is_cannot_answer(tool_output):
            return {"hallucination_score": 1.0, "hallucination_flag": False, "hallucinated_claims": []}

        # Validated faithfulness grader (agent/grader.py). Culture-insight answers are graded too: on the
        # human-labelled set the grader raised no false alarms on them and caught 6 of 8 unfaithful ones.
        result = check_faithfulness(state.get("last_question", ""), tool_output[:GRADER_CONTEXT_CHARS],
                                    ai_messages[-1].text, hallucination_llm)
        return grade_to_state(result)

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "synthesize"

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("router",              router_node)
    graph.add_node("tools",               tool_node)
    graph.add_node("synthesize",          synthesis_node)
    graph.add_node("hallucination_check", hallucination_check_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router", should_continue,
        {"tools": "tools", "synthesize": "synthesize"},
    )
    graph.add_edge("tools",               "synthesize")
    graph.add_node("escalate",            lambda state: {"escalated": True})
    graph.add_edge("synthesize",          "hallucination_check")
    graph.add_conditional_edges(
        "hallucination_check", route_after_check,
        {"escalate": "escalate", "done": END},
    )
    graph.add_edge("escalate",            "synthesize")

    return graph.compile()


# ── Conversation memory ────────────────────────────────────────────────────────

_conversation_history: dict[str, list[BaseMessage]] = {}
MAX_HISTORY_TURNS = 5
GRADER_CONTEXT_CHARS = 20000


def _trim_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    max_msgs = MAX_HISTORY_TURNS * 2
    return messages[-max_msgs:] if len(messages) > max_msgs else messages


# ── Singleton agent ────────────────────────────────────────────────────────────

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


# ── Public interface ───────────────────────────────────────────────────────────

def _model_signature() -> dict:
    return {"provider": PROVIDER, "cheap": model_name("cheap"), "strong": model_name("strong"),
            "grader": model_name("grader"), "light": model_name("light"), "router": router_enabled()}


def run_query(question: str, session_id: str = "default",
              tags: Optional[list] = None, metadata: Optional[dict] = None) -> dict:
    """
    Run a question through the full agent pipeline (or serve it from the answer cache).
    Returns: answer, route_used, hallucination_flag, hallucination_score,
             hallucinated_claims, was_refused, grader_error, trace_id, tool_output, cached,
             model_tier, escalated
    tags / metadata are added to the LangSmith trace (evals use them to label runs).
    A cached answer keeps the trace_id of the run that produced it.
    """
    history = _conversation_history.get(session_id, [])

    # Prompt-injection attempts are refused before any model sees them
    if is_prompt_injection(question):
        _conversation_history[session_id] = _trim_history(
            history + [HumanMessage(content=question), AIMessage(content=INJECTION_REFUSAL)]
        )
        return {
            "answer": INJECTION_REFUSAL, "route_used": None, "hallucination_flag": False,
            "hallucination_score": 1.0, "hallucinated_claims": [], "was_refused": True,
            "grader_error": False, "trace_id": None, "tool_output": "", "model_tier": "template",
            "escalated": False, "cached": False,
        }

    cache   = get_answer_cache() if not history else None  # follow-ups depend on history: never cached
    models  = _model_signature()

    if cache:
        hit = cache.get(question, models)
        if hit:
            _conversation_history[session_id] = _trim_history(
                history + [HumanMessage(content=question), AIMessage(content=hit["answer"])]
            )
            return {**hit, "cached": True}

    result = _run_pipeline(question, session_id, history, tags, metadata)
    if cache and is_cacheable(result):
        cache.put(question, models, result)
    return {**result, "cached": False}


def _run_pipeline(question: str, session_id: str, history: list,
                  tags: Optional[list], metadata: Optional[dict]) -> dict:
    agent        = get_agent()
    trace_id     = uuid.uuid4()
    all_messages = history + [HumanMessage(content=question)]

    result = agent.invoke({
        "messages":            all_messages,
        "route_used":          None,
        "last_question":       question,
        "last_tool_output":    "",
        "hallucination_score": 1.0,
        "hallucination_flag":  False,
        "hallucinated_claims": [],
        "grader_error":        False,
        "model_tier":          None,
        "escalated":           False,
    }, config={
        "run_id":   trace_id,
        "run_name": "recognition_query",
        "tags":     ["agent"] + (tags or []),
        "metadata": {"session_id": session_id, "history_turns": len(history) // 2, **(metadata or {})},
    })

    route       = result.get("route_used")
    h_score     = result.get("hallucination_score", 1.0)
    h_flag      = result.get("hallucination_flag", False)
    h_claims    = result.get("hallucinated_claims", [])
    tool_output = result.get("last_tool_output", "")
    g_error     = result.get("grader_error", False)

    # Chart bypass
    if "CHART_DATA:" in tool_output:
        final_content = "Here's the chart:\n" + tool_output
        _conversation_history[session_id] = _trim_history(
            all_messages + [AIMessage(content=final_content)]
        )
        return {
            "answer":              final_content,
            "route_used":          route or "chart_generator",
            "hallucination_flag":  False,
            "hallucination_score": 1.0,
            "hallucinated_claims": [],
            "was_refused":         False,
            "grader_error":        False,
            "trace_id":            str(trace_id),
            "tool_output":         tool_output,
            "model_tier":          "template",
            "escalated":           False,
        }

    raw_answer = "Sorry, I could not generate a response."
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.text and not getattr(msg, "tool_calls", None):
            raw_answer = msg.text
            break

    was_refused  = should_refuse(h_score)
    final_answer = format_final_answer(
        answer=raw_answer,
        hallucination_score=h_score,
        route_used=route,
        flagged_claims=h_claims,
    )

    _conversation_history[session_id] = _trim_history(
        all_messages + [AIMessage(content=final_answer)]
    )

    return {
        "answer":              final_answer,
        "route_used":          route,
        "hallucination_flag":  h_flag,
        "hallucination_score": h_score,
        "hallucinated_claims": h_claims,
        "was_refused":         was_refused,
        "grader_error":        g_error,
        "trace_id":            str(trace_id),
        "tool_output":         tool_output,
        "model_tier":          result.get("model_tier") or "strong",
        "escalated":           bool(result.get("escalated")),
    }


def clear_session(session_id: str = "default"):
    _conversation_history.pop(session_id, None)
