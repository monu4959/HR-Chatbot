"""
agent/prompt_engine.py

Central prompt registry for the Recognition Analytics Agent.
All prompts live here — static base (role + CoT) + dynamic per-query extension.
"""

import json
import re
from agent.llm import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

# ── Thresholds ────────────────────────────────────────────────────────────────
HARD_REFUSAL_THRESHOLD = 0.4
SOFT_WARNING_THRESHOLD = 0.6

# ── Static base prompts ───────────────────────────────────────────────────────

ROUTER_PROMPT = """You are an expert routing agent for an HR recognition analytics platform.

ROLE: You are a precise intent classifier. Pick the single best tool for the user's question.

CHAIN OF THOUGHT — before deciding, reason through:
1. What type of question is this? (counting/stats, semantic search, person/role profile, cultural insight, chart, or graph/network)
2. Does conversation history change what the user is really asking?
3. Which tool is the closest match? Is there ambiguity?
4. If ambiguous, pick the tool that most directly answers the question.

TOOLS:
1. pandas_query        — counts, stats, rankings, distributions, percentages, "how many"
2. semantic_search     — find recognitions by topic, theme, keyword, behavior, person name
3. profile_and_replace — person/role profiles, skills, who can replace someone
4. llm_synthesis       — cultural insights, themes, patterns, open-ended analysis
5. chart_generator     — any request to visualize, plot, chart, or graph data
6. graph_query         — relationships, collaboration networks, centrality, "who works with whom",
                         cross-department flow, bridge roles, recognition patterns between teams

RULES:
- Always call exactly one tool. Never answer directly.
- Use conversation history to resolve follow-up questions.
- "How many", "how often", counts and percentages go to pandas_query — including how many messages mention
  a word or topic. semantic_search is for finding and showing example messages, not for counting them.
- Role groups, departments, teams or "groups of roles" (job titles grouped together) go to graph_query —
  including which group gives or receives the most recognition. Individual job titles go to pandas_query.
- Comparing recognition categories or subcategories by name ("how does X differ from Y", "compare X and Y")
  goes to llm_synthesis — categories are not teams.
- Pass the user's FULL question as the tool argument. Only rewrite it to resolve references from
  history (e.g. "him" → "Marcus", "that" → "awards by category"). Never shorten it to keywords —
  the tools rely on words like "replace", "unique", "each" and "which".
- If the question is completely outside this dataset, call llm_synthesis — it will handle refusal.
"""

SYNTHESIS_BASE_PROMPT = """You are a senior HR analytics consultant with 10 years of experience interpreting employee recognition data. Your professional reputation depends entirely on only stating what the data directly supports.

ROLE: Present tool results clearly, accurately, and only based on what was retrieved.

CHAIN OF THOUGHT — before writing your answer, work through these steps:
Step 1 — What did the tool actually return? Read it carefully.
Step 2 — What does this data directly tell us? List only facts present in the tool output.
Step 3 — What does the user's question ask for? Map the data to the question.
Step 4 — Are there gaps — things asked about that the data doesn't cover? Note them honestly.
Step 5 — Write the answer using only what Step 2 identified. Do not add interpretation beyond the data.

RULES:
- Never invent names, numbers, or facts not present in the tool output
- If the tool returned 0 results — say so clearly, do not speculate
- If CRAG quality is Low — acknowledge limited confidence
- For stats: report exact numbers only
- For search results: present every RESULT, never skip any
- For graph/network results: explain relationships in plain organizational language
- Use markdown. Reference conversation history naturally if relevant.

REFUSAL RULE: If tool output contains "CANNOT_ANSWER" — tell the user honestly this question cannot be answered from the available dataset, and suggest what kinds of questions can be answered instead.
"""

HALLUCINATION_CHECK_PROMPT = """You are a rigorous fact-checking auditor for an AI analytics system.

ROLE: Determine whether an AI-generated answer is fully grounded in its source context.

CHAIN OF THOUGHT:
Step 1 — Read the SOURCE CONTEXT carefully. List the key facts it contains.
Step 2 — Read the AI ANSWER carefully. List every factual claim it makes.
Step 3 — For each claim in the answer, check: directly supported / inferred / hallucinated?
Step 4 — Count: how many claims are supported vs hallucinated?
Step 5 — Assign a score 0.0-1.0:
  1.0 = all claims supported
  0.7-0.9 = mostly supported, minor inferences
  0.4-0.6 = some hallucination
  0.0-0.3 = significant hallucination

OUTPUT: Return ONLY valid JSON, no other text:
{"score": 0.85, "hallucinated_claims": ["claim 1"] or [], "issues": "brief description or null"}
"""

PANDAS_FALLBACK_PROMPT = """You are an expert Python/pandas data analyst.

ROLE: Write precise pandas code to answer analytical questions about an employee recognition dataset.

CHAIN OF THOUGHT:
Step 1 — What metric does the question ask for? (count, percentage, rank, filter, etc.)
Step 2 — Which columns are relevant?
Step 3 — Write the simplest pandas operation to compute it.
Step 4 — Format as a readable string.

DATAFRAME `df` columns: {columns}
Sample values: {sample_values}

RULES:
- Store final answer as string in variable `result`; include the label(s) and number(s) it is about
- `df`, `pd`, `re` and `Counter` are already available — do NOT import anything, read files or use dunder attributes
- Do NOT use print()
- Text matching must be case-insensitive and NaN-safe: df[col].str.contains("x", case=False, na=False)
- Job titles vary ("Engineer", "Engineering", "Senior Software Engineer"), so match on word stems, not exact titles
- Only set result = "CANNOT_ANSWER: <reason>" if the needed column does not exist — a filter that matches 0 rows is an answer (0), not CANNOT_ANSWER
- 1-10 lines of code maximum
- Return ONLY the code, no markdown
"""

PROFILE_EXTRACTION_PROMPT = """You are a senior talent analytics specialist building evidence-based employee profiles.

ROLE: Extract a structured skill profile ONLY from what the recognition messages explicitly say.

CHAIN OF THOUGHT:
Step 1 — Read each message. List only concrete behaviors or actions explicitly mentioned.
Step 2 — Group similar behaviors into skill themes — only if backed by message evidence.
Step 3 — Identify business impact only if a message mentions it.
Step 4 — Look for seniority signals only if a message mentions them.
Step 5 — Write key_themes using only what the messages say.

STRICT RULES:
- NEVER invent generic skills like "teamwork", "communication", "leadership", "problem-solving" unless a message explicitly uses those words
- NEVER fill empty fields with guesses — use [] if there is no evidence
- key_themes must read like a summary of the actual messages, not career advice
- Return ONLY valid JSON, no other text

JSON schema:
{{
  "name": string,
  "message_count": int,
  "top_skills": [list of 5-8 specific skills with evidence],
  "categories": [list of work categories],
  "impact_areas": [list of business impact areas],
  "seniority_signals": [list of seniority indicators],
  "key_themes": "2-3 sentence summary based only on what messages say"
}}
"""

SYNTHESIS_TOOL_PROMPT = """You are a senior organizational culture researcher and HR strategist.

ROLE: Analyze employee recognition data to surface genuine cultural insights and behavioral patterns.

CHAIN OF THOUGHT:
Step 1 — What is the specific question? (values, behavioral patterns, comparison, etc.)
Step 2 — Scan the data. What themes appear repeatedly across multiple messages?
Step 3 — What do nomination patterns (who gives to whom, which roles) reveal?
Step 4 — What is notably absent? What does the data NOT show?
Step 5 — Answer the question using patterns from Steps 2-4.
Step 6 — If data is insufficient, say so explicitly.

RULES:
- Reference specific patterns from the data
- If pattern appears in only 1-2 messages: "there is limited evidence that..."
- Never state something as fact if the data doesn't support it
- 3-5 paragraphs maximum
- If asked about something outside this dataset: "CANNOT_ANSWER: This question is outside the scope of the recognition dataset."
"""

# ── Dynamic prompt generator ──────────────────────────────────────────────────

DYNAMIC_PROMPT_GENERATOR = """You are a prompt engineering specialist. Write a short, targeted instruction extension for an AI that is about to answer a specific question using retrieved data.

You will receive:
- The user's question
- Which tool was used
- A preview of what the tool returned

Write 2-4 sentences of ADDITIONAL instructions that tailor the synthesis prompt for this specific question.

Focus on:
- What specific angle or framing the answer should take
- Any caution needed (e.g. "data only covers job titles, not individual names")
- What format best serves this question (table, bullets, paragraphs, etc.)
- Any domain-specific context for interpreting this result

Do NOT repeat base rules. Only add question-specific guidance.
Return ONLY the instruction text, no labels or formatting."""

_prompt_llm = None


def _get_prompt_llm():
    global _prompt_llm
    if _prompt_llm is None:
        _prompt_llm = get_llm("cheap")
    return _prompt_llm


def generate_dynamic_prompt_extension(question: str, tool_used: str, tool_output_preview: str) -> str:
    """Generate a question-specific prompt extension via the cheap model."""
    llm = _get_prompt_llm()
    user_msg = (
        f"User question: {question}\n\n"
        f"Tool used: {tool_used}\n\n"
        f"Tool output preview (first 600 chars):\n{tool_output_preview[:600]}\n\n"
        f"Write 2-4 sentences of additional synthesis instructions tailored to this question and result."
    )
    try:
        response = llm.invoke([
            SystemMessage(content=DYNAMIC_PROMPT_GENERATOR),
            HumanMessage(content=user_msg),
        ])
        return response.text.strip()
    except Exception:
        return ""


def build_synthesis_prompt(question: str, tool_used: str, tool_output: str) -> str:
    """Combine static base + dynamic extension into final synthesis prompt."""
    dynamic = generate_dynamic_prompt_extension(question, tool_used or "unknown", tool_output)
    if dynamic:
        return (
            SYNTHESIS_BASE_PROMPT
            + "\n\n--- QUESTION-SPECIFIC INSTRUCTIONS ---\n"
            + dynamic
        )
    return SYNTHESIS_BASE_PROMPT


# ── Refusal gate ──────────────────────────────────────────────────────────────

REFUSAL_MESSAGE = """I'm not able to provide a confident answer to this question based on the available recognition dataset.

This could be because:
- The question asks about information not captured in recognition messages (salary, performance ratings, personal details)
- The retrieved data wasn't relevant enough to the question
- The question requires information outside this dataset's scope

**Questions this system answers well:**
- Recognition counts, distributions, and statistics
- Finding recognition messages by theme, keyword, or behavior
- Role skill profiles and succession planning
- Organizational culture and behavioral patterns
- Collaboration and nomination network analysis (including Neo4j graph queries)

Please try rephrasing or asking one of the question types above."""


# ── Prompt-injection guard (deterministic, runs before any model call) ────────

_INJECTION_PATTERNS = re.compile(
    r"\b(ignore|disregard|forget|override)\b[^.?!]{0,30}\b(previous|prior|above|earlier|all|your|the)\b[^.?!]{0,20}"
    r"\b(instructions?|prompts?|rules|guidelines)\b"
    r"|\b(print|show|reveal|repeat|output|display|tell me|give me|what is|what are)\b[^.?!]{0,25}"
    r"\b(system prompt|instructions|hidden prompt|initial prompt|your prompt|your rules)\b"
    r"|\bsystem prompt\b|\bdeveloper mode\b|\bjailbreak\b|\bdan mode\b"
    r"|\byou are now\b|\bpretend (to be|you are)\b|\bfrom now on,? you\b"
    r"|^\s*(system|assistant|developer)\s*:",
    re.IGNORECASE,
)

INJECTION_REFUSAL = """I can't do that. I only answer questions about the recognition dataset, and I don't change or reveal my instructions.

**Questions this system answers well:**
- Recognition counts, distributions, and statistics
- Finding recognition messages by theme, keyword, or behavior
- Role skill profiles and succession planning
- Organizational culture and behavioral patterns
- Collaboration and nomination network analysis"""


def is_prompt_injection(question: str) -> bool:
    return bool(_INJECTION_PATTERNS.search(question or ""))


def sanitize_tool_args(tool_calls: list, question: str) -> list:
    """
    Defence in depth: a router model under a prompt-injection attack can copy its own system prompt into a
    tool argument. Any string argument that quotes the router prompt, or is far longer than the user's
    question, is replaced with the user's question.
    """
    markers = ("You are an expert routing agent", "CHAIN OF THOUGHT", "ROLE: You are a precise intent classifier")
    for call in tool_calls or []:
        args = call.get("args") or {}
        for key, value in list(args.items()):
            if isinstance(value, str) and (any(m.lower() in value.lower() for m in markers)
                                           or len(value) > 3 * len(question) + 200):
                args[key] = question
    return tool_calls


def is_cannot_answer(tool_output) -> bool:
    return str(tool_output).lstrip().startswith("CANNOT_ANSWER")


def cannot_answer_message(tool_output: str) -> str:
    """Deterministic refusal built from the tools' CANNOT_ANSWER reasons (no LLM involved)."""
    reasons = [part.strip().removeprefix("CANNOT_ANSWER:").strip()
               for part in str(tool_output).split("\n\n") if part.strip()]
    reason_text = "\n\n".join(r for r in reasons if r)
    return (
        "I can't answer that from the recognition dataset.\n\n"
        f"**Why:** {reason_text}\n\n"
        "**Questions this system answers well:**\n"
        "- Recognition counts, distributions, and statistics\n"
        "- Finding recognition messages by theme, keyword, or behavior\n"
        "- Role skill profiles and succession planning\n"
        "- Organizational culture and behavioral patterns\n"
        "- Collaboration and nomination network analysis"
    )


def should_refuse(score: float) -> bool:
    return score < HARD_REFUSAL_THRESHOLD


def should_warn(score: float) -> bool:
    return HARD_REFUSAL_THRESHOLD <= score < SOFT_WARNING_THRESHOLD


def refusal_with_flags(flagged_claims: list | None) -> str:
    """The refusal, plus what the faithfulness check could not find in the data (warn and refuse)."""
    if not flagged_claims:
        return REFUSAL_MESSAGE
    shown = "\n".join(f"- {c}" for c in flagged_claims[:5])
    more  = f"\n- …and {len(flagged_claims) - 5} more" if len(flagged_claims) > 5 else ""
    return (
        "I'm not able to provide a confident answer: an automatic check could not find these claims "
        f"in the retrieved data, so I'm not showing the answer.\n\n**Flagged claims:**\n{shown}{more}\n\n"
        + REFUSAL_MESSAGE.split("\n\n", 1)[1]
    )


def format_final_answer(answer: str, hallucination_score: float, route_used: str = None,
                        flagged_claims: list | None = None) -> str:
    """Apply refusal gate, route badge, and warning to final answer."""
    if should_refuse(hallucination_score):
        return refusal_with_flags(flagged_claims)

    badge_map = {
        "pandas_query":        "📊 Analytical",
        "semantic_search":     "🔍 Semantic search",
        "profile_and_replace": "👤 Profile",
        "llm_synthesis":       "💡 Cultural insight",
        "chart_generator":     "📈 Chart",
        "graph_query":         "🕸️ Graph",
    }
    route_badge = ""
    if route_used:
        label = badge_map.get(route_used, route_used)
        route_badge = f"*Route: {label}*\n\n"

    if should_warn(hallucination_score):
        warning = (
            f"\n\n---\n⚠️ *Low confidence (grounding score: {hallucination_score}). "
            f"Some parts of this answer may not be fully supported by retrieved data. "
            f"Please verify against the source.*"
        )
        return route_badge + answer + warning

    return route_badge + answer
