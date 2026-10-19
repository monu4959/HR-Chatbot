"""
agent/model_router.py

Decides how much model each answer gets, from signals the pipeline already has (which tool ran,
what it returned, whether it's a follow-up) — no extra model call.

  template  No model call: the tool output is already the answer, so it is only reformatted
            (headings, bullets, quotes — never new facts). No grader either: a verbatim tool
            output is faithful by construction.
  light     The "light" model writes the answer without the dynamic-prompt call. The grader still
            checks it, and a failing grade escalates the question to the strong path.
  strong    The full pipeline: dynamic prompt + strong model + grader.

Why these rules (from the eval baseline and the LangSmith step timings):
  - Rule-based pandas results are exact; making a model retype them only adds risk — the 3B baseline
    garbled correct numbers in 7 cases — and ~9 s per question.
  - Semantic search spent ~39 s per question retyping its result list; graph results are short lists.
  - Profiles and culture insights are open-ended and spend their time inside the tool, so they stay strong.
  - On an 8 GB GPU a second model can't stay loaded next to qwen2.5:7b (reloading 7B costs ~11 s),
    so by default the light model IS the strong model, minus the extra call. On a provider where models
    don't compete for one GPU (e.g. Claude Haiku), set <PROVIDER>_MODEL_LIGHT to a cheaper model.

MODEL_ROUTER=on|off (default on). Off = every answer takes the strong path.
"""

import os
import re

from tools.pandas_tool import GENERATED_CODE_PREFIX

TEMPLATE_ROUTES   = {"pandas_query", "graph_query", "semantic_search"}
LIGHT_ROUTES      = {"pandas_query"}   # generated-code pandas results
TEMPLATE_MAX_CHARS = 12000
LIGHT_MAX_CHARS    = 4000
SEARCH_PREVIEW_CHARS = 400  # tools/rag_tool.py shows the first 400 characters of each message


def router_enabled() -> bool:
    return os.getenv("MODEL_ROUTER", "on").strip().lower() not in ("off", "0", "false", "no")


def pick_tier(route: str | None, tool_output: str, has_history: bool) -> str:
    if not router_enabled() or not route or not tool_output:
        return "strong"
    generated = tool_output.startswith(GENERATED_CODE_PREFIX)
    if route in TEMPLATE_ROUTES and not generated and len(tool_output) <= TEMPLATE_MAX_CHARS:
        return "template"
    if route in LIGHT_ROUTES and generated and not has_history and len(tool_output) <= LIGHT_MAX_CHARS:
        return "light"
    return "strong"


# ── Template rendering (formatting only) ──────────────────────────────────────

def _render_search(tool_output: str) -> str:
    head = re.search(r"Found (\d+) relevant recognitions for '(.*?)' \(CRAG quality: (\w+)", tool_output)
    lines = []
    if head:
        n, query, quality = head.groups()
        lines.append(f"Found **{n}** recognitions matching *{query}* (retrieval quality: {quality}).")
        if quality == "Low":
            lines.append("_These matches are weak — the dataset may not cover this topic well._")
    blocks = re.findall(r"RESULT \d+ \[relevance: [\d.]+\]:\n(.*?)(?=\nRESULT \d+ |\nNote:|\nINSTRUCTION:|\Z)",
                        tool_output, re.DOTALL)
    for i, block in enumerate(blocks, 1):
        f = dict(re.findall(r"^\s*(Recipient|Nominator|Category|Message): (.*)$", block, re.MULTILINE))
        message = f.get("Message", "").strip()
        if len(message) >= SEARCH_PREVIEW_CHARS:  # the search tool cut it off
            message += "…"
        lines.append(f"\n**{i}. {f.get('Recipient', 'N/A')}** — nominated by {f.get('Nominator', 'N/A')} · "
                     f"{f.get('Category', 'N/A')}\n> {message}")
    note = re.search(r"Note: (.*)", tool_output)
    if note:
        lines.append(f"\n_{note.group(1).strip()}_")
    return "\n".join(lines)


def _render_lines(tool_output: str) -> str:
    out = []
    for line in tool_output.splitlines():
        heading = re.match(r"^=== (.*) ===$", line.strip())
        if heading:
            out.append(f"\n**{heading.group(1)}**")
        elif re.match(r"^\s+- ", line):
            out.append("- " + line.strip()[2:])
        elif re.match(r"^\s{2,}\S", line):
            out.append("- " + line.strip())
        else:
            out.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def render_template(route: str, tool_output: str) -> str:
    if route == "semantic_search" and tool_output.startswith("SEARCH COMPLETE"):
        return _render_search(tool_output)
    return _render_lines(tool_output)
