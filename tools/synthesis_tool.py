"""
tools/synthesis_tool.py

Cultural insight synthesis over the recognition dataset.
Stratified sampling + dynamic category keyword matching + multi-call for broad questions.
"""

import os
import sys
import re
import random
import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm import get_llm, max_prompt_chars
from agent.prompt_engine import SYNTHESIS_TOOL_PROMPT

DATA_PATH = os.getenv("DATA_PATH", "./data/annotated_results.csv")


def _load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    return df


def _get_all_keywords(df: pd.DataFrame) -> list[str]:
    """Build keyword list dynamically from actual CSV category/subcategory values."""
    keywords = set()
    for val in list(df["category_name"].dropna().unique()) + list(df["subcategory_name"].dropna().unique()):
        for word in val.lower().replace("&", "").replace("-", " ").split():
            if len(word) > 3:
                keywords.add(word)
    return sorted(keywords)


def _match_keyword(q_lower: str, keywords: list[str]) -> str | None:
    """Pick the most specific keyword in the question: longest first, then earliest — deterministic across runs."""
    hits = [kw for kw in keywords if kw in q_lower]
    return min(hits, key=lambda kw: (-len(kw), q_lower.index(kw))) if hits else None


def _clean_cannot_answer(text: str) -> str:
    """If the model refused, keep only the CANNOT_ANSWER sentence — not the analysis it kept writing after it."""
    i = text.find("CANNOT_ANSWER")
    if i < 0:
        return text
    reason = text[i:].split("\n")[0]
    m = re.match(r"(CANNOT_ANSWER:.*?[.!?])(\s|$)", reason)
    return m.group(1) if m else reason.strip()


def _sample_messages(df: pd.DataFrame, n: int = 300) -> str:
    """Stratified sample across categories."""
    sampled = df.groupby("category_name", group_keys=False).apply(
        lambda x: x.sample(
            min(len(x), max(5, int(n * len(x) / len(df)))),
            random_state=42
        )
    )
    lines = []
    for _, row in sampled.iterrows():
        lines.append(
            f"[{row['category_name']} > {row['subcategory_name']}] "
            f"From: {row['nominator_title']} To: {row['recipient_title']}\n"
            f"{str(row['message'])[:300]}"
        )
    return "\n\n---\n\n".join(lines)


def _fit_budget(lines: list[str], budget: int | None) -> list[str]:
    """Deterministically sample lines down to a character budget (only applies to small-context models)."""
    if not budget or sum(len(l) for l in lines) <= budget:
        return lines
    picked, used = [], 0
    for i in random.Random(42).sample(range(len(lines)), len(lines)):
        if used + len(lines[i]) > budget:
            continue
        picked.append(i)
        used += len(lines[i])
    return [lines[i] for i in sorted(picked)]


def _messages_for_category(df: pd.DataFrame, category: str) -> str:
    """All messages matching a category keyword, sampled down only if they exceed the model's context budget."""
    filtered = df[
        df["category_name"].str.lower().str.contains(category.lower(), na=False) |
        df["subcategory_name"].str.lower().str.contains(category.lower(), na=False)
    ]
    lines = []
    for _, row in filtered.iterrows():
        lines.append(
            f"[{row['subcategory_name']}] {row['nominator_title']} → {row['recipient_title']}:\n"
            f"{str(row['message'])[:300]}"
        )
    return "\n\n---\n\n".join(_fit_budget(lines, max_prompt_chars()))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().replace("&", " and ")).strip()


def _named_groups(df: pd.DataFrame, question: str) -> list[tuple[str, str]]:
    """Every category / subcategory named in full in the question ('&' and 'and' both match), longest first."""
    nq, found = _norm(question), []
    labels = [(c, "category_name") for c in df["category_name"].dropna().unique()] + \
             [(s, "subcategory_name") for s in df["subcategory_name"].dropna().unique()]
    for label, col in sorted(labels, key=lambda x: -len(x[0])):
        n = _norm(label)
        if n in nq:
            found.append((label, col))
            nq = nq.replace(n, " ")  # a subcategory name is not also matched inside a longer one
    return found


def _comparison_prompt(df: pd.DataFrame, question: str, groups: list[tuple[str, str]]) -> str:
    budget = max_prompt_chars()
    per_group = budget // len(groups) if budget else None
    sections = []
    for label, col in groups:
        rows = df[df[col] == label]
        lines = [f"{r['nominator_title']} → {r['recipient_title']} [{r['subcategory_name']}]:\n{str(r['message'])[:300]}"
                 for _, r in rows.iterrows()]
        sections.append(f"### {label} ({len(rows)} recognitions)\n\n" + "\n\n---\n\n".join(_fit_budget(lines, per_group)))
    return (f"{SYNTHESIS_TOOL_PROMPT}\n\nYou are comparing {len(groups)} groups of recognitions from this dataset. "
            f"Both are in scope — compare what is recognized in each, citing patterns from the messages.\n\n"
            + "\n\n".join(sections) + f"\n\nQuestion: {question}")


def _multi_call_synthesis(df: pd.DataFrame, question: str, llm) -> str:
    """Per-category synthesis then combine for broad questions."""
    summaries = []
    for cat in df["category_name"].unique():
        cat_msgs = _messages_for_category(df, cat)
        if not cat_msgs:
            continue
        resp = llm.invoke([HumanMessage(content=(
            f"You are analyzing the '{cat}' recognition category.\n\n"
            f"Data:\n{cat_msgs}\n\n"
            f"Summarize in 2-3 sentences: what themes, behaviors, and patterns appear? "
            f"Only state what the data shows."
        ))])
        summaries.append(f"**{cat}:**\n{resp.text}")

    combined = "\n\n".join(summaries)
    resp = llm.invoke([HumanMessage(content=(
        f"{SYNTHESIS_TOOL_PROMPT}\n\n"
        f"Per-category insights:\n{combined}\n\n"
        f"Question: {question}\n\n"
        f"Using the category insights above, answer the question comprehensively."
    ))])
    return resp.text


@tool
def llm_synthesis(question: str) -> str:
    """
    Use this tool for open-ended insight questions that require understanding
    patterns, themes, culture, or making comparisons across the dataset.

    Examples:
    - What does this organization value most?
    - What themes appear in leadership recognitions?
    - How do senior leaders write differently from peers?
    - What cultural patterns exist in this dataset?
    - Compare how Engineering vs Marketing recognitions differ
    - What behaviors are most celebrated in this company?
    """
    llm = get_llm("cheap")
    df  = _load_df()
    q_lower = question.lower()

    # Comparison of named categories / subcategories: give the model each group's messages side by side.
    # (Matching a single keyword showed it only one group, and it then called the comparison out of scope.)
    groups = _named_groups(df, question)
    if len(groups) >= 2:
        return _clean_cannot_answer(llm.invoke([HumanMessage(content=_comparison_prompt(df, question, groups))]).text)

    # Dynamic keyword matching from actual CSV values
    target_category = _match_keyword(q_lower, _get_all_keywords(df))

    broad_triggers = ["overall", "organization", "company", "everything",
                      "all categories", "whole", "general", "across"]
    is_broad = any(t in q_lower for t in broad_triggers) or target_category is None

    if is_broad:
        return _clean_cannot_answer(_multi_call_synthesis(df, question, llm))

    messages_text = _messages_for_category(df, target_category)
    if not messages_text:
        messages_text = _sample_messages(df, n=300)

    prompt = (
        f"{SYNTHESIS_TOOL_PROMPT}\n\n"
        f"Recognition data:\n{messages_text}\n\n"
        f"Question: {question}"
    )
    return _clean_cannot_answer(llm.invoke([HumanMessage(content=prompt)]).text)
