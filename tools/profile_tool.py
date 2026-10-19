"""
tools/profile_tool.py

Person/role skill profiles + succession planning.
Uses CoT prompt from prompt_engine + persistent JSON cache.
"""

import os
import re
import sys
import json
import hashlib
import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm import get_llm, model_name, parse_json_reply
from agent.prompt_engine import PROFILE_EXTRACTION_PROMPT

DATA_PATH          = os.getenv("DATA_PATH",          "./data/annotated_results.csv")
PROFILE_CACHE_PATH = os.getenv("PROFILE_CACHE_PATH", "./data/profile_cache.json")

_profile_cache: dict = {}
_cache_loaded:  bool = False


def _load_cache():
    global _profile_cache, _cache_loaded
    if _cache_loaded:
        return
    if os.path.exists(PROFILE_CACHE_PATH):
        try:
            with open(PROFILE_CACHE_PATH, "r") as f:
                _profile_cache = json.load(f)
        except Exception:
            _profile_cache = {}
    _cache_loaded = True


def _save_cache():
    os.makedirs(os.path.dirname(os.path.abspath(PROFILE_CACHE_PATH)), exist_ok=True)
    with open(PROFILE_CACHE_PATH, "w") as f:
        json.dump(_profile_cache, f, indent=2)


@lru_cache(maxsize=1)
def _load_df():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    return df


def _get_messages_for_person(name: str) -> list[str]:
    df = _load_df()
    return [
        msg for msg in df["message"].dropna()
        if re.search(rf'\b{re.escape(name.lower())}\b', msg.lower())
    ]


def _get_messages_for_role(role: str) -> list[str]:
    df = _load_df()
    return df[
        df["recipient_title"].str.lower().str.contains(role.lower(), na=False)
    ]["message"].dropna().tolist()


_QUESTION_WORDS = {
    "who", "what", "when", "where", "how", "replace", "replaced", "replacement",
    "left", "leaves", "similar", "skills", "best", "can", "could", "would",
    "the", "a", "an", "is", "are", "has", "have", "if", "and", "or", "for",
    "tell", "me", "about", "find", "show", "give", "which", "whose", "list",
}


@lru_cache(maxsize=1)
def _title_words() -> frozenset:
    """Every word used in a job title — 'Senior', 'Director', 'Engineering' are roles, not people."""
    df = _load_df()
    titles = pd.concat([df["recipient_title"], df["nominator_title"]]).dropna()
    return frozenset(w for t in titles for w in re.findall(r"[a-z]+", t.lower()))


def _candidate_names(query: str) -> list[str]:
    words = [re.sub(r"(['’]s)$", "", w.strip("?,.!:;\"()")) for w in query.split()]
    return [w for w in words
            if len(w) > 2 and w[0].isupper()
            and w.lower() not in _QUESTION_WORDS and w.lower() not in _title_words()]


def _is_replacement(query: str) -> bool:
    q = query.lower()
    return (any(kw in q for kw in ["replace", "replacement", "succeed", "successor", "left", "leaves",
                                   "leave", "quit", "resign", "backup", "back up", "similar to", "cover for"])
            or bool(re.search(r"\b(someone|somebody|people|person|anyone)\s+like\b", q)))


def _as_text_list(value) -> list[str]:
    """Flatten whatever shape the model returned (str, nested lists, {skill, evidence} dicts) into strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        parts = [str(v) for v in value.values() if v not in (None, "", [])]
        return [f"{parts[0]} ({', '.join(parts[1:])})" if len(parts) > 1 else (parts[0] if parts else "")]
    if isinstance(value, (list, tuple)):
        return [item for v in value for item in _as_text_list(v)]
    return [str(value)]


def _cache_key(name: str, messages: list[str]) -> str:
    """Profiles depend on the messages and on the model that wrote them — key on both."""
    digest = hashlib.sha256("\n".join(messages).encode("utf-8")).hexdigest()[:12]
    return f"{name}|{model_name('cheap')}|{digest}"


def _build_profile(name: str, messages: list[str]) -> dict:
    _load_cache()
    key = _cache_key(name, messages)
    if key in _profile_cache:
        return _profile_cache[key]

    llm           = get_llm("cheap", json_mode=True)
    messages_text = "\n\n---\n\n".join(messages[:15])

    prompt = PROFILE_EXTRACTION_PROMPT.format() + f'\n\nRecognition messages mentioning "{name}":\n\n{messages_text}'
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        profile = parse_json_reply(response.text)
    except Exception:
        profile = {
            "name":              name,
            "message_count":     len(messages),
            "top_skills":        ["Could not parse profile"],
            "categories":        [],
            "impact_areas":      [],
            "seniority_signals": [],
            "key_themes":        f"Found {len(messages)} messages mentioning {name} but could not parse profile."
        }

    if not isinstance(profile, dict):
        profile = {"name": name, "message_count": len(messages), "key_themes": str(profile)}

    _profile_cache[key] = profile
    _save_cache()
    return profile


def _similar_roles(role_name: str, df: pd.DataFrame) -> str:
    """Role-based substitution: other recipient titles whose recognitions read most like this role's."""
    from tools.rag_tool import _get_collection, _get_embedder
    import numpy as np
    embedder   = _get_embedder()
    collection = _get_collection()
    role_df    = df[df["recipient_title"].str.lower().str.contains(role_name.lower(), na=False)]
    profile_text = " | ".join(
        f"{row['category_name']}: {row['message'][:100]}"
        for _, row in role_df.iterrows()
    )[:2000]
    profile_vec = embedder.encode([profile_text])[0].tolist()
    results = collection.query(
        query_embeddings=[profile_vec],
        n_results=50,
        include=["metadatas", "distances"],
    )
    role_scores: dict = {}
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        r = meta.get("recipient_title", "")
        if r and role_name.lower() not in r.lower():
            role_scores.setdefault(r, []).append(1 - dist)
    ranked = sorted(
        [(r, round(float(np.mean(s)) * 100, 1)) for r, s in role_scores.items()],
        key=lambda x: x[1], reverse=True
    )[:10]
    lines = [
        f"=== Roles Most Similar to '{role_name}' ===",
        f"Based on {len(role_df)} recognition records\n",
    ]
    for i, (r, score) in enumerate(ranked, 1):
        lines.append(f"  {i}. {r} — {score}% similarity")
    return "\n".join(lines)


@tool
def profile_and_replace(query: str) -> str:
    """
    Use this tool when the user asks about a specific person's contributions,
    skills, or impact — OR asks who could replace someone, succeed them,
    or fill a similar role.

    Also handles role-based substitution queries like "who can replace a Director of Engineering".

    Examples:
    - What has Sarah contributed?
    - Who can replace Matt?
    - If Emma left, who would be the best replacement?
    - Who could replace a Senior Director?
    - What are Kaleigh's key strengths?
    """
    df = _load_df()

    candidate_names = _candidate_names(query)
    is_replacement  = _is_replacement(query)

    # Role-based substitution (e.g. "who could replace a Director of Engineering")
    role_patterns = [
        r"replace\s+(?:a|an|the)?\s*([A-Za-z ,&/]+?)(?:\?|$|\.)",
        r"if\s+(?:a|an|the)?\s*([A-Za-z ,&/]+?)\s+(?:left|quit|resigned)",
        r"similar to\s+(?:a|an|the)?\s*([A-Za-z ,&/]+?)(?:\?|$|\.)",
        r"backup\s+(?:for\s+)?(?:a|an|the)?\s*([A-Za-z ,&/]+?)(?:\?|$|\.)",
    ]

    if is_replacement and not candidate_names:
        for pattern in role_patterns:
            m = re.search(pattern, query, re.IGNORECASE)
            if m and _get_messages_for_role(m.group(1).strip()):
                return _similar_roles(m.group(1).strip(), df)

    # A bare role title (the router sometimes shortens "Who could replace a Senior Director?" to
    # "Senior Director") — for this tool the useful answer is still the similar-roles analysis
    bare_role = re.sub(r"^(?:a|an|the)\s+", "", query.strip(" ?.!"), flags=re.IGNORECASE)
    if not candidate_names and bare_role and _get_messages_for_role(bare_role):
        return _similar_roles(bare_role, df)

    if not candidate_names:
        return "CANNOT_ANSWER: Could not identify a person's name or role in the query. Please include a first name (e.g. 'What has Sarah contributed?') or a role title (e.g. 'Who can replace a Director of Engineering?')."

    target_name = candidate_names[0]
    messages    = _get_messages_for_person(target_name)

    if not messages:
        # Show who IS in the dataset so the user can try a valid name
        df2 = _load_df()
        import re as _re
        all_names = set()
        pat = _re.compile(r'(?:^|\n)([A-Z][a-z]{2,12})(?:\s*,|\s+and\s+[A-Z])', _re.MULTILINE)
        for msg in df2["message"].dropna():
            for m in pat.findall(msg):
                all_names.add(m.strip())
        non = {"Team","Hi","Dear","Thanks","Thank","Hello","Hey","All","Folks","Guys",
               "Gents","Girls","Ladies","Lads","Monday","Tuesday","Wednesday","Thursday",
               "Friday","January","February","March","April","May","June","July",
               "August","September","October","November","December","Well"}
        counts = {n: sum(1 for msg in df2["message"].dropna()
                         if _re.search(rf'\b{n.lower()}\b', msg.lower()))
                  for n in all_names - non}
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:15]
        names_str = ", ".join(f"{n} ({c})" for n, c in top)
        return (
            f"CANNOT_ANSWER: No recognition messages found mentioning '{target_name}'. "
            f"They may not appear in this dataset.\n\n"
            f"People with the most recognition data: {names_str}"
        )

    MIN_MESSAGES = 3
    if len(messages) < MIN_MESSAGES:
        return (
            f"CANNOT_ANSWER: Only {len(messages)} recognition message(s) mention '{target_name}' — "
            f"not enough data to build a reliable profile (minimum {MIN_MESSAGES} required). "
            f"Try someone with more recognition history."
        )

    target_profile = _build_profile(target_name, messages)

    if not is_replacement:
        result  = f"Profile for {target_name} (based on {target_profile.get('message_count', len(messages))} recognition messages):\n\n"
        result += f"Key themes: {target_profile.get('key_themes', 'N/A')}\n\n"
        result += f"Top skills: {', '.join(_as_text_list(target_profile.get('top_skills')))}\n"
        result += f"Work categories: {', '.join(_as_text_list(target_profile.get('categories')))}\n"
        result += f"Impact areas: {', '.join(_as_text_list(target_profile.get('impact_areas')))}\n"
        result += f"Seniority signals: {', '.join(_as_text_list(target_profile.get('seniority_signals')))}\n"
        return result

    # Build candidate profiles
    all_names      = set()
    direct_address = re.compile(r'(?:^|\n)([A-Z][a-z]{2,12})(?:\s*,|\s+and\s+[A-Z])', re.MULTILINE)
    for msg in df["message"].dropna():
        for match in direct_address.findall(msg):
            all_names.add(match.strip())

    non_names = {"Team", "Hi", "Dear", "Thanks", "Thank", "Hello", "Hey", "All", "Folks",
                 "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                 "January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"}
    all_names = {n for n in all_names if n not in non_names and n != target_name}

    counts     = [(n, len(_get_messages_for_person(n))) for n in all_names]
    top_cands  = [n for n, c in sorted(counts, key=lambda x: x[1], reverse=True)[:12] if c >= 2]
    cand_profs = [_build_profile(n, _get_messages_for_person(n)) for n in top_cands]

    llm          = get_llm("cheap")
    target_text  = json.dumps(target_profile, indent=2)
    cands_text   = json.dumps(cand_profs, indent=2)

    rank_prompt = (
        f"You are a senior HR analyst. A person named {target_name} may be leaving.\n\n"
        f"CHAIN OF THOUGHT:\n"
        f"Step 1 — Review {target_name}'s profile. List their core skills and impact areas.\n"
        f"Step 2 — For each candidate, identify skill overlap with {target_name}.\n"
        f"Step 3 — Rank top 3 by skill coverage. Identify specific gaps.\n"
        f"Step 4 — Assign fit scores only after completing Steps 1-3.\n\n"
        f"{target_name}'s profile:\n{target_text}\n\n"
        f"Candidates:\n{cands_text}\n\n"
        f"Present top 3 replacement candidates. For each: name, specific skill overlaps, gaps, fit score /10.\n"
        f"Only reference skills that appear in the profiles — do not invent attributes."
    )

    response = llm.invoke([HumanMessage(content=rank_prompt)])
    result   = f"Replacement analysis for {target_name}:\n\n"
    result  += f"{target_name}'s summary: {target_profile.get('key_themes', '')}\n\n"
    result  += "--- Top Replacement Candidates ---\n\n"
    result  += response.text
    return result
