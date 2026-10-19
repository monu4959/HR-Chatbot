"""
tools/pandas_tool.py

Analytical queries over the recognition CSV.
Handles: counts, stats, rankings, distributions, names, people,
         awards, message length, cross-filters, and LLM fallback.
"""

import os
import re
import sys
import builtins
import pandas as pd
from collections import Counter
from langchain_core.tools import tool
from langsmith import traceable
from langchain_core.messages import HumanMessage
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm import get_llm
from agent.prompt_engine import PANDAS_FALLBACK_PROMPT

DATA_PATH = os.getenv("DATA_PATH", "./data/annotated_results.csv")

_SKIP_WORDS = {
    "You","Your","Team","The","This","Thanks","Thank","Hello","Hey","Dear","Hi",
    "All","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday",
    "January","February","March","April","May","June","July","August","September",
    "October","November","December","Well","With","What","That","They","Their",
    "From","Over","When","Where","Will","Have","Been","Each","Just","Also","Very",
    "Here","There","And","But","For","Not","Can","Our","Has","Had","Its","New",
    "Year","Christmas","Wishing","Looking","Cheers","Customer","Please","Really",
    "Great","Good","Work","Time","Always","Never","Every","Through","Across",
    "During","These","Those","Would","Could","Should","Happy","Many","Much",
    "Some","Such","Both","Into","Only","Made","Make","Take","Than","More","Most",
    "Way","Day","Get","Got","Let","Put","See","Now","Any","Big","Key","End",
    "Top","Best","High","True","Full","Next","Last","First","Second","Third",
    "Done","Sure","Hard","Long","Know","Show","Give","Keep","Call","Part",
    "Role","Back","Move","Hold","Come","Send","Want","Need","Feel","Find",
    "Join","Help","Lead","Drive","Build","Share","Bring","Stand","Start",
    "Continue","Support","Deliver","Ensure","Provide","Achieve","Enable",
    "Improve","Create","Develop","Impact","Focus","Manage","Across","Within"
}


@lru_cache(maxsize=1)
def load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df["msg_len"] = df["message"].str.len()
    return df


def _norm(text: str) -> str:
    """Lowercase, '&' -> 'and', collapse whitespace — so 'Creative and Brand Impact' matches 'Creative & Brand Impact'."""
    return re.sub(r"\s+", " ", str(text).lower().replace("&", " and ")).strip()


# The dataset has no date column, so any time-based question must be refused rather than answered with the total
_TIME_PATTERN = re.compile(
    r"\b(19|20)\d{2}\b"
    r"|\b(january|february|march|april|june|july|august|september|october|november|december)\b"
    r"|\b(last|this|next|previous|past) (year|month|week|quarter)\b"
    r"|\bper (year|month|week|quarter|day|hour)\b|\bover time\b|\btrend(s|ing)?\b|\bdates?\b"
    r"|\b(weekends?|weekdays?|mondays?|tuesdays?|wednesdays?|thursdays?|fridays?|saturdays?|sundays?)\b"
    r"|\b(mornings?|afternoons?|evenings?|nights?|overnight|hours?|time of (day|year))\b"
    r"|\b(daily|weekly|monthly|quarterly|yearly|annually|seasons?|seasonal|holidays?)\b|\bq[1-4]\b"
    r"|\bwhen (was|were|did|do|does|is|are)\b"
)

# "How many recognitions mention the word 'customer'?" — a text search, answerable even if the word is "weekend"
_MENTION_TERM = re.compile(
    r"\b(?:mention|mentions|mentioning|mentioned|contain|contains|containing|include|includes|including|say|says|use|uses)\b"
    r"(?:\s+the)?(?:\s+(?:word|term|phrase|keyword))?\s+[\"'‘“]([^\"'’”]{2,40})[\"'’”]"
    r"|\bthe (?:word|term|phrase|keyword) ([a-z][a-z\-]{1,30})\b"
)
_COUNT_INTENT = re.compile(r"\b(how many|how often|count|number of|percentage|percent|share|fraction|proportion)\b")

# Superlatives over one dimension: "Which subcategory has the most recognitions?" / "...the fewest?"
_SUPERLATIVE_DIMENSIONS = [   # order matters: "subcategor" before "categor"
    ("subcategor", "subcategory_name", "subcategory"),
    ("categor",    "category_name",    "category"),
    ("award",      "award_title",      "award title"),
    ("nominator",  "nominator_title",  "nominator job title"),
    ("recipient",  "recipient_title",  "recipient job title"),
]
_MOST_WORDS   = re.compile(r"\b(most|largest|highest|biggest|greatest|top|maximum)\b")
_FEWEST_WORDS = re.compile(r"\b(fewest|least|smallest|lowest|minimum|bottom)\b")

# Role groups (job titles grouped into departments, as in tools/graph_tool.py)
_GROUP_WORDS   = re.compile(r"\b(role ?groups?|groups? of (job )?(roles|titles)|departments?|teams?|business functions?)\b")
_GIVE_WORDS    = re.compile(r"\b(give|gives|giving|gave|given out|nominate|nominates|nominating|send|sends|sent)\b")
_RECEIVE_WORDS = re.compile(r"\b(receive|receives|receiving|received|get|gets|recogni[sz]ed)\b")

_COUNT_WORDS = ["how many", "count", "number of", "total", "awards in", "recognitions in",
                "percentage", "share", "%", "proportion"]

# Words that mean the question is narrower than "how many rows are there"
_NOT_A_TOTAL = ["categor", "recipient", "nominator", "title", "role", "job", "unique", "distinct",
                "each", "per ", " by ", "average", "mention", "name", "people", "person", "message"]

_MOST_RECOGNIZED = re.compile(
    r"(most|top)\s+recogni[sz]ed|recogni[sz]ed\s+(the\s+)?most|receives?\s+(the\s+)?most"
    r"|most\s+(awards|recognitions?)\s+received|gets\s+(the\s+)?most\s+(awards|recognitions?)"
)
_MOST_NOMINATING = re.compile(
    r"nominates?\s+(the\s+)?most|gives?\s+(the\s+)?most|most\s+nominations|top\s+nominator"
    r"|who\s+nominates|most\s+active\s+nominator|most\s+nominator"
)


_HOW_MANY_OF = re.compile(r"\b(how many|number of)\s+(unique\s+|distinct\s+|different\s+)?subcategor")
_COMPARISON  = re.compile(r"\b(which|per|each|most|largest|highest|fewest|least|smallest|by)\b")


def _names_something(question: str) -> bool:
    """True if a capitalized word (team, person, title) appears after the sentence-initial capital."""
    words = re.findall(r"\b[A-Z][a-zA-Z]+\b", question.strip()[1:])
    return any(w != "I" for w in words)


def _extract_names_from_messages(df: pd.DataFrame) -> Counter:
    all_names = []
    for msg in df["message"].dropna():
        found = re.findall(r'\b([A-Z][a-z]{2,14})\b', str(msg))
        all_names.extend([n for n in found if n not in _SKIP_WORDS])
    return Counter(all_names)


# ── LLM fallback (sandboxed) ──────────────────────────────────────────────────

_SAFE_BUILTINS = {
    name: getattr(builtins, name) for name in [
        "len", "sum", "min", "max", "round", "sorted", "list", "dict", "set", "tuple", "str",
        "int", "float", "abs", "enumerate", "zip", "range", "any", "all", "map", "filter",
        "isinstance", "bool", "reversed",
    ]
}
_BLOCKED_TOKENS = ["import", "__", "open(", "exec(", "eval(", "compile(", "globals", "locals",
                   "getattr", "setattr", "delattr", "os.", "sys.", "subprocess", "to_csv", "to_pickle",
                   "read_csv", "read_", "input("]

# Prefix on results computed by model-written code, so they can be told apart from the exact rule-based answers
GENERATED_CODE_PREFIX = "Result (computed with generated pandas code): "


def _extract_code(reply: str) -> str:
    """Pull code out of a model reply, preferring a fenced block if the model added prose around it."""
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", reply, re.DOTALL)
    return (fenced.group(1) if fenced else reply).strip()


def _run_generated_code(code: str, df: pd.DataFrame) -> tuple[bool, str]:
    """Run model-written pandas code with no imports, no file access and a small builtin whitelist."""
    bad = [t for t in _BLOCKED_TOKENS if t in code]
    if bad:
        return False, f"blocked token(s) in generated code: {bad}"
    local_vars = {"df": df.copy(), "pd": pd, "re": re, "Counter": Counter}
    try:
        exec(code, {"__builtins__": _SAFE_BUILTINS}, local_vars)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    if "result" not in local_vars:
        return False, "code did not set `result`"
    return True, str(local_vars["result"])


@traceable(name="pandas_llm_fallback", process_inputs=lambda inputs: {"question": inputs["question"]})
def _llm_pandas_fallback(question: str, df: pd.DataFrame) -> str:
    llm     = get_llm("cheap")
    cols    = df.columns.tolist()
    samples = {col: df[col].dropna().unique()[:5].tolist() for col in cols if col != "message"}
    prompt  = PANDAS_FALLBACK_PROMPT.format(columns=cols, sample_values=samples)
    prompt += f"\n\nQuestion: {question}"

    messages = [HumanMessage(content=prompt)]
    error = ""
    for _ in range(2):  # one retry, showing the model its own error
        response = llm.invoke(messages)
        code     = _extract_code(response.text)
        ok, out  = _run_generated_code(code, df)
        if ok:
            # Marked so the model router never shows a generated-code result verbatim (see agent/model_router.py)
            return out if out.startswith("CANNOT_ANSWER") else GENERATED_CODE_PREFIX + out
        error = out
        messages += [response, HumanMessage(content=f"That code failed: {out}\nReturn corrected code only.")]

    return (
        f"CANNOT_ANSWER: Could not compute an answer for this question ({error}). "
        f"Dataset columns: {', '.join(cols)}."
    )


def _count_mentions(df: pd.DataFrame, q: str) -> str | None:
    """How many messages contain a quoted word, or 'the word X' (case-insensitive substring match)."""
    m = _MENTION_TERM.search(q)
    if not m or not _COUNT_INTENT.search(q):
        return None
    term = (m.group(1) or m.group(2)).strip()
    n    = int(df["message"].str.contains(re.escape(term), case=False, na=False).sum())
    pct  = round(n / len(df) * 100, 1)
    return (f"{n} of {len(df)} recognition messages ({pct}%) mention '{term}' "
            f"(case-insensitive match anywhere in the message text).")


def _superlative(df: pd.DataFrame, q: str) -> str | None:
    """'Which subcategory has the most recognitions?' — but not 'most distinct subcategories' (a different question)."""
    if not re.search(r"\b(which|what)\b", q) or re.search(r"\b(distinct|unique|different)\b", q):
        return None
    direction = "fewest" if _FEWEST_WORDS.search(q) else "most" if _MOST_WORDS.search(q) else None
    if direction is None:
        return None
    for key, col, label in _SUPERLATIVE_DIMENSIONS:
        if key in q:
            counts = df[col].value_counts().sort_values(ascending=(direction == "fewest"), kind="stable")
            name, n = counts.index[0], int(counts.iloc[0])
            lines = "\n".join(f"  - {k}: {v}" for k, v in counts.head(5).items())
            return (f"The {label} with the {direction} recognitions is '{name}' ({n}).\n"
                    f"{label.capitalize()}s with the {direction} recognitions:\n{lines}")
    return None


def _group_totals(df: pd.DataFrame, q: str) -> str | None:
    """Recognitions given / received per role group — the same grouping as the graph tool's role groups."""
    if not _GROUP_WORDS.search(q):
        return None
    give, receive = bool(_GIVE_WORDS.search(q)), bool(_RECEIVE_WORDS.search(q))
    if not (give or receive):
        return None
    from tools.graph_tool import classify_role
    titled = df.dropna(subset=["nominator_title", "recipient_title"])
    parts = []
    for wanted, col, verb in [(give, "nominator_title", "gives"), (receive, "recipient_title", "receives")]:
        if not wanted:
            continue
        counts = titled[col].map(classify_role).value_counts()
        lines  = "\n".join(f"  - {k}: {v}" for k, v in counts.items())
        parts.append(f"The role group that {verb} the most recognitions is '{counts.index[0]}' ({int(counts.iloc[0])}).\n"
                     f"Recognitions {'given' if verb == 'gives' else 'received'} by role group "
                     f"(job titles grouped into departments):\n{lines}")
    return "\n\n".join(parts)


def _count_for_label(df: pd.DataFrame, q: str) -> str | None:
    """Count for a specific category or subcategory named in the question (longest name wins)."""
    nq = _norm(q)
    labels = [(s, "subcategory_name") for s in df["subcategory_name"].dropna().unique()] + \
             [(c, "category_name") for c in df["category_name"].dropna().unique()]
    for label, col in sorted(labels, key=lambda x: -len(x[0])):
        if _norm(label) not in nq:
            continue
        rows = df[df[col] == label]
        if col == "category_name" and "subcategor" in q:
            subs = rows["subcategory_name"].value_counts()
            lines = "\n".join(f"  - {s}: {c}" for s, c in subs.items())
            return f"'{label}' has {len(subs)} subcategories:\n{lines}"
        if any(w in q for w in _COUNT_WORDS):
            pct = round(len(rows) / len(df) * 100, 1)
            kind = "subcategory" if col == "subcategory_name" else "category"
            return f"There are {len(rows)} recognitions ({pct}%) in the {kind} '{label}'."
    return None


@tool
def pandas_query(question: str) -> str:
    """
    Use this tool for ANY question involving counts, numbers, statistics,
    rankings, distributions, names, people, awards, or filtering the dataset.

    Examples:
    - How many awards are in each category?
    - Which job title gets recognized the most?
    - How many people are mentioned in the dataset?
    - Who are the most mentioned people?
    - What are all the unique award titles?
    - What is the average message length?
    - Which award title is most common?
    - How many recognitions did Engineering receive?
    - List all subcategories
    - What is the longest recognition message?
    - How many times is Sarah mentioned?
    - How many messages mention the word 'customer'?
    - Which subcategory has the fewest recognitions?
    """
    df = load_dataframe()
    q  = question.lower()

    # ── Messages that mention a word (before the date guard: "mention 'weekend'" is a text search) ──
    mention_answer = _count_mentions(df, q)
    if mention_answer:
        return mention_answer

    # ── No dates in the data ─────────────────────────────────────────────────
    if _TIME_PATTERN.search(q):
        return (
            "CANNOT_ANSWER: The dataset has no date or time information, so questions about "
            f"specific years, months or trends over time can't be answered. Columns: {', '.join(df.columns)}."
        )

    # ── A specific category / subcategory named in the question ──────────────
    # Checked before any total/breakdown rule, which would otherwise swallow "how many recognitions are in X"
    label_answer = _count_for_label(df, q)
    if label_answer:
        return label_answer

    # ── Role groups / departments giving or receiving the most ───────────────
    group_answer = _group_totals(df, q)
    if group_answer:
        return group_answer

    # ── "Which <dimension> has the most / fewest recognitions?" ──────────────
    superlative_answer = _superlative(df, q)
    if superlative_answer:
        return superlative_answer

    breakdown_words = ["how many", "count", "distribution", "breakdown", "each", "per ", " by "]

    # ── Subcategories (before categories: "subcategory" contains "category") ─
    if _HOW_MANY_OF.search(q) and "subcategor" in q and not _COMPARISON.search(q):
        return f"There are {df['subcategory_name'].nunique()} subcategories."

    if "list" in q and "subcategor" in q:
        subs = df["subcategory_name"].unique().tolist()
        return "Subcategories:\n" + "\n".join(f"  - {s}" for s in sorted(subs))

    if "subcategor" in q and any(w in q for w in breakdown_words):
        counts = df["subcategory_name"].value_counts()
        result = "Recognition count by subcategory:\n"
        for sub, count in counts.items():
            result += f"  - {sub}: {count}\n"
        return result

    # ── Categories ───────────────────────────────────────────────────────────
    if re.search(r"\b(how many|number of|total)\s+(unique\s+|distinct\s+|different\s+)?categor", q):
        cats = df["category_name"].unique().tolist()
        return f"There are {df['category_name'].nunique()} categories: {', '.join(cats)}."

    if "list" in q and "categor" in q:
        cats = df["category_name"].unique().tolist()
        return "Categories:\n" + "\n".join(f"  - {c}" for c in cats)

    if "categor" in q and any(w in q for w in breakdown_words):
        counts = df["category_name"].value_counts()
        result = f"Recognition count by category (total: {len(df)}):\n"
        for cat, count in counts.items():
            pct = round(count / len(df) * 100, 1)
            result += f"  - {cat}: {count} ({pct}%)\n"
        return result

    # ── Total counts — only when nothing narrows the question ────────────────
    if any(w in q for w in ["how many total", "total number", "total rows",
                             "how many records", "total awards", "total recognitions",
                             "how many recognitions", "how many awards"]) \
            and not any(w in q for w in _NOT_A_TOTAL) and not _names_something(question):
        return f"There are {len(df)} total recognitions in the dataset."

    # ── Names / people ────────────────────────────────────────────────────────
    if any(w in q for w in ["how many people", "how many names", "how many person",
                             "number of people", "number of names", "unique people",
                             "unique names", "total people", "total names"]):
        name_counts = _extract_names_from_messages(df)
        return (
            f"There are approximately {len(name_counts)} unique names mentioned "
            f"across the {len(df)} recognition messages."
        )

    if any(w in q for w in ["most mentioned", "top names", "most common names",
                             "most mentioned people", "most mentioned names",
                             "who appears most", "who is mentioned most",
                             "popular names", "frequent names"]):
        name_counts = _extract_names_from_messages(df)
        top = name_counts.most_common(15)
        result = "Most mentioned names in recognition messages:\n"
        for name, count in top:
            result += f"  - {name}: mentioned {count} times\n"
        return result

    if any(w in q for w in ["list all names", "list names", "all names",
                             "all people", "list all people", "everyone mentioned"]):
        name_counts = _extract_names_from_messages(df)
        top = name_counts.most_common(30)
        result = f"Top 30 names in recognition messages (out of ~{len(name_counts)} unique):\n"
        for name, count in top:
            result += f"  - {name} ({count} mentions)\n"
        return result

    # Specific name mention count
    name_match = re.search(r'\b([A-Z][a-z]{2,14})\b', question)
    if name_match and any(w in q for w in ["how many times", "how often", "mentioned",
                                            "appears", "count of", "number of times"]):
        name = name_match.group(1)
        if name not in _SKIP_WORDS:
            count = sum(1 for msg in df["message"].dropna()
                        if re.search(rf'\b{re.escape(name)}\b', str(msg), re.IGNORECASE))
            return f"'{name}' is mentioned in {count} recognition messages out of {len(df)} total."

    # ── Award titles ──────────────────────────────────────────────────────────
    if any(w in q for w in ["how many award", "unique award", "number of award",
                             "total award title"]):
        return f"There are {df['award_title'].nunique()} unique award titles across {len(df)} records."

    if any(w in q for w in ["most common award", "top award", "popular award",
                             "most used award", "award distribution"]):
        counts = df["award_title"].value_counts().head(15)
        result = "Most common award titles:\n"
        for title, count in counts.items():
            result += f"  - {title}: {count}\n"
        return result

    if any(w in q for w in ["list award", "all award", "what award"]):
        counts = df["award_title"].value_counts().head(20)
        result = f"Top 20 award titles (out of {df['award_title'].nunique()} unique):\n"
        for title, count in counts.items():
            result += f"  - {title}: {count}\n"
        return result

    # ── Job titles ────────────────────────────────────────────────────────────
    if _MOST_NOMINATING.search(q):
        counts = df["nominator_title"].value_counts().head(10)
        result = "Top 10 nominator job titles:\n"
        for title, count in counts.items():
            result += f"  - {title}: {count}\n"
        return result

    if _MOST_RECOGNIZED.search(q):
        counts = df["recipient_title"].value_counts().head(10)
        result = "Top 10 most recognized job titles:\n"
        for title, count in counts.items():
            result += f"  - {title}: {count}\n"
        return result

    if "nominator" in q and any(w in q for w in ["unique", "distinct", "how many", "number of"]):
        return f"There are {df['nominator_title'].nunique()} unique nominator job titles."

    if any(w in q for w in ["unique recipient", "unique job title", "how many role",
                             "how many job title", "number of role", "different role"]):
        return (
            f"There are {df['recipient_title'].nunique()} unique recipient job titles "
            f"and {df['nominator_title'].nunique()} unique nominator job titles."
        )

    if any(w in q for w in ["list job title", "list role", "all job title",
                             "list recipient", "all recipient"]):
        titles = df["recipient_title"].value_counts().head(20)
        result = f"Top 20 recipient job titles (out of {df['recipient_title'].nunique()} unique):\n"
        for title, count in titles.items():
            result += f"  - {title}: {count}\n"
        return result

    if any(w in q for w in ["list nominator", "all nominator"]):
        titles = df["nominator_title"].value_counts().head(20)
        result = f"Top 20 nominator job titles (out of {df['nominator_title'].nunique()} unique):\n"
        for title, count in titles.items():
            result += f"  - {title}: {count}\n"
        return result

    # ── Message length ────────────────────────────────────────────────────────
    if any(w in q for w in ["average message", "avg message", "mean message",
                             "average length", "avg length", "mean length"]):
        return (
            f"Average message: {int(df['msg_len'].mean())} chars. "
            f"Median: {int(df['msg_len'].median())}. "
            f"Min: {int(df['msg_len'].min())}. "
            f"Max: {int(df['msg_len'].max())}."
        )

    if any(w in q for w in ["longest message", "longest recognition"]):
        row = df.loc[df["msg_len"].idxmax()]
        return (
            f"Longest message: {row['msg_len']} characters.\n"
            f"Award: {row['award_title']}\n"
            f"Category: {row['category_name']}\n"
            f"Preview: {str(row['message'])[:400]}..."
        )

    if any(w in q for w in ["shortest message", "shortest recognition"]):
        row = df.loc[df["msg_len"].idxmin()]
        return (
            f"Shortest message: {row['msg_len']} characters.\n"
            f"Award: {row['award_title']}\n"
            f"Message: {str(row['message'])}"
        )

    if any(w in q for w in ["message length", "length distribution", "length by category"]):
        result = "Average message length by category:\n"
        for cat, grp in df.groupby("category_name"):
            result += f"  - {cat}: {int(grp['msg_len'].mean())} chars avg\n"
        return result

    # ── Unique counts (nominator before the generic "job title" match) ────────
    # Comparisons ("which category has the most distinct subcategories") are not plain unique counts
    if ("unique" in q or "distinct" in q) and not _COMPARISON.search(q):
        if "award" in q:
            return f"There are {df['award_title'].nunique()} unique award titles."
        if "nominator" in q:
            return f"There are {df['nominator_title'].nunique()} unique nominator job titles."
        if "recipient" in q:
            return f"There are {df['recipient_title'].nunique()} unique recipient job titles."
        if "job title" in q or "role" in q:
            return (
                f"There are {df['recipient_title'].nunique()} unique recipient job titles "
                f"and {df['nominator_title'].nunique()} unique nominator job titles."
            )
        if "subcategor" in q:
            return f"There are {df['subcategory_name'].nunique()} unique subcategories."
        if "categor" in q:
            return f"There are {df['category_name'].nunique()} unique categories."
        if "name" in q or "people" in q or "person" in q:
            name_counts = _extract_names_from_messages(df)
            return f"There are approximately {len(name_counts)} unique names in the messages."

    # ── Dataset overview ──────────────────────────────────────────────────────
    if any(w in q for w in ["overview", "summary", "describe", "what columns",
                             "what data", "dataset info", "tell me about the data"]):
        name_counts = _extract_names_from_messages(df)
        return (
            f"Dataset overview:\n"
            f"  - Total records: {len(df)}\n"
            f"  - Categories: {df['category_name'].nunique()}\n"
            f"  - Subcategories: {df['subcategory_name'].nunique()}\n"
            f"  - Unique award titles: {df['award_title'].nunique()}\n"
            f"  - Unique recipient job titles: {df['recipient_title'].nunique()}\n"
            f"  - Unique nominator job titles: {df['nominator_title'].nunique()}\n"
            f"  - Unique names in messages: ~{len(name_counts)}\n"
            f"  - Avg message length: {int(df['msg_len'].mean())} chars\n"
            f"  - Columns: {', '.join(df.columns.tolist())}\n"
        )

    # ── Category + subcategory combo ─────────────────────────────────────────
    if "categor" in q and "subcategor" in q and not _COMPARISON.search(q):
        return (
            f"There are {df['category_name'].nunique()} categories and "
            f"{df['subcategory_name'].nunique()} subcategories."
        )

    return _llm_pandas_fallback(question, df)