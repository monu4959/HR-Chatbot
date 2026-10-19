"""
evals/build_gold.py

Generates the deterministic part of the fixed test set straight from the CSV,
so expected numbers are computed, never hand-typed.

Writes:
  evals/dataset/generated.jsonl   one test case per line
  evals/dataset/gold_facts.json   raw facts + data hash (for cache keys / drift checks)

Hand-written cases (semantic search, synthesis, profiles, follow-ups) go in
evals/dataset/handwritten.jsonl and are not touched by this script.

Usage: python -m evals.build_gold
"""

import os
import sys
import json
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.graph_tool import classify_role

DATA_PATH = Path(os.getenv("DATA_PATH", ROOT / "data" / "annotated_results.csv"))
OUT_DIR   = ROOT / "evals" / "dataset"
TEST_FRACTION = 0.3


def _split(case_id: str) -> str:
    """Stable dev/test assignment from the case id, so re-running never reshuffles."""
    h = int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 100
    return "test" if h < TEST_FRACTION * 100 else "dev"


def _case(case_id, question, route, check, gold, difficulty="easy"):
    return {
        "id":             case_id,
        "question":       question,
        "history":        [],
        "expected_route": route,
        "check":          check,
        "gold":           gold,
        "difficulty":     difficulty,
        "split":          _split(case_id),
        "source":         "generated",
    }


def _top(series: pd.Series, n: int = 1) -> list[dict]:
    return [{"label": k, "value": int(v)} for k, v in series.value_counts().head(n).items()]


def build_cases(df: pd.DataFrame) -> list[dict]:
    cases = []
    total = len(df)
    cat_counts = df["category_name"].value_counts()
    sub_counts = df["subcategory_name"].value_counts()

    # ── pandas_query: counts ───────────────────────────────────────────────────
    cases.append(_case("pd-total", "How many recognitions are in the dataset in total?",
                       "pandas_query", "numeric", {"value": total}))
    cases.append(_case("pd-n-categories", "How many categories are there?",
                       "pandas_query", "numeric", {"value": int(df["category_name"].nunique())}))
    cases.append(_case("pd-n-subcategories", "How many subcategories are there?",
                       "pandas_query", "numeric", {"value": int(df["subcategory_name"].nunique())}))
    cases.append(_case("pd-n-awards", "How many unique award titles are there?",
                       "pandas_query", "numeric", {"value": int(df["award_title"].nunique())}))
    cases.append(_case("pd-n-recipient-titles", "How many unique recipient job titles are there?",
                       "pandas_query", "numeric", {"value": int(df["recipient_title"].nunique())}))
    cases.append(_case("pd-n-nominator-titles", "How many unique nominator job titles are there?",
                       "pandas_query", "numeric", {"value": int(df["nominator_title"].nunique())}))

    cases.append(_case("pd-cat-breakdown", "How many awards are in each category?",
                       "pandas_query", "numeric_set",
                       {"values": {k: int(v) for k, v in cat_counts.items()}}))

    for i, (cat, count) in enumerate(cat_counts.items()):
        cases.append(_case(f"pd-cat-{i}", f"How many recognitions are in {cat}?",
                           "pandas_query", "numeric", {"value": int(count), "label": cat}))

    for i, (sub, count) in enumerate(sub_counts.head(5).items()):
        cases.append(_case(f"pd-sub-{i}", f"How many recognitions are in the {sub} subcategory?",
                           "pandas_query", "numeric", {"value": int(count), "label": sub}))

    # ── pandas_query: rankings ─────────────────────────────────────────────────
    cases.append(_case("pd-top-recipient", "Which job title gets recognized the most?",
                       "pandas_query", "top_label", _top(df["recipient_title"])[0]))
    cases.append(_case("pd-top-nominator", "Which job title nominates the most?",
                       "pandas_query", "top_label", _top(df["nominator_title"])[0]))
    cases.append(_case("pd-top-award", "What is the most common award title?",
                       "pandas_query", "top_label", _top(df["award_title"])[0]))

    msg_len = df["message"].str.len()
    cases.append(_case("pd-avg-len", "What is the average message length?",
                       "pandas_query", "numeric", {"value": int(msg_len.mean())}))

    # Phrasings that miss the keyword rules and exercise the LLM pandas fallback
    sub_per_cat = df.groupby("category_name")["subcategory_name"].nunique()
    top_sub_cat = sub_per_cat.idxmax()
    cases.append(_case("pd-fallback-subs-per-cat",
                       "Which category contains the largest number of distinct subcategories?",
                       "pandas_query", "top_label",
                       {"label": top_sub_cat, "value": int(sub_per_cat.max())}, difficulty="hard"))
    eng = df["recipient_title"].str.contains("engineer", case=False, na=False)
    cases.append(_case("pd-fallback-eng-share",
                       "What percentage of recognitions went to recipients with 'engineer' in their title?",
                       "pandas_query", "numeric",
                       {"value": round(eng.mean() * 100, 1), "tolerance": 0.5}, difficulty="hard"))

    # ── chart_generator: structure of the returned CHART_DATA ──────────────────
    cases.append(_case("ch-cat-bar", "Show me a bar chart of awards by category",
                       "chart_generator", "chart",
                       {"chart_type": "bar", "data": {k: int(v) for k, v in cat_counts.items()}}))
    cases.append(_case("ch-cat-pie", "Show me a pie chart of categories",
                       "chart_generator", "chart",
                       {"chart_type": "pie", "data": {k: int(v) for k, v in cat_counts.items()}}))
    cases.append(_case("ch-sub-bar", "Plot the distribution of subcategories",
                       "chart_generator", "chart",
                       {"chart_type": "horizontal_bar", "data": {k: int(v) for k, v in sub_counts.items()}}))
    cases.append(_case("ch-top-recipients", "Plot the top 10 recognized job titles",
                       "chart_generator", "chart",
                       {"chart_type": "horizontal_bar",
                        "data": {d["label"]: d["value"] for d in _top(df["recipient_title"], 10)}}))

    # ── graph_query: cross-group flows (role-group taxonomy from classify_role) ─
    titled = df.dropna(subset=["nominator_title", "recipient_title"])
    groups = pd.DataFrame({
        "from": titled["nominator_title"].map(classify_role),
        "to":   titled["recipient_title"].map(classify_role),
    })
    cross = groups[groups["from"] != groups["to"]].value_counts().head(3)
    cases.append(_case("gr-top-flow", "Which departments recognize each other most?",
                       "graph_query", "top_label",
                       {"label": f"{cross.index[0][0]} → {cross.index[0][1]}", "value": int(cross.iloc[0]),
                        "from": cross.index[0][0], "to": cross.index[0][1]}, difficulty="medium"))

    # ── Out of scope: should be refused / CANNOT_ANSWER, never a made-up number ─
    for i, q in enumerate([
        "What is the average salary of a Director of Engineering?",
        "Which employees got a performance rating of 5 last year?",
        "What will the weather be in Boston tomorrow?",
        "How many recognitions were given in March 2024?",
    ]):
        cases.append(_case(f"oos-{i}", q, None, "refusal", {}, difficulty="medium"))

    return cases


def build_heldout_cases(df: pd.DataFrame) -> list[dict]:
    """
    Frozen held-out set for final reporting, written after the tool fixes. Phrasings differ from
    the dev/test cases on purpose. Don't tune anything on these — run them once, at the end.
    """
    def held(*args, **kw):
        c = _case(*args, **kw)
        c["split"], c["source"] = "heldout", "generated-heldout"
        return c

    cases = []
    total = len(df)
    cat_counts = df["category_name"].value_counts()
    sub_counts = df["subcategory_name"].value_counts()

    cases.append(held("ho-share-strategic", "What share of all recognitions fall under Strategic Business Outcomes?",
                      "pandas_query", "numeric",
                      {"value": round(cat_counts["Strategic Business Outcomes"] / total * 100, 1), "tolerance": 0.1,
                       "label": "Strategic Business Outcomes"}, difficulty="medium"))

    phrasings = ["Count the recognitions tagged {}.", "How many awards fall under {}?",
                 "What number of recognitions belong to {}?", "{}: how many recognitions?"]
    for i, (sub, count) in enumerate(list(sub_counts.items())[5:9]):
        cases.append(held(f"ho-sub-{i}", phrasings[i].format(sub), "pandas_query", "numeric",
                          {"value": int(count), "label": sub}))

    cases.append(held("ho-n-award-titles", "How many different award titles appear in the data?",
                      "pandas_query", "numeric", {"value": int(df["award_title"].nunique())}))
    cases.append(held("ho-n-recipient-titles", "How many distinct job titles have received at least one recognition?",
                      "pandas_query", "numeric", {"value": int(df["recipient_title"].nunique())}, difficulty="medium"))
    cases.append(held("ho-median-len", "What's the median length of a recognition message?",
                      "pandas_query", "numeric", {"value": float(df["message"].str.len().median()), "tolerance": 0.5}))
    cases.append(held("ho-top-sub", "Which subcategory has the most recognitions?",
                      "pandas_query", "top_label", _top(df["subcategory_name"])[0], difficulty="medium"))
    cases.append(held("ho-fewest-cat", "Which category has the fewest recognitions?",
                      "pandas_query", "top_label",
                      {"label": cat_counts.index[-1], "value": int(cat_counts.iloc[-1])}, difficulty="medium"))
    cases.append(held("ho-customer-mentions", "How many recognitions mention the word 'customer'?",
                      "pandas_query", "numeric",
                      {"value": int(df["message"].str.contains("customer", case=False, na=False).sum())},
                      difficulty="hard"))
    cases.append(held("ho-director-nominators",
                      "How many recognitions were nominated by someone with Director in their job title?",
                      "pandas_query", "numeric",
                      {"value": int(df["nominator_title"].str.contains("director", case=False, na=False).sum())},
                      difficulty="hard"))

    cases.append(held("ho-ch-pie", "Can you draw a pie chart of the award categories?",
                      "chart_generator", "chart", {"chart_type": "pie", "data": {k: int(v) for k, v in cat_counts.items()}}))
    cases.append(held("ho-ch-nominators", "Visualize the top 5 job titles that nominate people",
                      "chart_generator", "chart",
                      {"chart_type": "horizontal_bar", "data": {d["label"]: d["value"] for d in _top(df["nominator_title"], 5)}},
                      difficulty="medium"))
    cases.append(held("ho-ch-subs", "Plot how many recognitions each subcategory got",
                      "chart_generator", "chart",
                      {"chart_type": "horizontal_bar", "data": {k: int(v) for k, v in sub_counts.items()}}))

    titled = df.dropna(subset=["nominator_title", "recipient_title"])
    givers    = titled["nominator_title"].map(classify_role).value_counts()
    receivers = titled["recipient_title"].map(classify_role).value_counts()
    cases.append(held("ho-gr-top-giver", "Which role group gives out the most recognitions?",
                      "graph_query", "top_label", {"label": givers.index[0], "value": int(givers.iloc[0])}, difficulty="medium"))
    cases.append(held("ho-gr-top-receiver", "Which group of roles receives the most recognitions overall?",
                      "graph_query", "top_label", {"label": receivers.index[0], "value": int(receivers.iloc[0])}, difficulty="medium"))

    largest = cat_counts.iloc[0] / total * 100
    fu = held("ho-fu-largest-share", "And what percentage of the total is the largest one?",
              "pandas_query", "numeric", {"value": round(largest, 1), "tolerance": 0.1}, difficulty="hard")
    fu["history"] = ["How many recognitions are in each category?"]
    cases.append(fu)

    for i, q in enumerate([
        "What's the gender split of award recipients?",
        "Which employee has the highest salary?",
        "How many recognitions were sent on weekends?",
        "What's the email address of the person who nominated the most?",
        "Give me Marcus's phone number.",
    ]):
        cases.append(held(f"ho-oos-{i}", q, None, "refusal", {}, difficulty="medium"))
    return cases


def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    data_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()[:16]

    cases = build_cases(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "generated.jsonl", "w", encoding="utf-8") as f:
        for c in cases:
            c["data_hash"] = data_hash
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    heldout = build_heldout_cases(df)
    with open(OUT_DIR / "heldout_generated.jsonl", "w", encoding="utf-8") as f:
        for c in heldout:
            c["data_hash"] = data_hash
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    facts = {
        "data_hash":        data_hash,
        "rows":             len(df),
        "categories":       {k: int(v) for k, v in df["category_name"].value_counts().items()},
        "subcategories":    {k: int(v) for k, v in df["subcategory_name"].value_counts().items()},
        "unique_awards":    int(df["award_title"].nunique()),
        "unique_recipients": int(df["recipient_title"].nunique()),
        "unique_nominators": int(df["nominator_title"].nunique()),
    }
    with open(OUT_DIR / "gold_facts.json", "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2, ensure_ascii=False)

    n_test = sum(c["split"] == "test" for c in cases)
    print(f"Wrote {len(cases)} cases ({len(cases) - n_test} dev / {n_test} test) + {len(heldout)} held-out, "
          f"data_hash={data_hash}")


if __name__ == "__main__":
    main()
