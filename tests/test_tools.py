"""
Deterministic tool tests — no LLM calls (the pandas LLM fallback is stubbed to fail loudly).
Each test pins a bug found by the eval baseline.
"""

import json
import os

import pandas as pd
import pytest

os.environ["NEO4J_URI"] = ""  # always exercise the NetworkX fallback

import tools.pandas_tool as pt
import tools.profile_tool as prof
import tools.rag_tool as rag
import tools.synthesis_tool as syn
from tools.chart_tool import chart_generator
from tools.graph_tool import graph_query, _extract_n, _build_group_graph, _build_title_graph

DF = pd.read_csv("data/annotated_results.csv")
CATS = DF["category_name"].value_counts()
SUBS = DF["subcategory_name"].value_counts()


@pytest.fixture(autouse=True)
def no_llm_fallback(monkeypatch):
    def boom(question, df):
        raise AssertionError(f"fell through to the LLM fallback: {question!r}")
    monkeypatch.setattr(pt, "_llm_pandas_fallback", boom)


def ask(question):
    return pt.pandas_query.invoke({"question": question})


def chart(question):
    return json.loads(chart_generator.invoke({"question": question}).split("CHART_DATA:", 1)[1])


# ── pandas_query ──────────────────────────────────────────────────────────────

def test_total_count():
    assert "1000" in ask("How many recognitions are in the dataset in total?")


@pytest.mark.parametrize("cat", CATS.index)
def test_single_category_count_is_not_the_total(cat):
    assert str(CATS[cat]) in ask(f"How many recognitions are in {cat}?")


def test_category_name_matches_with_and_instead_of_ampersand():
    assert str(CATS["Creative & Brand Impact"]) in ask("how many awards in creative and brand impact")


@pytest.mark.parametrize("sub", SUBS.index[:5])
def test_single_subcategory_count(sub):
    assert str(SUBS[sub]) in ask(f"How many recognitions are in the {sub} subcategory?")


def test_category_breakdown_lists_every_category():
    out = ask("How many awards are in each category?")
    assert all(f"{c}: {n}" in out for c, n in CATS.items())


def test_subcategory_breakdown_is_not_the_category_breakdown():
    out = ask("How many recognitions are in each subcategory?")
    assert f"{SUBS.index[0]}: {SUBS.iloc[0]}" in out


def test_subcategories_within_a_category():
    cat = "Creative & Brand Impact"
    n = DF[DF["category_name"] == cat]["subcategory_name"].nunique()
    assert f"{n} subcategories" in ask(f"How many subcategories does {cat} have?")


def test_unique_nominator_titles_is_not_recipient_count():
    assert str(DF["nominator_title"].nunique()) in ask("How many unique nominator job titles are there?")


@pytest.mark.parametrize("q", ["Which job title gets recognized the most?", "Which role receives the most recognition?"])
def test_most_recognized_title_phrasings(q):
    assert DF["recipient_title"].value_counts().index[0] in ask(q)


def test_top_nominator_phrasing():
    assert DF["nominator_title"].value_counts().index[0] in ask("Which job title nominates the most?")


@pytest.mark.parametrize("q", ["How many recognitions were given in March 2024?",
                               "How many awards were given last year?"])
def test_time_questions_are_refused_because_there_are_no_dates(q):
    assert ask(q).startswith("CANNOT_ANSWER")


@pytest.mark.parametrize("code", ["import os\nresult = os.getcwd()",
                                  "result = __import__('os').getcwd()",
                                  "result = open('data/annotated_results.csv').read()"])
def test_generated_code_cannot_escape_the_sandbox(code):
    ok, out = pt._run_generated_code(code, DF)
    assert not ok


def test_generated_code_still_runs_normal_pandas():
    ok, out = pt._run_generated_code("result = str(len(df[df['category_name'].str.contains('creative', case=False)]))", DF)
    assert ok and out == str(CATS["Creative & Brand Impact"])


def test_code_is_extracted_from_a_fenced_reply():
    reply = "Here you go:\n```python\nresult = '42'\n```\nThis counts rows."
    assert pt._extract_code(reply) == "result = '42'"


# ── chart_generator ───────────────────────────────────────────────────────────

def test_subcategory_chart_is_not_the_category_chart():
    c = chart("Plot the distribution of subcategories")
    assert c["chart_type"] == "horizontal_bar"
    assert {d["label"]: d["value"] for d in c["data"]} == SUBS.to_dict()


def test_pie_chart_of_subcategories():
    c = chart("Show me a pie chart of subcategories")
    assert c["chart_type"] == "pie" and len(c["data"]) == len(SUBS)


def test_category_bar_and_pie_unchanged():
    assert chart("Show me a bar chart of awards by category")["chart_type"] == "bar"
    assert chart("Show me a pie chart of categories")["chart_type"] == "pie"


def test_chart_respects_top_n():
    assert len(chart("Plot the top 5 recognized job titles")["data"]) == 5


# ── graph_query (NetworkX) ────────────────────────────────────────────────────

def test_extract_n_ignores_number_words_inside_other_words():
    assert _extract_n("which roles does anyone recognize often") == 10
    assert _extract_n("top 3 flows") == 3
    assert _extract_n("show the five strongest links") == 5


def test_group_graph_has_only_groups_and_most_common_category():
    G = _build_group_graph()
    assert all(not n.islower() for n in G.nodes)  # group names, no raw titles
    assert G.number_of_nodes() <= 11
    for _, _, d in G.edges(data=True):
        assert d["top_category"] in CATS.index


def test_title_graph_bridges_are_not_all_zero():
    out = graph_query.invoke({"question": "Who are the bridge roles that span multiple departments?"})
    scores = [float(line.rsplit(" ", 1)[1]) for line in out.splitlines() if "bridge score" in line]
    assert scores and max(scores) > 0


def test_receivers_and_mutual_pairs_have_networkx_output():
    out = graph_query.invoke({"question": "Who receives the most recognition?"})
    assert "received" in out
    out = graph_query.invoke({"question": "Are there mutual recognition pairs?"})
    assert "↔" in out


def test_centrality_question_does_not_also_dump_flows():
    out = graph_query.invoke({"question": "Who are the most central hubs in the recognition network?"})
    assert "Central" in out or "Connected" in out
    assert "Flows" not in out


# ── profile_and_replace ───────────────────────────────────────────────────────

def test_role_words_are_not_treated_as_person_names():
    assert prof._candidate_names("Who could replace a Senior Director?") == []
    assert prof._candidate_names("What has Marcus contributed?") == ["Marcus"]
    assert prof._candidate_names("If Emma left, who would be the best replacement?") == ["Emma"]


def test_profile_fields_tolerate_nested_model_output():
    assert prof._as_text_list(["a", ["b", "c"], {"skill": "d", "evidence": "e"}, 3]) == ["a", "b", "c", "d (e)", "3"]
    assert prof._as_text_list("single") == ["single"]
    assert prof._as_text_list(None) == []


def test_like_only_means_replacement_in_someone_like():
    assert prof._is_replacement("Who is someone like Marcus?")
    assert not prof._is_replacement("What would Marcus like to improve?")


# ── semantic_search CRAG ──────────────────────────────────────────────────────

def test_crag_ignores_filler_words():
    q = "Find recognitions about mentorship"
    assert rag._keyword_overlap(q, "this is about nothing in particular") == 0
    assert rag._keyword_overlap(q, "thanks for mentoring the new hires") == 1


# ── llm_synthesis helpers ─────────────────────────────────────────────────────

def test_category_keyword_choice_is_deterministic_and_specific():
    kws = syn._get_all_keywords(DF)
    assert syn._match_keyword("themes in leadership and mentorship recognitions", kws) == "leadership"
    assert syn._match_keyword("nothing relevant here", kws) is None


def test_cannot_answer_is_trimmed_to_the_reason():
    raw = "CANNOT_ANSWER: This is outside the scope of the recognition dataset. While the insights provided..."
    assert syn._clean_cannot_answer(raw) == "CANNOT_ANSWER: This is outside the scope of the recognition dataset."
    assert syn._clean_cannot_answer("A normal answer.") == "A normal answer."


# ── agent refusal bypass ──────────────────────────────────────────────────────

def test_cannot_answer_message_keeps_the_reason_and_drops_the_marker():
    from agent.prompt_engine import cannot_answer_message, is_cannot_answer
    out = "CANNOT_ANSWER: No recognition messages found mentioning 'Zebedee'.\n\nPeople with the most recognition data: Marcus (46)"
    msg = cannot_answer_message(out)
    assert is_cannot_answer(out) and not is_cannot_answer("There are 48 recognitions.")
    assert "Zebedee" in msg and "Marcus (46)" in msg and "CANNOT_ANSWER" not in msg


# ── regressions from the post-fix eval run (router rewords questions) ─────────

@pytest.mark.parametrize("q", ["How many categories are there?", "How many unique categories are there?",
                               "How many distinct categories exist?"])
def test_category_count_phrasings(q):
    assert f"There are {DF['category_name'].nunique()} categories" in ask(q)


def test_comparison_questions_are_not_answered_as_plain_unique_counts():
    with pytest.raises(AssertionError, match="fell through"):
        ask("Which category contains the largest number of distinct subcategories?")


def test_bare_role_title_gets_similar_roles():
    assert "Roles Most Similar to 'Senior Director'" in prof.profile_and_replace.invoke({"query": "Senior Director"})


def test_ranked_counts_breaks_ties_alphabetically():
    from tools.counts import ranked_counts
    s = pd.Series(["b", "a", "c", "a", "b", "d"])
    assert list(ranked_counts(s).items()) == [("a", 2), ("b", 2), ("c", 1), ("d", 1)]
    assert list(ranked_counts(s, ascending=True).items()) == [("c", 1), ("d", 1), ("a", 2), ("b", 2)]
