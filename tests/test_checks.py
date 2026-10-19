import json

from evals.checks import check_answer, check_tool
from evals.run import load_cases

FACTS = {"categories": {"A": 3, "B": 1}}


def _case(check, gold):
    return {"id": "t", "check": check, "gold": gold}


def test_numeric_handles_thousands_separator_and_tolerance():
    assert check_answer(_case("numeric", {"value": 1000}), "There are 1,000 recognitions.", False, FACTS)[0]
    assert not check_answer(_case("numeric", {"value": 48}), "There are 1000 recognitions.", False, FACTS)[0]
    assert check_answer(_case("numeric", {"value": 12.3, "tolerance": 0.5}), "About 12.6% of them", False, FACTS)[0]


def test_numeric_set_requires_every_value():
    case = _case("numeric_set", {"values": {"A": 494, "B": 48}})
    assert check_answer(case, "A: 494, B: 48", False, FACTS)[0]
    ok, detail = check_answer(case, "A: 494", False, FACTS)
    assert not ok and "B" in detail


def test_top_label_graph_flow_needs_both_ends():
    case = _case("top_label", {"label": "X → Y", "from": "Engineering", "to": "Leadership"})
    assert check_answer(case, "engineering recognizes leadership most", False, FACTS)[0]
    assert not check_answer(case, "Engineering is busiest", False, FACTS)[0]


def test_chart_compares_type_and_data():
    chart = "CHART_DATA:" + json.dumps({"chart_type": "pie", "data": [{"label": "A", "value": 3}, {"label": "B", "value": 1}]})
    assert check_answer(_case("chart", {"chart_type": "pie", "data_from": "categories"}), chart, False, FACTS)[0]
    assert not check_answer(_case("chart", {"chart_type": "bar", "data_from": "categories"}), chart, False, FACTS)[0]
    assert not check_answer(_case("chart", {"chart_type": "pie", "data_from": "categories"}), "no chart", False, FACTS)[0]


def test_refusal_accepts_gate_or_wording_and_rejects_made_up_answers():
    case = _case("refusal", {})
    assert check_answer(case, "anything", True, FACTS)[0]
    assert check_answer(case, "Salary data is not available in this dataset.", False, FACTS)[0]
    assert not check_answer(case, "| Technology | 175,000 |", False, FACTS)[0]


def test_rubric_tool_check_is_separate_from_answer():
    case = _case("rubric", {"must_include_any": ["mentor"], "tool_must_include_any": ["Profile for"]})
    assert check_answer(case, "Great mentorship shown", False, FACTS)[0]
    assert check_tool(case, "Profile for Marcus ...", FACTS) == (True, "")
    assert check_tool(_case("rubric", {}), "whatever", FACTS) == (None, "")


def test_dataset_loads_with_unique_ids_and_known_checks():
    cases = load_cases("all", None, None)
    assert len(cases) == len({c["id"] for c in cases})
    assert {c["check"] for c in cases} <= {"numeric", "numeric_set", "top_label", "chart", "refusal", "rubric"}
    assert {c["split"] for c in cases} == {"dev", "test"}
