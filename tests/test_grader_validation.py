from evals.grader_validation.metrics import agreement
from evals.graders import strip_badge, reference_text, _parse


def test_perfect_agreement():
    m = agreement([(True, True), (False, False), (True, True), (False, False)])
    assert m["accuracy"] == 1.0 and m["kappa"] == 1.0 and m["fail_recall"] == 1.0


def test_grader_that_always_passes_has_zero_kappa_and_catches_nothing():
    m = agreement([(True, True), (False, True), (True, True), (False, True)])
    assert m["accuracy"] == 0.5 and m["kappa"] == 0.0 and m["fail_recall"] == 0.0 and m["pass_recall"] == 1.0


def test_confusion_counts():
    m = agreement([(False, False), (False, True), (True, False), (True, True)])
    assert m["confusion"] == {"caught_bad": 1, "passed_good": 1, "failed_good": 1, "missed_bad": 1}


def test_strip_badge_removes_route_line_and_warning_footer():
    a = "*Route: 📊 Analytical*\n\nThere are 48.\n\n---\n⚠️ *Low confidence (grounding score: 0.5).*"
    assert strip_badge(a) == "There are 48."


def test_parse_tolerates_fences_and_prose():
    assert _parse('Sure:\n```json\n{"verdict": "pass", "reason": "ok"}\n```')["verdict"] == "pass"


def test_reference_text_covers_every_check_type():
    facts = {"categories": {}}
    for case in [{"check": "numeric", "gold": {"value": 48, "label": "X"}},
                 {"check": "numeric_set", "gold": {"values": {"A": 1}}},
                 {"check": "top_label", "gold": {"label": "T", "value": 3}},
                 {"check": "refusal", "gold": {}},
                 {"check": "rubric", "gold": {"must_include_any": ["mentor"]}}]:
        assert reference_text(case, facts)


def test_number_screen_flags_changed_numbers_but_not_list_numbering():
    from evals.graders import numbers_not_in_data
    data = "There are 926 unique award titles across 1000 records."
    assert numbers_not_in_data("There are 933 unique award titles.", data) == ["933"]
    assert numbers_not_in_data("There are 926 award titles across 1,000 records.", data) == []
    assert numbers_not_in_data("1. Engineering: 926\n2. Leadership: 1000", data) == []


def test_number_screen_ignores_numbers_inside_words_and_other_list_styles():
    from evals.graders import numbers_not_in_data
    data = "There are 48 recognitions."
    assert numbers_not_in_data("Network analysis (including Neo4j graph queries)", data) == []
    assert numbers_not_in_data("1) Engineering\n2) Leadership\n**3.** Sales", data) == []
    assert numbers_not_in_data("There are 61 recognitions.", data) == ["61"]
