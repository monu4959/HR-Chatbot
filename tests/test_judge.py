import evals.judge as judge


def test_judge_record_skips_charts_and_errors():
    assert judge.judge_record({"answer": "Here's the chart:\nCHART_DATA:{}"}, {"question": "q"}, {}) == \
        {"judge_correct": None, "judge_faithful": None}
    assert judge.judge_record({"error": "boom", "answer": "x"}, {"question": "q"}, {})["judge_correct"] is None


def test_judge_records_and_summary(monkeypatch):
    monkeypatch.setattr(judge, "grade_correctness", lambda q, ref, a: {"verdict": "pass" if "48" in a else "fail", "reason": "r"})
    monkeypatch.setattr(judge, "grade_faithfulness", lambda q, t, a: {"verdict": "pass" if "48" in t else "fail", "reason": "r"})
    case = {"id": "c1", "question": "How many?", "check": "numeric", "gold": {"value": 48}}
    recs = [{"id": "c1", "answer": "There are 48.", "tool_output": "48 rows", "answer_pass": False},
            {"id": "c2", "answer": "There are 7.",  "tool_output": "1000 rows", "answer_pass": False}]
    judge.judge_records(recs, {"c1": case, "c2": {**case, "id": "c2"}}, {}, workers=2)
    assert recs[0]["judge_correct"] is True and recs[1]["judge_faithful"] is False
    s = judge.judge_summary(recs, "m")
    assert s["correct_rate"] == 0.5 and s["disagree_with_keyword_checks"] == ["c1"] and s["unfaithful"] == ["c2"]
