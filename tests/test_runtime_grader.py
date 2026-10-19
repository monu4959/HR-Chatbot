import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agent.grader import check_faithfulness, FAITHFULNESS_PROMPT
from agent.graph import grade_to_state, route_after_check
from agent.prompt_engine import format_final_answer, REFUSAL_MESSAGE
from evals.graders import FAITHFULNESS_PROMPT as EVAL_PROMPT


def _llm(reply: dict | str):
    return FakeListChatModel(responses=[reply if isinstance(reply, str) else json.dumps(reply)])


def test_app_and_evals_grade_with_the_same_prompt():
    assert FAITHFULNESS_PROMPT == EVAL_PROMPT


def test_fail_verdict_collects_the_unsupported_claims():
    reply = {"claims": [{"claim": "There are 933 award titles", "status": "unsupported", "evidence": "DATA says 926"},
                        {"claim": "Data covers 1000 records", "status": "supported", "evidence": "1000 records"}],
             "verdict": "fail", "reason": "933 is not in the data"}
    out = check_faithfulness("How many award titles?", "926 unique award titles across 1000 records",
                             "There are 933 award titles.", _llm(reply))
    assert out["verdict"] == "fail" and out["unsupported"] == ["There are 933 award titles"]


def test_unparseable_reply_is_a_grader_error_not_a_pass():
    out = check_faithfulness("q", "data", "answer", _llm("I think it looks fine!"))
    assert out["verdict"] is None and "error" in out


def test_verdicts_map_to_refuse_pass_and_kept_on_error():
    fail = grade_to_state({"verdict": "fail", "unsupported": ["X"]})
    assert fail["hallucination_score"] == 0.0 and fail["hallucination_flag"] and fail["hallucinated_claims"] == ["X"]
    ok = grade_to_state({"verdict": "pass", "unsupported": ["ignored"]})
    assert ok == {"hallucination_score": 1.0, "hallucination_flag": False, "hallucinated_claims": [], "grader_error": False}
    err = grade_to_state({"verdict": None, "error": "boom"})
    assert err["hallucination_score"] == 1.0 and err["grader_error"]


def test_failed_light_answer_escalates_before_refusing():
    state = {"model_tier": "light", "escalated": False, "grader_error": False,
             **grade_to_state({"verdict": "fail", "unsupported": ["X"]})}
    assert route_after_check(state) == "escalate"


def test_refusal_shows_flagged_claims():
    out = format_final_answer("There are 933.", 0.0, "pandas_query", flagged_claims=["There are 933 award titles"])
    assert "Flagged claims" in out and "933" in out and "There are 933." not in out
    assert "Questions this system answers well" in out
    assert format_final_answer("x", 0.0) == REFUSAL_MESSAGE      # no claims → the plain refusal
    assert format_final_answer("Fine.", 1.0, "pandas_query").endswith("Fine.")
