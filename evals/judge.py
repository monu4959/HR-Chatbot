"""
evals/judge.py

Second scorer for eval runs: the validated graders (evals/graders.py, v3) running on the "judge" tier
(JUDGE_PROVIDER / JUDGE_MODEL — Claude Haiku by default). On the human-labelled holdout, Haiku agreed with
the labeller at kappa 0.90 (correctness) and 0.81 (faithfulness), above the keyword checks (0.81).

The keyword checks stay the primary score (deterministic, free); the judge adds
  judge_correct   does the answer match the reference?           (catches what keyword rules miss)
  judge_faithful  is every claim supported by the tool output?   (catches fabrication even when the answer is right)
and the run summary lists every case where the judge and the keyword checks disagree, for review.
"""

from concurrent.futures import ThreadPoolExecutor

from evals.graders import grade_correctness, grade_faithfulness, reference_text


def _verdict(out: dict) -> bool | None:
    return None if out.get("verdict") is None else out["verdict"] == "pass"


def judge_record(rec: dict, case: dict, facts: dict) -> dict:
    """Judge fields for one eval record (None where the judge doesn't apply or failed)."""
    answer = rec.get("answer") or ""
    if "error" in rec or not answer or "CHART_DATA:" in answer:  # charts are checked exactly by the keyword checks
        return {"judge_correct": None, "judge_faithful": None}
    correct = grade_correctness(case["question"], reference_text(case, facts), answer)
    faithful = (grade_faithfulness(case["question"], rec["tool_output"], answer) if rec.get("tool_output")
                else {"verdict": None, "reason": "no tool output"})
    return {
        "judge_correct":         _verdict(correct),
        "judge_correct_reason":  correct.get("reason") or correct.get("error", ""),
        "judge_faithful":        _verdict(faithful),
        "judge_faithful_reason": faithful.get("reason") or faithful.get("error", ""),
    }


def judge_records(records: list[dict], cases: dict, facts: dict, workers: int = 8) -> None:
    """Add judge fields to every record in place (API calls run in parallel)."""
    todo = [r for r in records if r["id"] in cases]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rec, fields in zip(todo, pool.map(lambda r: judge_record(r, cases[r["id"]], facts), todo)):
            rec.update(fields)


def judge_summary(records: list[dict], model: str) -> dict | None:
    judged = [r for r in records if r.get("judge_correct") is not None or r.get("judge_faithful") is not None]
    if not judged:
        return None
    rate = lambda key: (round(sum(r[key] for r in judged if r.get(key) is not None)
                              / max(1, sum(r.get(key) is not None for r in judged)), 3))
    return {
        "model":         model,
        "n_judged":      len(judged),
        "correct_rate":  rate("judge_correct"),
        "faithful_rate": rate("judge_faithful"),
        "n_errors":      sum(r.get("judge_correct") is None for r in judged),
        "disagree_with_keyword_checks": [r["id"] for r in judged
                                         if r.get("judge_correct") is not None and r["judge_correct"] != bool(r["answer_pass"])],
        "unfaithful": [r["id"] for r in judged if r.get("judge_faithful") is False],
    }
