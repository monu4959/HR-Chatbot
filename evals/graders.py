"""
evals/graders.py

LLM judges, validated against human labels in evals/grader_validation/.

  faithfulness  Is every factual claim in the answer supported by the tool output?
                (Can run at serving time — needs no gold answer.)
  correctness   Does the answer correctly answer the question, given the reference
                computed from the data? (Evals only — needs the gold answer.)

Both return a binary verdict ("pass" / "fail") with per-claim reasons. Binary verdicts are
easier to validate against humans than 0–1 scores, and the prompts contain no example
score for a small model to copy.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm import get_llm, parse_json_reply
from agent.grader import FAITHFULNESS_PROMPT, strip_badge, numbers_not_in_data, _with_number_screen

FAITHFULNESS_PROMPT_V1 = """You are auditing an AI analytics assistant. You get the DATA a tool returned and the ANSWER the assistant wrote from it.

Decide whether every factual claim in the ANSWER is supported by the DATA.

How to judge:
1. List each factual claim in the ANSWER: numbers, percentages, counts, names, job titles, categories, rankings, quotes, and statements about what the data shows.
2. For each claim decide:
   - "supported": stated in the DATA, or follows from it by simple arithmetic or reordering
   - "unsupported": not in the DATA, contradicts it, or adds specifics the DATA does not contain
3. Do NOT count as claims: formatting, headings, a "Route:" line, offers to help further, or general caveats.
4. Saying information is missing or that the question can't be answered is supported when the DATA does not contain that information.

The verdict is "fail" if ANY claim is unsupported, otherwise "pass".

Return ONLY a JSON object with these keys:
  "claims":  a list of objects, each with "claim" (string), "status" ("supported" or "unsupported") and "evidence" (string: the DATA text it relies on, or why it is unsupported)
  "verdict": "pass" or "fail"
  "reason":  one sentence explaining the verdict"""

CORRECTNESS_PROMPT_V1 = """You are grading an AI analytics assistant against a REFERENCE computed directly from the dataset.

Decide whether the ANSWER correctly answers the QUESTION.

How to judge:
- If the REFERENCE gives exact values, the ANSWER must state those values (ignore formatting such as "1,000" vs "1000", or rounding within the stated tolerance). A different number, a missing required value, or the wrong top item is "fail".
- If the REFERENCE says the question cannot be answered from the data, the ANSWER must decline or say the information is not available, and must not invent data. Making up an answer is "fail".
- If the REFERENCE describes an open-ended question, "pass" means the ANSWER addresses the question directly and stays about the requested topic; "fail" means it is off-topic, answers a different question, or refuses without reason.
- Extra correct detail is fine. Judge only correctness, not style or length.

Return ONLY a JSON object with these keys:
  "verdict": "pass" or "fail"
  "reason":  one sentence explaining the verdict"""

# ── v2: revised on the calibration half of the human labels (evals/grader_validation) ─────────────
# Faithfulness v1 said "all claims supported" without checking numbers, accepted "the data doesn't contain X"
# when it did, and flagged true statements about missing data. v2 gets a code-computed list of answer
# numbers that don't appear in the DATA, and explicit rules for claims about missing information.
# RESULT on calibration: worse than v1 (faithfulness κ 0.48 → 0.32, correctness κ 0.69 → 0.48). The number screen
# had false flags ("4" from "Neo4j"), dropping "general caveats" from the ignore list made it fail harmless
# caveats, and the longer rule set made the 7B grader stricter and less accurate. Kept as a recorded negative result.

FAITHFULNESS_PROMPT_V2 = """You are auditing an AI analytics assistant. You get the DATA a tool returned and the ANSWER the assistant wrote from it.

Decide whether every factual claim in the ANSWER is supported by the DATA. Check it — do not assume.

How to judge:
1. List each factual claim in the ANSWER: numbers, percentages, counts, names, job titles, categories, rankings, quotes, and statements about what the data shows or does not show.
2. For each claim decide:
   - "supported": stated in the DATA, or follows from it by simple arithmetic, rounding or reordering
   - "unsupported": not in the DATA, contradicts it, or adds specifics the DATA does not contain (dates, places, people, totals, endings of quotes the DATA cut off)
3. NUMBERS NOT FOUND IN THE DATA lists every number in the ANSWER that does not appear anywhere in the DATA (found by code). Rule on each one: it is unsupported unless it follows from DATA numbers by simple arithmetic or rounding.
4. Claims about missing information:
   - "The data does not contain X" / "cannot be answered" is supported ONLY if the DATA really lacks X.
   - It is unsupported if the DATA does contain X (e.g. the DATA lists matching results or an analysis, but the ANSWER says there is none).
   - A true statement that something is not in the DATA is supported — it is not an invented claim.
5. Do NOT count as claims: formatting, headings, a "Route:" line, offers to help further, or generic advice about what kinds of questions the system can answer.

The verdict is "fail" if ANY claim is unsupported, otherwise "pass".

Return ONLY a JSON object with these keys:
  "claims":  a list of objects, each with "claim" (string), "status" ("supported" or "unsupported") and "evidence" (string: a short exact quote from the DATA, or why it is unsupported)
  "verdict": "pass" or "fail"
  "reason":  one sentence explaining the verdict"""

# Correctness v1 accepted refusals of answerable questions, failed full lists for not restating the count,
# and failed open-ended answers for being specific.

CORRECTNESS_PROMPT_V2 = """You are grading an AI analytics assistant against a REFERENCE computed directly from the dataset.

Decide whether the ANSWER correctly answers the QUESTION.

How to judge:
- Exact values in the REFERENCE: the ANSWER must give those values for the thing the question asks about. Ignore formatting ("1,000" vs "1000") and rounding within the stated tolerance. A different number for that thing, a missing required value, or the wrong top item is "fail". If the question asks "how many" and the ANSWER lists exactly that many items, the count is given.
- Compare carefully: take the number the ANSWER gives and the number the REFERENCE gives, and check they are the same.
- Declining ("cannot answer", "not available", "not able to provide a confident answer") is correct ONLY when the REFERENCE says the question cannot be answered from the data. If the REFERENCE gives a value or describes an answerable question, declining is "fail".
- If the REFERENCE says the question cannot be answered, the ANSWER must decline or say the information is not available, and must not invent data. Making up an answer is "fail".
- Open-ended REFERENCE: "pass" if the ANSWER addresses the question and stays on the requested topic. Specific examples, numbers or extra detail do not make it wrong — this grader judges correctness, not whether the details are supported.
- Judge only correctness, not style or length.

Return ONLY a JSON object with these keys:
  "verdict": "pass" or "fail"
  "reason":  one sentence explaining the verdict"""

# ── v3: v1 plus minimal, targeted additions (lesson from v2: a 7B grader gets worse with more rules) ──────

# v3 faithfulness lives in agent/grader.py (the app runs it); it must stay v1 plus exactly these two rules
_FAITHFULNESS_V3 = FAITHFULNESS_PROMPT_V1.replace(
    "4. Saying information is missing or that the question can't be answered is supported when the DATA does not contain that information.",
    "4. Saying information is missing or that the question can't be answered is supported when the DATA does not contain that information"
    " — but unsupported when the DATA does contain it (e.g. the DATA lists results or an analysis and the ANSWER says there is none).\n"
    "5. A list of NUMBERS NOT FOUND IN THE DATA follows the ANSWER (found by code). A number on that list that the ANSWER states as a fact"
    " about the data is unsupported, unless it follows from DATA numbers by simple arithmetic or rounding.",
)

assert FAITHFULNESS_PROMPT == _FAITHFULNESS_V3, "agent/grader.py prompt drifted from the validated v3"

CORRECTNESS_PROMPT = CORRECTNESS_PROMPT_V1.replace(
    "- Extra correct detail is fine. Judge only correctness, not style or length.",
    "- If the REFERENCE gives a value, an ANSWER that declines or says the information is not available is \"fail\".\n"
    "- If the question asks how many and the ANSWER lists exactly that many items, the count is given.\n"
    "- Extra correct detail is fine. Judge only correctness, not style or length.",
)
assert FAITHFULNESS_PROMPT != FAITHFULNESS_PROMPT_V1 and CORRECTNESS_PROMPT != CORRECTNESS_PROMPT_V1


_parse = parse_json_reply


def _judge(system_prompt: str, user_msg: str, model_tier: str = "judge") -> dict:
    llm = get_llm(model_tier, json_mode=True)
    try:
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_msg)])
        parsed   = _parse(response.text)
        verdict  = str(parsed.get("verdict", "")).strip().lower()
        if verdict not in ("pass", "fail"):
            raise ValueError(f"bad verdict: {verdict!r}")
        parsed["verdict"] = verdict
        return parsed
    except Exception as e:
        return {"verdict": None, "error": f"{type(e).__name__}: {e}"}


def grade_faithfulness(question: str, tool_output: str, answer: str) -> dict:
    return _judge(FAITHFULNESS_PROMPT, _with_number_screen(question, tool_output, answer))


# v2 (negative result; see the note above FAITHFULNESS_PROMPT_V2)
def grade_faithfulness_v2(question: str, tool_output: str, answer: str) -> dict:
    return _judge(FAITHFULNESS_PROMPT_V2, _with_number_screen(question, tool_output, answer))


def grade_correctness_v2(question: str, reference: str, answer: str) -> dict:
    return _judge(CORRECTNESS_PROMPT_V2,
                  f"QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nANSWER:\n{strip_badge(answer)}")


def grade_correctness(question: str, reference: str, answer: str) -> dict:
    return _judge(CORRECTNESS_PROMPT,
                  f"QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nANSWER:\n{strip_badge(answer)}")


# v1, kept so earlier validation results stay reproducible (evals/grader_validation, graders *_v1)
def grade_faithfulness_v1(question: str, tool_output: str, answer: str) -> dict:
    return _judge(FAITHFULNESS_PROMPT_V1,
                  f"QUESTION:\n{question}\n\nDATA:\n{tool_output}\n\nANSWER:\n{strip_badge(answer)}")


def grade_correctness_v1(question: str, reference: str, answer: str) -> dict:
    return _judge(CORRECTNESS_PROMPT_V1,
                  f"QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nANSWER:\n{strip_badge(answer)}")


def reference_text(case: dict, facts: dict) -> str:
    """Human- and judge-readable reference built from a test case's gold."""
    check, gold = case["check"], case.get("gold", {})
    if check == "numeric":
        tol = f" (within ±{gold['tolerance']})" if gold.get("tolerance") else ""
        about = f" for '{gold['label']}'" if gold.get("label") else ""
        return f"The correct value{about} is {gold['value']}{tol}."
    if check == "numeric_set":
        return "The correct counts are: " + "; ".join(f"{k}: {v}" for k, v in gold["values"].items()) + "."
    if check == "top_label":
        if "from" in gold:
            return f"The strongest cross-group flow is {gold['from']} → {gold['to']} ({gold['value']} recognitions)."
        return f"The correct top item is '{gold['label']}' ({gold['value']})."
    if check == "refusal":
        return ("The dataset cannot answer this question (it has only recognition messages, award titles, "
                "job titles and categories — no dates, salaries, ratings or personal details). "
                "A correct answer declines or says the information is not available.")
    if check == "chart":
        return f"A {gold['chart_type']} chart of {gold.get('data_from', 'the requested data')}."
    if check == "rubric":
        topics = gold.get("must_include_any") or gold.get("tool_must_include_any") or []
        hint = f" It should be about: {', '.join(topics)}." if topics else ""
        return f"Open-ended question; there is no single exact answer.{hint}"
    return ""
