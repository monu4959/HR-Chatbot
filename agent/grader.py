"""
agent/grader.py

The faithfulness grader the app runs on every graded answer (hallucination_check in agent/graph.py).
It is the version validated against human labels in evals/grader_validation (v3, chosen on the
calibration half): on the 43-item holdout it agreed with the human at kappa 0.40 (catches 61% of
unfaithful answers, passes 80% of faithful ones) — the old 0–1 score agreed at kappa 0.00.
evals/graders.py imports this prompt, so the app and the evals grade with exactly the same text.

A code-computed list of answer numbers that appear nowhere in the tool output is passed to the model,
because most fabrications in this domain are numbers and a 7B model rarely checks them unaided.
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm import parse_json_reply

FAITHFULNESS_PROMPT = 'You are auditing an AI analytics assistant. You get the DATA a tool returned and the ANSWER the assistant wrote from it.\n\nDecide whether every factual claim in the ANSWER is supported by the DATA.\n\nHow to judge:\n1. List each factual claim in the ANSWER: numbers, percentages, counts, names, job titles, categories, rankings, quotes, and statements about what the data shows.\n2. For each claim decide:\n   - "supported": stated in the DATA, or follows from it by simple arithmetic or reordering\n   - "unsupported": not in the DATA, contradicts it, or adds specifics the DATA does not contain\n3. Do NOT count as claims: formatting, headings, a "Route:" line, offers to help further, or general caveats.\n4. Saying information is missing or that the question can\'t be answered is supported when the DATA does not contain that information — but unsupported when the DATA does contain it (e.g. the DATA lists results or an analysis and the ANSWER says there is none).\n5. A list of NUMBERS NOT FOUND IN THE DATA follows the ANSWER (found by code). A number on that list that the ANSWER states as a fact about the data is unsupported, unless it follows from DATA numbers by simple arithmetic or rounding.\n\nThe verdict is "fail" if ANY claim is unsupported, otherwise "pass".\n\nReturn ONLY a JSON object with these keys:\n  "claims":  a list of objects, each with "claim" (string), "status" ("supported" or "unsupported") and "evidence" (string: the DATA text it relies on, or why it is unsupported)\n  "verdict": "pass" or "fail"\n  "reason":  one sentence explaining the verdict'


def strip_badge(answer: str) -> str:
    """Remove the UI decorations format_final_answer adds (route badge, low-confidence footer)."""
    text = re.sub(r"^\*Route:[^\n]*\*\s*", "", answer or "")
    return re.split(r"\n\n---\n⚠️", text)[0].strip()


def _number_tokens(text: str) -> list[str]:
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text or "")
    # list numbering ("1. ", "**2. ", "3) ", "#4 ") is not a claim
    text = re.sub(r"(?m)^\s*(?:[-*>#|]\s*)*\**#?\d+[.)]\**\s", "", text)
    # standalone numbers only: the 4 in "Neo4j" or the 3 in "III3" is part of a word
    return re.findall(r"(?<![A-Za-z\d.])\d+(?:\.\d+)?(?![A-Za-z\d])", text)


def numbers_not_in_data(answer: str, data: str) -> list[str]:
    """Numbers stated in the answer that appear nowhere in the tool output — candidates for fabrication."""
    have, out = {float(n) for n in _number_tokens(data)}, []
    for n in _number_tokens(answer):
        if float(n) not in have and n not in out:
            out.append(n)
    return out


def _with_number_screen(question: str, tool_output: str, answer: str) -> str:
    answer  = strip_badge(answer)
    missing = numbers_not_in_data(answer, tool_output)
    screen  = ", ".join(missing) if missing else "(none — every number in the ANSWER appears in the DATA)"
    return (f"QUESTION:\n{question}\n\nDATA:\n{tool_output}\n\nANSWER:\n{answer}\n\n"
            f"NUMBERS NOT FOUND IN THE DATA:\n{screen}")



def check_faithfulness(question: str, tool_output: str, answer: str, llm) -> dict:
    """
    Returns {"verdict": "pass" | "fail" | None, "unsupported": [claim, ...], "reason": str}.
    verdict None means the grader itself failed (bad JSON, model error) — the caller decides what to do.
    """
    try:
        response = llm.invoke([SystemMessage(content=FAITHFULNESS_PROMPT),
                               HumanMessage(content=_with_number_screen(question, tool_output, answer))])
        parsed  = parse_json_reply(response.text)
        verdict = str(parsed.get("verdict", "")).strip().lower()
        if verdict not in ("pass", "fail"):
            raise ValueError(f"bad verdict: {verdict!r}")
        unsupported = [str(c.get("claim", "")).strip() for c in parsed.get("claims", []) or []
                       if isinstance(c, dict) and str(c.get("status", "")).lower() == "unsupported" and c.get("claim")]
        return {"verdict": verdict, "unsupported": unsupported, "reason": str(parsed.get("reason", ""))}
    except Exception as e:
        return {"verdict": None, "unsupported": [], "reason": "", "error": f"{type(e).__name__}: {e}"}
