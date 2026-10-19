"""
evals/checks.py

Deterministic checks for eval cases — no LLM calls, so they're cheap and stable.
Each check is applied to the final answer, and (where it makes sense) to the raw
tool output too, so a failure can be blamed on the tool or on synthesis.

Check types (case["check"]):
  numeric      gold.value appears in the text (gold.tolerance for floats)
  numeric_set  every number in gold.values appears
  top_label    gold.label appears (graph flows: gold.from and gold.to both appear)
  chart        CHART_DATA matches gold.chart_type and data (or gold.data_from: categories|subcategories)
  refusal      the answer declines instead of making something up
  rubric       must_include_any / must_not_include on the answer, tool_must_include_any on the tool output
"""

import re
import json

REFUSAL_PHRASES = [
    "cannot be answered", "can't be answered", "cannot answer", "can't answer", "cannot_answer",
    "not able to", "unable to", "outside the scope", "outside of the scope", "not available in",
    "does not contain", "doesn't contain", "does not include", "doesn't include", "not included in",
    "no information", "not captured", "not present in", "isn't available", "is not available",
    "not part of", "don't have", "do not have", "no data",
    "cannot provide", "can't provide", "does not specify", "doesn't specify", "not been recognized",
    "no recognition messages", "not found in",
]


def _numbers(text: str) -> list[float]:
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text or "")
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _has_number(text: str, value: float, tolerance: float = 0) -> bool:
    return any(abs(n - value) <= tolerance for n in _numbers(text))


def _contains(text: str, needle: str) -> bool:
    return needle.lower() in (text or "").lower()


def _parse_chart(text: str):
    if not text or "CHART_DATA:" not in text:
        return None
    try:
        return json.loads(text.split("CHART_DATA:", 1)[1].strip())
    except json.JSONDecodeError:
        return None


def is_refusal(text: str) -> bool:
    return any(p in (text or "").lower() for p in REFUSAL_PHRASES)


def _check_text(case: dict, text: str, facts: dict) -> tuple[bool, str]:
    check, gold = case["check"], case.get("gold", {})

    if check == "numeric":
        ok = _has_number(text, gold["value"], gold.get("tolerance", 0))
        return ok, "" if ok else f"expected {gold['value']}"

    if check == "numeric_set":
        missing = [k for k, v in gold["values"].items() if not _has_number(text, v)]
        return not missing, "" if not missing else f"missing values for: {', '.join(missing)}"

    if check == "top_label":
        labels = [gold["from"], gold["to"]] if "from" in gold else [gold["label"]]
        missing = [l for l in labels if not _contains(text, l)]
        return not missing, "" if not missing else f"missing label(s): {', '.join(missing)}"

    if check == "chart":
        chart = _parse_chart(text)
        if chart is None:
            return False, "no CHART_DATA"
        expected = gold.get("data") or facts[gold["data_from"]]
        got = {d["label"]: d["value"] for d in chart.get("data", [])}
        if chart.get("chart_type") != gold["chart_type"]:
            return False, f"chart_type {chart.get('chart_type')} != {gold['chart_type']}"
        return got == expected, "" if got == expected else "chart data differs from gold"

    if check == "rubric":
        if gold.get("must_include_any") and not any(_contains(text, w) for w in gold["must_include_any"]):
            return False, f"none of {gold['must_include_any']}"
        bad = [w for w in gold.get("must_not_include", []) if _contains(text, w)]
        return not bad, "" if not bad else f"contains forbidden: {bad}"

    raise ValueError(f"unknown check type: {check}")


def check_answer(case: dict, answer: str, was_refused: bool, facts: dict) -> tuple[bool, str]:
    if case["check"] == "refusal":
        ok = was_refused or is_refusal(answer)
        return ok, "" if ok else "answered instead of refusing"
    return _check_text(case, answer, facts)


def combine(case: dict, answer: tuple[bool, str], tool: tuple[bool | None, str]) -> tuple[bool, str]:
    """A rubric that only sets tool criteria has nothing to check in the answer itself — the answer can only
    be right if the tool returned what was needed."""
    gold = case.get("gold", {})
    if case["check"] == "rubric" and not gold.get("must_include_any") and tool[0] is False:
        return False, "tool output lacked what the answer needed: " + tool[1]
    return answer


def check_tool(case: dict, tool_output: str, facts: dict) -> tuple[bool | None, str]:
    """Same check on the raw tool output. None = not applicable for this case."""
    gold = case.get("gold", {})
    if case["check"] == "rubric":
        needles = gold.get("tool_must_include_any")
        if not needles:
            return None, ""
        ok = any(_contains(tool_output, w) for w in needles)
        return ok, "" if ok else f"tool output has none of {needles}"
    if case["check"] == "refusal" or not tool_output:
        return None, ""
    return _check_text(case, tool_output, facts)
