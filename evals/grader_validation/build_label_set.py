"""
evals/grader_validation/build_label_set.py

Builds the set of answers a human labels to validate the graders. No model calls: it reuses
answers already produced by eval runs, plus deliberately broken copies of correct answers
(a changed number, or an invented sentence) so the set contains known failures.

  python -m evals.grader_validation.build_label_set

Writes evals/grader_validation/items.jsonl. Labels live separately in labels.jsonl, so
rebuilding items never touches labels (item ids are stable).
"""

import re
import sys
import json
import random
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evals.run import load_cases, RESULTS_DIR, DATASET_DIR
from evals.graders import reference_text, strip_badge

OUT        = Path(__file__).resolve().parent / "items.jsonl"
RUNS       = ["baseline_qwen2.5-7b", "baseline_llama3.2-3b", "fixed_qwen2.5-7b", "fixed_llama3.2-3b"]
N_REAL     = 80
N_PERTURB  = 20
SEED       = 7

_INVENTED_CLAIMS = [
    "Notably, the VP of Sales received the most recognitions overall, with 57 awards.",
    "Most of these recognitions were given in the fourth quarter of 2024.",
    "This pattern is strongest in the Berlin office, which accounts for 38% of them.",
    "Overall, 72% of these were peer-to-peer recognitions between engineers.",
]


def _calibration_or_holdout(item_id: str) -> str:
    """Half the labels are for tuning grader prompts, half are held out for reporting."""
    return "holdout" if int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % 2 else "calibration"


def _real_items(cases: dict, facts: dict) -> list[dict]:
    items, seen = [], set()
    for run in RUNS:
        path = RESULTS_DIR / run / "cases.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if "error" in r or r["id"] not in cases or "CHART_DATA:" in r.get("answer", ""):
                continue
            key = (r["id"], re.sub(r"\s+", " ", strip_badge(r["answer"])).strip().lower())
            if key in seen:
                continue
            seen.add(key)
            case = cases[r["id"]]
            items.append({
                "item_id":       f"{run}:{r['id']}",
                "case_id":       r["id"],
                "question":      case["question"],
                "history":       case.get("history", []),
                "tool_output":   r.get("tool_output", ""),
                "tool_output_truncated": len(r.get("tool_output", "")) >= 4000,
                "answer":        r["answer"],
                "reference":     reference_text(case, facts),
                "source":        run,
                "route_used":    r.get("route_used"),
                "perturbation":  None,
                # what the existing automatic checks said (hidden from the labeller)
                "keyword_check": r["answer_pass"],
                "old_grader_score_in_run": r.get("hallucination_score"),
            })
    return items


def _perturb(item: dict, rng: random.Random, cases: dict) -> dict | None:
    case   = cases[item["case_id"]]
    answer = item["answer"]
    gold   = case.get("gold", {})
    value  = gold.get("value")

    if case["check"] in ("numeric", "top_label") and value is not None:
        token = str(int(value)) if float(value).is_integer() else str(value)
        if re.search(rf"\b{re.escape(token)}\b", answer):
            wrong = str(int(float(token)) + rng.choice([7, 13, 21]))
            new = re.sub(rf"\b{re.escape(token)}\b", wrong, answer, count=1)
            return {**item, "item_id": f"perturb-number:{item['case_id']}:{item['source']}",
                    "answer": new, "perturbation": f"changed {token} -> {wrong}"}

    if case["check"] in ("rubric", "numeric_set", "numeric", "top_label"):
        claim = rng.choice(_INVENTED_CLAIMS)
        return {**item, "item_id": f"perturb-claim:{item['case_id']}:{item['source']}",
                "answer": answer.rstrip() + "\n\n" + claim, "perturbation": f"added: {claim}"}
    return None


def main():
    cases = {c["id"]: c for c in load_cases("all", None, None)}
    facts = json.loads((DATASET_DIR / "gold_facts.json").read_text(encoding="utf-8"))
    rng   = random.Random(SEED)

    real = _real_items(cases, facts)
    # Keep every answer the keyword checks failed (they're the interesting ones), fill up with passes
    fails  = [i for i in real if not i["keyword_check"]]
    passes = [i for i in real if i["keyword_check"]]
    rng.shuffle(fails); rng.shuffle(passes)
    chosen = (fails + passes)[:N_REAL] if len(fails) < N_REAL else fails[:N_REAL]

    # Perturb answers that were correct, so the failure is known by construction
    pool, perturbed = [i for i in passes if i not in chosen] + [i for i in chosen if i["keyword_check"]], []
    rng.shuffle(pool)
    for item in pool:
        if len(perturbed) >= N_PERTURB:
            break
        p = _perturb(item, rng, cases)
        if p and p["item_id"] not in {x["item_id"] for x in perturbed}:
            perturbed.append(p)

    items = chosen + perturbed
    rng.shuffle(items)
    for it in items:
        it["split"] = _calibration_or_holdout(it["item_id"])

    OUT.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    n_hold = sum(i["split"] == "holdout" for i in items)
    print(f"Wrote {len(items)} items ({len(chosen)} real: {sum(not i['keyword_check'] for i in chosen)} "
          f"keyword-fail / {sum(bool(i['keyword_check']) for i in chosen)} keyword-pass; {len(perturbed)} perturbed) "
          f"| calibration {len(items) - n_hold} / holdout {n_hold} -> {OUT}")


if __name__ == "__main__":
    main()
