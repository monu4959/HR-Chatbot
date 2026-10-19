"""
evals/grader_validation/validate_grader.py

Measures how well each automatic grader agrees with the human labels in labels.jsonl.

  python -m evals.grader_validation.validate_grader                  # all graders, all labelled items
  python -m evals.grader_validation.validate_grader --split holdout  # report numbers from here
  python -m evals.grader_validation.validate_grader --no-llm         # keyword checks only (instant)

Graders compared (each against the human label it is meant to predict):
  keyword_checks  evals/checks.py                       -> human "correct"
  old_grader      the original 0–1 hallucination check  -> human "faithful" (and "correct": it gated refusals)
  faithfulness    evals/graders.py                      -> human "faithful"
  correctness     evals/graders.py                      -> human "correct"

LLM outputs are cached in grader_outputs.jsonl (keyed by grader, model, prompt and item), so re-running
after labelling more items only grades the new ones. Writes report.json.
"""

import re
import sys
import json
import hashlib
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm import get_llm, model_name, parse_json_reply
from agent.prompt_engine import HALLUCINATION_CHECK_PROMPT, SOFT_WARNING_THRESHOLD
from evals.checks import check_answer
from evals.graders import (FAITHFULNESS_PROMPT, CORRECTNESS_PROMPT, FAITHFULNESS_PROMPT_V1, CORRECTNESS_PROMPT_V1,
                           FAITHFULNESS_PROMPT_V2, CORRECTNESS_PROMPT_V2,
                           grade_faithfulness, grade_correctness, grade_faithfulness_v1, grade_correctness_v1,
                           grade_faithfulness_v2, grade_correctness_v2)
from evals.grader_validation.metrics import agreement
from evals.run import load_cases, DATASET_DIR

HERE        = Path(__file__).resolve().parent
ITEMS_PATH  = HERE / "items.jsonl"
LABELS_PATH = HERE / "labels.jsonl"
CACHE_PATH  = HERE / "grader_outputs.jsonl"
REPORT_PATH = HERE / "report.json"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def _old_grader(item: dict) -> dict:
    """The original runtime check (agent/graph.py before step 3), re-run with the current grader model."""
    if item.get("route_used") == "llm_synthesis":
        return {"verdict": "pass", "score": 0.8, "note": "runtime skipped llm_synthesis"}
    if not item["tool_output"] or item["tool_output"].lstrip().startswith("CANNOT_ANSWER"):
        return {"verdict": "pass", "score": 1.0, "note": "runtime skipped (no grounding needed)"}
    llm = get_llm("judge", json_mode=True)
    try:
        resp  = llm.invoke([SystemMessage(content=HALLUCINATION_CHECK_PROMPT),
                            HumanMessage(content=f"SOURCE CONTEXT:\n{item['tool_output'][:20000]}\n\nAI ANSWER:\n{item['answer']}")])
        score = float(parse_json_reply(resp.text).get("score", 1.0))
        return {"verdict": "pass" if score >= SOFT_WARNING_THRESHOLD else "fail", "score": score}
    except Exception as e:
        # The runtime treated a grader failure as a pass — keep that behaviour, but record it
        return {"verdict": "pass", "score": 1.0, "error": f"{type(e).__name__}: {e}"}


LLM_GRADERS = {
    "old_grader":   (HALLUCINATION_CHECK_PROMPT, _old_grader),
    "faithfulness": (FAITHFULNESS_PROMPT, lambda it: grade_faithfulness(it["question"], it["tool_output"], it["answer"])),
    "correctness":  (CORRECTNESS_PROMPT,  lambda it: grade_correctness(it["question"], it["reference"], it["answer"])),
    "faithfulness_v1": (FAITHFULNESS_PROMPT_V1, lambda it: grade_faithfulness_v1(it["question"], it["tool_output"], it["answer"])),
    "correctness_v1":  (CORRECTNESS_PROMPT_V1,  lambda it: grade_correctness_v1(it["question"], it["reference"], it["answer"])),
    "faithfulness_v2": (FAITHFULNESS_PROMPT_V2, lambda it: grade_faithfulness_v2(it["question"], it["tool_output"], it["answer"])),
    "correctness_v2":  (CORRECTNESS_PROMPT_V2,  lambda it: grade_correctness_v2(it["question"], it["reference"], it["answer"])),
}

# Which human label each grader is meant to predict
TARGETS = {
    "keyword_checks": ["correct"],
    "old_grader":     ["faithful", "correct"],
    "faithfulness":   ["faithful"],
    "correctness":    ["correct"],
    "faithfulness_v1": ["faithful"],
    "correctness_v1":  ["correct"],
    "faithfulness_v2": ["faithful"],
    "correctness_v2":  ["correct"],
}


def _cache_key(grader: str, prompt: str, item_id: str) -> str:
    p = hashlib.sha256(prompt.encode()).hexdigest()[:10]
    return f"{grader}|{model_name('judge')}|{p}|{item_id}"


def run_llm_graders(items: list[dict], graders: list[str], workers: int = 1) -> dict:
    """workers > 1 grades in parallel threads — useful for API providers, pointless for one local GPU."""
    cache = {c["key"]: c for c in _jsonl(CACHE_PATH)}
    todo  = [(g, it) for g in graders for it in items if _cache_key(g, LLM_GRADERS[g][0], it["item_id"]) not in cache]
    if todo:
        print(f"Grading {len(todo)} new (grader, item) pairs with {model_name('judge')} ({workers} worker(s))...")
    lock = threading.Lock()

    def grade(job):
        g, it = job
        return g, it, LLM_GRADERS[g][1](it)

    with open(CACHE_PATH, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (g, it, out) in enumerate(pool.map(grade, todo), 1):
            rec = {"key": _cache_key(g, LLM_GRADERS[g][0], it["item_id"]), "grader": g, "item_id": it["item_id"], **out}
            with lock:
                cache[rec["key"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")
    return {(g, it["item_id"]): cache[_cache_key(g, LLM_GRADERS[g][0], it["item_id"])]
            for g in graders for it in items}


def keyword_verdicts(items: list[dict]) -> dict:
    """Stored check result for real answers; re-checked for perturbed ones (their text changed)."""
    cases = {c["id"]: c for c in load_cases("all", None, None)}
    facts = json.loads((DATASET_DIR / "gold_facts.json").read_text(encoding="utf-8"))
    out = {}
    for it in items:
        ok = it["keyword_check"] if not it["perturbation"] else \
            check_answer(cases[it["case_id"]], it["answer"], False, facts)[0]
        out[("keyword_checks", it["item_id"])] = {"verdict": "pass" if ok else "fail"}
    return out


def label_provenance(labels: dict) -> dict:
    """How the labels were made, and how well Claude's drafts agree with the human where the human was blind."""
    drafts = {d["item_id"]: d for d in _jsonl(HERE / "drafts.jsonl")}
    blind  = [l for l in labels.values() if l.get("mode", "blind") == "blind"]
    review = [l for l in labels.values() if l.get("mode") == "reviewed"]
    vs = {}
    for field in ("faithful", "correct"):
        pairs = [(l[field] == "yes", drafts[l["item_id"]][field] == "yes") for l in blind
                 if l["item_id"] in drafts and "unsure" not in (l[field], drafts[l["item_id"]][field])]
        vs[field] = agreement(pairs)
    return {"blind": len(blind), "reviewed": len(review),
            "reviewed_changed": sum(bool(l.get("changed")) for l in review),
            "draft_vs_blind_human": vs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="all", choices=["calibration", "holdout", "all"])
    ap.add_argument("--graders", default="old_grader,faithfulness_v1,correctness_v1,faithfulness_v2,correctness_v2,faithfulness,correctness")
    ap.add_argument("--no-llm", action="store_true", help="only score the keyword checks")
    ap.add_argument("--workers", type=int, default=1, help="parallel grading threads (API providers only)")
    ap.add_argument("--blind-only", action="store_true",
                    help="use only labels made without a Claude suggestion (robustness check against anchoring)")
    args = ap.parse_args()

    items  = {it["item_id"]: it for it in _jsonl(ITEMS_PATH)}
    labels = {l["item_id"]: l for l in _jsonl(LABELS_PATH) if l["item_id"] in items}
    if not labels:
        sys.exit("No labels yet — run `streamlit run evals/grader_validation/label_app.py` first.")
    if args.blind_only:
        labels = {i: l for i, l in labels.items() if l.get("mode", "blind") == "blind"}
    labelled = [items[i] for i in labels if args.split == "all" or items[i]["split"] == args.split]

    graders  = [] if args.no_llm else [g.strip() for g in args.graders.split(",") if g.strip()]
    verdicts = keyword_verdicts(labelled) | (run_llm_graders(labelled, graders, args.workers) if graders else {})

    report = {"split": args.split, "grader_model": model_name("judge"), "n_labelled": len(labelled), "results": {}}
    print(f"\n{len(labelled)} labelled items (split: {args.split}), grader model: {model_name('judge')}\n")
    print(f"{'grader':<17}{'vs human':<10}{'n':>4}{'acc':>7}{'kappa':>7}{'catch bad':>11}{'pass good':>11}{'errors':>8}")
    for g in ["keyword_checks"] + graders:
        for target in TARGETS[g]:
            pairs, errors = [], 0
            for it in labelled:
                human = labels[it["item_id"]][target]
                v = verdicts[(g, it["item_id"])]
                if human == "unsure":
                    continue
                if v.get("verdict") is None or v.get("error"):
                    errors += 1
                    if v.get("verdict") is None:
                        continue
                pairs.append((human == "yes", v["verdict"] == "pass"))
            m = agreement(pairs)
            m["errors"] = errors
            report["results"][f"{g} vs {target}"] = m
            if m["n"]:
                print(f"{g:<17}{target:<10}{m['n']:>4}{m['accuracy']:>7}{m['kappa']:>7}"
                      f"{str(m['fail_recall']):>11}{str(m['pass_recall']):>11}{errors:>8}")

    report["labels"] = label_provenance(labels)
    print(f"\nLabels: {report['labels']['blind']} blind, {report['labels']['reviewed']} reviewed Claude drafts "
          f"({report['labels']['reviewed_changed']} changed by you)")
    for field, m in report["labels"]["draft_vs_blind_human"].items():
        if m.get("n"):
            print(f"  Claude drafts vs your blind labels — {field}: n={m['n']} acc={m['accuracy']} kappa={m['kappa']}")

    perturbed = [it for it in labelled if it["perturbation"]]
    if perturbed:
        caught = sum(labels[it["item_id"]]["faithful"] == "no" for it in perturbed)
        report["perturbation_sanity"] = {"n": len(perturbed), "human_marked_unfaithful": caught}
        print(f"\nSanity: you marked {caught}/{len(perturbed)} deliberately broken answers as unfaithful.")

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n'catch bad' = share of answers you judged bad that the grader failed; "
          f"'pass good' = share you judged good that it passed.\nReport: {REPORT_PATH}")


if __name__ == "__main__":
    main()
