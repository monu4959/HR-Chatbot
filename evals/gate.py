"""
evals/gate.py

Regression gate for LLM eval runs: compares a run against evals/baseline.json using the limits
in evals/thresholds.json, and exits 1 if the run is worse. Used locally and by the self-hosted
GitHub workflow (.github/workflows/llm-eval.yml).

  python -m evals.gate --run fixed_qwen2.5-7b                    # check a saved run
  python -m evals.gate --run fixed_qwen2.5-7b --update-baseline  # accept it as the new baseline
  python -m evals.run --split all --gate                         # run + check in one go

What fails the gate:
  - answer pass rate or route accuracy below the absolute floors
  - any agent errors (crashes)
  - answer pass rate dropping more than max_drop_vs_baseline below the baseline
  - a route's pass rate dropping more than max_route_drop_vs_baseline (routes with >= min_route_n cases)
  - any failure in a must-pass check type (refusals: the agent must never make up data)
Cases that passed in the baseline and fail now are always listed, but only fail the gate
through the rates above — single cases flip run to run on a 49-case set.
"""

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR     = ROOT / "evals" / "results"
BASELINE_PATH   = ROOT / "evals" / "baseline.json"
THRESHOLDS_PATH = ROOT / "evals" / "thresholds.json"
_EPS            = 1e-9  # so a drop of exactly the tolerance (0.05 vs 0.0500000001) is allowed


def load_run(run_name: str) -> tuple[dict, list[dict]]:
    out_dir = RESULTS_DIR / run_name
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    cases   = [json.loads(l) for l in (out_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return summary, cases


def make_baseline(summary: dict, cases: list[dict]) -> dict:
    return {
        "run_name":          summary["run_name"],
        "split":             summary["split"],
        "provider":          summary.get("provider"),
        "model_cheap":       summary.get("model_cheap"),
        "model_strong":      summary.get("model_strong"),
        "model_grader":      summary.get("model_grader"),
        "data_hash":         summary.get("data_hash"),
        "answer_pass_rate":  summary["answer_pass_rate"],
        "route_accuracy":    summary["route_accuracy"],
        "by_expected_route": summary["by_expected_route"],
        "cases":             {c["id"]: bool(c["answer_pass"]) for c in cases},
    }


def gate(summary: dict, cases: list[dict], baseline: dict | None, t: dict) -> tuple[list[str], list[str]]:
    """Returns (failures, notes). An empty failures list means the run passes."""
    failures, notes = [], []
    rate, route_acc = summary["answer_pass_rate"] or 0, summary["route_accuracy"] or 0

    if rate < t["min_answer_pass_rate"]:
        failures.append(f"answer pass rate {rate:.1%} < floor {t['min_answer_pass_rate']:.0%}")
    if route_acc < t["min_route_accuracy"]:
        failures.append(f"route accuracy {route_acc:.1%} < floor {t['min_route_accuracy']:.0%}")
    if summary["n_errors"] > t["max_errors"]:
        failures.append(f"{summary['n_errors']} agent error(s) (max {t['max_errors']})")

    for c in cases:
        if c["check"] in t["must_pass_checks"] and not c["answer_pass"]:
            failures.append(f"must-pass case failed: {c['id']} ({c['check']}) — {c.get('answer_detail') or c.get('error', '')}")

    # Second scorer (evals/judge.py): absolute floors, applied only when the run was judged
    judge = summary.get("judge")
    if judge:
        for key, floor in (("correct_rate", "min_judge_correct_rate"), ("faithful_rate", "min_judge_faithful_rate")):
            if floor in t and judge.get(key) is not None and judge[key] < t[floor]:
                failures.append(f"judge {key.replace('_rate', '')} rate {judge[key]:.1%} < floor {t[floor]:.0%}")
        if judge.get("disagree_with_keyword_checks"):
            notes.append(f"judge disagrees with the keyword checks on: {', '.join(judge['disagree_with_keyword_checks'])}")

    if baseline is None:
        notes.append("no baseline yet — only absolute floors were checked (use --update-baseline to set one)")
        return failures, notes

    if baseline["split"] != summary["split"]:
        failures.append(f"split mismatch: run is '{summary['split']}', baseline is '{baseline['split']}'")
        return failures, notes
    for key in ("model_cheap", "model_strong", "data_hash"):
        if baseline.get(key) != summary.get(key):
            notes.append(f"{key} differs from baseline ({baseline.get(key)} -> {summary.get(key)})")

    drop = baseline["answer_pass_rate"] - rate
    if drop > t["max_drop_vs_baseline"] + _EPS:
        failures.append(f"answer pass rate fell {drop:.1%} vs baseline ({baseline['answer_pass_rate']:.1%} -> {rate:.1%})")

    for route, base in baseline["by_expected_route"].items():
        now = summary["by_expected_route"].get(route)
        if not now or base["n"] < t["min_route_n"] or base["answer_pass"] is None:
            continue
        r_drop = base["answer_pass"] - (now["answer_pass"] or 0)
        if r_drop > t["max_route_drop_vs_baseline"] + _EPS:
            name = "out-of-scope (no tool expected)" if route in ("null", "None") else route
            failures.append(f"route '{name}' fell {r_drop:.1%} ({base['answer_pass']:.1%} -> {now['answer_pass']:.1%})")

    regressed = [c["id"] for c in cases if baseline["cases"].get(c["id"]) and not c["answer_pass"]]
    fixed     = [c["id"] for c in cases if baseline["cases"].get(c["id"]) is False and c["answer_pass"]]
    if regressed:
        notes.append(f"{len(regressed)} case(s) passed in baseline but fail now: {', '.join(regressed)}")
    if fixed:
        notes.append(f"{len(fixed)} case(s) newly passing: {', '.join(fixed)}")
    return failures, notes


def check_run(run_name: str, update_baseline: bool = False) -> bool:
    summary, cases = load_run(run_name)
    t        = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8")) if BASELINE_PATH.exists() else None

    failures, notes = gate(summary, cases, baseline, t)
    base_desc = f"{baseline['run_name']} ({baseline['answer_pass_rate']:.1%})" if baseline else "none"
    print(f"\nGate: run {run_name} ({summary['answer_pass_rate']:.1%}) vs baseline {base_desc}")
    for n in notes:
        print(f"  note: {n}")
    for f in failures:
        print(f"  FAIL: {f}")
    print("  RESULT: " + ("PASS" if not failures else "FAIL"))

    if update_baseline:
        if failures and baseline is not None:
            print("  Not updating the baseline: the run fails the gate. Fix it, or delete evals/baseline.json to reset.")
        else:
            BASELINE_PATH.write_text(json.dumps(make_baseline(summary, cases), indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  Baseline updated -> {BASELINE_PATH}")
    return not failures


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name under evals/results/")
    ap.add_argument("--update-baseline", action="store_true", help="save this run as the new baseline if it passes")
    args = ap.parse_args()
    sys.exit(0 if check_run(args.run, args.update_baseline) else 1)


if __name__ == "__main__":
    main()
