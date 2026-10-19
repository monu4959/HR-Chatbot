"""
evals/run.py

Runs the fixed test set through the full agent and scores it.

  python -m evals.run                     # dev split
  python -m evals.run --split test        # test split (seen during the tool fixes)
  python -m evals.run --split heldout     # frozen held-out set: run once, for final numbers ("all" = dev+test)
  python -m evals.run --ids pd-total,ch-cat-pie
  python -m evals.run --route pandas_query --limit 5
  python -m evals.run --fail-under 0.8    # exit 1 if answer pass rate < 80% (for CI)
  python -m evals.run --split all --gate  # exit 1 on regression vs evals/baseline.json (see evals/gate.py)
  python -m evals.run --rescore baseline_qwen2.5-7b   # re-apply current checks to a saved run, no model calls

Writes evals/results/<timestamp>_<split>/{cases.jsonl,summary.json}.
Every case is traced in LangSmith with tags ["eval", "eval:<run name>"] and metadata eval_case=<id>.
"""

import os
import sys
import json
import time
import uuid
import argparse
import statistics
import traceback
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # tools resolve ./data and ./chroma_db relative to the project root

from evals.build_gold import _split
from evals.checks import check_answer, check_tool, combine

DATASET_DIR = ROOT / "evals" / "dataset"
RESULTS_DIR = ROOT / "evals" / "results"


def load_cases(split: str, ids: set | None, route: str | None) -> list[dict]:
    cases = []
    for path in sorted(DATASET_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            c.setdefault("split", _split(c["id"]))
            c.setdefault("history", [])
            c.setdefault("difficulty", "medium")
            c["file"] = path.name
            cases.append(c)

    seen = set()
    for c in cases:
        if c["id"] in seen:
            raise ValueError(f"duplicate case id: {c['id']}")
        seen.add(c["id"])

    if ids:
        return [c for c in cases if c["id"] in ids]
    if split == "all":
        cases = [c for c in cases if c["split"] != "heldout"]  # held-out only ever runs explicitly
    else:
        cases = [c for c in cases if c["split"] == split]
    if route:
        cases = [c for c in cases if c.get("expected_route") == route]
    return cases


def run_case(case: dict, run_name: str, facts: dict) -> dict:
    from agent.graph import run_query, clear_session

    session = f"eval-{uuid.uuid4()}"
    trace = {"tags": ["eval", f"eval:{run_name}"], "metadata": {"eval_run": run_name, "eval_case": case["id"]}}
    rec = {"id": case["id"], "split": case["split"], "check": case["check"], "difficulty": case["difficulty"],
           "expected_route": case.get("expected_route"), "question": case["question"]}
    start = time.time()
    try:
        for turn in case["history"]:
            run_query(turn, session_id=session, **trace)
        r = run_query(case["question"], session_id=session, **trace)
    except Exception as e:
        rec.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(),
                   answer_pass=False, latency_s=round(time.time() - start, 2))
        return rec
    finally:
        clear_session(session)

    answer_ok, answer_detail = check_answer(case, r["answer"], r["was_refused"], facts)
    tool_ok, tool_detail     = check_tool(case, r.get("tool_output", ""), facts)
    answer_ok, answer_detail = combine(case, (answer_ok, answer_detail), (tool_ok, tool_detail))
    route_ok = None if not case.get("expected_route") else r["route_used"] == case["expected_route"]

    rec.update(
        route_used=r["route_used"], route_pass=route_ok,
        answer_pass=answer_ok, answer_detail=answer_detail,
        tool_pass=tool_ok, tool_detail=tool_detail,
        hallucination_score=r["hallucination_score"], was_refused=r["was_refused"],
        grader_error=r["grader_error"], trace_id=r["trace_id"],
        model_tier=r.get("model_tier"), escalated=r.get("escalated", False),
        latency_s=round(time.time() - start, 2),
        answer=r["answer"], tool_output=(r.get("tool_output") or "")[:20000],
    )
    return rec


def _rate(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def summarize(records: list[dict], meta: dict) -> dict:
    by = lambda key: {
        k: {"n": len(rs), "answer_pass": _rate(r["answer_pass"] for r in rs)}
        for k, rs in sorted(_group(records, key).items(), key=lambda kv: str(kv[0]))
    }
    ok = [r for r in records if "error" not in r]
    scores = [r["hallucination_score"] for r in ok if r.get("hallucination_score") is not None]
    lat = sorted(r["latency_s"] for r in records)
    return {
        **meta,
        "n_cases":            len(records),
        "n_errors":           len(records) - len(ok),
        "answer_pass_rate":   _rate(r["answer_pass"] for r in records),
        "tool_pass_rate":     _rate(r.get("tool_pass") for r in ok),
        "route_accuracy":     _rate(r.get("route_pass") for r in ok),
        "by_split":           by("split"),
        "by_check":           by("check"),
        "by_expected_route":  by("expected_route"),
        "by_difficulty":      by("difficulty"),
        "by_model_tier":      {k: {**v, "mean_latency_s": round(statistics.mean(r["latency_s"] for r in _group(records, "model_tier")[k]), 1)}
                               for k, v in by("model_tier").items()},
        "n_escalated":        sum(bool(r.get("escalated")) for r in ok),
        "grader": {
            "mean_score":      round(statistics.mean(scores), 3) if scores else None,
            "stdev_score":     round(statistics.pstdev(scores), 3) if scores else None,
            "distinct_scores": sorted(set(scores)),
            "n_grader_errors": sum(bool(r.get("grader_error")) for r in ok),
            "n_refused":       sum(bool(r.get("was_refused")) for r in ok),
        },
        "latency_s": {
            "mean": round(statistics.mean(lat), 1) if lat else None,
            "p95":  lat[int(0.95 * (len(lat) - 1))] if lat else None,
        },
    }


def rescore(run_name: str, facts: dict):
    """Re-apply the current checks to a saved run (tool_output is stored truncated to 20000 chars; runs before 2026-10-18 used 4000)."""
    out_dir = RESULTS_DIR / run_name
    cases   = {c["id"]: c for c in load_cases("all", None, None)}
    records = [json.loads(l) for l in (out_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    changed = []
    for rec in records:
        if "error" in rec or rec["id"] not in cases:
            continue
        before = rec["answer_pass"]
        rec["answer_pass"], rec["answer_detail"] = check_answer(cases[rec["id"]], rec["answer"], rec["was_refused"], facts)
        rec["tool_pass"], rec["tool_detail"] = check_tool(cases[rec["id"]], rec["tool_output"], facts)
        rec["answer_pass"], rec["answer_detail"] = combine(cases[rec["id"]], (rec["answer_pass"], rec["answer_detail"]),
                                                           (rec["tool_pass"], rec["tool_detail"]))
        if rec["answer_pass"] != before:
            changed.append(f"{rec['id']}: {_mark(before)} -> {_mark(rec['answer_pass'])}")
    old = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    meta = {k: old[k] for k in ("run_name", "split", "provider", "model_cheap", "model_strong",
                                "model_grader", "data_hash", "timestamp", "llm_cache", "model_router", "model_light") if k in old}
    summary = summarize(records, {**meta, "rescored_at": datetime.now().isoformat(timespec="seconds")})
    (out_dir / "cases.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{run_name}: answer pass rate {old['answer_pass_rate']} -> {summary['answer_pass_rate']}")
    for c in changed:
        print("  ", c)


def _group(records, key):
    groups = defaultdict(list)
    for r in records:
        groups[r.get(key)].append(r)
    return groups


def _mark(v):
    return {True: "PASS", False: "FAIL", None: " -- "}[v]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "heldout", "all"])
    ap.add_argument("--ids", help="comma-separated case ids (ignores --split/--route)")
    ap.add_argument("--route", help="only cases with this expected_route")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--name", help="run name (default: timestamp)")
    ap.add_argument("--fail-under", type=float, help="exit 1 if answer pass rate is below this")
    ap.add_argument("--no-llm-cache", action="store_true",
                    help="call the models fresh (default: reuse identical model calls from .cache/; the answer cache is always off in evals)")
    ap.add_argument("--gate", action="store_true", help="after the run, check it against evals/baseline.json (exit 1 on regression)")
    ap.add_argument("--rescore", metavar="RUN_NAME", help="re-apply current checks to a saved run and exit")
    ap.add_argument("--router", choices=["on", "off"], help="model router (agent/model_router.py); default: MODEL_ROUTER env, else on")
    ap.add_argument("--resume", action="store_true", help="keep finished cases from an interrupted run with the same --name")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the second scorer (evals/judge.py; JUDGE_PROVIDER/JUDGE_MODEL, Claude Haiku by default)")
    ap.add_argument("--judge-workers", type=int, default=8, help="parallel judge calls")
    args = ap.parse_args()

    if args.rescore:
        rescore(args.rescore, json.loads((DATASET_DIR / "gold_facts.json").read_text(encoding="utf-8")))
        return

    # Evals must exercise the pipeline: never serve whole answers from the answer cache.
    # The LLM cache (identical model calls) stays on unless --no-llm-cache.
    os.environ["ANSWER_CACHE"] = "off"
    if args.no_llm_cache:
        os.environ["LLM_CACHE"] = "off"
    if args.router:
        os.environ["MODEL_ROUTER"] = args.router

    from agent.llm import PROVIDER, model_name
    from agent.model_router import router_enabled
    facts = json.loads((DATASET_DIR / "gold_facts.json").read_text(encoding="utf-8"))

    cases = load_cases(args.split, set(args.ids.split(",")) if args.ids else None, args.route)
    if args.limit:
        cases = cases[:args.limit]
    stale = {c["id"] for c in cases if c.get("data_hash") not in (None, facts["data_hash"])}
    if stale:
        sys.exit(f"Gold is stale for {len(stale)} case(s) — re-run `python -m evals.build_gold`.")

    run_name = args.name or datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{args.split}"
    out_dir  = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.no_llm_cache:
        # A fresh run must not reuse profiles another run built (tools/profile_tool.py keeps them on disk);
        # set before agent.graph imports the profile tool.
        os.environ["PROFILE_CACHE_PATH"] = str(out_dir / "profile_cache.json")
    print(f"Run {run_name}: {len(cases)} cases | provider={PROVIDER} cheap={model_name('cheap')} "
          f"strong={model_name('strong')} light={model_name('light')} router={'on' if router_enabled() else 'off'}\n")

    records, done = [], {}
    cases_path = out_dir / "cases.jsonl"
    if args.resume and cases_path.exists():
        for line in cases_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if "error" not in rec:
                    done[rec["id"]] = rec
        print(f"Resuming: {len(done)} finished case(s) kept.\n")

    with open(cases_path, "w", encoding="utf-8") as f:
        for i, case in enumerate(cases, 1):
            rec = done.get(case["id"]) or run_case(case, run_name, facts)
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            detail = rec.get("error") or rec.get("answer_detail") or ""
            print(f"[{i:>3}/{len(cases)}] {_mark(rec['answer_pass'])}  route:{_mark(rec.get('route_pass'))}  "
                  f"tool:{_mark(rec.get('tool_pass'))}  {rec['latency_s']:>5.1f}s  {case['id']:<26} {detail[:70]}")

    if not args.no_judge:
        from evals.judge import judge_records
        print(f"\nJudging {len(records)} answers with {model_name('judge')}...")
        judge_records(records, {c["id"]: c for c in cases}, facts, workers=args.judge_workers)
        cases_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")

    summary = summarize(records, {
        "run_name": run_name, "split": args.split, "provider": PROVIDER,
        "model_cheap": model_name("cheap"), "model_strong": model_name("strong"), "model_grader": model_name("grader"),
        "data_hash": facts["data_hash"], "timestamp": datetime.now().isoformat(timespec="seconds"),
        "llm_cache": not args.no_llm_cache, "model_router": router_enabled(), "model_light": model_name("light"),
    })
    from evals.judge import judge_summary
    summary["judge"] = judge_summary(records, model_name("judge"))
    from agent.cache import cache_stats
    stats = cache_stats()
    summary["llm_cache_hits"], summary["llm_cache_misses"] = stats["llm_hits"], stats["llm_misses"]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers
        wait_for_all_tracers()
    except Exception:
        pass

    print(f"\nAnswer pass rate: {summary['answer_pass_rate']}   tool pass rate: {summary['tool_pass_rate']}   "
          f"route accuracy: {summary['route_accuracy']}   errors: {summary['n_errors']}")
    print(f"Grader: mean {summary['grader']['mean_score']}, stdev {summary['grader']['stdev_score']}, "
          f"distinct scores {summary['grader']['distinct_scores']}")
    print(f"Latency: mean {summary['latency_s']['mean']}s, p95 {summary['latency_s']['p95']}s")
    print("Model tiers: " + ", ".join(f"{k}: {v['n']} cases ({v['answer_pass']:.0%} pass, {v['mean_latency_s']}s avg)"
                                      for k, v in summary["by_model_tier"].items() if v["answer_pass"] is not None)
          + f" | escalated: {summary['n_escalated']}")
    j = summary.get("judge")
    if j:
        print(f"Judge ({j['model']}): correct {j['correct_rate']:.1%}, faithful {j['faithful_rate']:.1%}, "
              f"errors {j['n_errors']} | disagrees with keyword checks on: {', '.join(j['disagree_with_keyword_checks']) or 'none'}"
              + (f" | judged unfaithful: {', '.join(j['unfaithful'])}" if j["unfaithful"] else ""))
    print(f"Results: {out_dir}")

    if args.fail_under is not None and (summary["answer_pass_rate"] or 0) < args.fail_under:
        sys.exit(1)

    if args.gate:
        from evals.gate import check_run
        if not check_run(run_name):
            sys.exit(1)


if __name__ == "__main__":
    main()
