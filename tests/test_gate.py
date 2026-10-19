from evals.gate import gate, make_baseline

T = {"min_answer_pass_rate": 0.8, "min_route_accuracy": 0.9, "max_errors": 0, "max_drop_vs_baseline": 0.05,
     "max_route_drop_vs_baseline": 0.15, "min_route_n": 3, "must_pass_checks": ["refusal"]}


def _run(passes: list[bool], checks=None, route="pandas_query", errors=0, split="all"):
    checks = checks or ["numeric"] * len(passes)
    cases = [{"id": f"c{i}", "check": ch, "answer_pass": p, "answer_detail": ""} for i, (p, ch) in enumerate(zip(passes, checks))]
    rate = sum(passes) / len(passes)
    summary = {"run_name": "r", "split": split, "answer_pass_rate": rate, "route_accuracy": 1.0, "n_errors": errors,
               "by_expected_route": {route: {"n": len(passes), "answer_pass": rate}},
               "model_cheap": "m", "model_strong": "m", "data_hash": "h"}
    return summary, cases


def test_identical_run_passes():
    s, c = _run([True] * 9 + [False])
    assert gate(s, c, make_baseline(s, c), T)[0] == []


def test_drop_beyond_tolerance_fails_and_lists_regressed_cases():
    base = make_baseline(*_run([True] * 10))
    s, c = _run([True] * 8 + [False] * 2)
    failures, notes = gate(s, c, base, T)
    assert any("fell" in f for f in failures)
    assert any("c8" in n and "c9" in n for n in notes)


def test_small_drop_within_tolerance_passes_but_is_reported():
    base = make_baseline(*_run([True] * 20))
    s, c = _run([True] * 19 + [False])
    failures, notes = gate(s, c, base, T)
    assert failures == [] and any("c19" in n for n in notes)


def test_any_refusal_failure_fails_even_with_a_high_rate():
    s, c = _run([True] * 19 + [False], checks=["numeric"] * 19 + ["refusal"])
    assert any("must-pass" in f for f in gate(s, c, None, T)[0])


def test_floors_and_errors_apply_without_a_baseline():
    s, c = _run([True] * 7 + [False] * 3, errors=1)
    failures, notes = gate(s, c, None, T)
    assert any("floor" in f for f in failures) and any("error" in f for f in failures)
    assert any("no baseline" in n for n in notes)


def test_split_mismatch_is_refused():
    base = make_baseline(*_run([True] * 10, split="dev"))
    s, c = _run([True] * 10, split="all")
    assert any("split mismatch" in f for f in gate(s, c, base, T)[0])


def test_judge_floors_apply_only_when_the_run_was_judged():
    t = {**T, "min_judge_correct_rate": 0.85, "min_judge_faithful_rate": 0.85}
    s, c = _run([True] * 10)
    assert gate(s, c, None, t)[0] == []                       # no judge block → nothing to check
    s["judge"] = {"correct_rate": 0.8, "faithful_rate": 0.9, "disagree_with_keyword_checks": ["c3"]}
    failures, notes = gate(s, c, None, t)
    assert any("judge correct rate" in f for f in failures) and not any("faithful" in f for f in failures)
    assert any("c3" in n for n in notes)
