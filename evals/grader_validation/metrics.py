"""Agreement metrics between a grader and human labels. "fail" is the positive class (catching bad answers)."""


def agreement(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs = [(human_pass, grader_pass), ...]"""
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    tp = sum(not h and not g for h, g in pairs)   # bad answer, grader failed it
    tn = sum(h and g for h, g in pairs)           # good answer, grader passed it
    fp = sum(h and not g for h, g in pairs)       # good answer wrongly failed
    fn = sum(not h and g for h, g in pairs)       # bad answer let through

    p_o = (tp + tn) / n
    h_pass, g_pass = (tn + fp) / n, (tn + fn) / n
    p_e = h_pass * g_pass + (1 - h_pass) * (1 - g_pass)
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0

    ratio = lambda a, b: round(a / b, 3) if b else None
    return {
        "n":              n,
        "accuracy":       round(p_o, 3),
        "kappa":          round(kappa, 3),
        "fail_recall":    ratio(tp, tp + fn),   # share of bad answers the grader catches (TPR)
        "pass_recall":    ratio(tn, tn + fp),   # share of good answers the grader lets through (TNR)
        "fail_precision": ratio(tp, tp + fp),   # when the grader says fail, how often it's right
        "confusion":      {"caught_bad": tp, "passed_good": tn, "failed_good": fp, "missed_bad": fn},
    }
