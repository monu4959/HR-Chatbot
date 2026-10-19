"""
tools/counts.py

value_counts with a deterministic tie-break: count first, then label alphabetically.

pandas' own order for equal counts comes from numpy's sort, which uses CPU-specific SIMD code — the same
data gave a different "top 10 job titles" on GitHub's runners than on a laptop. Every ranking in the tools
and the gold answers goes through here, so answers and test cases are identical on every machine.
"""

import pandas as pd


def ranked_counts(values: pd.Series | pd.DataFrame, ascending: bool = False) -> pd.Series:
    counts = values.value_counts()
    return counts.sort_index(kind="stable").sort_values(ascending=ascending, kind="stable")
