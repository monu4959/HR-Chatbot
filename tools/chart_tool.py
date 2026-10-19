import os
import re
import json
import pandas as pd
from langchain_core.tools import tool
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "./data/annotated_results.csv")

# (dimension, column, default chart type, default top-n, title, axis label)
# Order matters: "subcategor" must be checked before "categor", since one contains the other.
_DIMENSIONS = [
    ("subcategory", "subcategory_name", "horizontal_bar", None, "Recognition count by subcategory",            "Subcategory"),
    ("category",    "category_name",    "bar",            None, "Recognition count by category",               "Category"),
    ("nominator",   "nominator_title",  "horizontal_bar", 10,   "Top {n} job titles that give most recognitions", "Job title"),
    ("recipient",   "recipient_title",  "horizontal_bar", 10,   "Top {n} most recognized job titles",           "Job title"),
    ("award",       "award_title",      "horizontal_bar", 10,   "Top {n} award titles",                         "Award title"),
]

_DIMENSION_KEYWORDS = {
    "subcategory": ["subcategor"],
    "category":    ["categor"],
    "nominator":   ["nominator", "nominates", "gives", "giver"],
    "recipient":   ["recipient", "recognized", "received", "receives", "job title"],
    "award":       ["award title", "award name"],
}


@lru_cache(maxsize=1)
def _load_df():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    return df


def _chart_json(chart_type: str, title: str, x_label: str, y_label: str, data: list) -> str:
    """Returns a JSON string the frontend can detect and render as a chart."""
    return "CHART_DATA:" + json.dumps({
        "chart_type": chart_type,
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "data": data
    })


def _top_n(q: str):
    m = re.search(r"\btop\s+(\d+)\b", q)
    return int(m.group(1)) if m else None


@tool
def chart_generator(question: str) -> str:
    """
    Use this tool when the user asks to VISUALIZE, PLOT, CHART, or SHOW A GRAPH of data.
    Returns structured chart data that gets rendered as an interactive chart.

    Examples:
    - Show me a bar chart of awards by category
    - Plot the distribution of subcategories
    - Visualize which job titles get recognized most
    - Graph the breakdown of categories
    - Chart how many awards each subcategory has
    - Show me a pie chart of categories
    - Plot the top 10 recognized job titles
    - Visualize nominator job titles
    """
    df = _load_df()
    q  = question.lower()

    # 1. What to chart (falls back to categories)
    dim = next((d for d in _DIMENSIONS if any(k in q for k in _DIMENSION_KEYWORDS[d[0]])), _DIMENSIONS[1])
    name, column, default_type, default_n, title, label = dim

    # 2. How to draw it
    if "pie" in q or "donut" in q:
        chart_type = "pie"
    elif "horizontal" in q:
        chart_type = "horizontal_bar"
    else:
        chart_type = default_type

    n      = _top_n(q) or default_n
    counts = df[column].value_counts()
    if n:
        counts = counts.head(n)
    data = [{"label": k, "value": int(v)} for k, v in counts.items()]

    if chart_type == "pie":
        return _chart_json("pie", title.format(n=len(data)).replace("count by", "distribution by"),
                           label, "Count", data)
    if chart_type == "bar":
        return _chart_json("bar", title.format(n=len(data)), label, "Number of recognitions", data)
    return _chart_json("horizontal_bar", title.format(n=len(data)), "Number of recognitions", label, data)
