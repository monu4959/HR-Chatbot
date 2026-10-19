"""
tools/graph_tool.py

Neo4j AuraDB graph queries with NetworkX fallback.
Neo4j graph schema (from neo4j_loader.py):
  Nodes: RoleGroup {name, total_received, top_category}
         Role      {title, group}
         Category  {name}
         Subcategory {name, category}
  Relationships:
    (RoleGroup)-[:RECOGNIZED {count, top_category}]->(RoleGroup)
    (Role)-[:RECEIVED {award_title, category, subcategory}]->(Role)
"""

import os
import re
import json
import pandas as pd
import networkx as nx
from collections import Counter
from functools import lru_cache
from langchain_core.tools import tool
from dotenv import load_dotenv

from tools.counts import ranked_counts

load_dotenv()

DATA_PATH        = os.getenv("DATA_PATH",        "./data/annotated_results.csv")
NEO4J_URI        = os.getenv("NEO4J_URI",        "")
NEO4J_USERNAME   = os.getenv("NEO4J_USERNAME",   os.getenv("NEO4J_USER", "neo4j"))
NEO4J_PASSWORD   = os.getenv("NEO4J_PASSWORD",   "")

_driver        = None
_neo4j_failed  = False


def _get_driver():
    global _driver, _neo4j_failed
    if _neo4j_failed:
        return None
    if _driver is not None:
        return _driver
    if not NEO4J_URI or not NEO4J_PASSWORD:
        return None
    try:
        from neo4j import GraphDatabase
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
            connection_timeout=5,
        )
        _driver.verify_connectivity()
        print("[graph_tool] Neo4j AuraDB connected.")
        return _driver
    except Exception as e:
        print(f"[graph_tool] Neo4j unavailable: {e} -- switching to NetworkX fallback.")
        _neo4j_failed = True
        _driver = None
        return None


def _run_cypher(cypher: str, params: dict = {}) -> list:
    driver = _get_driver()
    if not driver:
        return []
    with driver.session() as session:
        return [dict(r) for r in session.run(cypher, **params)]


_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _extract_n(query: str, default: int = 10) -> int:
    m = re.search(r"\btop\s+(\d+)\b", query) or re.search(r"\b(\d+)\s+(?:most|strongest|top|biggest)\b", query)
    if m:
        return int(m.group(1))
    for w, n in _NUMBER_WORDS.items():
        # word boundary: "anyone" / "often" are not numbers, and neither is the "two" in "two-way"
        if re.search(rf"\b{w}\b(?![-\s]way)", query):
            return n
    return default


# ── NetworkX fallback ─────────────────────────────────────────────────────────

def classify_role(title: str) -> str:
    if not title or pd.isna(title):
        return "Other"
    t = title.lower()
    if any(k in t for k in ["customer service", "customer success", "helpdesk"]):
        return "Customer Service"
    if any(k in t for k in ["engineer", "software", "qa", "technical", "devops", "architect"]):
        return "Engineering"
    if any(k in t for k in ["design", "creative", "ux", "ui", "copywriter", "content"]):
        return "Design/Creative"
    if any(k in t for k in ["account", "finance", "payroll", "buyer", "procurement", "financial"]):
        return "Finance/Admin"
    if any(k in t for k in ["director", "vp", "head of", "chief", "president", "senior director"]):
        return "Leadership"
    if any(k in t for k in ["hr", "hx", "people", "talent", "recruit"]):
        return "People/HR"
    if any(k in t for k in ["market", "brand", "media", "pr ", "communications"]):
        return "Marketing"
    if any(k in t for k in ["product", "program", "project"]):
        return "Product/Program"
    if any(k in t for k in ["data", "analy", "insight", "research"]):
        return "Data/Analytics"
    if any(k in t for k in ["sales", "business develop", "revenue"]):
        return "Sales/BD"
    return "Other"


@lru_cache(maxsize=1)
def _load_edges() -> pd.DataFrame:
    """One row per recognition: nominator/recipient title and their role groups (rows with a missing title dropped)."""
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["nominator_title", "recipient_title"])
    return pd.DataFrame({
        "nom":      df["nominator_title"].str.strip(),
        "rec":      df["recipient_title"].str.strip(),
        "nom_g":    df["nominator_title"].map(classify_role),
        "rec_g":    df["recipient_title"].map(classify_role),
        "category": df["category_name"].str.strip(),
    })


@lru_cache(maxsize=1)
def _build_group_graph() -> nx.DiGraph:
    """Role-group graph (mirrors Neo4j RoleGroup-[:RECOGNIZED]->RoleGroup): weight = count, top_category = most common."""
    G = nx.DiGraph()
    for (u, v), grp in _load_edges().groupby(["nom_g", "rec_g"]):
        G.add_edge(u, v, weight=len(grp), top_category=Counter(grp["category"]).most_common(1)[0][0])
    return G


@lru_cache(maxsize=1)
def _build_title_graph() -> nx.DiGraph:
    """Job-title graph for centrality / bridge questions; each node carries its role group."""
    G = nx.DiGraph()
    for row in _load_edges().itertuples():
        G.add_node(row.nom, group=row.nom_g)
        G.add_node(row.rec, group=row.rec_g)
        if G.has_edge(row.nom, row.rec):
            G[row.nom][row.rec]["weight"] += 1
        else:
            G.add_edge(row.nom, row.rec, weight=1)
    return G


def _nx_top_relationships(n: int = 10) -> list:
    G = _build_group_graph()
    edges = [(u, v, d["weight"], d["top_category"]) for u, v, d in G.edges(data=True)]
    return sorted(edges, key=lambda x: x[2], reverse=True)[:n]


def _nx_cross_team(n: int = 10) -> list:
    return [e for e in _nx_top_relationships(10_000) if e[0] != e[1]][:n]


def _nx_mutual() -> list:
    G = _build_group_graph()
    pairs = []
    for u, v, d in G.edges(data=True):
        if u < v and G.has_edge(v, u):
            pairs.append((u, v, d["weight"], G[v][u]["weight"]))
    return sorted(pairs, key=lambda p: p[2] + p[3], reverse=True)


def _nx_most_central(n: int = 10) -> list:
    G = _build_title_graph()
    centrality = nx.degree_centrality(G)
    return sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:n]


def _nx_bridges(n: int = 10) -> list:
    G = _build_title_graph()
    btw = nx.betweenness_centrality(G.to_undirected())
    return sorted(btw.items(), key=lambda x: x[1], reverse=True)[:n]


def _nx_totals(direction: str, n: int = 10) -> list:
    edges = _load_edges()
    col = "nom_g" if direction == "given" else "rec_g"
    return list(ranked_counts(edges[col]).head(n).items())


# ── Main tool ─────────────────────────────────────────────────────────────────

_FLOW_WORDS     = ["strongest", "between", "each other", "flow", "cross-department", "cross department",
                   "department", "relationship", "whom", "pairs of teams", "team to team"]
_MUTUAL_WORDS   = ["mutual", "bidirectional", "each other", "both ways", "reciprocal", "reciprocate",
                   "two-way", "two way", "back and forth", "both directions", "in return"]
_CENTRAL_WORDS  = ["central", "influential", "hub", "important", "key player", "connected"]
_BRIDGE_WORDS   = ["bridge", "span", "connector", "cross-team", "cross team", "broker"]
# "gives (out) the most", "nominates the most people", "receives the most recognition", "gets recognized most"
_GIVER_RE    = re.compile(r"\b(top giver|most active giver|who gives)\b|\b(give|gives|giving|gave|nominate|nominates|send|sends)\b"
                          r"(?:\s+out)?[^.?!]{0,30}\bmost\b|\bmost\b[^.?!]{0,20}\b(givers?|nominators?)\b")
_RECEIVER_RE = re.compile(r"\b(top receiver|most recogni[sz]ed|who receives)\b|\b(receive|receives|received|get|gets)\b"
                          r"[^.?!]{0,30}\bmost\b|\brecogni[sz]ed\s+(the\s+)?most\b")


@tool
def graph_query(question: str) -> str:
    """
    Use this tool for relationship and network questions about the recognition dataset.
    Best for understanding who recognizes whom, collaboration patterns, influential roles,
    and cross-department recognition flows.

    Uses Neo4j AuraDB if available, NetworkX otherwise.

    Examples:
    - Which departments recognize each other most?
    - Who are the most central hubs in the recognition network?
    - What is the recognition flow between engineering and product?
    - Who are the bridge roles that span multiple departments?
    - Which groups give and receive the most recognition?
    - Which role group (department) gives out the most recognitions?
    - Which teams have a two-way recognition relationship?
    - Are there mutual recognition pairs (both ways)?
    """
    q      = question.lower()
    driver = _get_driver()
    n      = _extract_n(q)
    output = []

    wants = {
        "mutual":   any(w in q for w in _MUTUAL_WORDS),
        "central":  any(w in q for w in _CENTRAL_WORDS),
        "bridge":   any(w in q for w in _BRIDGE_WORDS),
        "giver":    bool(_GIVER_RE.search(q)),
        "receiver": bool(_RECEIVER_RE.search(q)),
    }
    # Flows are the default answer when no more specific intent was asked for
    wants["flows"] = any(w in q for w in _FLOW_WORDS) or not any(wants.values())

    # ── Strongest relationships / cross-department flow ───────────────────────
    if wants["flows"]:
        if driver:
            records = _run_cypher("""
                MATCH (a:RoleGroup)-[r:RECOGNIZED]->(b:RoleGroup)
                WHERE a.name <> b.name
                RETURN a.name AS from_group, b.name AS to_group,
                       r.count AS count, r.top_category AS top_category
                ORDER BY r.count DESC LIMIT $n
            """, {"n": n})
            output.append(f"=== Top {n} Cross-Department Recognition Flows ===\n")
            for rec in records:
                output.append(
                    f"  {rec['from_group']} → {rec['to_group']}: "
                    f"{rec['count']} recognitions (mainly: {rec['top_category']})"
                )
        else:
            output.append(f"=== Top {n} Cross-Team Recognition Flows ===\n")
            for u, v, w, cat in _nx_cross_team(n):
                output.append(f"  {u} → {v}: {w} recognitions (mainly: {cat})")
        output.append("")

    # ── Mutual / bidirectional recognition ────────────────────────────────────
    if wants["mutual"]:
        if driver:
            records = _run_cypher("""
                MATCH (a:RoleGroup)-[r1:RECOGNIZED]->(b:RoleGroup)-[r2:RECOGNIZED]->(a)
                WHERE elementId(a) < elementId(b)
                RETURN a.name AS group1, b.name AS group2,
                       r1.count AS a_to_b, r2.count AS b_to_a,
                       r1.count + r2.count AS total
                ORDER BY total DESC
            """)
            pairs = [(r["group1"], r["group2"], r["a_to_b"], r["b_to_a"]) for r in records]
        else:
            pairs = _nx_mutual()
        output.append("=== Mutual Recognition Pairs (Both Directions) ===\n")
        for a, b, ab, ba in pairs[:n]:
            output.append(f"  {a} ↔ {b}: {ab + ba} total ({ab} + {ba})")
        output.append("")

    # ── Most central / influential ────────────────────────────────────────────
    if wants["central"]:
        if driver:
            records = _run_cypher("""
                MATCH (g:RoleGroup)
                OPTIONAL MATCH (g)-[out:RECOGNIZED]->()
                OPTIONAL MATCH ()-[in:RECOGNIZED]->(g)
                WITH g,
                     coalesce(sum(out.count), 0) AS given,
                     coalesce(sum(in.count), 0)  AS received
                RETURN g.name AS group,
                       given, received,
                       given + received AS total_connections
                ORDER BY total_connections DESC LIMIT $n
            """, {"n": n})
            output.append("=== Most Central Role Groups ===\n")
            for rec in records:
                output.append(
                    f"  {rec['group']}: {rec['total_connections']} total "
                    f"(gave {rec['given']}, received {rec['received']})"
                )
        else:
            G = _build_title_graph()
            output.append("=== Most Connected Job Titles (by degree centrality) ===\n")
            for name, score in _nx_most_central(n):
                output.append(f"  {name} [{G.nodes[name]['group']}]: centrality {round(score, 4)}, "
                              f"{G.degree(name)} distinct connections")
        output.append("")

    # ── Bridge roles ──────────────────────────────────────────────────────────
    if wants["bridge"]:
        if driver:
            records = _run_cypher("""
                MATCH (n:RoleGroup)
                WITH n, COUNT { (n)-[:RECOGNIZED]->() } AS out_deg,
                        COUNT { ()-[:RECOGNIZED]->(n) } AS in_deg
                WHERE out_deg > 0 AND in_deg > 0
                ORDER BY (out_deg + in_deg) DESC LIMIT $n
                RETURN n.name AS group, out_deg, in_deg
            """, {"n": n})
            output.append("=== Bridge Role Groups (span across teams) ===\n")
            for rec in records:
                output.append(
                    f"  {rec['group']}: gave {rec['out_deg']}, received {rec['in_deg']}"
                )
        else:
            G = _build_title_graph()
            output.append("=== Bridge Job Titles (by betweenness centrality) ===\n")
            for name, score in _nx_bridges(n):
                output.append(f"  {name} [{G.nodes[name]['group']}]: bridge score {round(score, 4)}")
        output.append("")

    # ── Top givers ────────────────────────────────────────────────────────────
    if wants["giver"]:
        if driver:
            records = _run_cypher("""
                MATCH (g:RoleGroup)-[r:RECOGNIZED]->()
                RETURN g.name AS group, sum(r.count) AS total_given
                ORDER BY total_given DESC LIMIT $n
            """, {"n": n})
            top = [(r["group"], r["total_given"]) for r in records]
        else:
            top = _nx_totals("given", n)
        output.append("=== Groups That Give the Most Recognitions ===\n")
        for name, total in top:
            output.append(f"  {name}: {total} given")
        output.append("")

    # ── Top receivers ─────────────────────────────────────────────────────────
    if wants["receiver"]:
        if driver:
            records = _run_cypher("""
                MATCH ()-[r:RECOGNIZED]->(g:RoleGroup)
                RETURN g.name AS group, sum(r.count) AS total_received
                ORDER BY total_received DESC LIMIT $n
            """, {"n": n})
            top = [(r["group"], r["total_received"]) for r in records]
        else:
            top = _nx_totals("received", n)
        output.append("=== Groups That Receive the Most Recognitions ===\n")
        for name, total in top:
            output.append(f"  {name}: {total} received")
        output.append("")

    return "\n".join(output).strip() if output else "No graph data available."
