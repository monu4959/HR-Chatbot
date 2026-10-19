import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import httpx
import uuid
import json
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

API_URL   = "http://localhost:8000"
DATA_PATH = os.getenv("DATA_PATH", "./data/annotated_results.csv")

st.set_page_config(
    page_title="Recognition Analytics",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CATEGORY_COLORS = {
    "Organizational & Team Enablement": "#6366f1",
    "Operational Excellence & Delivery": "#f59e0b",
    "Strategic Business Outcomes":       "#10b981",
    "Creative & Brand Impact":           "#f43f5e",
}
CATEGORY_SHORT = {
    "Organizational & Team Enablement": "Team Enablement",
    "Operational Excellence & Delivery": "Ops Excellence",
    "Strategic Business Outcomes":       "Business Outcomes",
    "Creative & Brand Impact":           "Creative & Brand",
}
FONT_CLR = "#b4b8cc"
GRID_CLR = "rgba(255,255,255,0.06)"
FONT     = "Inter, sans-serif"

def base_layout(**kw):
    d = dict(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=FONT_CLR, size=12),
        margin=dict(l=20, r=20, t=40, b=20),
        hoverlabel=dict(bgcolor="#1e1f2e", font_size=12),
    )
    d.update(kw)
    return d

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

.stApp, [data-testid="stAppViewContainer"] {
    background-color: #0c0d14 !important;
    color: #b4b8cc;
    font-family: 'Inter', sans-serif;
}
section[data-testid="stSidebar"]         { display: none !important; }
[data-testid="collapsedControl"]         { display: none !important; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
header[data-testid="stHeader"]           { display: none !important; }

/* ── Top bar ── */
.top-bar {
    display: flex;
    align-items: center;
    gap: 0;
    background: #0f1019;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding: 0 28px;
    height: 52px;
    position: sticky;
    top: 0;
    z-index: 999;
}
.top-logo {
    font-size: 14px;
    font-weight: 700;
    color: #e0e2ed;
    letter-spacing: -0.3px;
    margin-right: 40px;
    display: flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
}
.top-logo-dot { color: #6366f1; font-size: 18px; }

.top-nav {
    display: flex;
    align-items: center;
    gap: 4px;
    height: 100%;
}
.top-nav-item {
    display: flex;
    align-items: center;
    gap: 7px;
    height: 52px;
    padding: 0 16px;
    font-size: 13px;
    font-weight: 500;
    color: #5f6380;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.15s;
    white-space: nowrap;
}
.top-nav-item:hover { color: #b4b8cc; }
.top-nav-item.active {
    color: #a5b4fc;
    border-bottom-color: #6366f1;
}

/* ── Page content ── */
.page-wrap { padding: 28px 32px; }

/* ── KPI cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] {
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #5f6380 !important;
}
[data-testid="stMetricValue"] {
    font-size: 26px !important;
    font-weight: 700 !important;
    font-family: 'SF Mono', 'Fira Code', monospace !important;
    color: #e0e2ed !important;
}
[data-testid="stMetricDelta"] { display: none !important; }

/* ── Insight box ── */
.insight-box {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 12px;
    line-height: 1.65;
    color: #8b8fa3;
}
.insight-box strong { color: #e0e2ed; }

/* ── Page headings ── */
.page-title {
    font-size: 15px;
    font-weight: 600;
    color: #e0e2ed;
    margin: 0 0 4px;
}
.page-subtitle {
    font-size: 12px;
    color: #5f6380;
    margin-bottom: 20px;
}

/* ── Chat ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
}
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #e0e2ed !important;
    font-size: 13px !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] {
    border-top: 1px solid rgba(255,255,255,0.06) !important;
    background: #0c0d14 !important;
    padding: 12px 32px !important;
}

/* ── Quick prompt buttons ── */
.stButton button {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #8b8fa3 !important;
    font-size: 11px !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
    transition: all 0.15s !important;
}
.stButton button:hover {
    background: rgba(99,102,241,0.1) !important;
    border-color: rgba(99,102,241,0.3) !important;
    color: #a5b4fc !important;
}

/* ── General ── */
h1,h2,h3,h4 { color: #e0e2ed !important; font-family: 'Inter', sans-serif !important; }
.stMarkdown p { color: #b4b8cc; }
.stCaption { color: #5f6380 !important; font-size: 11px !important; }
[data-testid="stMultiSelect"] label,
[data-testid="stTextInput"] label { font-size: 11px !important; color: #5f6380 !important; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

if "page"     not in st.session_state: st.session_state.page     = "chatbot"
if "messages" not in st.session_state: st.session_state.messages = []
if "session_id" not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
if "explorer_page" not in st.session_state: st.session_state.explorer_page = 0

# ── Top navigation bar ────────────────────────────────────────────────────────

NAV_ITEMS = [
    ("chatbot",  "💬", "Chatbot"),
    ("overview", "📊", "Overview"),
    ("patterns", "🔀", "Patterns"),
    ("explorer", "📋", "Explorer"),
]

st.markdown('<div class="top-bar"><div class="top-logo"><span class="top-logo-dot">◆</span>Recognition Analytics</div><div class="top-nav">', unsafe_allow_html=True)

cols = st.columns([2] + [1] * len(NAV_ITEMS) + [6])
for i, (key, icon, label) in enumerate(NAV_ITEMS):
    active = "active" if st.session_state.page == key else ""
    with cols[i + 1]:
        if st.button(f"{icon} {label}", key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

st.markdown('</div></div>', unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    for p in [Path(DATA_PATH), Path("data/annotated_results.csv"), Path("annotated_results.csv")]:
        if p.exists():
            df = pd.read_csv(p)
            df["msg_len"] = df["message"].str.len()
            df["category_short"] = df["category_name"].map(CATEGORY_SHORT)
            def rg(t):
                if pd.isna(t): return "Other"
                t = t.lower()
                if any(k in t for k in ["customer service","helpdesk","customer success"]): return "Customer Service"
                if any(k in t for k in ["engineer","software","qa","technical","devops","architect"]): return "Engineering"
                if any(k in t for k in ["design","creative","ux","copywriter","content"]): return "Design/Creative"
                if any(k in t for k in ["finance","payroll","procurement","financial","account"]): return "Finance/Admin"
                if any(k in t for k in ["director","vp","head of","chief","president"]): return "Leadership"
                if any(k in t for k in ["hr","hx","people","talent","recruit"]): return "People/HR"
                if any(k in t for k in ["market","brand","media","communications"]): return "Marketing"
                if any(k in t for k in ["product","program","project"]): return "Product/Program"
                if any(k in t for k in ["data","analy","insight","research"]): return "Data/Analytics"
                if any(k in t for k in ["sales","business develop","revenue"]): return "Sales/BD"
                return "Other"
            df["role_group"] = df["recipient_title"].apply(rg)
            return df
    return None

df = load_data()

# ── Chart renderer ────────────────────────────────────────────────────────────

def parse_chart(answer):
    if "CHART_DATA:" not in answer: return answer, None
    parts = answer.split("CHART_DATA:", 1)
    try: return parts[0].strip(), json.loads(parts[1].strip())
    except: return answer, None

def render_chart(cj, key="chart"):
    ct     = cj.get("chart_type")
    title  = cj.get("title", "")
    xl     = cj.get("x_label", "")
    yl     = cj.get("y_label", "")
    data   = cj.get("data", [])
    if not data: return
    labels = [d["label"] for d in data]
    values = [d["value"] for d in data]
    dff    = pd.DataFrame({"label": labels, "value": values})
    COLS   = ["#6366f1","#f59e0b","#10b981","#f43f5e","#3b82f6","#8b5cf6","#ec4899","#14b8a6","#f97316","#84cc16"]

    if ct == "bar":
        fig = px.bar(dff, x="label", y="value", title=title,
                     labels={"label": xl, "value": yl},
                     color="label", color_discrete_sequence=COLS, text="value")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_tickangle=-20)
    elif ct == "horizontal_bar":
        dff = dff.sort_values("value", ascending=True)
        fig = px.bar(dff, x="value", y="label", orientation="h", title=title,
                     labels={"label": yl, "value": xl},
                     color="value", color_continuous_scale=["#4f46e5","#818cf8"], text="value")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, yaxis=dict(automargin=True))
    elif ct == "pie":
        fig = px.pie(dff, names="label", values="value", title=title,
                     color_discrete_sequence=COLS, hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label")
    else:
        fig = px.bar(dff, x="label", y="value", title=title, color_discrete_sequence=COLS)

    fig.update_layout(**base_layout(height=400, margin=dict(t=50, b=40, l=20, r=20)))
    st.plotly_chart(fig, use_container_width=True, key=key)

# ── Pages ─────────────────────────────────────────────────────────────────────

page = st.session_state.page
st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

# ════════════════════════════════════════════
# CHATBOT
# ════════════════════════════════════════════
if page == "chatbot":
    st.markdown("<p class='page-title'>AI Chatbot</p>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Ask anything — stats, charts, cultural insights, succession planning, or network analysis</p>", unsafe_allow_html=True)

    # Quick prompts
    QUICK = [
        "How many awards are in each category?",
        "What does this org value most?",
        "Which departments recognize each other most?",
        "Who can replace a Senior Director?",
        "Find recognitions about mentorship",
        "Show me a bar chart of awards by category",
    ]
    cols = st.columns(3)
    for i, q in enumerate(QUICK):
        if cols[i % 3].button(q, key=f"qp_{i}", use_container_width=True):
            st.session_state["prefill"] = q

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Chat history
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg.get("text"):  st.markdown(msg["text"])
            if msg.get("chart"): render_chart(msg["chart"], key=f"hist_chart_{i}")
            if msg.get("meta"):  st.caption(msg["meta"])

    # Input
    prefill    = st.session_state.pop("prefill", "")
    user_input = st.chat_input("Ask a question or request a chart...")
    question   = user_input or (prefill if prefill else None)

    if question:
        st.session_state.messages.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = httpx.post(f"{API_URL}/query",
                                      json={"question": question, "session_id": st.session_state.session_id},
                                      timeout=120)
                    if resp.status_code == 200:
                        data       = resp.json()
                        raw        = data["answer"]
                        api_time   = data["time_taken_seconds"]
                        route      = data.get("route_used", "")
                        h_score    = data.get("hallucination_score", 1.0)
                        was_refused = data.get("was_refused", False)

                        text_part, chart_data = parse_chart(raw)

                        if text_part:  st.markdown(text_part)
                        if chart_data: render_chart(chart_data, key=f"new_chart_{len(st.session_state.messages)}")

                        meta = f"⏱ {api_time}s"
                        if data.get("cached"): meta += "  ·  ⚡ cached"
                        if data.get("model_tier"): meta += f"  ·  model: {data['model_tier']}" + (" (escalated)" if data.get("escalated") else "")
                        if route:   meta += f"  ·  route: {route}"
                        if h_score: meta += f"  ·  grounding: {h_score}"
                        st.caption(meta)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "text": text_part,
                            "chart": chart_data,
                            "meta": meta,
                        })
                    else:
                        st.error(f"API error {resp.status_code}: {resp.text}")
                except httpx.ConnectError:
                    st.error("Cannot connect to API. Make sure the server is running: `uvicorn api.main:app --reload --port 8000`")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    if st.session_state.messages:
        c1, _, _ = st.columns([1, 4, 1])
        with c1:
            if st.button("🗑 Clear chat", type="secondary"):
                st.session_state.messages = []
                try: httpx.post(f"{API_URL}/session/clear", json={"session_id": st.session_state.session_id}, timeout=5)
                except: pass
                st.rerun()

# ════════════════════════════════════════════
# OVERVIEW
# ════════════════════════════════════════════
elif page == "overview":
    if df is None:
        st.error("Could not load dataset. Check your DATA_PATH in .env")
        st.stop()

    TOTAL = len(df)
    cat_summary = (
        df.groupby("category_name")
        .agg(count=("message","size"), avg_len=("msg_len","mean"))
        .reset_index().sort_values("count", ascending=False)
    )
    cat_summary["pct"] = (cat_summary["count"] / TOTAL * 100).round(1)
    sub_summary = (
        df.groupby(["category_name","subcategory_name"])
        .agg(count=("message","size"))
        .reset_index().sort_values("count", ascending=False)
    )
    sub_summary["pct"] = (sub_summary["count"] / TOTAL * 100).round(1)

    st.markdown("<p class='page-title'>Recognition overview</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='page-subtitle'>{TOTAL:,} messages · {df['category_name'].nunique()} categories · {df['subcategory_name'].nunique()} subcategories</p>", unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Records",      f"{TOTAL:,}")
    k2.metric("Categories",   df["category_name"].nunique())
    k3.metric("Subcategories",df["subcategory_name"].nunique())
    k4.metric("Median Length",f"{int(df['msg_len'].median())} chars")
    k5.metric("Top Category", f"{cat_summary.iloc[0]['pct']}%")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    col_bar, col_sun = st.columns(2)

    with col_bar:
        st.markdown("#### Category distribution")
        st.caption("Recognition count per category")
        fig = go.Figure()
        for _, row in cat_summary.iloc[::-1].iterrows():
            color = CATEGORY_COLORS.get(row["category_name"], "#6366f1")
            fig.add_trace(go.Bar(
                y=[row["category_name"]], x=[row["count"]], orientation="h",
                marker=dict(color=color, opacity=0.85, line=dict(width=0)),
                text=f"  {row['count']}  ({row['pct']}%)",
                textposition="outside", textfont=dict(size=11, color=FONT_CLR),
                hovertemplate=f"<b>{row['category_name']}</b><br>{row['count']} ({row['pct']}%)<extra></extra>",
                showlegend=False,
            ))
        fig.update_layout(**base_layout(
            height=260, margin=dict(l=260, r=80, t=10, b=10),
            xaxis=dict(visible=False),
            yaxis=dict(tickfont=dict(size=11), automargin=True), bargap=0.3,
        ))
        st.plotly_chart(fig, use_container_width=True, key="ov_bar")
        top2 = cat_summary.iloc[0]["pct"] + cat_summary.iloc[1]["pct"]
        ratio = round(cat_summary.iloc[0]["count"] / cat_summary.iloc[-1]["count"])
        st.markdown(
            f'<div class="insight-box"><span style="color:#6366f1;font-weight:600">Team Enablement</span> dominates at '
            f'{cat_summary.iloc[0]["pct"]}% — <strong>{ratio}×</strong> the size of '
            f'{cat_summary.iloc[-1]["category_name"]} ({cat_summary.iloc[-1]["pct"]}%). '
            f'Top two = <strong>{top2:.1f}%</strong> of all recognition.</div>',
            unsafe_allow_html=True
        )

    with col_sun:
        st.markdown("#### Full taxonomy")
        st.caption("Inner ring = categories · outer ring = subcategories")
        fig_sun = px.sunburst(
            sub_summary, path=["category_name","subcategory_name"],
            values="count", color="category_name", color_discrete_map=CATEGORY_COLORS,
        )
        fig_sun.update_traces(
            insidetextorientation="radial", textfont=dict(size=10),
            hovertemplate="<b>%{label}</b><br>%{value} records<br>%{percentRoot:.1%}<extra></extra>",
        )
        fig_sun.update_layout(**base_layout(height=400, margin=dict(l=10,r=10,t=10,b=10)), showlegend=False)
        st.plotly_chart(fig_sun, use_container_width=True, key="ov_sun")

# ════════════════════════════════════════════
# PATTERNS
# ════════════════════════════════════════════
elif page == "patterns":
    if df is None:
        st.error("Could not load dataset.")
        st.stop()

    TOTAL = len(df)
    cat_summary = (
        df.groupby("category_name")
        .agg(count=("message","size"), avg_len=("msg_len","mean"))
        .reset_index().sort_values("count", ascending=False)
    )

    st.markdown("<p class='page-title'>Recognition patterns</p>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>How recognition varies by role, category, and message length</p>", unsafe_allow_html=True)

    col_cross, col_box = st.columns(2)

    with col_cross:
        st.markdown("#### Do roles get recognized differently?")
        st.caption("Proportion of each role group's recognition by category")
        role_cat = pd.crosstab(df["role_group"], df["category_name"])
        role_cat = role_cat[role_cat.sum(axis=1) >= 15]
        role_cat_pct = (role_cat.div(role_cat.sum(axis=1), axis=0) * 100).round(1)
        if "Organizational & Team Enablement" in role_cat_pct.columns:
            role_cat_pct = role_cat_pct.sort_values("Organizational & Team Enablement", ascending=True)
        fig_cross = go.Figure()
        for cat_name in CATEGORY_COLORS:
            if cat_name in role_cat_pct.columns:
                fig_cross.add_trace(go.Bar(
                    y=role_cat_pct.index, x=role_cat_pct[cat_name],
                    name=CATEGORY_SHORT.get(cat_name, cat_name), orientation="h",
                    marker=dict(color=CATEGORY_COLORS[cat_name], opacity=0.85),
                    hovertemplate=f"<b>%{{y}}</b><br>{CATEGORY_SHORT.get(cat_name,cat_name)}: %{{x:.1f}}%<extra></extra>",
                ))
        fig_cross.update_layout(**base_layout(
            height=400, margin=dict(l=130,r=20,t=10,b=10), barmode="stack",
            xaxis=dict(ticksuffix="%", range=[0,105], showgrid=True, gridcolor=GRID_CLR, tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=11), automargin=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
        ))
        st.plotly_chart(fig_cross, use_container_width=True, key="pat_cross")

    with col_box:
        st.markdown("#### Are some categories more detailed?")
        st.caption("Message length distribution by category (characters)")
        fig_box = go.Figure()
        for cat_name in cat_summary["category_name"]:
            cat_data = df[df["category_name"] == cat_name]["msg_len"]
            color = CATEGORY_COLORS.get(cat_name, "#6366f1")
            r,g,b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
            fig_box.add_trace(go.Box(
                y=cat_data, name=CATEGORY_SHORT.get(cat_name, cat_name),
                marker=dict(color=color, opacity=0.6, size=3),
                line=dict(color=color, width=1.5),
                fillcolor=f"rgba({r},{g},{b},0.15)",
                boxmean=True, hoverinfo="y+name",
            ))
        fig_box.update_layout(**base_layout(
            height=400, margin=dict(l=50,r=20,t=10,b=10),
            showlegend=False,
            yaxis=dict(title="Characters", showgrid=True, gridcolor=GRID_CLR, tickfont=dict(size=10)),
            xaxis=dict(tickfont=dict(size=10)),
        ))
        st.plotly_chart(fig_box, use_container_width=True, key="pat_box")

        cat_medians = df.groupby("category_name")["msg_len"].median().sort_values(ascending=False)
        lcat = cat_medians.index[0]; lmed = int(cat_medians.iloc[0])
        scat = cat_medians.index[-1]; smed = int(cat_medians.iloc[-1])
        lcolor = CATEGORY_COLORS.get(lcat,"#10b981")
        scolor = CATEGORY_COLORS.get(scat,"#f59e0b")
        st.markdown(
            f'<div class="insight-box"><span style="color:{lcolor};font-weight:600">'
            f'{CATEGORY_SHORT.get(lcat,lcat)}</span> messages are longest (median {lmed} chars). '
            f'<span style="color:{scolor};font-weight:600">{CATEGORY_SHORT.get(scat,scat)}</span> '
            f'is shortest at {smed} chars.</div>', unsafe_allow_html=True
        )

# ════════════════════════════════════════════
# EXPLORER
# ════════════════════════════════════════════
elif page == "explorer":
    if df is None:
        st.error("Could not load dataset.")
        st.stop()

    TOTAL = len(df)
    st.markdown("<p class='page-title'>Message Explorer</p>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Search and filter individual recognition messages</p>", unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        filter_cat = st.multiselect("Filter by category", options=df["category_name"].unique().tolist(),
                                     default=[], placeholder="All categories")
    with col_f2:
        sub_opts   = (df[df["category_name"].isin(filter_cat)]["subcategory_name"].unique().tolist()
                      if filter_cat else df["subcategory_name"].unique().tolist())
        filter_sub = st.multiselect("Filter by subcategory", options=sorted(sub_opts),
                                     default=[], placeholder="All subcategories")
    with col_f3:
        search_text = st.text_input("Search messages", placeholder="keyword...")

    filtered = df.copy()
    if filter_cat:   filtered = filtered[filtered["category_name"].isin(filter_cat)]
    if filter_sub:   filtered = filtered[filtered["subcategory_name"].isin(filter_sub)]
    if search_text:  filtered = filtered[filtered["message"].str.contains(search_text, case=False, na=False)]

    st.markdown(f"<p style='font-size:12px;color:#5f6380'>Showing {len(filtered)} of {TOTAL} records</p>",
                unsafe_allow_html=True)

    page_size   = 15
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    if st.session_state.explorer_page >= total_pages:
        st.session_state.explorer_page = 0
    p    = st.session_state.explorer_page
    rows = filtered.iloc[p * page_size : p * page_size + page_size]

    for _, row in rows.iterrows():
        color     = CATEGORY_COLORS.get(row["category_name"], "#6366f1")
        cat_short = CATEGORY_SHORT.get(row["category_name"], row["category_name"])
        msg       = row["message"][:380] + "..." if len(row["message"]) > 380 else row["message"]
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.06);'
            f'border-left:3px solid {color};border-radius:10px;padding:16px 20px;margin-bottom:10px">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:8px">'
            f'<div><span style="font-weight:600;color:#e0e2ed;font-size:13px">{row["award_title"]}</span>'
            f'<span style="color:#5f6380;font-size:11px;margin-left:10px">→ {row["recipient_title"]}</span></div>'
            f'<span style="color:#5f6380;font-size:10px;font-family:monospace">{row["msg_len"]} chars</span></div>'
            f'<p style="font-size:12px;color:#8b8fa3;line-height:1.65;margin:0 0 10px">{msg}</p>'
            f'<span style="background:{color}18;color:{color};border-radius:5px;padding:2px 10px;font-size:10px;font-weight:600;margin-right:6px">{cat_short}</span>'
            f'<span style="background:{color}10;color:{color}cc;border-radius:5px;padding:2px 10px;font-size:10px">{row["subcategory_name"]}</span>'
            f'</div>', unsafe_allow_html=True
        )

    if total_pages > 1:
        c1, c2, c3 = st.columns([1, 3, 1])
        with c1:
            if st.button("← Prev", disabled=(p == 0)):
                st.session_state.explorer_page = max(0, p - 1); st.rerun()
        with c2:
            st.markdown(f"<p style='text-align:center;color:#5f6380;font-size:12px;padding-top:8px'>Page {p+1} of {total_pages}</p>",
                        unsafe_allow_html=True)
        with c3:
            if st.button("Next →", disabled=(p >= total_pages - 1)):
                st.session_state.explorer_page = min(total_pages - 1, p + 1); st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
