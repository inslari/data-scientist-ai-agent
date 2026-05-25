"""
Data Scientist Agent — Streamlit Dashboard
==========================================
Professional dark-theme dashboard for the AI-powered data science pipeline.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Path setup
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.graph import build_graph # type: ignore[import-not-found]
from src.state import AgentState # type: ignore[import-not-found]
from src.history_tracker import get_experiments # type: ignore[import-not-found]

# Page config
st.set_page_config(
    page_title="DS Agent",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Global CSS
DARK_VARS = """
    --bg:        #0a0a0f;
    --bg2:       #111118;
    --bg3:       #16161f;
    --border:    rgba(255,255,255,0.06);
    --border2:   rgba(255,255,255,0.12);
    --text:      #e8e8f0;
    --text2:     #8888aa;
    --text3:     #4a4a6a;
    --scrollbar: #4a4a6a;
"""

LIGHT_VARS = """
    --bg:        #f4f4f8;
    --bg2:       #ffffff;
    --bg3:       #ebebf2;
    --border:    rgba(0,0,0,0.07);
    --border2:   rgba(0,0,0,0.14);
    --text:      #1a1a2e;
    --text2:     #5a5a7a;
    --text3:     #aaaacc;
    --scrollbar: #aaaacc;
"""

theme_vars = DARK_VARS if st.session_state.theme == "dark" else LIGHT_VARS

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Syne:wght@400;500;600;700;800&display=swap');

:root {{
    {theme_vars}
    --accent:    #6c63ff;
    --accent2:   #ff6584;
    --accent3:   #43e97b;
    --accent4:   #f7971e;
    --mono:      'DM Mono', monospace;
    --sans:      'Syne', sans-serif;
}}

html, body, [data-testid="stAppViewContainer"] {{
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}}

[data-testid="stSidebar"] {{
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}}

[data-testid="stSidebar"] * {{ color: var(--text) !important; }}

/* Hide default streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stDecoration"] {{ display: none; }}

/* Inputs */
.stTextArea textarea, .stTextInput input {{
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
}}
.stTextArea textarea:focus, .stTextInput input:focus {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(108,99,255,0.15) !important;
}}

/* Buttons */
.stButton > button {{
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.8rem !important;
    transition: all 0.2s ease !important;
}}
.stButton > button:hover {{
    background: #8075ff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.35) !important;
}}

/* Metric cards */
[data-testid="stMetric"] {{
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
}}
[data-testid="stMetricLabel"] {{ color: var(--text2) !important; font-family: var(--mono) !important; font-size: 0.75rem !important; }}
[data-testid="stMetricValue"] {{ color: var(--text) !important; font-family: var(--sans) !important; font-weight: 700 !important; }}
[data-testid="stMetricDelta"] {{ font-family: var(--mono) !important; }}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {{
    background: var(--bg2) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
}}
[data-testid="stTabs"] [role="tab"] {{
    color: var(--text2) !important;
    font-family: var(--sans) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    border-radius: 7px !important;
    transition: all 0.2s !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    background: var(--accent) !important;
    color: white !important;
}}

/* Expander */
[data-testid="stExpander"] {{
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}}

/* Code blocks */
code {{ font-family: var(--mono) !important; }}

/* Divider */
hr {{ border-color: var(--border) !important; }}

/* Custom scrollbar */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: var(--scrollbar); border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)

# Plotly theme
is_dark = st.session_state.theme == "dark"

# Theme-aware text colors for charts
CHART_TEXT_COLOR = "#e8e8f0" if is_dark else "#1a1a2e"
CHART_LABEL_COLOR = "#8888aa" if is_dark else "#5a5a7a"
CHART_DIM_COLOR  = "#4a4a6a" if is_dark else "#aaaacc"
INFO_BG = "rgba(255,255,255,0.03)" if is_dark else "rgba(0,0,0,0.04)"
STATUS_BG = "#111118" if is_dark else "#f0f0f8"
STATUS_BORDER = "rgba(255,255,255,0.06)" if is_dark else "rgba(0,0,0,0.10)"
TRACE_BORDER = "rgba(255,255,255,0.03)" if is_dark else "rgba(0,0,0,0.06)"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        family="DM Mono, monospace",
        color=CHART_LABEL_COLOR,
        size=11,
    ),
    margin=dict(l=10, r=10, t=30, b=10),
)
AXIS_STYLE = dict(
    gridcolor="rgba(255,255,255,0.04)" if is_dark else "rgba(0,0,0,0.06)",
    zerolinecolor="rgba(255,255,255,0.08)" if is_dark else "rgba(0,0,0,0.10)",
    tickfont=dict(color=CHART_LABEL_COLOR),
)
COLORS = ["#6c63ff", "#ff6584", "#43e97b", "#f7971e", "#38b2f4", "#f093fb"]


# SIDEBAR

with st.sidebar:
    # Logo + Theme toggle
    logo_col, toggle_col = st.columns([3, 1])
    with logo_col:
        st.markdown(f"""
        <div style="padding: 1rem 0 1.5rem 0;">
            <div style="font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800; 
                        letter-spacing: -0.02em; color: {'#e8e8f0' if is_dark else '#1a1a2e'};">
                ⬡ DS Agent
            </div>
            <div style="font-family: 'DM Mono', monospace; font-size: 0.7rem; 
                        color: #6c63ff; margin-top: 2px; letter-spacing: 0.08em;">
                AI-POWERED ANALYTICS
            </div>
        </div>
        """, unsafe_allow_html=True)
    with toggle_col:
        st.markdown("<div style='padding-top: 1.2rem;'>", unsafe_allow_html=True)
        theme_icon = "☀️" if is_dark else "🌙"
        if st.button(theme_icon, key="theme_toggle", help="Toggle light/dark theme"):
            st.session_state.theme = "light" if is_dark else "dark"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <hr style="border-color: {'rgba(255,255,255,0.06)' if is_dark else 'rgba(0,0,0,0.07)'}; margin-bottom: 1.5rem;">
    """, unsafe_allow_html=True)

    st.markdown(f"<p style='font-family: DM Mono; font-size: 0.72rem; color: {'#8888aa' if is_dark else '#5a5a7a'}; letter-spacing: 0.08em; margin-bottom: 0.5rem;'>QUERY</p>", unsafe_allow_html=True)

    query = st.text_area(
        label="query",
        label_visibility="collapsed",
        value="Predict the income level of individuals in this dataset. What factors influence whether a person earns over 50K?",
        height=120,
        placeholder="Describe what you want to analyze...",
        key="query_input",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("⬡ Run Analysis", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family: 'DM Mono', monospace; font-size: 0.7rem; color: {'#4a4a6a' if is_dark else '#aaaacc'}; line-height: 1.8;">
        <div style="color: {'#8888aa' if is_dark else '#5a5a7a'}; font-size: 0.65rem; letter-spacing: 0.08em; margin-bottom: 0.5rem;">PIPELINE</div>
        <div>01 · Intent Analyzer</div>
        <div>02 · DB Context Loader</div>
        <div>03 · Analysis Planner</div>
        <div>04 · Execution Layer</div>
        <div>05 · Feature Planner</div>
        <div>06 · Feature Engineering</div>
        <div>07 · AutoML</div>
        <div>08 · Evaluation</div>
        <div>09 · Report Generator</div>
    </div>
    """, unsafe_allow_html=True)


# HEADER

st.markdown(f"""
<div style="padding: 2rem 0 1rem 0; border-bottom: 1px solid {'rgba(255,255,255,0.06)' if is_dark else 'rgba(0,0,0,0.07)'}; margin-bottom: 2rem;">
    <h1 style="font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; 
               letter-spacing: -0.03em; color: {'#e8e8f0' if is_dark else '#1a1a2e'}; margin: 0;">
        Data Scientist Agent
    </h1>
    <p style="font-family: 'DM Mono', monospace; font-size: 0.78rem; color: {'#8888aa' if is_dark else '#5a5a7a'}; 
              margin: 0.4rem 0 0 0;">
        Fully autonomous ML pipeline — from query to insights
    </p>
</div>
""", unsafe_allow_html=True)


# PIPELINE RUNNER

def run_pipeline(user_query: str) -> AgentState:
    """Runs the full LangGraph pipeline and returns final state."""
    app = build_graph()
    initial_state: AgentState = {
        "user_query": user_query,
        "messages": [],
    }
    return asyncio.run(app.ainvoke(initial_state))


# LOADING ANIMATION

def show_loading():
    """Animated node-by-node loading sequence."""

    raw_query = st.session_state.get("query_input", "").lower()
    sql_keywords = ["getir", "listele", "göster", "kaç", "filtrele", "sırala",
                    "select", "fetch", "show", "how many", "top ", "filter"]
    is_likely_sql = any(kw in raw_query for kw in sql_keywords)

    if is_likely_sql:
        nodes = [
            ("01", "Intent Analyzer",  "Classifying as SQL query..."),
            ("02", "SQL Generator",    "Writing SELECT query with GPT-4o..."),
            ("03", "MCP Executor",     "Running query on database..."),
            ("04", "Chart Planner",    "Deciding best visualization type..."),
        ]
    else:
        nodes = [
            ("01", "Intent Analyzer",      "Classifying analysis intent..."),
            ("02", "DB Context Loader",    "Fetching schema & running intelligence engine..."),
            ("03", "Analysis Planner",     "Building execution plan..."),
            ("04", "Execution Layer",      "Fetching data & profiling..."),
            ("05", "Feature Planner",      "LLM designing feature strategy..."),
            ("06", "Feature Engineering",  "Encoding, scaling, selecting features..."),
            ("07", "AutoML",               "Training & comparing 6 models with 5-fold CV..."),
            ("08", "Evaluation",           "Computing metrics & reliability score..."),
            ("09", "Report Generator",     "Synthesizing insights with GPT-4o..."),
        ]
    
    # Total steps
    total = len(nodes)
    
    placeholder = st.empty()
    for i, (num, name, desc) in enumerate(nodes):
        progress = (i + 1) / total
        completed = nodes[:i]
        current = (num, name, desc)


    placeholder = st.empty()
    for i, (num, name, desc) in enumerate(nodes):
        progress = (i + 1) / len(nodes)
        completed = nodes[:i]
        current = (num, name, desc)

        html = f"""
        <div style="background: #111118; border: 1px solid rgba(255,255,255,0.06);
                    border-radius: 12px; padding: 1.5rem 2rem; font-family: 'DM Mono', monospace;">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.2rem;">
                <div style="font-family: 'Syne', sans-serif; font-size: 0.85rem; 
                            color: #6c63ff; font-weight: 600; letter-spacing: 0.1em;">
                    RUNNING PIPELINE
                </div>
                <div style="flex: 1; height: 2px; background: rgba(255,255,255,0.06); border-radius: 1px;">
                    <div style="height: 100%; width: {progress*100:.0f}%; background: linear-gradient(90deg, #6c63ff, #ff6584);
                                border-radius: 1px; transition: width 0.3s ease;"></div>
                </div>
                <div style="font-size: 0.7rem; color: #8888aa;">{i+1}/9</div>
            </div>
        """

        # Completed nodes
        for cn, cname, _ in completed:
            html += f"""
            <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.4rem; opacity: 0.5;">
                <span style="color: #43e97b; font-size: 0.7rem;">✓</span>
                <span style="color: #4a4a6a; font-size: 0.72rem;">{cn} · {cname}</span>
            </div>
            """

        # Current node
        html += f"""
        <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.4rem;
                    padding: 0.5rem 0.8rem; background: rgba(108,99,255,0.08); 
                    border-radius: 6px; border-left: 2px solid #6c63ff;">
            <span style="color: #6c63ff; font-size: 0.7rem; animation: pulse 1s infinite;">◉</span>
            <div>
                <div style="color: #e8e8f0; font-size: 0.75rem; font-weight: 500;">{current[0]} · {current[1]}</div>
                <div style="color: #8888aa; font-size: 0.68rem; margin-top: 1px;">{current[2]}</div>
            </div>
        </div>
        </div>
        """

        placeholder.markdown(html, unsafe_allow_html=True)
        time.sleep(0.4)

    placeholder.empty()


# DASHBOARD SECTIONS

def render_metric_card(label: str, value: str, delta: str = "", color: str = "#6c63ff") -> str:
    text_color = "#1a1a2e" if st.session_state.theme == "light" else "#e8e8f0"
    label_color = "#5a5a7a" if st.session_state.theme == "light" else "#8888aa"
    bg_color = "#ebebf2" if st.session_state.theme == "light" else "#16161f"
    border_color = "rgba(0,0,0,0.07)" if st.session_state.theme == "light" else "rgba(255,255,255,0.06)"
    
    return f"""
    <div style="background: {bg_color}; border: 1px solid {border_color}; border-radius: 10px;
                padding: 1rem 1.2rem; border-top: 2px solid {color};">
        <div style="font-family: 'DM Mono', monospace; font-size: 0.68rem; color: {label_color}; 
                    letter-spacing: 0.08em; margin-bottom: 0.4rem;">{label.upper()}</div>
        <div style="font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700; 
                    color: {text_color};">{value}</div>
        {f'<div style="font-family: DM Mono; font-size: 0.7rem; color: {color}; margin-top: 2px;">{delta}</div>' if delta else ''}
    </div>
    """


def section_header(title: str, subtitle: str = ""):
    text_color = "#1a1a2e" if st.session_state.theme == "light" else "#e8e8f0"
    sub_color = "#5a5a7a" if st.session_state.theme == "light" else "#8888aa"
    st.markdown(f"""
    <div style="margin: 2rem 0 1rem 0;">
        <div style="font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; 
                    color: {text_color}; letter-spacing: -0.01em;">{title}</div>
        {f'<div style="font-family: DM Mono; font-size: 0.72rem; color: {sub_color}; margin-top: 2px;">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def render_sql_results(result: AgentState):
    """SQL Query intent için özel dashboard."""

    chart    = result.get("sql_query_chart", {})
    columns  = result.get("sql_query_columns", [])
    rows     = result.get("sql_query_results", [])
    sql_raw  = result.get("sql_query_raw", "")
    summary  = result.get("sql_query_summary", "")
    query    = result.get("user_query", "")

    chart_type = chart.get("type", "table")
    title      = chart.get("title", "Sorgu Sonuçları")

    labels   = chart.get("labels", [])
    datasets = chart.get("datasets", [])
    n_rows   = len(rows)
    n_cols   = len(columns)

   
    tick_angle = -45 if len(labels) > 10 else 0

   
    chart_height = max(420, min(len(labels) * 18, 700)) if labels else 420

    # Plotly interaktivite config 
    PLOTLY_CONFIG = {
        "scrollZoom":           True,
        "displayModeBar":       True,
        "displaylogo":          False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "modeBarButtonsToAdd":  ["zoom2d", "pan2d", "resetScale2d", "toImage"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "ds_agent_chart",
            "scale": 2,
        },
    }

    # Status bar         
    st.markdown(f"""
    <div style="background: {STATUS_BG}; border: 1px solid {STATUS_BORDER}; border-radius: 10px;
                padding: 1rem 1.5rem; display: flex; align-items: center; gap: 2rem;
                margin-bottom: 2rem; flex-wrap: wrap;">
        <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #43e97b;">
            ✓ QUERY COMPLETE
        </div>
        <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: {CHART_LABEL_COLOR};">
            INTENT: <span style="color: #6c63ff;">SQL_QUERY</span>
        </div>
        <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: {CHART_LABEL_COLOR};">
            ROWS: <span style="color: {CHART_TEXT_COLOR};">{n_rows:,}</span>
        </div>
        <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: {CHART_LABEL_COLOR};">
            COLS: <span style="color: {CHART_TEXT_COLOR};">{n_cols}</span>
        </div>
        <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: {CHART_LABEL_COLOR};">
            CHART: <span style="color: {CHART_TEXT_COLOR};">{chart_type.upper()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    warning  = result.get("sql_query_warning", "")

    # Warning for requested data 
    if warning:
        st.markdown(f"""
        <div style="background: rgba(247,151,30,0.08); border: 1px solid rgba(247,151,30,0.35);
                    border-left: 4px solid #f7971e; border-radius: 10px;
                    padding: 1rem 1.5rem; margin-bottom: 1.2rem;">
            <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem;
                        color: #f7971e; letter-spacing: 0.08em; margin-bottom: 0.5rem;">
                ⚡ YAKLAŞIK SONUÇ — İSTENEN VERİ DOĞRUDAN MEVCUT DEĞİL
            </div>
            <div style="font-family: 'DM Mono', monospace; font-size: 0.8rem;
                        color: {CHART_TEXT_COLOR}; line-height: 1.7;">
                {warning}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Summary of query
    if summary:
        st.markdown(f"""
        <div style="background: rgba(108,99,255,0.06); border: 1px solid rgba(108,99,255,0.18);
                    border-left: 3px solid #6c63ff; border-radius: 8px;
                    padding: 0.8rem 1.2rem; margin-bottom: 1.5rem;
                    font-family: 'DM Mono', monospace; font-size: 0.8rem;
                    color: {CHART_TEXT_COLOR}; line-height: 1.7;">
            {summary}
        </div>
        """, unsafe_allow_html=True)

    # SQL code 
    with st.expander("🗂 Generated SQL", expanded=False):
        st.code(sql_raw, language="sql")

    st.markdown("<br>", unsafe_allow_html=True)

    # Warning for chart type 
    if chart_type == "warning":
        warning_msg = chart.get("summary", "Bu sorgu için uygun kolon bulunamadı.")
        st.markdown(f"""
        <div style="background: rgba(255,101,132,0.07); border: 1px solid rgba(255,101,132,0.3);
                    border-left: 4px solid #ff6584; border-radius: 10px;
                    padding: 1.5rem 2rem; margin-bottom: 1rem;">
            <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem;
                        color: #ff6584; letter-spacing: 0.08em; margin-bottom: 0.8rem;">⚠ UYGUN KOLON BULUNAMADI</div>
            <div style="font-family: 'DM Mono', monospace; font-size: 0.82rem;
                        color: {CHART_TEXT_COLOR}; line-height: 1.8;">
                {warning_msg}
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Chart and Table beside each other  
    col_chart, col_table = st.columns([1.4, 1], gap="large")

    with col_chart:
        section_header(title, f"Görselleştirme · {chart_type}")

        # 1x1: Single numeric value -> big metric card 
        if n_rows == 1 and n_cols == 1:
            val = list(rows[0].values())[0]
            col_name = columns[0]
            try:
                val_f = float(val)
                display_val = f"{val_f:,.4f}" if val_f != int(val_f) else f"{int(val_f):,}"
            except (TypeError, ValueError):
                display_val = str(val)

            st.markdown(f"""
            <div style="background: rgba(108,99,255,0.08); border: 1px solid rgba(108,99,255,0.25);
                        border-radius: 16px; padding: 3rem 2rem; text-align: center; margin-top: 1rem;">
                <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem;
                            color: {CHART_LABEL_COLOR}; letter-spacing: 0.1em; margin-bottom: 1rem;">
                    {col_name.upper().replace('_', ' ')}
                </div>
                <div style="font-family: 'Syne', sans-serif; font-size: 3.5rem; font-weight: 800;
                            color: #6c63ff; letter-spacing: -0.03em; line-height: 1;">
                    {display_val}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 1 row, many columns -> horizontal metric cards 
        elif n_rows == 1 and n_cols > 1:
            row_data = rows[0]
            card_cols = st.columns(min(n_cols, 4))
            for i, (k, v) in enumerate(row_data.items()):
                try:
                    v_f = float(v)
                    display_v = f"{v_f:,.4f}" if v_f != int(v_f) else f"{int(v_f):,}"
                except (TypeError, ValueError):
                    display_v = str(v)
                with card_cols[i % 4]:
                    st.markdown(render_metric_card(
                        k.replace("_", " ").upper(),
                        display_v,
                        color=COLORS[i % len(COLORS)]
                    ), unsafe_allow_html=True)

        # Table fallback 
        if chart_type == "table" or not labels:
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True, height=chart_height)
            else:
                st.info("Sonuç bulunamadı.")

        # Pie    
        elif chart_type == "pie":
            # If there are too many categories in Pie, drop to table — pie >12 categories unreadable
            if len(labels) > 12:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True, height=chart_height)
            else:
                values = datasets[0]["data"] if datasets else []
                fig = go.Figure(go.Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=COLORS * (len(labels) // len(COLORS) + 1)),
                    hole=0.5,
                    textfont=dict(family="DM Mono", size=11),
                    textinfo="label+percent+value",
                    hovertemplate="<b>%{label}</b><br>Değer: %{value}<br>Oran: %{percent}<extra></extra>",
                ))
                fig.update_layout(
                    **PLOTLY_LAYOUT,
                    height=chart_height,
                    showlegend=True,
                    legend=dict(font=dict(family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        # Bar
        elif chart_type == "bar":
            fig = go.Figure()
            for i, ds in enumerate(datasets):
                y_vals = ds["data"]
                fig.add_trace(go.Bar(
                    x=labels,
                    y=y_vals,
                    name=ds.get("label", ""),
                    marker_color=COLORS[i % len(COLORS)],
                    marker_line_width=0,
                    text=[
                        f"{v:,.2f}" if isinstance(v, float) else str(v)
                        for v in y_vals
                    ],
                    textposition="outside",
                    textfont=dict(family="DM Mono", size=10, color=CHART_TEXT_COLOR),
                    hovertemplate="<b>%{x}</b><br>%{y:,.4f}<extra></extra>",
                ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                height=chart_height,
                xaxis=dict(
                    **{k: v for k, v in AXIS_STYLE.items() if k != "tickfont"},
                    tickangle=tick_angle,
                    tickfont=dict(
                        family="DM Mono, monospace",
                        size=9 if len(labels) > 20 else 11,
                        color=CHART_LABEL_COLOR,
                    ),
                ),
                yaxis=dict(**AXIS_STYLE),
                legend=dict(font=dict(family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)"),
                showlegend=len(datasets) > 1,
                bargap=0.15,
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        # Line
        elif chart_type == "line":
            fig = go.Figure()
            for i, ds in enumerate(datasets):
                fig.add_trace(go.Scatter(
                    x=labels,
                    y=ds["data"],
                    mode="lines+markers",
                    name=ds.get("label", ""),
                    line=dict(color=COLORS[i % len(COLORS)], width=2),
                    marker=dict(size=6),
                    hovertemplate="<b>%{x}</b><br>%{y:,.4f}<extra></extra>",
                ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                height=chart_height,
                xaxis=dict(
                    **AXIS_STYLE,
                    tickangle=tick_angle,
                    tickfont=dict(
                        size=9 if len(labels) > 20 else 11,
                        color=CHART_LABEL_COLOR,
                    ),
                ),
                yaxis=dict(**AXIS_STYLE),
                legend=dict(font=dict(family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)"),
                showlegend=len(datasets) > 1,
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        # Scatter
        elif chart_type == "scatter":
            y_vals = datasets[0]["data"] if datasets else []
            fig = go.Figure(go.Scatter(
                x=labels,
                y=y_vals,
                mode="markers",
                marker=dict(
                    color=COLORS[0],
                    size=8,
                    opacity=0.75,
                    line=dict(color="rgba(255,255,255,0.1)", width=1),
                ),
                hovertemplate="<b>%{x}</b><br>%{y:,.4f}<extra></extra>",
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                height=chart_height,
                xaxis=dict(
                    **AXIS_STYLE,
                    tickangle=tick_angle,
                    tickfont=dict(
                        size=9 if len(labels) > 20 else 11,
                        color=CHART_LABEL_COLOR,
                    ),
                ),
                yaxis=dict(**AXIS_STYLE),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with col_table:
        section_header("Raw Data", f"{n_rows:,} satır · {n_cols} kolon")
        if rows:
            df = pd.DataFrame(rows)
            for col in df.columns:
                if pd.api.types.is_float_dtype(df[col]):
                    df[col] = df[col].map(lambda x: f"{x:,.4f}" if pd.notna(x) else "")
            st.dataframe(df, use_container_width=True, hide_index=True, height=chart_height)
        else:
            st.info("Sonuç bulunamadı.")

def render_results(result: AgentState):
    """Renders the full dashboard from pipeline results."""
    
    if result.get("intent") == "sql_query":
        render_sql_results(result)
        return

    metrics = result.get("metrics", {})
    ev = result.get("evaluation_report", {})
    feature_report = result.get("feature_report", {})
    profiling = result.get("profiling_report", {})
    feature_plan = result.get("feature_plan", {})
    report = result.get("report", "")
    model_name = result.get("model_name", "unknown")
    intent = result.get("intent", "")

    test_m = ev.get("test_metrics", {})
    ovf = ev.get("overfitting", {})
    reliability = ev.get("reliability_score", {})
    fi = ev.get("feature_importance", {})
    fi_summary = ev.get("feature_importance_summary", {})
    confusion = ev.get("confusion_matrix", {})
    roc = ev.get("roc_curve", {})
    calibration = ev.get("calibration", {})
    per_class = ev.get("per_class_metrics", {})
    all_scores = metrics.get("all_model_scores", {})

    # Status bar
    grade = reliability.get("grade", "?")
    score = reliability.get("score", 0)
    grade_color = {"A": "#43e97b", "B": "#6c63ff", "C": "#f7971e", "D": "#ff6584", "F": "#ff6584"}.get(grade, "#8888aa")

    st.markdown(f"""
    <div style="background: {STATUS_BG}; border: 1px solid {STATUS_BORDER}; border-radius: 10px;
                padding: 1rem 1.5rem; display: flex; align-items: center; gap: 2rem; 
                margin-bottom: 2rem; flex-wrap: wrap;">
        <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #43e97b;">
            ✓ PIPELINE COMPLETE
        </div>
        <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: {CHART_LABEL_COLOR};">
            MODEL: <span style="color: {CHART_TEXT_COLOR};">{model_name.upper()}</span>
        </div>
        <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: {CHART_LABEL_COLOR};">
            INTENT: <span style="color: #6c63ff;">{intent.upper()}</span>
        </div>
        <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: {CHART_LABEL_COLOR};">
            RELIABILITY: <span style="color: {grade_color}; font-weight: 700;">{score}/100 · {grade}</span>
        </div>
        <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: {CHART_LABEL_COLOR};">
            FEATURES: <span style="color: {CHART_TEXT_COLOR};">{feature_report.get('n_features', '?')}</span>
        </div>
        <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: {CHART_LABEL_COLOR};">
            TARGET: <span style="color: {CHART_TEXT_COLOR};">{feature_report.get('target_name', '?').upper()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview",
        "🎯 Model Performance",
        "🔍 Feature Analysis",
        "⚖️ Fairness & Bias",
        "🧪 Data & Profiling",
        "🗺️ Pipeline Trace",
        "🗄️ Audit Trail",
    ])

    # TAB 1 — OVERVIEW
    with tab1:
        section_header("Key Metrics", "Core performance indicators from test set")

        # Metric cards
        auc = test_m.get("roc_auc", test_m.get("roc_auc_ovr", 0))
        acc = test_m.get("accuracy", 0)
        f1 = test_m.get("f1_weighted", 0)
        ll = test_m.get("log_loss", None)

        task = metrics.get("task", ev.get("task", "classification"))

        if task == "clustering":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(render_metric_card("Silhouette Score",
                    f"{ev.get('silhouette_score', 0):.4f}", "↑ Higher is better (-1 to 1)", "#6c63ff"), unsafe_allow_html=True)
            with c2:
                st.markdown(render_metric_card("N Clusters",
                    str(ev.get("n_clusters", "?")), "Optimal clusters found", "#43e97b"), unsafe_allow_html=True)
            with c3:
                st.markdown(render_metric_card("Algorithm",
                    str(ev.get("best_algorithm", "?")).upper(), "Best performing", "#f7971e"), unsafe_allow_html=True)
            with c4:
                db = ev.get("davies_bouldin", None)
                st.markdown(render_metric_card("Davies-Bouldin",
                    f"{db:.4f}" if db else "N/A", "↓ Lower is better", "#ff6584"), unsafe_allow_html=True)

        elif task == "anomaly_detection":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(render_metric_card("Anomalies Found",
                    str(ev.get("n_anomalies", "?")), "Total anomalies detected", "#ff6584"), unsafe_allow_html=True)
            with c2:
                st.markdown(render_metric_card("Anomaly Rate",
                    f"{ev.get('anomaly_pct', 0):.1f}%", "% of dataset", "#f7971e"), unsafe_allow_html=True)
            with c3:
                st.markdown(render_metric_card("Contamination",
                    str(ev.get("best_contamination", "?")), "Optimal threshold", "#6c63ff"), unsafe_allow_html=True)
            with c4:
                st.markdown(render_metric_card("Normal Records",
                    str(ev.get("n_normal", "?")), "Clean data points", "#43e97b"), unsafe_allow_html=True)

        elif task == "regression":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(render_metric_card("R²",
                    f"{test_m.get('r2', 0):.4f}", "↑ Higher is better", "#6c63ff"), unsafe_allow_html=True)
            with c2:
                st.markdown(render_metric_card("RMSE",
                    f"{test_m.get('rmse', 0):.4f}", "↓ Lower is better", "#ff6584"), unsafe_allow_html=True)
            with c3:
                st.markdown(render_metric_card("MAE",
                    f"{test_m.get('mae', 0):.4f}", "Mean Absolute Error", "#f7971e"), unsafe_allow_html=True)
            with c4:
                st.markdown(render_metric_card("CV R²",
                    f"{metrics.get('cv_r2', 0):.4f}", f"±{metrics.get('cv_r2_std', 0):.4f}", "#43e97b"), unsafe_allow_html=True)

        else:  # classification
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(render_metric_card("ROC-AUC", f"{auc:.4f}", "↑ Higher is better", "#6c63ff"), unsafe_allow_html=True)
            with c2:
                st.markdown(render_metric_card("Accuracy", f"{acc:.2%}", f"{'↑ ' if acc > 0.8 else '↓ '}{acc:.4f}", "#43e97b"), unsafe_allow_html=True)
            with c3:
                st.markdown(render_metric_card("F1 Weighted", f"{f1:.4f}", "Precision × Recall balance", "#f7971e"), unsafe_allow_html=True)
            with c4:
                if ll:
                    st.markdown(render_metric_card("Log Loss", f"{ll:.4f}", "↓ Lower is better", "#ff6584"), unsafe_allow_html=True)
                else:
                    st.markdown(render_metric_card("Reliability", f"{score}/100", f"Grade {grade}", grade_color), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1])

        # Left panel — task-aware
        with col_left:
            if task == "clustering":
                section_header("Cluster Distribution", "Sample count and proportion per cluster")
                cluster_dist = ev.get("cluster_distribution", metrics.get("cluster_distribution", {}))
                if cluster_dist:
                    _lc = [k for k in cluster_dist.keys() if k != "noise"]
                    _cc = [cluster_dist[k]["count"] for k in _lc]
                    _pc = [cluster_dist[k].get("pct", 0) for k in _lc]
                    fig_cd = go.Figure()
                    fig_cd.add_trace(go.Bar(
                        x=_lc, y=_cc,
                        marker_color=COLORS[:len(_lc)], marker_line_width=0,
                        text=[f"{p:.1f}%" for p in _pc], textposition="outside",
                        textfont=dict(family="DM Mono", size=11, color=CHART_TEXT_COLOR),
                    ))
                    fig_cd.update_layout(**PLOTLY_LAYOUT, height=220,
                        xaxis=dict(**AXIS_STYLE), yaxis=dict(**AXIS_STYLE), showlegend=False,
                        title=dict(text=f"Total: {sum(_cc):,} samples", font=dict(size=11, family="DM Mono"), x=0),
                    )
                    st.plotly_chart(fig_cd, use_container_width=True)
                _db = ev.get("davies_bouldin")
                _ch = ev.get("calinski_harabasz")
                st.markdown(f"""
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:{CHART_LABEL_COLOR};
                            padding:0.6rem 0.8rem;background:{INFO_BG};border-radius:6px;">
                    Silhouette: <span style="color:#6c63ff">{ev.get('silhouette_score',0):.4f}</span>
                    &nbsp;|&nbsp; Davies-Bouldin: <span style="color:{CHART_TEXT_COLOR}">{f'{_db:.4f}' if _db else 'N/A'}</span>
                    &nbsp;|&nbsp; Calinski-Harabasz: <span style="color:{CHART_TEXT_COLOR}">{f'{_ch:.0f}' if _ch else 'N/A'}</span>
                </div>
                """, unsafe_allow_html=True)

            elif task == "anomaly_detection":
                section_header("Anomaly Distribution", "Normal vs detected anomalies")
                _na = ev.get("n_anomalies", 0); _nn = ev.get("n_normal", 0)
                fig_ad = go.Figure(go.Pie(
                    labels=["Normal", "Anomaly"], values=[_nn, _na],
                    marker=dict(colors=["#43e97b", "#ff6584"]),
                    hole=0.55, textfont=dict(family="DM Mono", size=11),
                ))
                fig_ad.update_layout(**PLOTLY_LAYOUT, height=220, showlegend=True,
                    legend=dict(font=dict(family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_ad, use_container_width=True)
                _st = ev.get("score_threshold", 0)
                st.markdown(f"""
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:{CHART_LABEL_COLOR};
                            padding:0.6rem 0.8rem;background:{INFO_BG};border-radius:6px;">
                    Anomaly rate: <span style="color:#ff6584">{ev.get('anomaly_pct',0):.2f}%</span>
                    &nbsp;|&nbsp; Contamination: <span style="color:{CHART_TEXT_COLOR}">{ev.get('best_contamination','?')}</span>
                    &nbsp;|&nbsp; Score threshold: <span style="color:{CHART_TEXT_COLOR}">{_st:.4f if isinstance(_st, float) else _st}</span>
                </div>
                """, unsafe_allow_html=True)

            else:
                section_header("Overfitting Analysis", "Train vs Test performance gap")
                train_s = ovf.get("train_score", 0); test_s = ovf.get("test_score", 0)
                gap = ovf.get("gap", 0); status = ovf.get("status", "unknown")
                status_color = {"normal": "#43e97b", "mild_overfitting": "#f7971e",
                               "overfitting_detected": "#ff6584", "underfitting": "#38b2f4"}.get(status, "#8888aa")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=["Train", "Test"], y=[train_s, test_s],
                    marker_color=["#6c63ff", "#43e97b"], marker_line_width=0,
                    text=[f"{train_s:.4f}", f"{test_s:.4f}"], textposition="outside",
                    textfont=dict(family="DM Mono", size=11, color=CHART_TEXT_COLOR),
                ))
                fig.update_layout(**PLOTLY_LAYOUT, height=220,
                    title=dict(text=f"<span style='color:{status_color}'>{status.replace('_',' ').title()}</span> · gap: {gap:.4f}",
                              font=dict(size=11, family="DM Mono"), x=0),
                    xaxis=dict(**AXIS_STYLE),
                    yaxis=dict(**AXIS_STYLE, range=[max(0, min(train_s, test_s) - 0.1), 1.05]),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"""
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:{CHART_LABEL_COLOR};
                            padding:0.6rem 0.8rem;background:{INFO_BG};border-radius:6px;">
                    {ovf.get('description', '')}
                </div>
                """, unsafe_allow_html=True)

        # Right panel — task-aware
        with col_right:
            if task == "clustering":
                section_header("Algorithm Comparison", "Silhouette score by algorithm")
                _algo_sc = ev.get("all_algorithm_scores", {})
                if _algo_sc:
                    _best_algo = ev.get("best_algorithm", "")
                    _sa = dict(sorted(_algo_sc.items(), key=lambda x: float(x[1]), reverse=True))
                    _an = list(_sa.keys()); _av = [float(v) for v in _sa.values()]
                    _bc_a = [COLORS[0] if a == _best_algo else "#2a2a3a" for a in _an]
                    fig_algo = go.Figure()
                    fig_algo.add_trace(go.Bar(
                        x=_av, y=_an, orientation="h",
                        marker_color=_bc_a, marker_line_width=0,
                        text=[f"{v:.4f}" for v in _av], textposition="outside",
                        textfont=dict(family="DM Mono", size=10, color=CHART_TEXT_COLOR),
                    ))
                    fig_algo.update_layout(**PLOTLY_LAYOUT, height=220,
                        xaxis=dict(**AXIS_STYLE, range=[max(-1, min(_av) - 0.05), 1.0]),
                        yaxis=dict(**AXIS_STYLE), showlegend=False,
                        title=dict(text=f"Best: {_best_algo.upper()}", font=dict(size=11, family="DM Mono"), x=0),
                    )
                    st.plotly_chart(fig_algo, use_container_width=True)

            elif task == "anomaly_detection":
                section_header("Contamination Scan", "Anomaly % at each threshold")
                _cscan = ev.get("contamination_scan", {})
                if _cscan:
                    _bc_s = str(ev.get("best_contamination", ""))
                    _ck = list(_cscan.keys())
                    _cp = [_cscan[k]["anomaly_pct"] for k in _ck]
                    _cn = [_cscan[k]["n_anomalies"] for k in _ck]
                    _bc_cont = [COLORS[0] if k == _bc_s else "#2a2a3a" for k in _ck]
                    fig_cont = go.Figure()
                    fig_cont.add_trace(go.Bar(
                        x=[f"cont={k}" for k in _ck], y=_cp,
                        marker_color=_bc_cont, marker_line_width=0,
                        text=[f"{p:.1f}%" for p in _cp], textposition="outside",
                        textfont=dict(family="DM Mono", size=10, color=CHART_TEXT_COLOR),
                    ))
                    fig_cont.update_layout(**PLOTLY_LAYOUT, height=220,
                        xaxis=dict(**AXIS_STYLE), yaxis=dict(**AXIS_STYLE, title="Anomaly %"),
                        showlegend=False,
                        title=dict(text=f"Best contamination: {_bc_s}", font=dict(size=11, family="DM Mono"), x=0),
                    )
                    st.plotly_chart(fig_cont, use_container_width=True)

            else:
                section_header("Model Comparison", "5-fold cross-validation ROC-AUC")
                if all_scores:
                    sorted_scores = dict(sorted(all_scores.items(), key=lambda x: float(x[1]), reverse=True))
                    models = list(sorted_scores.keys())
                    scores = [float(v) for v in sorted_scores.values()]
                    bar_colors = [COLORS[0] if m == model_name else ("#2a2a3a" if is_dark else "#d0d0e0") for m in models]
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        x=scores, y=models, orientation="h",
                        marker_color=bar_colors, marker_line_width=0,
                        text=[f"{s:.4f}" for s in scores], textposition="outside",
                        textfont=dict(family="DM Mono", size=10, color=CHART_TEXT_COLOR),
                    ))
                    fig2.update_layout(**PLOTLY_LAYOUT, height=220,
                        xaxis=dict(**AXIS_STYLE, range=[min(scores) - 0.02, max(scores) + 0.02]),
                        yaxis=dict(**AXIS_STYLE), showlegend=False,
                    )
                    st.plotly_chart(fig2, use_container_width=True)

        # Reliability factors
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Reliability Breakdown", f"Score: {score}/100 · Grade: {grade}")
        factors = reliability.get("factors", [])
        if factors:
            cols = st.columns(min(len(factors), 4))
            for i, factor in enumerate(factors):
                with cols[i % 4]:
                    st.markdown(f"""
                    <div style="background: rgba(67,233,123,0.06); border: 1px solid rgba(67,233,123,0.15);
                                border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.5rem;">
                        <span style="color: #43e97b; font-size: 0.7rem; font-family: DM Mono;">✓ {factor}</span>
                    </div>
                    """, unsafe_allow_html=True)

    # TAB 2 — MODEL PERFORMANCE
    with tab2:
        if task == "clustering":
            col_l2, col_r2 = st.columns(2)
            with col_l2:
                section_header("Cluster Size Distribution", "Samples per cluster")
                _cd2 = ev.get("cluster_distribution", {})
                if _cd2:
                    _lc2 = list(_cd2.keys())
                    _cc2 = [_cd2[k]["count"] for k in _lc2]
                    _pc2 = [_cd2[k].get("pct", 0) for k in _lc2]
                    fig_cbar = go.Figure()
                    fig_cbar.add_trace(go.Bar(
                        x=_lc2, y=_cc2, marker_color=COLORS[:len(_lc2)], marker_line_width=0,
                        text=[f"{p:.1f}%" for p in _pc2], textposition="outside",
                        textfont=dict(family="DM Mono", size=11, color=CHART_TEXT_COLOR),
                    ))
                    fig_cbar.update_layout(**PLOTLY_LAYOUT, height=300,
                        xaxis=dict(**AXIS_STYLE), yaxis=dict(**AXIS_STYLE, title="Count"), showlegend=False,
                    )
                    st.plotly_chart(fig_cbar, use_container_width=True)

            with col_r2:
                section_header("Silhouette Score Gauge", "Cluster separation quality")
                _sil = ev.get("silhouette_score", 0)
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=_sil,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Silhouette Score", "font": {"family": "DM Mono", "size": 13, "color": CHART_LABEL_COLOR}},
                    number={"font": {"family": "DM Mono", "size": 30, "color": CHART_TEXT_COLOR}, "suffix": ""},
                    gauge={
                        "axis": {"range": [-1, 1], "tickfont": {"family": "DM Mono", "size": 10}},
                        "bar": {"color": "#6c63ff"},
                        "steps": [
                            {"range": [-1, 0], "color": "#1a1218"},
                            {"range": [0, 0.5], "color": "#151525"},
                            {"range": [0.5, 1], "color": "#151a30"},
                        ],
                        "threshold": {"line": {"color": "#43e97b", "width": 2}, "value": 0.5},
                    },
                ))
                fig_gauge.update_layout(**PLOTLY_LAYOUT, height=260)
                st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            _m3 = st.columns(3)
            with _m3[0]:
                st.markdown(render_metric_card("N Clusters", str(ev.get("n_clusters","?")), "Optimal found", "#6c63ff"), unsafe_allow_html=True)
            with _m3[1]:
                _db2 = ev.get("davies_bouldin")
                st.markdown(render_metric_card("Davies-Bouldin", f"{_db2:.4f}" if _db2 else "N/A", "↓ Lower is better", "#f7971e"), unsafe_allow_html=True)
            with _m3[2]:
                _ch2 = ev.get("calinski_harabasz")
                st.markdown(render_metric_card("Calinski-Harabasz", f"{_ch2:.0f}" if _ch2 else "N/A", "↑ Higher is better", "#43e97b"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            _as2 = ev.get("all_algorithm_scores", {})
            if _as2:
                section_header("Algorithm Comparison", "Silhouette score per algorithm tested")
                _adf = pd.DataFrame([
                    {"Algorithm": k.upper(), "Silhouette": round(v, 4),
                     "Status": "BEST" if k == ev.get("best_algorithm") else ""}
                    for k, v in sorted(_as2.items(), key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(_adf, use_container_width=True, hide_index=True)

        elif task == "anomaly_detection":
            col_l2, col_r2 = st.columns(2)
            with col_l2:
                section_header("Detection Breakdown", "Normal vs anomaly records")
                _na2 = ev.get("n_anomalies", 0); _nn2 = ev.get("n_normal", 0)
                fig_pie2 = go.Figure(go.Pie(
                    labels=["Normal", "Anomaly"], values=[_nn2, _na2],
                    marker=dict(colors=["#43e97b", "#ff6584"]),
                    hole=0.55, textfont=dict(family="DM Mono", size=12), textinfo="label+percent+value",
                ))
                fig_pie2.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=True,
                    legend=dict(font=dict(family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)"),
                    title=dict(text=f"Total: {_na2+_nn2:,} records", font=dict(size=11, family="DM Mono"), x=0),
                )
                st.plotly_chart(fig_pie2, use_container_width=True)

            with col_r2:
                section_header("Contamination Scan", "Anomaly count at each threshold")
                _cs2 = ev.get("contamination_scan", {})
                if _cs2:
                    _bc2 = str(ev.get("best_contamination", ""))
                    _ck2 = list(_cs2.keys())
                    _cp2 = [_cs2[k]["anomaly_pct"] for k in _ck2]
                    _cn2 = [_cs2[k]["n_anomalies"] for k in _ck2]
                    fig_scan = go.Figure()
                    fig_scan.add_trace(go.Bar(
                        x=[f"cont={k}" for k in _ck2], y=_cp2,
                        marker_color=[COLORS[0] if k == _bc2 else ("#2a2a3a" if is_dark else "#d0d0e0") for k in _ck2],
                        marker_line_width=0,
                        text=[f"{p:.1f}%<br>({n} pts)" for p, n in zip(_cp2, _cn2)],
                        textposition="outside",
                        textfont=dict(family="DM Mono", size=10, color=CHART_TEXT_COLOR),
                    ))
                    fig_scan.update_layout(**PLOTLY_LAYOUT, height=300,
                        xaxis=dict(**AXIS_STYLE), yaxis=dict(**AXIS_STYLE, title="Anomaly %"), showlegend=False,
                        title=dict(text=f"Selected contamination={_bc2} (highlighted)",
                                  font=dict(size=11, family="DM Mono"), x=0),
                    )
                    st.plotly_chart(fig_scan, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            _fi_a = ev.get("feature_importance", {})
            if _fi_a:
                section_header("Anomaly Drivers (Feature Importance)", "Features most responsible for anomaly detection")
                _tot_a = sum(_fi_a.values()) or 1
                _top_a = dict(list(_fi_a.items())[:15])
                _fi_ap = {k: round(v / _tot_a * 100, 2) for k, v in _top_a.items()}
                _feats_a = list(_fi_ap.keys())[::-1]
                _vals_a = [_fi_ap[f] for f in _feats_a]
                fig_fi_a = go.Figure()
                fig_fi_a.add_trace(go.Bar(
                    x=_vals_a, y=_feats_a, orientation="h",
                    marker_color=[f"rgba(255,101,132,{0.35 + 0.65 * v / max(_vals_a)})" for v in _vals_a],
                    marker_line_width=0,
                    text=[f"{v:.1f}%" for v in _vals_a], textposition="outside",
                    textfont=dict(family="DM Mono", size=10, color=CHART_TEXT_COLOR),
                ))
                fig_fi_a.update_layout(**PLOTLY_LAYOUT,
                    height=max(250, len(_feats_a) * 28), showlegend=False,
                    xaxis=dict(**AXIS_STYLE, title="Relative Importance (%)"),
                    yaxis=dict(**AXIS_STYLE),
                )
                st.plotly_chart(fig_fi_a, use_container_width=True)

        else:
            # Classification / Regression — existing charts
            col_l, col_r = st.columns(2)
            with col_l:
                section_header("Confusion Matrix", "Actual vs Predicted")
                if confusion:
                    cm_matrix = confusion.get("matrix", [])
                    labels = [str(l) for l in confusion.get("labels", [])]
                    if cm_matrix:
                        fig_cm = px.imshow(
                            cm_matrix, x=labels, y=labels,
                            color_continuous_scale=[[0, "#0a0a0f"], [0.5, "#2a1f5a"], [1, "#6c63ff"]],
                            text_auto=True,
                        )
                        fig_cm.update_traces(textfont=dict(family="DM Mono", size=14, color="white"))
                        fig_cm.update_layout(**PLOTLY_LAYOUT, height=300,
                            xaxis=dict(**AXIS_STYLE, title="Predicted"),
                            yaxis=dict(**AXIS_STYLE, title="Actual"),
                            coloraxis_showscale=False,
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)

            with col_r:
                section_header("ROC Curve", f"AUC = {auc:.4f}")
                if roc:
                    fpr_vals = roc.get("fpr", []); tpr_vals = roc.get("tpr", [])
                    opt_fpr = roc.get("optimal_fpr"); opt_tpr = roc.get("optimal_tpr")
                    opt_thresh = roc.get("optimal_threshold")
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                        line=dict(color="rgba(255,255,255,0.15)", dash="dash", width=1), showlegend=False))
                    fig_roc.add_trace(go.Scatter(x=fpr_vals, y=tpr_vals, mode="lines", fill="tozeroy",
                        line=dict(color="#6c63ff", width=2), fillcolor="rgba(108,99,255,0.1)", name=f"AUC = {auc:.4f}"))
                    if opt_fpr and opt_tpr:
                        fig_roc.add_trace(go.Scatter(x=[opt_fpr], y=[opt_tpr], mode="markers",
                            marker=dict(color="#ff6584", size=10, symbol="star"), name=f"Optimal = {opt_thresh}"))
                    fig_roc.update_layout(**PLOTLY_LAYOUT, height=300,
                        xaxis=dict(**AXIS_STYLE, title="False Positive Rate"),
                        yaxis=dict(**AXIS_STYLE, title="True Positive Rate"),
                        legend=dict(font=dict(family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)"),
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if per_class:
                section_header("Per-Class Performance", "Precision, Recall, F1 by class")
                pc_df = pd.DataFrame(per_class).T.reset_index()
                pc_df.columns = ["Class", "Precision", "Recall", "F1", "Support"]
                fig_pc = go.Figure()
                for metric, color in [("Precision", "#6c63ff"), ("Recall", "#43e97b"), ("F1", "#f7971e")]:
                    fig_pc.add_trace(go.Bar(name=metric, x=pc_df["Class"].astype(str), y=pc_df[metric],
                        marker_color=color, marker_line_width=0))
                fig_pc.update_layout(**PLOTLY_LAYOUT, height=250, barmode="group",
                    xaxis=dict(**AXIS_STYLE), yaxis=dict(**AXIS_STYLE),
                    legend=dict(font=dict(family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_pc, use_container_width=True)

            if calibration:
                section_header("Model Calibration", f"Brier Score: {calibration.get('brier_score')}")
                prob_true = calibration.get("prob_true", []); prob_pred = calibration.get("prob_pred", [])
                is_cal = calibration.get("is_well_calibrated", False)
                fig_cal = go.Figure()
                fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(color="rgba(255,255,255,0.2)", dash="dash", width=1), name="Perfect calibration", showlegend=True))
                fig_cal.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers",
                    line=dict(color="#f7971e", width=2), marker=dict(size=7, color="#f7971e"), name="Model calibration"))
                fig_cal.update_layout(**PLOTLY_LAYOUT, height=250,
                    xaxis=dict(**AXIS_STYLE, title="Mean Predicted Probability"),
                    yaxis=dict(**AXIS_STYLE, title="Fraction of Positives"),
                    legend=dict(font=dict(family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)"),
                    title=dict(text=f"{'✓ Well calibrated' if is_cal else '⚠ Poorly calibrated'}",
                        font=dict(size=11, family="DM Mono", color="#43e97b" if is_cal else "#f7971e"), x=0),
                )
                st.plotly_chart(fig_cal, use_container_width=True)

    # TAB 3 — FEATURE ANALYSIS
    with tab3:
        if fi:
            section_header("Feature Importance", f"Top features · {fi_summary.get('top5_total_contribution_pct', '?')}% of total impact from top 5")

            top_n = 20
            top_fi = dict(list(fi.items())[:top_n])
            total_fi = sum(top_fi.values()) or 1
            fi_pct = {k: round(v / total_fi * 100, 2) for k, v in top_fi.items()}

            fig_fi = go.Figure()
            feats = list(fi_pct.keys())[::-1]
            vals = [fi_pct[f] for f in feats]
            bar_colors_fi = [
                f"rgba(108,99,255,{0.4 + 0.6 * v / max(vals)})" for v in vals
            ]

            fig_fi.add_trace(go.Bar(
                x=vals, y=feats, orientation="h",
                marker_color=bar_colors_fi, marker_line_width=0,
                text=[f"{v:.1f}%" for v in vals],
                textposition="outside",
                textfont=dict(family="DM Mono", size=10, color=CHART_TEXT_COLOR),
            ))
            fig_fi.update_layout(**PLOTLY_LAYOUT,
                height=max(300, len(feats) * 28),
                showlegend=False,
                xaxis=dict(**AXIS_STYLE, title="Relative Importance (%)"),
                yaxis=dict(**AXIS_STYLE),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            # Insignificant features
            insig = fi_summary.get("insignificant_features", [])
            insig_count = fi_summary.get("insignificant_features_count", 0)
            if insig_count > 0:
                with st.expander(f"⚠ {insig_count} Insignificant Features (<1% contribution)"):
                    st.markdown(f"""
                    <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #8888aa; line-height: 1.8;">
                        {' · '.join(insig)}
                    </div>
                    """, unsafe_allow_html=True)

            # SHAP Analysis
            shap_info = ev.get("shap_importance", {})
            if shap_info and shap_info.get("all_shap_values"):
                st.markdown("<br>", unsafe_allow_html=True)
                section_header("SHAP Explainability", "Global feature impact based on mean |SHAP| values")
                shap_vals = shap_info.get("all_shap_values", {})
                
                # Plot SHAP bar chart
                top_shap = dict(list(shap_vals.items())[:top_n])
                total_shap = sum(top_shap.values()) or 1
                shap_pct = {k: round(v / total_shap * 100, 2) for k, v in top_shap.items()}
                
                fig_shap = go.Figure()
                feats_s = list(shap_pct.keys())[::-1]
                vals_s = [shap_pct[f] for f in feats_s]
                bar_colors_shap = [f"rgba(255,101,132,{0.4 + 0.6 * v / max(vals_s)})" for v in vals_s]
                
                fig_shap.add_trace(go.Bar(
                    x=vals_s, y=feats_s, orientation="h",
                    marker_color=bar_colors_shap, marker_line_width=0,
                    text=[f"{v:.1f}%" for v in vals_s],
                    textposition="outside",
                    textfont=dict(family="DM Mono", size=10, color=CHART_TEXT_COLOR),
                ))
                fig_shap.update_layout(**PLOTLY_LAYOUT,
                    height=max(300, len(feats_s) * 28),
                    showlegend=False,
                    xaxis=dict(**AXIS_STYLE, title="Relative SHAP Impact (%)"),
                    yaxis=dict(**AXIS_STYLE),
                )
                st.plotly_chart(fig_shap, use_container_width=True)

        else:
            if task == "clustering":
                section_header("Cluster Statistics", "Detailed per-cluster breakdown")
                _cd3 = ev.get("cluster_distribution", {})
                if _cd3:
                    _crows = []
                    for _cname, _cinfo in _cd3.items():
                        _crows.append({"Cluster": _cname, "Count": _cinfo.get("count", 0),
                                       "Share %": f"{_cinfo.get('pct', 0):.1f}%"})
                    st.dataframe(pd.DataFrame(_crows), use_container_width=True, hide_index=True)
                st.markdown(f"""
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:{CHART_LABEL_COLOR};
                            padding:0.8rem;background:{INFO_BG};border-radius:8px;
                            border-left:2px solid #6c63ff;margin-top:0.8rem;">
                    <strong style="color:{CHART_TEXT_COLOR};">Note:</strong> Feature importance is not available for
                    KMeans / DBSCAN clustering algorithms. Use the anomaly_detection intent with
                    IsolationForest to get feature-level importance scores.
                </div>
                """, unsafe_allow_html=True)
            elif task == "anomaly_detection":
                st.info("Feature importance will appear here once IsolationForest computes it.")
            else:
                st.info("Feature importance data not available.")

        # Feature Engineering summary
        if feature_report:
            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Feature Engineering Summary", "Transformations applied by the pipeline")

            if task in ("clustering", "anomaly_detection"):
                # Unsupervised — no train/test split, show total samples only
                _total_samples = feature_report.get("n_train", 0) + feature_report.get("n_test", 0)
                fr_cols = st.columns(2)
                with fr_cols[0]:
                    st.markdown(render_metric_card("Final Features", str(feature_report.get("n_features", "?")), "After selection", "#6c63ff"), unsafe_allow_html=True)
                with fr_cols[1]:
                    st.markdown(render_metric_card("Total Samples", str(_total_samples) if _total_samples else str(feature_report.get("n_train", "?")), "Full dataset used", "#43e97b"), unsafe_allow_html=True)
            else:
                fr_cols = st.columns(3)
                with fr_cols[0]:
                    st.markdown(render_metric_card("Final Features", str(feature_report.get("n_features", "?")), "After selection", "#6c63ff"), unsafe_allow_html=True)
                with fr_cols[1]:
                    st.markdown(render_metric_card("Train Samples", str(feature_report.get("n_train", "?")), "80% split", "#43e97b"), unsafe_allow_html=True)
                with fr_cols[2]:
                    st.markdown(render_metric_card("Test Samples", str(feature_report.get("n_test", "?")), "20% split", "#f7971e"), unsafe_allow_html=True)

                # Target distribution (supervised only)
                dist = feature_report.get("target_distribution_train", {})
                if dist:
                    st.markdown("<br>", unsafe_allow_html=True)
                    section_header("Target Distribution (Train)", "Class balance in training set")
                    labels_d = [str(k) for k in dist.keys()]
                    values_d = list(dist.values())
                    fig_dist = go.Figure(go.Pie(
                        labels=labels_d, values=values_d,
                        marker=dict(colors=COLORS[:len(labels_d)]),
                        hole=0.6,
                        textfont=dict(family="DM Mono", size=11),
                    ))
                    fig_dist.update_layout(**PLOTLY_LAYOUT, height=250, showlegend=True,
                        legend=dict(font=dict(family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)"),
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

    # TAB 4 — FAIRNESS & BIAS
    with tab4:
        bias_info = ev.get("bias_analysis", {})
        if bias_info:
            section_header("Demographic Fairness & Bias", "Disparate Impact and Performance Differences")
            for demographic_col, metrics_dict in bias_info.items():
                st.markdown(f"#### 🎭 {demographic_col.upper()}")
                for encoded_col, m in metrics_dict.items():
                    with st.expander(f"{encoded_col}"):
                        diff = m.get('accuracy_diff', m.get('rmse_diff', 0))
                        di = m.get('disparate_impact')
                        
                        cols = st.columns(3)
                        cols[0].metric(label="Group 1 Metric", value=m.get('accuracy_group_1', m.get('rmse_group_1')))
                        cols[1].metric(label="Group 0 Metric", value=m.get('accuracy_group_0', m.get('rmse_group_0')))
                        cols[2].metric(label="Difference", value=diff)
                        
                        if di is not None:
                            status = "🟢 Fair" if 0.8 <= di <= 1.25 else "🔴 High Bias"
                            st.markdown(f"**Disparate Impact Ratio**: {di:.3f} &nbsp;&nbsp;&nbsp; {status}")
        else:
            st.info("No demographic bias detected or no demographic features available.")

    # TAB 5 — DATA & PROFILING
    with tab5:
        if profiling:
            shape = profiling.get("shape", {})
            summary = profiling.get("summary", {})
            columns = profiling.get("columns", {})

            section_header("Dataset Overview", "Raw data statistics from execution layer")

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.markdown(render_metric_card("Rows", f"{shape.get('rows', '?'):,}", "Total records", "#6c63ff"), unsafe_allow_html=True)
            with d2:
                st.markdown(render_metric_card("Columns", str(shape.get("columns", "?")), "Total features", "#43e97b"), unsafe_allow_html=True)
            with d3:
                missing_rate = summary.get("missing_rate", 0)
                st.markdown(render_metric_card("Missing Rate", f"{missing_rate:.2%}", "Null values ratio", "#f7971e" if missing_rate > 0.05 else "#43e97b"), unsafe_allow_html=True)
            with d4:
                dups = summary.get("duplicate_rows", 0)
                st.markdown(render_metric_card("Duplicates", str(dups), "Removed during cleaning", "#ff6584" if dups > 0 else "#43e97b"), unsafe_allow_html=True)

            # Column type distribution
            st.markdown("<br>", unsafe_allow_html=True)
            num_cols = len(summary.get("numeric_columns", []))
            cat_cols = len(summary.get("categorical_columns", []))
            dt_cols = len(summary.get("datetime_columns", []))

            fig_types = go.Figure(go.Pie(
                labels=["Numeric", "Categorical", "Datetime"],
                values=[num_cols, cat_cols, dt_cols],
                marker=dict(colors=["#6c63ff", "#ff6584", "#43e97b"]),
                hole=0.5,
                textfont=dict(family="DM Mono", size=11),
            ))
            fig_types.update_layout(**PLOTLY_LAYOUT, height=220, showlegend=True,
                legend=dict(font=dict(family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)"),
                title=dict(text="Column Type Distribution", font=dict(size=11, family="DM Mono"), x=0),
            )
            st.plotly_chart(fig_types, use_container_width=True)

            # Profiling recommendations
            recs = profiling.get("recommendations", [])
            if recs:
                section_header("Profiling Recommendations", "Automated data quality insights")
                for rec in recs:
                    icon = "⚠" if "outlier" in rec.lower() or "missing" in rec.lower() else "ℹ"
                    color = "#f7971e" if icon == "⚠" else "#6c63ff"
                    st.markdown(f"""
                    <div style="display: flex; align-items: flex-start; gap: 0.6rem; margin-bottom: 0.4rem;
                                padding: 0.5rem 0.8rem; background: {INFO_BG}; 
                                border-radius: 6px; border-left: 2px solid {color};">
                        <span style="color: {color}; font-size: 0.7rem; margin-top: 1px;">{icon}</span>
                        <span style="font-family: DM Mono; font-size: 0.72rem; color: {CHART_LABEL_COLOR};">{rec}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # Column stats table
            if columns:
                section_header("Column Statistics", "Per-column profiling data")
                col_data = []
                for col_name, info in columns.items():
                    col_data.append({
                        "Column": col_name,
                        "Type": info.get("dtype", ""),
                        "Null %": f"{info.get('null_rate', 0):.1%}",
                        "Unique": info.get("unique_count", ""),
                        "Outliers": "⚠" if info.get("has_outliers") else "✓",
                        "Mean": f"{info.get('mean', ''):.3f}" if info.get("mean") is not None else "",
                        "Std": f"{info.get('std', ''):.3f}" if info.get("std") is not None else "",
                    })
                df_cols = pd.DataFrame(col_data)
                st.dataframe(
                    df_cols, use_container_width=True, hide_index=True,
                    column_config={
                        "Outliers": st.column_config.TextColumn(width="small"),
                        "Null %": st.column_config.TextColumn(width="small"),
                    }
                )

    # TAB 6 — PIPELINE TRACE
    with tab6:
        section_header("Pipeline Execution Trace", "Step-by-step node outputs")

        nodes_trace = [
            ("01 · Intent", "intent_node", {
                "Intent": result.get("intent", ""),
                "Confidence": f"{result.get('intent_confidence', 0):.2%}",
            }),
            ("02 · DB Context", "db_context_node", {
                "Tables found": str(result.get("relevant_tables", [])),
                "Relationships": str(len(result.get("db_intelligence", {}).get("relationships", []))),
                "Capabilities": str(result.get("db_intelligence", {}).get("capabilities", [])),
            }),
            ("03 · Planner", "planner_node", {
                "Steps": str(len(result.get("analysis_plan", []))),
                "Plan": " → ".join([s.get("step", "") for s in result.get("analysis_plan", [])]),
            }),
            ("04 · Execution", "execution_node", {
                "Rows fetched": f"{len(result.get('sql_results', [])):,}",
                "Columns": str(profiling.get("shape", {}).get("columns", "")),
                "Recommendations": str(len(profiling.get("recommendations", []))),
            }),
            ("05 · Feature Plan", "feature_plan_node", {
                "Target": feature_plan.get("target", {}).get("column_name", ""),
                "Is derived": str(feature_plan.get("target", {}).get("is_derived", "")),
                "Aggregation": feature_plan.get("aggregation", {}).get("level", ""),
                "Feature groups": str(feature_plan.get("feature_groups", [])),
            }),
            ("06 · Feature Eng", "feature_engineering_node", {
                "Final features": str(feature_report.get("n_features", "")),
                "Train size": str(feature_report.get("n_train", "")),
                "Test size": str(feature_report.get("n_test", "")),
                "Target dist": str(feature_report.get("target_distribution_train", {})),
            }),
            ("07 · AutoML", "automl_node", {
                "Best model": model_name,
                **(
                    {
                        "Algorithm": str(metrics.get("best_algorithm", "?")),
                        "Silhouette": f"{metrics.get('silhouette_score', 0):.4f}",
                        "N Clusters": str(metrics.get("n_clusters", "?")),
                    } if intent == "clustering" else {
                        "N Anomalies": str(metrics.get("n_anomalies", "?")),
                        "Anomaly %": f"{metrics.get('anomaly_pct', 0):.2f}%",
                        "Contamination": str(metrics.get("best_contamination", "?")),
                    } if intent == "anomaly_detection" else {
                        "CV ROC-AUC": f"{metrics.get('cv_roc_auc', 0):.4f} ± {metrics.get('cv_roc_auc_std', 0):.4f}",
                        "Test accuracy": f"{test_m.get('accuracy', 0):.4f}",
                        "Test ROC-AUC": f"{test_m.get('roc_auc', test_m.get('roc_auc_ovr', 0)):.4f}",
                    }
                ),
            }),
            ("08 · Evaluation", "evaluation_node", {
                "Reliability score": f"{reliability.get('score', 0)}/100 (Grade {reliability.get('grade', '?')})",
                "Overfitting": ovf.get("status", ""),
                "Train-Test gap": f"{ovf.get('gap', 0):.4f}",
                "Calibration": f"Brier={calibration.get('brier_score', 'N/A')}",
            }),
            ("09 · Report", "report_node", {
                "Report length": f"{len(report):,} chars",
                "Status": "✓ Generated" if report else "⚠ Failed",
            }),
        ]

        for node_title, node_key, node_info in nodes_trace:
            with st.expander(node_title, expanded=False):
                for k, v in node_info.items():
                    st.markdown(f"""
                    <div style="display: flex; gap: 1rem; padding: 0.3rem 0; 
                                border-bottom: 1px solid {TRACE_BORDER};">
                        <div style="font-family: DM Mono; font-size: 0.7rem; color: {CHART_LABEL_COLOR}; 
                                    min-width: 140px;">{k}</div>
                        <div style="font-family: DM Mono; font-size: 0.7rem; color: {CHART_TEXT_COLOR};">{v}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # TAB 7 — FULL REPORT & EXPERIMENT HISTORY
    with tab7:
        if report:
            section_header("Generated Report", "LLM-synthesized analysis in Turkish")
            st.markdown(f"""
            <div style="background: {STATUS_BG}; border: 1px solid {STATUS_BORDER}; 
                        border-radius: 12px; padding: 2rem; font-family: 'DM Mono', monospace;
                        font-size: 0.82rem; line-height: 1.8; color: {CHART_TEXT_COLOR};">
            """, unsafe_allow_html=True)
            st.markdown(report)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Report not generated.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Experiment History", "Log of past runs from history.db")
        history = get_experiments()
        if history:
            df_history = pd.DataFrame(history)
            # Clean up the dataframe for display
            if "id" in df_history.columns:
                df_history = df_history.drop(columns=["id"])
            if "timestamp" in df_history.columns:
                df_history["timestamp"] = pd.to_datetime(df_history["timestamp"]).dt.strftime('%Y-%m-%d %H:%M:%S')
                
            st.dataframe(df_history, use_container_width=True, hide_index=True)
        else:
            st.info("No experiment history found yet.")


# MAIN FLOW

if "result" not in st.session_state:
    st.session_state.result = None

if run_btn and query.strip():
    with st.spinner(""):
        show_loading()
        with st.spinner("Finalizing pipeline..."):
            result = run_pipeline(query)
            st.session_state.result = result
    st.rerun()

elif not run_btn and st.session_state.result is None:
    # Welcome screen
    st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;
                min-height: 50vh; text-align: center; padding: 4rem 2rem;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">⬡</div>
        <h2 style="font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800;
                   letter-spacing: -0.03em; color: #e8e8f0; margin: 0 0 0.5rem 0;">
            Ready to Analyze
        </h2>
        <p style="font-family: 'DM Mono', monospace; font-size: 0.8rem; color: #8888aa; 
                  max-width: 400px; line-height: 1.7; margin: 0;">
            Type your analysis query in the sidebar and click <strong style="color: #6c63ff;">Run Analysis</strong> 
            to start the autonomous ML pipeline.
        </p>
        <div style="margin-top: 2rem; display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center;">
            <div style="background: rgba(108,99,255,0.08); border: 1px solid rgba(108,99,255,0.2);
                        border-radius: 8px; padding: 0.6rem 1rem; font-family: DM Mono; 
                        font-size: 0.7rem; color: #8888aa;">
                "Predict income level from census data"
            </div>
            <div style="background: rgba(108,99,255,0.08); border: 1px solid rgba(108,99,255,0.2);
                        border-radius: 8px; padding: 0.6rem 1rem; font-family: DM Mono; 
                        font-size: 0.7rem; color: #8888aa;">
                "Segment customers by purchase behavior"
            </div>
            <div style="background: rgba(108,99,255,0.08); border: 1px solid rgba(108,99,255,0.2);
                        border-radius: 8px; padding: 0.6rem 1rem; font-family: DM Mono; 
                        font-size: 0.7rem; color: #8888aa;">
                "Forecast monthly revenue trends"
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if st.session_state.result is not None:
    render_results(st.session_state.result)