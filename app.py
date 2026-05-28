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
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.graph import build_graph
from src.state import AgentState
from src.history_tracker import get_experiments

st.set_page_config(
    page_title="DS Agent",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Agent Chat"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

is_dark = st.session_state.theme == "dark"

DARK_VARS = """
    --bg:     #08080e;
    --bg2:    #0d0d16;
    --bg3:    #12121c;
    --border: rgba(108,99,255,0.12);
    --border2:rgba(108,99,255,0.22);
    --text:   #e8e8f0;
    --text2:  #7777aa;
    --text3:  #3a3a5a;
"""
LIGHT_VARS = """
    --bg:     #f2f2f8;
    --bg2:    #ffffff;
    --bg3:    #eaeaf4;
    --border: rgba(108,99,255,0.1);
    --border2:rgba(108,99,255,0.2);
    --text:   #1a1a2e;
    --text2:  #5a5a7a;
    --text3:  #aaaacc;
"""

theme_vars = DARK_VARS if is_dark else LIGHT_VARS

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {{
    {theme_vars}
    --accent:  #6c63ff;
    --pink:    #ff6584;
    --green:   #43e97b;
    --orange:  #f7971e;
    --mono:    'JetBrains Mono', monospace;
    --sans:    'Space Grotesk', sans-serif;
}}

html, body, [data-testid="stAppViewContainer"] {{
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}}

[data-testid="stSidebar"] {{
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: {'4px 0 32px rgba(0,0,0,0.5)' if is_dark else '4px 0 16px rgba(0,0,0,0.06)'} !important;
}}
[data-testid="stSidebar"] * {{ color: var(--text) !important; }}

#MainMenu, footer {{ visibility: hidden; }}
header {{ background: transparent !important; }}
[data-testid="stDecoration"] {{ display: none; }}

.stTextArea textarea, .stTextInput input {{
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
    transition: all 0.2s !important;
}}
.stTextArea textarea:focus, .stTextInput input:focus {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.1) !important;
}}

.stButton > button {{
    background: linear-gradient(135deg, #6c63ff, #9c56d8, #ff6584) !important;
    background-size: 200% 200% !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.03em !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.4) !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(108,99,255,0.55) !important;
}}
.stButton > button:active {{
    transform: translateY(0) scale(0.98) !important;
}}

[data-testid="stMetric"] {{
    background: {'rgba(108,99,255,0.06)' if is_dark else 'rgba(108,99,255,0.04)'} !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}}
[data-testid="stMetricLabel"] {{ color: var(--text2) !important; font-family: var(--mono) !important; font-size: 0.72rem !important; }}
[data-testid="stMetricValue"] {{ color: var(--text) !important; font-family: var(--sans) !important; font-weight: 600 !important; }}

[data-testid="stTabs"] [role="tablist"] {{
    background: var(--bg3) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
}}
[data-testid="stTabs"] [role="tab"] {{
    color: var(--text2) !important;
    font-family: var(--sans) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    border-radius: 9px !important;
    transition: all 0.2s !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg, #6c63ff, #9c56d8) !important;
    color: white !important;
    box-shadow: 0 2px 12px rgba(108,99,255,0.4) !important;
}}

[data-testid="stExpander"] {{
    background: {'rgba(255,255,255,0.02)' if is_dark else 'rgba(0,0,0,0.02)'} !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s !important;
}}
[data-testid="stExpander"]:hover {{
    border-color: var(--border2) !important;
}}

code {{ font-family: var(--mono) !important; font-size: 0.82em !important; }}
hr {{ border-color: var(--border) !important; }}

::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{
    background: {'rgba(108,99,255,0.25)' if is_dark else 'rgba(108,99,255,0.18)'};
    border-radius: 2px;
}}

@keyframes pulse-dot {{
    0%,100% {{ opacity:1; transform:scale(1); }}
    50%      {{ opacity:0.4; transform:scale(0.75); }}
}}
@keyframes scan {{
    0%   {{ transform:translateX(-100%); }}
    100% {{ transform:translateX(300%); }}
}}
@keyframes fade-up {{
    from {{ opacity:0; transform:translateY(6px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}
@media print {{
    @page {{
        margin: 0;
    }}
    * {{
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
    }}
    html, body, [data-testid="stAppViewContainer"], .stApp, .main, .block-container {{
        background: var(--bg) !important;
        color: var(--text) !important;
        height: auto !important;
        min-height: 100% !important;
        overflow: visible !important;
        display: block !important;
        position: relative !important;
    }}
    .main .block-container {{
        padding: 1.5cm 1.5cm !important;
        max-width: 100% !important;
    }}
    [data-testid="stSidebar"] {{ display: none !important; }}
    header {{ display: none !important; }}
    .stChatInputContainer {{ display: none !important; }}
    /* Hide the print button iframe itself */
    iframe {{ display: none !important; }}
    
    /* Mesajların ortadan ikiye bölünüp beyaz boşluk yaratmasını önler */
    [data-testid="stChatMessage"] {{
        page-break-inside: avoid !important;
        break-inside: avoid !important;
    }}
    
    /* Gereksiz boşlukları engellemek için */
    [data-testid="stVerticalBlock"] {{
        gap: 0 !important;
    }}
}}
</style>
""", unsafe_allow_html=True)

CHART_TEXT_COLOR  = "#e8e8f0" if is_dark else "#1a1a2e"
CHART_LABEL_COLOR = "#7777aa" if is_dark else "#5a5a7a"
INFO_BG    = "rgba(255,255,255,0.03)" if is_dark else "rgba(0,0,0,0.03)"
STATUS_BG  = "#0d0d16" if is_dark else "#f0f0f8"
STATUS_BORDER = "rgba(108,99,255,0.15)" if is_dark else "rgba(108,99,255,0.1)"
TRACE_BORDER  = "rgba(255,255,255,0.04)" if is_dark else "rgba(0,0,0,0.05)"
COLORS = ["#6c63ff","#ff6584","#43e97b","#f7971e","#38b2f4","#f093fb"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color=CHART_LABEL_COLOR, size=11),
    margin=dict(l=10, r=10, t=30, b=10),
)
AXIS_STYLE = dict(
    gridcolor="rgba(255,255,255,0.04)" if is_dark else "rgba(0,0,0,0.05)",
    zerolinecolor="rgba(255,255,255,0.08)" if is_dark else "rgba(0,0,0,0.08)",
    tickfont=dict(color=CHART_LABEL_COLOR),
)


# ── DB status ────────────────────────────────────────────────
@st.cache_data(ttl=60)
def check_db_status():
    try:
        from src.core.db_provider import db_manager
        from sqlalchemy import text
        stats = []
        for db_name, provider in db_manager.get_all_providers().items():
            try:
                with provider.connect() as conn:
                    tables_result = conn.execute(text(provider.get_table_names_query()))
                    tables = [r[0] for r in tables_result]
                    total_rows = 0
                    for table in tables:
                        count_result = conn.execute(text(provider.get_count_query(table)))
                        count_row = count_result.fetchone()
                        total_rows += count_row[0] if count_row else 0
                    stats.append({"db": db_name, "tables": len(tables), "rows": total_rows, "status": "ok"})
            except Exception as e:
                stats.append({"db": db_name, "status": "error", "error": str(e)})
        return stats
    except Exception as e:
        return [{"db": "System", "status": "error", "error": str(e)}]


# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    logo_col, toggle_col = st.columns([3, 1])
    with logo_col:
        st.markdown(f"""
        <div style="padding:1rem 0 1.2rem 0;">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:1.35rem;font-weight:700;
                      letter-spacing:-0.02em;color:{'#e8e8f0' if is_dark else '#1a1a2e'};">
            ⬡ DS Agent
          </div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                      background:linear-gradient(90deg,#6c63ff,#ff6584);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                      margin-top:2px;letter-spacing:0.1em;">
            AI-POWERED ANALYTICS
          </div>
        </div>
        """, unsafe_allow_html=True)
    with toggle_col:
        st.markdown("<div style='padding-top:1.1rem;'>", unsafe_allow_html=True)
        theme_ph = st.empty()
        if theme_ph.button("☀️" if is_dark else "🌙", key="theme_toggle", disabled=st.session_state.get("is_running", False)):
            st.session_state.theme = "light" if is_dark else "dark"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<p style='font-family:JetBrains Mono;font-size:0.6rem;color:{'#5555aa' if is_dark else '#7777aa'};letter-spacing:0.1em;margin-bottom:0.1rem;'>VIEW MODE</p>", unsafe_allow_html=True)
    st.session_state.view_mode = st.radio(
        "View Mode",
        ["Agent Chat", "Dashboard"],
        index=0 if st.session_state.get("view_mode", "Agent Chat") == "Agent Chat" else 1,
        horizontal=True,
        label_visibility="collapsed",
        disabled=st.session_state.get("is_running", False)
    )
    st.markdown("<br>", unsafe_allow_html=True)

    db_stats = check_db_status()
    st.markdown(f"<p style='font-family:JetBrains Mono;font-size:0.6rem;color:{'#5555aa' if is_dark else '#7777aa'};letter-spacing:0.1em;margin-bottom:0.4rem;'>DATABASE</p>", unsafe_allow_html=True)
    for stat in db_stats:
        if stat["status"] == "ok":
            st.markdown(f"""
            <div style="background:rgba(67,233,123,0.06);border:1px solid rgba(67,233,123,0.18);
                        border-radius:7px;padding:0.5rem 0.7rem;margin-bottom:0.4rem;">
              <div style="color:#43e97b;font-size:0.65rem;font-family:JetBrains Mono;">
                ✓ {stat['db'].upper()} · {stat['tables']}t · {stat['rows']:,}r
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:rgba(255,101,132,0.06);border:1px solid rgba(255,101,132,0.25);
                        border-radius:7px;padding:0.5rem 0.7rem;margin-bottom:0.4rem;">
              <div style="color:#ff6584;font-size:0.65rem;font-family:JetBrains Mono;">
                ⚠ {stat['db'].upper()} · Error
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color:{'rgba(108,99,255,0.12)' if is_dark else 'rgba(108,99,255,0.08)'};margin:0.8rem 0 1rem;'>", unsafe_allow_html=True)

    st.markdown(f"<p style='font-family:JetBrains Mono;font-size:0.6rem;color:{'#5555aa' if is_dark else '#7777aa'};letter-spacing:0.1em;margin-bottom:0.4rem;'>QUERY</p>", unsafe_allow_html=True)
    query = st.text_area(
        label="query", label_visibility="collapsed",
        value="Predict the income level of individuals in this dataset. What factors influence whether a person earns over 50K?",
        height=120, placeholder="Describe what you want to analyze...", key="query_input",
        disabled=st.session_state.get("is_running", False)
    )
    st.markdown("<br>", unsafe_allow_html=True)
    
    def on_run_clicked():
        if st.session_state.query_input.strip():
            st.session_state.is_running = True
            st.session_state.run_query = st.session_state.query_input

    run_btn = st.button("⬡ Run Analysis", use_container_width=True, on_click=on_run_clicked, disabled=st.session_state.get("is_running", False))
    st.markdown("<br>", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:2rem 0 1rem;border-bottom:1px solid {'rgba(108,99,255,0.12)' if is_dark else 'rgba(108,99,255,0.08)'};margin-bottom:2rem;">
  <h1 style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:700;
             letter-spacing:-0.03em;color:{'#e8e8f0' if is_dark else '#1a1a2e'};margin:0;">
    Data Scientist Agent
  </h1>
  <p style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:{'#5555aa' if is_dark else '#7777aa'};margin:0.4rem 0 0;">
    Fully autonomous ML pipeline — from query to insights
  </p>
</div>
""", unsafe_allow_html=True)


# ── PIPELINE RUNNER ──────────────────────────────────────────
def run_pipeline(user_query: str) -> AgentState:
    app = build_graph()
    initial_state: AgentState = {"user_query": user_query, "messages": []}

    node_names = {
        "intent_node":              ("01","Intent Analyzer",    "Classifying intent..."),
        "db_context_node":          ("02","DB Context",         "Schema + column stats..."),
        "sql_node":                 ("03","SQL Executor",       "Running query..."),
        "planner_node":             ("03","Analysis Planner",   "Building plan..."),
        "execution_node":           ("04","Execution",          "Fetching + profiling..."),
        "feature_plan_node":        ("05","Feature Planner",    "Feature strategy..."),
        "feature_engineering_node": ("06","Feature Eng.",       "Encoding + scaling..."),
        "automl_node":              ("07","AutoML",             "Training models..."),
        "evaluation_node":          ("08","Evaluation",         "Metrics + SHAP..."),
        "critic_node":              ("09","AI Critic",          "Checking quality..."),
        "report_node":              ("10","Report Generator",   "Writing report..."),
    }

    completed_nodes = []
    final_state     = dict(initial_state)
    content_ph      = st.empty()
    chat_container  = st.container()
    st.session_state.chat_history = []
    theme_ph.button("⏳", key="theme_toggle_disabled", disabled=True)

    def render_trace(current_node=None):
        intent   = final_state.get("intent","eda")
        is_sql   = intent == "sql_query"
        total    = 4 if is_sql else 10
        step     = min(len(completed_nodes)+1, total)
        progress = step / total
        lc = "#4a4a6a" if is_dark else "#9999bb"
        tc = "#e8e8f0" if is_dark else "#1a1a2e"
        bg = "#0d0d16" if is_dark else "#f5f5fb"
        bd = "rgba(108,99,255,0.18)" if is_dark else "rgba(108,99,255,0.12)"

        html = f"""
<div style="background:{bg};border:1px solid {bd};border-radius:14px;overflow:hidden;
            font-family:'JetBrains Mono',monospace;animation:fade-up 0.3s ease both;">
  <div style="background:{'rgba(108,99,255,0.07)' if is_dark else 'rgba(108,99,255,0.04)'};
              border-bottom:1px solid {bd};padding:0.6rem 1rem;
              display:flex;align-items:center;gap:0.5rem;">
    <div style="width:9px;height:9px;border-radius:50%;background:#ff6584;flex-shrink:0;"></div>
    <div style="width:9px;height:9px;border-radius:50%;background:#f7971e;flex-shrink:0;"></div>
    <div style="width:9px;height:9px;border-radius:50%;background:#43e97b;flex-shrink:0;"></div>
    <span style="margin-left:0.4rem;font-size:0.6rem;color:{lc};letter-spacing:0.08em;">
      DS AGENT · LIVE · {step}/{total}
    </span>
    <div style="flex:1;"></div>
    <div style="height:2px;width:70px;background:{'rgba(255,255,255,0.06)' if is_dark else 'rgba(0,0,0,0.06)'};
                border-radius:1px;overflow:hidden;">
      <div style="height:100%;width:{progress*100:.0f}%;
                  background:linear-gradient(90deg,#6c63ff,#43e97b);transition:width 0.4s;"></div>
    </div>
  </div>
  <div style="padding:0.9rem 1.1rem;">"""

        for cn in completed_nodes:
            info = node_names.get(cn,("","",cn,""))
            html += f"""
    <div style="display:flex;align-items:center;gap:0.55rem;margin-bottom:0.25rem;opacity:0.45;">
      <span style="color:#43e97b;font-size:0.68rem;min-width:12px;">✓</span>
      <span style="font-size:0.68rem;color:{lc};text-decoration:line-through;">{info[0]} · {info[1]}</span>
    </div>"""

        if current_node:
            info = node_names.get(current_node,("","",current_node,"Processing..."))
            html += f"""
    <div style="display:flex;align-items:flex-start;gap:0.55rem;padding:0.55rem 0.7rem;
                margin:0.25rem 0;background:rgba(108,99,255,0.09);
                border:1px solid rgba(108,99,255,0.22);border-left:2px solid #6c63ff;border-radius:7px;">
      <div style="width:9px;height:9px;border-radius:50%;background:#6c63ff;margin-top:3px;flex-shrink:0;
                  box-shadow:0 0 8px rgba(108,99,255,0.7);animation:pulse-dot 1.4s ease-in-out infinite;"></div>
      <div>
        <div style="font-size:0.72rem;color:{tc};font-weight:500;">{info[0]} · {info[1]}</div>
        <div style="font-size:0.62rem;color:#6c63ff;margin-top:1px;">{info[2]}</div>
      </div>
    </div>
    <div style="margin-top:0.3rem;padding:0 0.1rem;">
      <span style="font-size:0.62rem;color:{lc};border-right:1.5px solid #6c63ff;padding-right:2px;
                   animation:pulse-dot 0.9s ease-in-out infinite;">_</span>
    </div>"""

        html += """
  </div>
</div>
<style>
@keyframes fade-up{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse-dot{0%,100%{opacity:1}50%{opacity:.3}}
</style>"""
        content_ph.markdown(html, unsafe_allow_html=True)



    if st.session_state.view_mode == "Dashboard":
        render_trace("intent_node")

    async def process_stream():
        nonlocal final_state
        
        user_msg = {"name": "User", "content": user_query}
        st.session_state.chat_history.append(user_msg)
        if st.session_state.view_mode == "Agent Chat":
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_query)

        async for chunk in app.astream(initial_state):
            node_name = list(chunk.keys())[0]
            final_state.update(chunk[node_name])
            completed_nodes.append(node_name)
            
            # --- Chat Mode Handling ---
            msg = None
            if node_name == "intent_node":
                intent = chunk[node_name].get("intent", "unknown")
                msg = {"name": "Orchestrator", "content": f"Decomposing query into sub-tasks. Identified primary intent: **{intent.upper()}**."}
            elif node_name == "sql_node":
                query_raw = chunk[node_name].get("sql_query_raw", "")
                results = chunk[node_name].get("sql_query_results", [])
                msg = {"name": "SQL Agent", "content": "Executed SQL query against the database.", "code": query_raw, "dataframe": pd.DataFrame(results[:5]) if results else None}
            elif node_name == "execution_node":
                prof = chunk[node_name].get("profiling_report", {})
                df = pd.DataFrame([{"Kolon":c,"Tip":i.get("dtype",""),"Boşluk":f"{i.get('null_rate',0):.1%}"} for c,i in prof.get("columns",{}).items()]) if prof else None
                msg = {"name": "Analysis Agent", "content": "Veri seti çıkarıldı ve istatistiksel profili oluşturuldu. Nedenleri bulmak için gizli desenler taranıyor...", "dataframe": df}
            elif node_name == "automl_node":
                model_name = chunk[node_name].get("model_name", "Unknown")
                msg = {"name": "ML Agent", "content": f"Değişkenler arası ilişkiler analiz edildi ve en uygun Makine Öğrenmesi modeli eğitildi: **{model_name}**."}
            elif node_name == "evaluation_node":
                metrics = chunk[node_name].get("metrics", {})
                metrics_str = ", ".join([f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k,v in metrics.items() if not isinstance(v,(list,dict))])
                msg = {"name": "ML Agent", "content": f"Model doğrulama testi tamamlandı. Başarı Metrikleri:\n{metrics_str}"}
            elif node_name == "report_node":
                report = chunk[node_name].get("report", "")
                msg = {"name": "Reporting Agent", "content": report}
            
            if msg:
                st.session_state.chat_history.append(msg)
                if st.session_state.view_mode == "Agent Chat":
                    with chat_container:
                        render_chat_message(msg)
            # --------------------------

            next_node = {
                "intent_node":              "db_context_node",
                "db_context_node":          "sql_node" if final_state.get("intent")=="sql_query" else "planner_node",
                "planner_node":             "execution_node",
                "execution_node":           "feature_plan_node" if final_state.get("intent") in ("classification","regression","clustering") else "automl_node",
                "feature_plan_node":        "feature_engineering_node",
                "feature_engineering_node": "automl_node",
                "automl_node":              "evaluation_node",
                "evaluation_node":          "critic_node",
                "critic_node":              "report_node",
            }.get(node_name)
            
            if st.session_state.view_mode == "Dashboard":
                render_trace(next_node)
        return final_state

    final_state = asyncio.run(process_stream())
    content_ph.empty()
    return final_state


# ── AGENT CHAT HELPERS ───────────────────────────────────────
def get_avatar(name):
    return {"User": "👤", "Orchestrator": "🧠", "SQL Agent": "💻", "Analysis Agent": "📊", "ML Agent": "🤖", "Reporting Agent": "📝"}.get(name, "🤖")

def render_chat_message(msg):
    role = "user" if msg["name"] == "User" else "assistant"
    with st.chat_message(role, avatar=get_avatar(msg["name"])):
        if msg["name"] != "User":
            st.markdown(f"**{msg['name']}**")
        st.markdown(msg["content"])
        if msg.get("code"):
            st.code(msg["code"], language="sql")
        if msg.get("dataframe") is not None and not msg["dataframe"].empty:
            st.dataframe(msg["dataframe"], use_container_width=True, hide_index=True)
        if msg.get("chart") is not None:
            st.plotly_chart(msg["chart"], use_container_width=True)

def render_chat_history():
    history = st.session_state.get("chat_history", [])
    if not history:
        return
        
    col1, col2 = st.columns([8, 1])
    with col2:
        components.html(
            """
            <script>
            function printPage() {
                window.parent.print();
            }
            </script>
            <style>
            button {
                background: linear-gradient(135deg, #6c63ff, #9c56d8);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.5rem 0.2rem;
                font-family: 'Space Grotesk', sans-serif;
                font-weight: 600;
                font-size: 0.8rem;
                cursor: pointer;
                width: 100%;
                box-shadow: 0 4px 14px rgba(108,99,255,0.3);
                transition: all 0.2s;
            }
            button:hover {
                transform: translateY(-1px);
                box-shadow: 0 6px 20px rgba(108,99,255,0.4);
            }
            </style>
            <button onclick="printPage()">🖨️ PDF</button>
            """,
            height=45
        )

    for msg in history:
        render_chat_message(msg)

# ── HELPERS ──────────────────────────────────────────────────
def render_metric_card(label: str, value: str, delta: str = "", color: str = "#6c63ff") -> str:
    tc = "#e8e8f0" if is_dark else "#1a1a2e"
    lc = "#7777aa" if is_dark else "#5a5a7a"
    bg = "rgba(255,255,255,0.03)" if is_dark else "rgba(255,255,255,0.85)"
    bd = "rgba(108,99,255,0.12)" if is_dark else "rgba(108,99,255,0.08)"
    return f"""
<div style="background:{bg};border:1px solid {bd};border-top:2px solid {color};
            border-radius:12px;padding:1rem 1.1rem;
            transition:transform 0.2s,box-shadow 0.2s;cursor:default;"
     onmouseover="this.style.transform='translateY(-3px)';this.style.boxShadow='0 8px 28px rgba(108,99,255,0.14)'"
     onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none'">
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
              color:{lc};letter-spacing:0.1em;margin-bottom:0.4rem;text-transform:uppercase;">{label}</div>
  <div style="font-family:'Space Grotesk',sans-serif;font-size:1.7rem;
              font-weight:600;color:{tc};letter-spacing:-0.02em;line-height:1.1;">{value}</div>
  {f'<div style="font-family:JetBrains Mono;font-size:0.65rem;color:{color};margin-top:4px;opacity:0.85;">{delta}</div>' if delta else ''}
</div>"""


def section_header(title: str, subtitle: str = ""):
    tc = "#e8e8f0" if is_dark else "#1a1a2e"
    sc = "#7777aa" if is_dark else "#5a5a7a"
    st.markdown(f"""
<div style="margin:2rem 0 0.9rem;">
  <div style="display:flex;align-items:center;gap:0.55rem;margin-bottom:0.25rem;">
    <div style="width:3px;height:17px;background:linear-gradient(180deg,#6c63ff,#ff6584);
                border-radius:2px;flex-shrink:0;"></div>
    <div style="font-family:'Space Grotesk',sans-serif;font-size:0.95rem;
                font-weight:600;color:{tc};letter-spacing:-0.01em;">{title}</div>
  </div>
  {f'<div style="font-family:JetBrains Mono;font-size:0.68rem;color:{sc};padding-left:0.85rem;">{subtitle}</div>' if subtitle else ''}
</div>""", unsafe_allow_html=True)


# ── SQL RESULTS ──────────────────────────────────────────────
def render_sql_results(result: AgentState):
    chart      = result.get("sql_query_chart", {})
    columns    = result.get("sql_query_columns", [])
    rows       = result.get("sql_query_results", [])
    sql_raw    = result.get("sql_query_raw", "")
    summary    = result.get("sql_query_summary", "")
    chart_type = chart.get("type", "table")
    title      = chart.get("title", "Sorgu Sonuçları")
    labels     = chart.get("labels", [])
    datasets   = chart.get("datasets", [])
    n_rows     = len(rows)
    n_cols     = len(columns)
    tick_angle = -45 if len(labels) > 10 else 0
    ch_height  = max(420, min(len(labels)*18, 700)) if labels else 420

    PLOTLY_CONFIG = {
        "scrollZoom": True, "displayModeBar": True, "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d","select2d"],
        "modeBarButtonsToAdd": ["zoom2d","pan2d","resetScale2d","toImage"],
        "toImageButtonOptions": {"format":"png","filename":"ds_agent_chart","scale":2},
    }

    st.markdown(f"""
<div style="background:{STATUS_BG};border:1px solid {STATUS_BORDER};border-radius:10px;
            padding:0.85rem 1.4rem;display:flex;align-items:center;gap:2rem;
            margin-bottom:1.8rem;flex-wrap:wrap;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#43e97b;">✓ QUERY COMPLETE</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:{CHART_LABEL_COLOR};">
    INTENT: <span style="color:#6c63ff;">SQL_QUERY</span></div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:{CHART_LABEL_COLOR};">
    ROWS: <span style="color:{CHART_TEXT_COLOR};">{n_rows:,}</span></div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:{CHART_LABEL_COLOR};">
    COLS: <span style="color:{CHART_TEXT_COLOR};">{n_cols}</span></div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:{CHART_LABEL_COLOR};">
    CHART: <span style="color:{CHART_TEXT_COLOR};">{chart_type.upper()}</span></div>
</div>""", unsafe_allow_html=True)

    warning = result.get("sql_query_warning","")
    if warning:
        st.markdown(f"""
<div style="background:rgba(247,151,30,0.07);border:1px solid rgba(247,151,30,0.3);
            border-left:3px solid #f7971e;border-radius:9px;padding:0.9rem 1.2rem;margin-bottom:1rem;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
              color:#f7971e;letter-spacing:0.08em;margin-bottom:0.4rem;">⚡ YAKLAŞIK SONUÇ</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
              color:{CHART_TEXT_COLOR};line-height:1.7;">{warning}</div>
</div>""", unsafe_allow_html=True)

    if summary:
        st.markdown(f"""
<div style="background:rgba(108,99,255,0.06);border:1px solid rgba(108,99,255,0.16);
            border-left:3px solid #6c63ff;border-radius:8px;
            padding:0.8rem 1.1rem;margin-bottom:1.3rem;
            font-family:'JetBrains Mono',monospace;font-size:0.77rem;
            color:{CHART_TEXT_COLOR};line-height:1.7;">{summary}</div>""", unsafe_allow_html=True)

    with st.expander("🗂 Generated SQL", expanded=False):
        st.code(sql_raw, language="sql")

    st.markdown("<br>", unsafe_allow_html=True)

    if chart_type == "warning":
        st.markdown(f"""
<div style="background:rgba(255,101,132,0.06);border:1px solid rgba(255,101,132,0.25);
            border-left:3px solid #ff6584;border-radius:9px;padding:1.3rem 1.6rem;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
              color:#ff6584;letter-spacing:0.08em;margin-bottom:0.6rem;">⚠ UYGUN KOLON BULUNAMADI</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;
              color:{CHART_TEXT_COLOR};line-height:1.8;">{chart.get("summary","")}</div>
</div>""", unsafe_allow_html=True)
        return

    col_chart, col_table = st.columns([1.4, 1], gap="large")

    with col_chart:
        section_header(title, f"Görselleştirme · {chart_type}")

        if n_rows == 1 and n_cols == 1:
            val = list(rows[0].values())[0]
            col_name = columns[0]
            try:
                vf = float(val)
                dv = f"{vf:,.4f}" if vf != int(vf) else f"{int(vf):,}"
            except: dv = str(val)
            st.markdown(f"""
<div style="background:rgba(108,99,255,0.07);border:1px solid rgba(108,99,255,0.2);
            border-radius:16px;padding:3rem 2rem;text-align:center;margin-top:0.5rem;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
              color:{CHART_LABEL_COLOR};letter-spacing:0.1em;margin-bottom:0.8rem;">
    {col_name.upper().replace('_',' ')}</div>
  <div style="font-family:'Space Grotesk',sans-serif;font-size:3.5rem;font-weight:700;
              color:#6c63ff;letter-spacing:-0.03em;line-height:1;">{dv}</div>
</div>""", unsafe_allow_html=True)

        elif n_rows == 1 and n_cols > 1:
            row_data  = rows[0]
            card_cols = st.columns(min(n_cols, 4))
            for i, (k, v) in enumerate(row_data.items()):
                try:
                    vf = float(v); dv = f"{vf:,.4f}" if vf != int(vf) else f"{int(vf):,}"
                except: dv = str(v)
                with card_cols[i % 4]:
                    st.markdown(render_metric_card(k.replace("_"," ").upper(), dv, color=COLORS[i%len(COLORS)]), unsafe_allow_html=True)

        elif chart_type == "table" or not labels:
            if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=ch_height)
            else: st.info("Sonuç bulunamadı.")

        elif chart_type == "pie":
            if len(labels) > 12:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=ch_height)
            else:
                values = datasets[0]["data"] if datasets else []
                fig = go.Figure(go.Pie(
                    labels=labels, values=values,
                    marker=dict(colors=COLORS*(len(labels)//len(COLORS)+1)),
                    hole=0.5, textfont=dict(family="JetBrains Mono",size=11),
                    textinfo="label+percent+value",
                    hovertemplate="<b>%{label}</b><br>%{value}<br>%{percent}<extra></extra>",
                ))
                fig.update_layout(**PLOTLY_LAYOUT, height=ch_height, showlegend=True,
                    legend=dict(font=dict(family="JetBrains Mono",size=10), bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        elif chart_type == "bar":
            fig = go.Figure()
            for i, ds in enumerate(datasets):
                yv = ds["data"]
                fig.add_trace(go.Bar(
                    x=labels, y=yv, name=ds.get("label",""),
                    marker_color=COLORS[i%len(COLORS)], marker_line_width=0,
                    text=[f"{v:,.2f}" if isinstance(v,float) else str(v) for v in yv],
                    textposition="outside",
                    textfont=dict(family="JetBrains Mono",size=10,color=CHART_TEXT_COLOR),
                    hovertemplate="<b>%{x}</b><br>%{y:,.4f}<extra></extra>",
                ))
            fig.update_layout(**PLOTLY_LAYOUT, height=ch_height, bargap=0.15,
                xaxis=dict(**{k:v for k,v in AXIS_STYLE.items() if k!="tickfont"},
                           tickangle=tick_angle,
                           tickfont=dict(family="JetBrains Mono",size=9 if len(labels)>20 else 11,color=CHART_LABEL_COLOR)),
                yaxis=dict(**AXIS_STYLE),
                legend=dict(font=dict(family="JetBrains Mono",size=10),bgcolor="rgba(0,0,0,0)"),
                showlegend=len(datasets)>1)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        elif chart_type == "line":
            fig = go.Figure()
            for i, ds in enumerate(datasets):
                fig.add_trace(go.Scatter(
                    x=labels, y=ds["data"], mode="lines+markers",
                    name=ds.get("label",""),
                    line=dict(color=COLORS[i%len(COLORS)],width=2), marker=dict(size=5),
                    hovertemplate="<b>%{x}</b><br>%{y:,.4f}<extra></extra>",
                ))
            fig.update_layout(**PLOTLY_LAYOUT, height=ch_height,
                xaxis=dict(**AXIS_STYLE, tickangle=tick_angle,
                           tickfont=dict(size=9 if len(labels)>20 else 11,color=CHART_LABEL_COLOR)),
                yaxis=dict(**AXIS_STYLE),
                legend=dict(font=dict(family="JetBrains Mono",size=10),bgcolor="rgba(0,0,0,0)"),
                showlegend=len(datasets)>1)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        elif chart_type == "scatter":
            yv = datasets[0]["data"] if datasets else []
            fig = go.Figure(go.Scatter(
                x=labels, y=yv, mode="markers",
                marker=dict(color=COLORS[0],size=7,opacity=0.75,
                            line=dict(color="rgba(255,255,255,0.1)",width=1)),
                hovertemplate="<b>%{x}</b><br>%{y:,.4f}<extra></extra>",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=ch_height,
                xaxis=dict(**AXIS_STYLE,tickangle=tick_angle,
                           tickfont=dict(size=9 if len(labels)>20 else 11,color=CHART_LABEL_COLOR)),
                yaxis=dict(**AXIS_STYLE), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with col_table:
        section_header("Raw Data", f"{n_rows:,} satır · {n_cols} kolon")
        if rows:
            df = pd.DataFrame(rows)
            for col in df.columns:
                if pd.api.types.is_float_dtype(df[col]):
                    df[col] = df[col].map(lambda x: f"{x:,.4f}" if pd.notna(x) else "")
            st.dataframe(df, use_container_width=True, hide_index=True, height=ch_height)
        else:
            st.info("Sonuç bulunamadı.")


# ── RENDER RESULTS ───────────────────────────────────────────
def render_results(result: AgentState):
    if result.get("intent") == "sql_query":
        render_sql_results(result)
        return

    metrics       = result.get("metrics", {})
    ev            = result.get("evaluation_report", {})
    feature_report= result.get("feature_report", {})
    profiling     = result.get("profiling_report", {})
    feature_plan  = result.get("feature_plan", {})
    report        = result.get("report", "")
    model_name    = result.get("model_name", "unknown")
    intent        = result.get("intent", "")

    test_m    = ev.get("test_metrics", {})
    ovf       = ev.get("overfitting", {})
    reliability = ev.get("reliability_score", {})
    fi        = ev.get("feature_importance", {})
    fi_summary= ev.get("feature_importance_summary", {})
    confusion = ev.get("confusion_matrix", {})
    roc       = ev.get("roc_curve", {})
    calibration = ev.get("calibration", {})
    per_class = ev.get("per_class_metrics", {})
    all_scores= metrics.get("all_model_scores", {})
    grade     = reliability.get("grade", "?")
    score     = reliability.get("score", 0)
    grade_color = {"A":"#43e97b","B":"#6c63ff","C":"#f7971e","D":"#ff6584","F":"#ff6584"}.get(grade,"#7777aa")

    st.markdown(f"""
<div style="background:{STATUS_BG};border:1px solid {STATUS_BORDER};border-radius:10px;
            padding:0.85rem 1.4rem;display:flex;align-items:center;gap:2rem;
            margin-bottom:1.8rem;flex-wrap:wrap;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#43e97b;">✓ PIPELINE COMPLETE</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:{CHART_LABEL_COLOR};">
    MODEL: <span style="color:{CHART_TEXT_COLOR};">{model_name.upper()}</span></div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:{CHART_LABEL_COLOR};">
    INTENT: <span style="color:#6c63ff;">{intent.upper()}</span></div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:{CHART_LABEL_COLOR};">
    RELIABILITY: <span style="color:{grade_color};font-weight:600;">{score}/100 · {grade}</span></div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:{CHART_LABEL_COLOR};">
    FEATURES: <span style="color:{CHART_TEXT_COLOR};">{feature_report.get('n_features','?')}</span></div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:{CHART_LABEL_COLOR};">
    TARGET: <span style="color:{CHART_TEXT_COLOR};">{feature_report.get('target_name','?').upper()}</span></div>
</div>""", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
        "📊 Overview","🎯 Model Performance","🔍 Feature Analysis",
        "⚖️ Fairness & Bias","🧪 Data & Profiling","🗺️ Pipeline Trace","🗄️ Audit Trail",
    ])

    task = metrics.get("task", ev.get("task","classification"))
    auc  = test_m.get("roc_auc", test_m.get("roc_auc_ovr",0))
    acc  = test_m.get("accuracy",0)
    f1   = test_m.get("f1_weighted",0)
    ll   = test_m.get("log_loss",None)

    with tab1:
        section_header("Key Metrics","Core performance indicators from test set")
        if task == "clustering":
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.markdown(render_metric_card("Silhouette Score",f"{ev.get('silhouette_score',0):.4f}","↑ Higher is better","#6c63ff"),unsafe_allow_html=True)
            with c2: st.markdown(render_metric_card("N Clusters",str(ev.get("n_clusters","?")),"Optimal found","#43e97b"),unsafe_allow_html=True)
            with c3: st.markdown(render_metric_card("Algorithm",str(ev.get("best_algorithm","?")).upper(),"Best performing","#f7971e"),unsafe_allow_html=True)
            with c4:
                db=ev.get("davies_bouldin")
                st.markdown(render_metric_card("Davies-Bouldin",f"{db:.4f}" if db else "N/A","↓ Lower is better","#ff6584"),unsafe_allow_html=True)
        elif task == "anomaly_detection":
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.markdown(render_metric_card("Anomalies Found",str(ev.get("n_anomalies","?")),"Total detected","#ff6584"),unsafe_allow_html=True)
            with c2: st.markdown(render_metric_card("Anomaly Rate",f"{ev.get('anomaly_pct',0):.1f}%","% of dataset","#f7971e"),unsafe_allow_html=True)
            with c3: st.markdown(render_metric_card("Contamination",str(ev.get("best_contamination","?")),"Optimal threshold","#6c63ff"),unsafe_allow_html=True)
            with c4: st.markdown(render_metric_card("Normal Records",str(ev.get("n_normal","?")),"Clean data points","#43e97b"),unsafe_allow_html=True)
        elif task == "regression":
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.markdown(render_metric_card("R²",f"{test_m.get('r2',0):.4f}","↑ Higher is better","#6c63ff"),unsafe_allow_html=True)
            with c2: st.markdown(render_metric_card("RMSE",f"{test_m.get('rmse',0):.4f}","↓ Lower is better","#ff6584"),unsafe_allow_html=True)
            with c3: st.markdown(render_metric_card("MAE",f"{test_m.get('mae',0):.4f}","Mean Absolute Error","#f7971e"),unsafe_allow_html=True)
            with c4: st.markdown(render_metric_card("CV R²",f"{metrics.get('cv_r2',0):.4f}",f"±{metrics.get('cv_r2_std',0):.4f}","#43e97b"),unsafe_allow_html=True)
        else:
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.markdown(render_metric_card("ROC-AUC",f"{auc:.4f}","↑ Higher is better","#6c63ff"),unsafe_allow_html=True)
            with c2: st.markdown(render_metric_card("Accuracy",f"{acc:.2%}",f"{'↑' if acc>0.8 else '↓'} {acc:.4f}","#43e97b"),unsafe_allow_html=True)
            with c3: st.markdown(render_metric_card("F1 Weighted",f"{f1:.4f}","Precision × Recall","#f7971e"),unsafe_allow_html=True)
            with c4:
                if ll: st.markdown(render_metric_card("Log Loss",f"{ll:.4f}","↓ Lower is better","#ff6584"),unsafe_allow_html=True)
                else: st.markdown(render_metric_card("Reliability",f"{score}/100",f"Grade {grade}",grade_color),unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_right = st.columns(2)

        with col_left:
            if task == "clustering":
                section_header("Cluster Distribution","Sample count per cluster")
                cd = ev.get("cluster_distribution", metrics.get("cluster_distribution",{}))
                if cd:
                    lc2 = [k for k in cd if k!="noise"]
                    cc2 = [cd[k]["count"] for k in lc2]
                    pc2 = [cd[k].get("pct",0) for k in lc2]
                    fig_cd = go.Figure()
                    fig_cd.add_trace(go.Bar(x=lc2,y=cc2,marker_color=COLORS[:len(lc2)],marker_line_width=0,
                        text=[f"{p:.1f}%" for p in pc2],textposition="outside",
                        textfont=dict(family="JetBrains Mono",size=11,color=CHART_TEXT_COLOR)))
                    fig_cd.update_layout(**PLOTLY_LAYOUT,height=220,xaxis=dict(**AXIS_STYLE),yaxis=dict(**AXIS_STYLE),showlegend=False)
                    st.plotly_chart(fig_cd,use_container_width=True)
            elif task == "anomaly_detection":
                section_header("Anomaly Distribution","Normal vs detected")
                na=ev.get("n_anomalies",0); nn=ev.get("n_normal",0)
                fig_ad = go.Figure(go.Pie(labels=["Normal","Anomaly"],values=[nn,na],
                    marker=dict(colors=["#43e97b","#ff6584"]),hole=0.55,
                    textfont=dict(family="JetBrains Mono",size=11)))
                fig_ad.update_layout(**PLOTLY_LAYOUT,height=220,showlegend=True,
                    legend=dict(font=dict(family="JetBrains Mono",size=10),bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_ad,use_container_width=True)
            else:
                section_header("Overfitting Analysis","Train vs Test gap")
                ts=ovf.get("train_score",0); tss=ovf.get("test_score",0)
                gap=ovf.get("gap",0); stat=ovf.get("status","unknown")
                sc2={"normal":"#43e97b","mild_overfitting":"#f7971e","overfitting_detected":"#ff6584","underfitting":"#38b2f4"}.get(stat,"#7777aa")
                fig_ovf = go.Figure()
                fig_ovf.add_trace(go.Bar(x=["Train","Test"],y=[ts,tss],marker_color=["#6c63ff","#43e97b"],
                    marker_line_width=0,text=[f"{ts:.4f}",f"{tss:.4f}"],textposition="outside",
                    textfont=dict(family="JetBrains Mono",size=11,color=CHART_TEXT_COLOR)))
                fig_ovf.update_layout(**PLOTLY_LAYOUT,height=220,
                    title=dict(text=f"<span style='color:{sc2}'>{stat.replace('_',' ').title()}</span> · gap: {gap:.4f}",
                               font=dict(size=11,family="JetBrains Mono"),x=0),
                    xaxis=dict(**AXIS_STYLE),
                    yaxis=dict(**AXIS_STYLE,range=[max(0,min(ts,tss)-0.1),1.05]),showlegend=False)
                st.plotly_chart(fig_ovf,use_container_width=True)
                st.markdown(f"""<div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                    color:{CHART_LABEL_COLOR};padding:0.5rem 0.7rem;background:{INFO_BG};border-radius:6px;">
                    {ovf.get('description','')}</div>""", unsafe_allow_html=True)

        with col_right:
            if task == "clustering":
                section_header("Algorithm Comparison","Silhouette by algorithm")
                alg = ev.get("all_algorithm_scores",{})
                if alg:
                    ba = ev.get("best_algorithm","")
                    sa = dict(sorted(alg.items(),key=lambda x:float(x[1]),reverse=True))
                    an = list(sa.keys()); av = [float(v) for v in sa.values()]
                    fig_alg = go.Figure()
                    fig_alg.add_trace(go.Bar(x=av,y=an,orientation="h",
                        marker_color=[COLORS[0] if a==ba else "#1e1e2a" for a in an],marker_line_width=0,
                        text=[f"{v:.4f}" for v in av],textposition="outside",
                        textfont=dict(family="JetBrains Mono",size=10,color=CHART_TEXT_COLOR)))
                    fig_alg.update_layout(**PLOTLY_LAYOUT,height=220,
                        xaxis=dict(**AXIS_STYLE,range=[max(-1,min(av)-0.05),1.0]),
                        yaxis=dict(**AXIS_STYLE),showlegend=False)
                    st.plotly_chart(fig_alg,use_container_width=True)
            elif task == "anomaly_detection":
                section_header("Contamination Scan","Anomaly % at each threshold")
                cs = ev.get("contamination_scan",{})
                if cs:
                    bcs = str(ev.get("best_contamination",""))
                    ck = list(cs.keys()); cp = [cs[k]["anomaly_pct"] for k in ck]
                    fig_cs = go.Figure()
                    fig_cs.add_trace(go.Bar(x=[f"c={k}" for k in ck],y=cp,
                        marker_color=[COLORS[0] if k==bcs else "#1e1e2a" for k in ck],marker_line_width=0,
                        text=[f"{p:.1f}%" for p in cp],textposition="outside",
                        textfont=dict(family="JetBrains Mono",size=10,color=CHART_TEXT_COLOR)))
                    fig_cs.update_layout(**PLOTLY_LAYOUT,height=220,
                        xaxis=dict(**AXIS_STYLE),yaxis=dict(**AXIS_STYLE),showlegend=False)
                    st.plotly_chart(fig_cs,use_container_width=True)
            else:
                section_header("Model Comparison","5-fold CV ROC-AUC")
                if all_scores:
                    ss = dict(sorted(all_scores.items(),key=lambda x:float(x[1]),reverse=True))
                    ms = list(ss.keys()); sv = [float(v) for v in ss.values()]
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(x=sv,y=ms,orientation="h",
                        marker_color=[COLORS[0] if m==model_name else "#1e1e2a" for m in ms],marker_line_width=0,
                        text=[f"{s:.4f}" for s in sv],textposition="outside",
                        textfont=dict(family="JetBrains Mono",size=10,color=CHART_TEXT_COLOR)))
                    fig2.update_layout(**PLOTLY_LAYOUT,height=220,
                        xaxis=dict(**AXIS_STYLE,range=[min(sv)-0.02,max(sv)+0.02]),
                        yaxis=dict(**AXIS_STYLE),showlegend=False)
                    st.plotly_chart(fig2,use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Reliability Breakdown",f"Score: {score}/100 · Grade: {grade}")
        factors = reliability.get("factors",[])
        if factors:
            cols = st.columns(min(len(factors),4))
            for i,f in enumerate(factors):
                with cols[i%4]:
                    st.markdown(f"""<div style="background:rgba(67,233,123,0.05);border:1px solid rgba(67,233,123,0.15);
                        border-radius:7px;padding:0.5rem 0.7rem;margin-bottom:0.4rem;">
                        <span style="color:#43e97b;font-size:0.68rem;font-family:JetBrains Mono;">✓ {f}</span>
                    </div>""", unsafe_allow_html=True)

    with tab2:
        if task == "clustering":
            cl2,cr2 = st.columns(2)
            with cl2:
                section_header("Cluster Size Distribution","Samples per cluster")
                cd2 = ev.get("cluster_distribution",{})
                if cd2:
                    lc3=list(cd2.keys()); cc3=[cd2[k]["count"] for k in lc3]; pc3=[cd2[k].get("pct",0) for k in lc3]
                    fig_cb = go.Figure()
                    fig_cb.add_trace(go.Bar(x=lc3,y=cc3,marker_color=COLORS[:len(lc3)],marker_line_width=0,
                        text=[f"{p:.1f}%" for p in pc3],textposition="outside",
                        textfont=dict(family="JetBrains Mono",size=11,color=CHART_TEXT_COLOR)))
                    fig_cb.update_layout(**PLOTLY_LAYOUT,height=300,xaxis=dict(**AXIS_STYLE),yaxis=dict(**AXIS_STYLE),showlegend=False)
                    st.plotly_chart(fig_cb,use_container_width=True)
            with cr2:
                section_header("Silhouette Gauge","Cluster separation quality")
                sil=ev.get("silhouette_score",0)
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",value=sil,domain={"x":[0,1],"y":[0,1]},
                    title={"text":"Silhouette","font":{"family":"JetBrains Mono","size":13,"color":CHART_LABEL_COLOR}},
                    number={"font":{"family":"JetBrains Mono","size":30,"color":CHART_TEXT_COLOR}},
                    gauge={"axis":{"range":[-1,1],"tickfont":{"family":"JetBrains Mono","size":10}},
                           "bar":{"color":"#6c63ff"},
                           "steps":[{"range":[-1,0],"color":"#0d0d16"},{"range":[0,0.5],"color":"#12122a"},{"range":[0.5,1],"color":"#0d1a20"}],
                           "threshold":{"line":{"color":"#43e97b","width":2},"value":0.5}}))
                fig_g.update_layout(**PLOTLY_LAYOUT,height=260)
                st.plotly_chart(fig_g,use_container_width=True)
        elif task == "anomaly_detection":
            cl2,cr2 = st.columns(2)
            with cl2:
                section_header("Detection Breakdown","Normal vs anomaly")
                na2=ev.get("n_anomalies",0); nn2=ev.get("n_normal",0)
                fig_p2 = go.Figure(go.Pie(labels=["Normal","Anomaly"],values=[nn2,na2],
                    marker=dict(colors=["#43e97b","#ff6584"]),hole=0.55,
                    textfont=dict(family="JetBrains Mono",size=12),textinfo="label+percent+value"))
                fig_p2.update_layout(**PLOTLY_LAYOUT,height=300,showlegend=True,
                    legend=dict(font=dict(family="JetBrains Mono",size=10),bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_p2,use_container_width=True)
            with cr2:
                section_header("Contamination Scan","Anomaly % at each threshold")
                cs2=ev.get("contamination_scan",{})
                if cs2:
                    bc2=str(ev.get("best_contamination",""))
                    ck2=list(cs2.keys()); cp2=[cs2[k]["anomaly_pct"] for k in ck2]; cn2=[cs2[k]["n_anomalies"] for k in ck2]
                    fig_sc = go.Figure()
                    fig_sc.add_trace(go.Bar(x=[f"c={k}" for k in ck2],y=cp2,
                        marker_color=[COLORS[0] if k==bc2 else "#1e1e2a" for k in ck2],marker_line_width=0,
                        text=[f"{p:.1f}%({n})" for p,n in zip(cp2,cn2)],textposition="outside",
                        textfont=dict(family="JetBrains Mono",size=10,color=CHART_TEXT_COLOR)))
                    fig_sc.update_layout(**PLOTLY_LAYOUT,height=300,xaxis=dict(**AXIS_STYLE),yaxis=dict(**AXIS_STYLE),showlegend=False)
                    st.plotly_chart(fig_sc,use_container_width=True)
        else:
            cl,cr = st.columns(2)
            with cl:
                section_header("Confusion Matrix","Actual vs Predicted")
                if confusion:
                    cm_m=confusion.get("matrix",[]); lbs=[str(l) for l in confusion.get("labels",[])]
                    if cm_m:
                        fig_cm=px.imshow(cm_m,x=lbs,y=lbs,
                            color_continuous_scale=[[0,"#08080e"],[0.5,"#1e1050"],[1,"#6c63ff"]],text_auto=True)
                        fig_cm.update_traces(textfont=dict(family="JetBrains Mono",size=14,color="white"))
                        fig_cm.update_layout(**PLOTLY_LAYOUT,height=300,
                            xaxis=dict(**AXIS_STYLE,title="Predicted"),yaxis=dict(**AXIS_STYLE,title="Actual"),
                            coloraxis_showscale=False)
                        st.plotly_chart(fig_cm,use_container_width=True)
            with cr:
                section_header("ROC Curve",f"AUC = {auc:.4f}")
                if roc:
                    fpr=roc.get("fpr",[]); tpr=roc.get("tpr",[])
                    of=roc.get("optimal_fpr"); ot=roc.get("optimal_tpr"); oth=roc.get("optimal_threshold")
                    fig_roc=go.Figure()
                    fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                        line=dict(color="rgba(255,255,255,0.12)",dash="dash",width=1),showlegend=False))
                    fig_roc.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",fill="tozeroy",
                        line=dict(color="#6c63ff",width=2),fillcolor="rgba(108,99,255,0.08)",name=f"AUC={auc:.4f}"))
                    if of and ot:
                        fig_roc.add_trace(go.Scatter(x=[of],y=[ot],mode="markers",
                            marker=dict(color="#ff6584",size=9,symbol="star"),name=f"Opt={oth}"))
                    fig_roc.update_layout(**PLOTLY_LAYOUT,height=300,
                        xaxis=dict(**AXIS_STYLE,title="FPR"),yaxis=dict(**AXIS_STYLE,title="TPR"),
                        legend=dict(font=dict(family="JetBrains Mono",size=10),bgcolor="rgba(0,0,0,0)"))
                    st.plotly_chart(fig_roc,use_container_width=True)

            if per_class:
                section_header("Per-Class Performance","Precision, Recall, F1 by class")
                pc_df=pd.DataFrame(per_class).T.reset_index()
                pc_df.columns=["Class","Precision","Recall","F1","Support"]
                fig_pc=go.Figure()
                for m,c in [("Precision","#6c63ff"),("Recall","#43e97b"),("F1","#f7971e")]:
                    fig_pc.add_trace(go.Bar(name=m,x=pc_df["Class"].astype(str),y=pc_df[m],
                        marker_color=c,marker_line_width=0))
                fig_pc.update_layout(**PLOTLY_LAYOUT,height=250,barmode="group",
                    xaxis=dict(**AXIS_STYLE),yaxis=dict(**AXIS_STYLE),
                    legend=dict(font=dict(family="JetBrains Mono",size=10),bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_pc,use_container_width=True)

            if calibration:
                section_header("Model Calibration",f"Brier Score: {calibration.get('brier_score')}")
                pt=calibration.get("prob_true",[]); pp=calibration.get("prob_pred",[]); ic=calibration.get("is_well_calibrated",False)
                fig_cal=go.Figure()
                fig_cal.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                    line=dict(color="rgba(255,255,255,0.18)",dash="dash",width=1),name="Perfect",showlegend=True))
                fig_cal.add_trace(go.Scatter(x=pp,y=pt,mode="lines+markers",
                    line=dict(color="#f7971e",width=2),marker=dict(size=6,color="#f7971e"),name="Model"))
                fig_cal.update_layout(**PLOTLY_LAYOUT,height=250,
                    xaxis=dict(**AXIS_STYLE,title="Mean Predicted Probability"),
                    yaxis=dict(**AXIS_STYLE,title="Fraction of Positives"),
                    legend=dict(font=dict(family="JetBrains Mono",size=10),bgcolor="rgba(0,0,0,0)"),
                    title=dict(text=f"{'✓ Well calibrated' if ic else '⚠ Poorly calibrated'}",
                               font=dict(size=11,family="JetBrains Mono",color="#43e97b" if ic else "#f7971e"),x=0))
                st.plotly_chart(fig_cal,use_container_width=True)

    with tab3:
        if fi:
            section_header("Feature Importance",f"Top features · {fi_summary.get('top5_total_contribution_pct','?')}% from top 5")
            top_n=20; top_fi=dict(list(fi.items())[:top_n]); tfi=sum(top_fi.values()) or 1
            fi_pct={k:round(v/tfi*100,2) for k,v in top_fi.items()}
            fig_fi=go.Figure()
            feats=list(fi_pct.keys())[::-1]; vals=[fi_pct[f] for f in feats]
            fig_fi.add_trace(go.Bar(x=vals,y=feats,orientation="h",
                marker_color=[f"rgba(108,99,255,{0.35+0.65*v/max(vals)})" for v in vals],marker_line_width=0,
                text=[f"{v:.1f}%" for v in vals],textposition="outside",
                textfont=dict(family="JetBrains Mono",size=10,color=CHART_TEXT_COLOR)))
            fig_fi.update_layout(**PLOTLY_LAYOUT,height=max(300,len(feats)*28),showlegend=False,
                xaxis=dict(**AXIS_STYLE,title="Relative Importance (%)"),yaxis=dict(**AXIS_STYLE))
            st.plotly_chart(fig_fi,use_container_width=True)

            insig=fi_summary.get("insignificant_features",[]); ic2=fi_summary.get("insignificant_features_count",0)
            if ic2>0:
                with st.expander(f"⚠ {ic2} Insignificant Features (<1% contribution)"):
                    st.markdown(f"<div style='font-family:JetBrains Mono;font-size:0.7rem;color:#7777aa;line-height:1.8;'>{' · '.join(insig)}</div>",unsafe_allow_html=True)

            shap_info=ev.get("shap_importance",{})
            if shap_info and shap_info.get("all_shap_values"):
                section_header("SHAP Explainability","Global feature impact · mean |SHAP|")
                sv=shap_info.get("all_shap_values",{}); ts2=sum(sv.values()) or 1
                sp={k:round(v/ts2*100,2) for k,v in list(sv.items())[:top_n]}
                fig_sh=go.Figure()
                fs=list(sp.keys())[::-1]; vs=[sp[f] for f in fs]
                fig_sh.add_trace(go.Bar(x=vs,y=fs,orientation="h",
                    marker_color=[f"rgba(255,101,132,{0.35+0.65*v/max(vs)})" for v in vs],marker_line_width=0,
                    text=[f"{v:.1f}%" for v in vs],textposition="outside",
                    textfont=dict(family="JetBrains Mono",size=10,color=CHART_TEXT_COLOR)))
                fig_sh.update_layout(**PLOTLY_LAYOUT,height=max(300,len(fs)*28),showlegend=False,
                    xaxis=dict(**AXIS_STYLE,title="Relative SHAP Impact (%)"),yaxis=dict(**AXIS_STYLE))
                st.plotly_chart(fig_sh,use_container_width=True)
        else:
            if task=="clustering":
                section_header("Cluster Statistics","Per-cluster breakdown")
                cd3=ev.get("cluster_distribution",{})
                if cd3:
                    st.dataframe(pd.DataFrame([{"Cluster":k,"Count":v.get("count",0),"Share":f"{v.get('pct',0):.1f}%"} for k,v in cd3.items()]),use_container_width=True,hide_index=True)
            else:
                st.info("Feature importance data not available.")

        if feature_report:
            section_header("Feature Engineering Summary","Transformations applied")
            if task in ("clustering","anomaly_detection"):
                ts3=feature_report.get("n_train",0)+feature_report.get("n_test",0)
                fc=st.columns(2)
                with fc[0]: st.markdown(render_metric_card("Final Features",str(feature_report.get("n_features","?")),"After selection","#6c63ff"),unsafe_allow_html=True)
                with fc[1]: st.markdown(render_metric_card("Total Samples",str(ts3) if ts3 else str(feature_report.get("n_train","?")),"Full dataset","#43e97b"),unsafe_allow_html=True)
            else:
                fc=st.columns(3)
                with fc[0]: st.markdown(render_metric_card("Final Features",str(feature_report.get("n_features","?")),"After selection","#6c63ff"),unsafe_allow_html=True)
                with fc[1]: st.markdown(render_metric_card("Train Samples",str(feature_report.get("n_train","?")),"80% split","#43e97b"),unsafe_allow_html=True)
                with fc[2]: st.markdown(render_metric_card("Test Samples",str(feature_report.get("n_test","?")),"20% split","#f7971e"),unsafe_allow_html=True)
                dist=feature_report.get("target_distribution_train",{})
                if dist:
                    section_header("Target Distribution (Train)","Class balance")
                    ld=[str(k) for k in dist.keys()]; vd=list(dist.values())
                    fig_d=go.Figure(go.Pie(labels=ld,values=vd,marker=dict(colors=COLORS[:len(ld)]),hole=0.6,textfont=dict(family="JetBrains Mono",size=11)))
                    fig_d.update_layout(**PLOTLY_LAYOUT,height=250,showlegend=True,legend=dict(font=dict(family="JetBrains Mono",size=10),bgcolor="rgba(0,0,0,0)"))
                    st.plotly_chart(fig_d,use_container_width=True)

    with tab4:
        bi=ev.get("bias_analysis",{})
        if bi:
            section_header("Demographic Fairness & Bias","Disparate Impact · Performance Differences")
            for dc,md in bi.items():
                st.markdown(f"#### 🎭 {dc.upper()}")
                for ec,m in md.items():
                    with st.expander(ec):
                        diff=m.get('accuracy_diff',m.get('rmse_diff',0)); di=m.get('disparate_impact')
                        cols=st.columns(3)
                        cols[0].metric("Group 1 Metric",m.get('accuracy_group_1',m.get('rmse_group_1')))
                        cols[1].metric("Group 0 Metric",m.get('accuracy_group_0',m.get('rmse_group_0')))
                        cols[2].metric("Difference",diff)
                        if di is not None:
                            st.markdown(f"**Disparate Impact Ratio**: {di:.3f} &nbsp;&nbsp; {'🟢 Fair' if 0.8<=di<=1.25 else '🔴 High Bias'}")
        else:
            st.info("No demographic bias detected or no demographic features available.")

    with tab5:
        if profiling:
            shape=profiling.get("shape",{}); summary=profiling.get("summary",{}); cols_p=profiling.get("columns",{})
            section_header("Dataset Overview","Raw data statistics")
            d1,d2,d3,d4=st.columns(4)
            with d1: st.markdown(render_metric_card("Rows",f"{shape.get('rows','?'):,}","Total records","#6c63ff"),unsafe_allow_html=True)
            with d2: st.markdown(render_metric_card("Columns",str(shape.get("columns","?")),"Total features","#43e97b"),unsafe_allow_html=True)
            with d3:
                mr=summary.get("missing_rate",0)
                st.markdown(render_metric_card("Missing Rate",f"{mr:.2%}","Null ratio","#f7971e" if mr>0.05 else "#43e97b"),unsafe_allow_html=True)
            with d4:
                dp=summary.get("duplicate_rows",0)
                st.markdown(render_metric_card("Duplicates",str(dp),"Removed","#ff6584" if dp>0 else "#43e97b"),unsafe_allow_html=True)

            nc=len(summary.get("numeric_columns",[])); cc=len(summary.get("categorical_columns",[])); dc2=len(summary.get("datetime_columns",[]))
            fig_t=go.Figure(go.Pie(labels=["Numeric","Categorical","Datetime"],values=[nc,cc,dc2],
                marker=dict(colors=["#6c63ff","#ff6584","#43e97b"]),hole=0.5,
                textfont=dict(family="JetBrains Mono",size=11)))
            fig_t.update_layout(**PLOTLY_LAYOUT,height=220,showlegend=True,
                legend=dict(font=dict(family="JetBrains Mono",size=10),bgcolor="rgba(0,0,0,0)"),
                title=dict(text="Column Type Distribution",font=dict(size=11,family="JetBrains Mono"),x=0))
            st.plotly_chart(fig_t,use_container_width=True)

            recs=profiling.get("recommendations",[])
            if recs:
                section_header("Profiling Recommendations","Data quality insights")
                for rec in recs:
                    ico="⚠" if "outlier" in rec.lower() or "missing" in rec.lower() else "ℹ"
                    c="#f7971e" if ico=="⚠" else "#6c63ff"
                    st.markdown(f"""<div style="display:flex;align-items:flex-start;gap:0.5rem;margin-bottom:0.35rem;
                        padding:0.45rem 0.7rem;background:{INFO_BG};border-radius:5px;border-left:2px solid {c};">
                        <span style="color:{c};font-size:0.68rem;">{ico}</span>
                        <span style="font-family:JetBrains Mono;font-size:0.7rem;color:{CHART_LABEL_COLOR};">{rec}</span>
                    </div>""", unsafe_allow_html=True)

            if cols_p:
                section_header("Column Statistics","Per-column profiling")
                cd_list=[{"Column":cn,"Type":info.get("dtype",""),"Null %":f"{info.get('null_rate',0):.1%}",
                          "Unique":info.get("unique_count",""),"Outliers":"⚠" if info.get("has_outliers") else "✓",
                          "Mean":f"{info.get('mean',''):.3f}" if info.get("mean") is not None else "",
                          "Std":f"{info.get('std',''):.3f}" if info.get("std") is not None else ""}
                         for cn,info in cols_p.items()]
                st.dataframe(pd.DataFrame(cd_list),use_container_width=True,hide_index=True,
                    column_config={"Outliers":st.column_config.TextColumn(width="small"),"Null %":st.column_config.TextColumn(width="small")})

    with tab6:
        section_header("Pipeline Execution Trace","Step-by-step node outputs")
        nodes_trace=[
            ("01 · Intent","intent_node",{"Intent":result.get("intent",""),"Confidence":f"{result.get('intent_confidence',0):.2%}"}),
            ("02 · DB Context","db_context_node",{"Tables":str(result.get("relevant_tables",[])),"Relationships":str(len(result.get("db_intelligence",{}).get("relationships",[]))),"Capabilities":str(result.get("db_intelligence",{}).get("capabilities",[]))}),
            ("03 · Planner","planner_node",{"Steps":str(len(result.get("analysis_plan",[]))),"Plan":" → ".join([s.get("step","") for s in result.get("analysis_plan",[])])}),
            ("04 · Execution","execution_node",{"Rows":f"{len(result.get('sql_results',[])):,}","Columns":str(profiling.get("shape",{}).get("columns","")),"Recs":str(len(profiling.get("recommendations",[])))}),
            ("05 · Feature Plan","feature_plan_node",{"Target":feature_plan.get("target",{}).get("column_name",""),"Derived":str(feature_plan.get("target",{}).get("is_derived","")),"Aggregation":feature_plan.get("aggregation",{}).get("level",""),"Groups":str(feature_plan.get("feature_groups",[]))}),
            ("06 · Feature Eng","feature_engineering_node",{"Features":str(feature_report.get("n_features","")),"Train":str(feature_report.get("n_train","")),"Test":str(feature_report.get("n_test","")),"Target dist":str(feature_report.get("target_distribution_train",{}))}),
            ("07 · AutoML","automl_node",{"Best model":model_name,**({"Algorithm":str(metrics.get("best_algorithm","?")),"Silhouette":f"{metrics.get('silhouette_score',0):.4f}","N Clusters":str(metrics.get("n_clusters","?"))} if intent=="clustering" else {"N Anomalies":str(metrics.get("n_anomalies","?")),"Anomaly %":f"{metrics.get('anomaly_pct',0):.2f}%","Contamination":str(metrics.get("best_contamination","?"))} if intent=="anomaly_detection" else {"CV ROC-AUC":f"{metrics.get('cv_roc_auc',0):.4f} ± {metrics.get('cv_roc_auc_std',0):.4f}","Test accuracy":f"{test_m.get('accuracy',0):.4f}","Test ROC-AUC":f"{test_m.get('roc_auc',test_m.get('roc_auc_ovr',0)):.4f}"})}),
            ("08 · Evaluation","evaluation_node",{"Reliability":f"{reliability.get('score',0)}/100 (Grade {reliability.get('grade','?')})","Overfitting":ovf.get("status",""),"Gap":f"{ovf.get('gap',0):.4f}","Brier":str(calibration.get('brier_score','N/A'))}),
            ("09 · Report","report_node",{"Length":f"{len(report):,} chars","Status":"✓ Generated" if report else "⚠ Failed"}),
        ]
        for nt,_,ni in nodes_trace:
            with st.expander(nt,expanded=False):
                for k,v in ni.items():
                    st.markdown(f"""<div style="display:flex;gap:1rem;padding:0.28rem 0;border-bottom:1px solid {TRACE_BORDER};">
                        <div style="font-family:JetBrains Mono;font-size:0.68rem;color:{CHART_LABEL_COLOR};min-width:130px;">{k}</div>
                        <div style="font-family:JetBrains Mono;font-size:0.68rem;color:{CHART_TEXT_COLOR};">{v}</div>
                    </div>""", unsafe_allow_html=True)

    with tab7:
        if report:
            section_header("Generated Report","LLM-synthesized analysis")
            st.markdown(f"""<div style="background:{STATUS_BG};border:1px solid {STATUS_BORDER};
                border-radius:12px;padding:2rem;font-family:'JetBrains Mono',monospace;
                font-size:0.8rem;line-height:1.85;color:{CHART_TEXT_COLOR};">""", unsafe_allow_html=True)
            st.markdown(report)
            st.markdown("</div>", unsafe_allow_html=True)

            if result.get("sql_results"):
                with st.expander("🔍 Veri Önizlemesi",expanded=True):
                    st.dataframe(pd.DataFrame(result["sql_results"][:5]),use_container_width=True,hide_index=True)
            pcols=result.get("profiling_report",{}).get("columns",{})
            if pcols:
                with st.expander("📊 Kolon İstatistikleri",expanded=False):
                    st.dataframe(pd.DataFrame([{"Kolon":cn,"Tip":i.get("dtype",""),"Boşluk":f"{i.get('null_rate',0):.1%}","Unique":i.get("unique_count",""),"Aykırı":"⚠" if i.get("has_outliers") else "✓"} for cn,i in pcols.items()]),use_container_width=True,hide_index=True)
            if result.get("metrics"):
                with st.expander("🎯 Model Metrikleri",expanded=False):
                    st.dataframe(pd.DataFrame([{"Metrik":k.replace('_',' ').title(),"Değer":f"{v:.4f}" if isinstance(v,float) else v} for k,v in result["metrics"].items() if not isinstance(v,(dict,list))]),use_container_width=True,hide_index=True)
        else:
            st.warning("Report not generated.")

        section_header("Experiment History","Log of past runs")
        history=get_experiments()
        if history:
            df_h=pd.DataFrame(history)
            if "id" in df_h.columns: df_h=df_h.drop(columns=["id"])
            if "timestamp" in df_h.columns: df_h["timestamp"]=pd.to_datetime(df_h["timestamp"]).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(df_h,use_container_width=True,hide_index=True)
        else:
            st.info("No experiment history found yet.")


if "is_running" not in st.session_state:
    st.session_state.is_running = False

if "result" not in st.session_state:
    st.session_state.result = None

if st.session_state.get("is_running") and st.session_state.get("run_query"):
    query_to_run = st.session_state.run_query
    st.session_state.run_query = None
    result = run_pipeline(query_to_run)
    st.session_state.result = result
    st.session_state.is_running = False
    st.rerun()

elif not st.session_state.get("is_running") and st.session_state.result is None:
    st.markdown(f"""
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
            min-height:55vh;text-align:center;padding:4rem 2rem;">
  <div style="font-size:2.8rem;margin-bottom:1rem;filter:drop-shadow(0 0 20px rgba(108,99,255,0.4));">⬡</div>
  <h2 style="font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:700;
             letter-spacing:-0.03em;color:{'#e8e8f0' if is_dark else '#1a1a2e'};margin:0 0 0.5rem;">
    Ready to Analyze
  </h2>
  <p style="font-family:'JetBrains Mono',monospace;font-size:0.77rem;
            color:{'#5555aa' if is_dark else '#7777aa'};max-width:380px;line-height:1.8;margin:0;">
    Write your query in the sidebar and click <strong style="color:#6c63ff;">Run Analysis</strong>
  </p>
  <div style="margin-top:2rem;display:flex;gap:0.8rem;flex-wrap:wrap;justify-content:center;">
    {''.join(f'<div style="background:rgba(108,99,255,0.06);border:1px solid rgba(108,99,255,0.15);border-radius:8px;padding:0.55rem 0.9rem;font-family:JetBrains Mono;font-size:0.65rem;color:{"#5555aa" if is_dark else "#7777aa"};">{t}</div>' for t in ['"Predict income level from census data"','"Segment customers by behavior"','"Forecast monthly revenue"'])}
  </div>
</div>""", unsafe_allow_html=True)

if st.session_state.result is not None:
    if st.session_state.view_mode == "Dashboard":
        render_results(st.session_state.result)
    else:
        render_chat_history()