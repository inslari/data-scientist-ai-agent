"""
Data Scientist Agent — Streamlit Dashboard
==========================================
Professional dark-theme dashboard for the AI-powered data science pipeline.

Run:
    streamlit run app.py
"""

from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "src"))

from src.graph import build_graph
from src.state import AgentState

# Page config
st.set_page_config(
    page_title="DS Agent",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Syne:wght@400;500;600;700;800&display=swap');

:root {
    --bg:        #0a0a0f;
    --bg2:       #111118;
    --bg3:       #16161f;
    --border:    rgba(255,255,255,0.06);
    --border2:   rgba(255,255,255,0.12);
    --accent:    #6c63ff;
    --accent2:   #ff6584;
    --accent3:   #43e97b;
    --accent4:   #f7971e;
    --text:      #e8e8f0;
    --text2:     #8888aa;
    --text3:     #4a4a6a;
    --mono:      'DM Mono', monospace;
    --sans:      'Syne', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Inputs */
.stTextArea textarea, .stTextInput input {
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(108,99,255,0.15) !important;
}

/* Buttons */
.stButton > button {
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
}
.stButton > button:hover {
    background: #8075ff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.35) !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: var(--text2) !important; font-family: var(--mono) !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: var(--sans) !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-family: var(--mono) !important; }

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--bg2) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stTabs"] [role="tab"] {
    color: var(--text2) !important;
    font-family: var(--sans) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    border-radius: 7px !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Code blocks */
code { font-family: var(--mono) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Custom scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--text3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# Plotly dark theme
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#8888aa", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
)
AXIS_STYLE = dict(
    gridcolor="rgba(255,255,255,0.04)",
    zerolinecolor="rgba(255,255,255,0.08)",
)
COLORS = ["#6c63ff", "#ff6584", "#43e97b", "#f7971e", "#38b2f4", "#f093fb"]

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 1.5rem 0;">
        <div style="font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800; 
                    letter-spacing: -0.02em; color: #e8e8f0;">
            ⬡ DS Agent
        </div>
        <div style="font-family: 'DM Mono', monospace; font-size: 0.7rem; 
                    color: #6c63ff; margin-top: 2px; letter-spacing: 0.08em;">
            AI-POWERED ANALYTICS
        </div>
    </div>
    <hr style="border-color: rgba(255,255,255,0.06); margin-bottom: 1.5rem;">
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-family: DM Mono; font-size: 0.72rem; color: #8888aa; letter-spacing: 0.08em; margin-bottom: 0.5rem;'>QUERY</p>", unsafe_allow_html=True)

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
    st.markdown("""
    <div style="font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #4a4a6a; line-height: 1.8;">
        <div style="color: #8888aa; font-size: 0.65rem; letter-spacing: 0.08em; margin-bottom: 0.5rem;">PIPELINE</div>
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


# Header
st.markdown("""
<div style="padding: 2rem 0 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 2rem;">
    <h1 style="font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; 
               letter-spacing: -0.03em; color: #e8e8f0; margin: 0;">
        Data Scientist Agent
    </h1>
    <p style="font-family: 'DM Mono', monospace; font-size: 0.78rem; color: #8888aa; 
              margin: 0.4rem 0 0 0;">
        Fully autonomous ML pipeline — from query to insights
    </p>
</div>
""", unsafe_allow_html=True)



# Pipeline Runner
def run_pipeline(user_query: str) -> AgentState:
    """Runs the full LangGraph pipeline and returns final state."""
    app = build_graph()
    initial_state: AgentState = {
        "user_query": user_query,
        "messages": [],
    }
    return app.invoke(initial_state)


# Loading Animation
def show_loading():
    """Animated node-by-node loading sequence."""
    nodes = [
        ("01", "Intent Analyzer", "Classifying analysis intent..."),
        ("02", "DB Context Loader", "Fetching schema & running intelligence engine..."),
        ("03", "Analysis Planner", "Building execution plan..."),
        ("04", "Execution Layer", "Fetching data & profiling..."),
        ("05", "Feature Planner", "LLM designing feature strategy..."),
        ("06", "Feature Engineering", "Encoding, scaling, selecting features..."),
        ("07", "AutoML", "Training & comparing 6 models with 5-fold CV..."),
        ("08", "Evaluation", "Computing metrics & reliability score..."),
        ("09", "Report Generator", "Synthesizing insights with GPT-4o..."),
    ]

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


# Dashboard Sections
def render_metric_card(label: str, value: str, delta: str = "", color: str = "#6c63ff") -> str:
    return f"""
    <div style="background: #16161f; border: 1px solid rgba(255,255,255,0.06); border-radius: 10px;
                padding: 1rem 1.2rem; border-top: 2px solid {color};">
        <div style="font-family: 'DM Mono', monospace; font-size: 0.68rem; color: #8888aa; 
                    letter-spacing: 0.08em; margin-bottom: 0.4rem;">{label.upper()}</div>
        <div style="font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700; 
                    color: #e8e8f0;">{value}</div>
        {f'<div style="font-family: DM Mono; font-size: 0.7rem; color: {color}; margin-top: 2px;">{delta}</div>' if delta else ''}
    </div>
    """


def section_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="margin: 2rem 0 1rem 0;">
        <div style="font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; 
                    color: #e8e8f0; letter-spacing: -0.01em;">{title}</div>
        {f'<div style="font-family: DM Mono; font-size: 0.72rem; color: #8888aa; margin-top: 2px;">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def render_results(result: AgentState):
    """Renders the full dashboard from pipeline results."""

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

    # Status Bar
    grade = reliability.get("grade", "?")
    score = reliability.get("score", 0)
    grade_color = {"A": "#43e97b", "B": "#6c63ff", "C": "#f7971e", "D": "#ff6584", "F": "#ff6584"}.get(grade, "#8888aa")

    st.markdown(f"""
    <div style="background: #111118; border: 1px solid rgba(255,255,255,0.06); border-radius: 10px;
                padding: 1rem 1.5rem; display: flex; align-items: center; gap: 2rem; 
                margin-bottom: 2rem; flex-wrap: wrap;">
        <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #43e97b;">
            ✓ PIPELINE COMPLETE
        </div>
        <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #8888aa;">
            MODEL: <span style="color: #e8e8f0;">{model_name.upper()}</span>
        </div>
        <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #8888aa;">
            INTENT: <span style="color: #6c63ff;">{intent.upper()}</span>
        </div>
        <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #8888aa;">
            RELIABILITY: <span style="color: {grade_color}; font-weight: 700;">{score}/100 · {grade}</span>
        </div>
        <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #8888aa;">
            FEATURES: <span style="color: #e8e8f0;">{feature_report.get('n_features', '?')}</span>
        </div>
        <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #8888aa;">
            TARGET: <span style="color: #e8e8f0;">{feature_report.get('target_name', '?').upper()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "🎯 Model Performance",
        "🔍 Feature Analysis",
        "🧪 Data & Profiling",
        "🗺️ Pipeline Trace",
        "📄 Full Report",
    ])

    # Tab 1 — Overview
    with tab1:
        section_header("Key Metrics", "Core performance indicators from test set")

        # Metric cards
        auc = test_m.get("roc_auc", test_m.get("roc_auc_ovr", 0))
        acc = test_m.get("accuracy", 0)
        f1 = test_m.get("f1_weighted", 0)
        ll = test_m.get("log_loss", None)

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

        # Overfitting radar / gauge
        with col_left:
            section_header("Overfitting Analysis", "Train vs Test performance gap")
            train_s = ovf.get("train_score", 0)
            test_s = ovf.get("test_score", 0)
            gap = ovf.get("gap", 0)
            status = ovf.get("status", "unknown")
            status_color = {"normal": "#43e97b", "mild_overfitting": "#f7971e",
                           "overfitting_detected": "#ff6584", "underfitting": "#38b2f4"}.get(status, "#8888aa")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Train", "Test"],
                y=[train_s, test_s],
                marker_color=["#6c63ff", "#43e97b"],
                marker_line_width=0,
                text=[f"{train_s:.4f}", f"{test_s:.4f}"],
                textposition="outside",
                textfont=dict(family="DM Mono", size=11, color="#e8e8f0"),
            ))
            fig.update_layout(**PLOTLY_LAYOUT,
                height=220,
                title=dict(text=f"<span style='color:{status_color}'>{status.replace('_', ' ').title()}</span> · gap: {gap:.4f}",
                          font=dict(size=11, family="DM Mono"), x=0),
                xaxis=dict(**AXIS_STYLE),
                yaxis=dict(**AXIS_STYLE, range=[max(0, min(train_s, test_s) - 0.1), 1.05]),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""
            <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #8888aa; 
                        padding: 0.6rem 0.8rem; background: rgba(255,255,255,0.03); border-radius: 6px;">
                {ovf.get('description', '')}
            </div>
            """, unsafe_allow_html=True)

        # Model comparison
        with col_right:
            section_header("Model Comparison", "5-fold cross-validation ROC-AUC")
            if all_scores:
                sorted_scores = dict(sorted(all_scores.items(), key=lambda x: float(x[1]), reverse=True))
                models = list(sorted_scores.keys())
                scores = [float(v) for v in sorted_scores.values()]
                bar_colors = [COLORS[0] if m == model_name else "#2a2a3a" for m in models]

                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=scores, y=models, orientation="h",
                    marker_color=bar_colors, marker_line_width=0,
                    text=[f"{s:.4f}" for s in scores],
                    textposition="outside",
                    textfont=dict(family="DM Mono", size=10, color="#e8e8f0"),
                ))
                fig2.update_layout(**PLOTLY_LAYOUT,
                    height=220,
                    xaxis=dict(**AXIS_STYLE, range=[min(scores) - 0.02, max(scores) + 0.02]),
                    yaxis=dict(**AXIS_STYLE),
                    showlegend=False,
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

    # Tab 2 — Model Performance
    with tab2:
        col_l, col_r = st.columns(2)

        # Confusion Matrix
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

        # ROC Curve
        with col_r:
            section_header("ROC Curve", f"AUC = {auc:.4f}")
            if roc:
                fpr_vals = roc.get("fpr", [])
                tpr_vals = roc.get("tpr", [])
                opt_fpr = roc.get("optimal_fpr", None)
                opt_tpr = roc.get("optimal_tpr", None)
                opt_thresh = roc.get("optimal_threshold", None)

                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines", line=dict(color="rgba(255,255,255,0.15)", dash="dash", width=1),
                    showlegend=False,
                ))
                fig_roc.add_trace(go.Scatter(
                    x=fpr_vals, y=tpr_vals,
                    mode="lines", fill="tozeroy",
                    line=dict(color="#6c63ff", width=2),
                    fillcolor="rgba(108,99,255,0.1)",
                    name=f"AUC = {auc:.4f}",
                ))
                if opt_fpr and opt_tpr:
                    fig_roc.add_trace(go.Scatter(
                        x=[opt_fpr], y=[opt_tpr],
                        mode="markers",
                        marker=dict(color="#ff6584", size=10, symbol="star"),
                        name=f"Optimal threshold = {opt_thresh}",
                    ))
                fig_roc.update_layout(**PLOTLY_LAYOUT, height=300,
                    xaxis=dict(**AXIS_STYLE, title="False Positive Rate"),
                    yaxis=dict(**AXIS_STYLE, title="True Positive Rate"),
                    legend=dict(font=dict(family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Per-class metrics
        if per_class:
            section_header("Per-Class Performance", "Precision, Recall, F1 by class")
            pc_df = pd.DataFrame(per_class).T.reset_index()
            pc_df.columns = ["Class", "Precision", "Recall", "F1", "Support"]

            fig_pc = go.Figure()
            for metric, color in [("Precision", "#6c63ff"), ("Recall", "#43e97b"), ("F1", "#f7971e")]:
                fig_pc.add_trace(go.Bar(
                    name=metric, x=pc_df["Class"].astype(str), y=pc_df[metric],
                    marker_color=color, marker_line_width=0,
                ))
            fig_pc.update_layout(**PLOTLY_LAYOUT, height=250, barmode="group",
                xaxis=dict(**AXIS_STYLE),
                yaxis=dict(**AXIS_STYLE),
                legend=dict(font=dict(family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_pc, use_container_width=True)

        # Calibration
        if calibration:
            section_header("Model Calibration", f"Brier Score: {calibration.get('brier_score')}")
            prob_true = calibration.get("prob_true", [])
            prob_pred = calibration.get("prob_pred", [])
            is_cal = calibration.get("is_well_calibrated", False)

            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color="rgba(255,255,255,0.2)", dash="dash", width=1),
                name="Perfect calibration", showlegend=True,
            ))
            fig_cal.add_trace(go.Scatter(
                x=prob_pred, y=prob_true, mode="lines+markers",
                line=dict(color="#f7971e", width=2),
                marker=dict(size=7, color="#f7971e"),
                name="Model calibration",
            ))
            fig_cal.update_layout(**PLOTLY_LAYOUT, height=250,
                xaxis=dict(**AXIS_STYLE, title="Mean Predicted Probability"),
                yaxis=dict(**AXIS_STYLE, title="Fraction of Positives"),
                legend=dict(font=dict(family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)"),
                title=dict(
                    text=f"{'✓ Well calibrated' if is_cal else '⚠ Poorly calibrated'}",
                    font=dict(size=11, family="DM Mono",
                             color="#43e97b" if is_cal else "#f7971e"), x=0,
                ),
            )
            st.plotly_chart(fig_cal, use_container_width=True)

    # Tab 3 — Feature Analysis
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
                textfont=dict(family="DM Mono", size=10, color="#e8e8f0"),
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
        else:
            st.info("Feature importance data not available.")

        # Feature Engineering summary
        if feature_report:
            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Feature Engineering Summary", "Transformations applied by the pipeline")

            fr_cols = st.columns(3)
            with fr_cols[0]:
                st.markdown(render_metric_card("Final Features", str(feature_report.get("n_features", "?")), "After selection", "#6c63ff"), unsafe_allow_html=True)
            with fr_cols[1]:
                st.markdown(render_metric_card("Train Samples", str(feature_report.get("n_train", "?")), "80% split", "#43e97b"), unsafe_allow_html=True)
            with fr_cols[2]:
                st.markdown(render_metric_card("Test Samples", str(feature_report.get("n_test", "?")), "20% split", "#f7971e"), unsafe_allow_html=True)

            # Target distribution
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

    # Tab 4 - Data & Profiling
    with tab4:
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
                                padding: 0.5rem 0.8rem; background: rgba(255,255,255,0.02); 
                                border-radius: 6px; border-left: 2px solid {color};">
                        <span style="color: {color}; font-size: 0.7rem; margin-top: 1px;">{icon}</span>
                        <span style="font-family: DM Mono; font-size: 0.72rem; color: #8888aa;">{rec}</span>
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

    # Tab 5 — Pipeline Trace
    with tab5:
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
                "CV ROC-AUC": f"{metrics.get('cv_roc_auc', 0):.4f} ± {metrics.get('cv_roc_auc_std', 0):.4f}",
                "Test accuracy": f"{test_m.get('accuracy', 0):.4f}",
                "Test ROC-AUC": f"{test_m.get('roc_auc', test_m.get('roc_auc_ovr', 0)):.4f}",
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
                                border-bottom: 1px solid rgba(255,255,255,0.03);">
                        <div style="font-family: DM Mono; font-size: 0.7rem; color: #8888aa; 
                                    min-width: 140px;">{k}</div>
                        <div style="font-family: DM Mono; font-size: 0.7rem; color: #e8e8f0;">{v}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # Full Report
    with tab6:
        if report:
            section_header("Generated Report", "LLM-synthesized analysis in Turkish")
            st.markdown(f"""
            <div style="background: #111118; border: 1px solid rgba(255,255,255,0.06); 
                        border-radius: 12px; padding: 2rem; font-family: 'DM Mono', monospace;
                        font-size: 0.82rem; line-height: 1.8; color: #c8c8d8;">
            """, unsafe_allow_html=True)
            st.markdown(report)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Report not generated.")

# Main Flow
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