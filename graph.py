from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8')

from langgraph.graph import END, START, StateGraph

from state import AgentState
from nodes.intent_node import intent_node
from nodes.db_context_node import db_context_node
from nodes.planner_node import planner_node
from nodes.execution_node import execution_node

# --- Sonraki adımlarda eklenecek node'lar ---
# from nodes.feature_engineering_node import feature_engineering_node
# from nodes.automl_node import automl_node
# from nodes.evaluation_node import evaluation_node
# from nodes.report_node import report_node


def build_graph() -> StateGraph:
    """
    Data Scientist Agent graph'ını oluşturur ve derler.
    
    Şu an aktif olan node'lar:
        START → intent_node → db_context_node → planner_node → END
    
    Sonraki adımlarda eklenecek:
        planner_node → tool_router → execution_node → feature_engineering_node
        → automl_node → evaluation_node → report_node → END
    """
    graph = StateGraph(AgentState)

    # --- Node'ları ekle ---
    graph.add_node("intent_node", intent_node)
    graph.add_node("db_context_node", db_context_node)
    graph.add_node("planner_node", planner_node)
    graph.add_node("execution_node", execution_node)

    # Sonraki adımlarda aktif edilecek:
    # graph.add_node("feature_engineering_node", feature_engineering_node)
    # graph.add_node("automl_node", automl_node)
    # graph.add_node("evaluation_node", evaluation_node)
    # graph.add_node("report_node", report_node)

    # --- Edge'leri bağla ---
    graph.add_edge(START, "intent_node")
    graph.add_edge("intent_node", "db_context_node")
    graph.add_edge("db_context_node", "planner_node")
    graph.add_edge("planner_node", "execution_node")
    graph.add_edge("execution_node", END)

    # Sonraki adımlarda:
    # graph.add_edge("planner_node", "execution_node")
    # graph.add_conditional_edges(
    #     "execution_node",
    #     route_after_execution,   # intent'e göre yönlendir
    #     {
    #         "feature_engineering": "feature_engineering_node",
    #         "stats_only": "evaluation_node",
    #     }
    # )
    # graph.add_edge("feature_engineering_node", "automl_node")
    # graph.add_edge("automl_node", "evaluation_node")
    # graph.add_edge("evaluation_node", "report_node")
    # graph.add_edge("report_node", END)

    return graph.compile()


# --- Conditional edge fonksiyonu (sonraki adım için hazır) ---
def route_after_execution(state: AgentState) -> str:
    """
    Execution node sonrasında intent'e göre yönlendirir.
    EDA/report → direkt evaluation'a git
    ML intent'leri → feature engineering'e git
    """
    intent = state.get("intent", "eda")
    if intent in ("classification", "regression", "clustering"):
        return "feature_engineering"
    return "stats_only"


# --- Çalıştırma ---
if __name__ == "__main__":
    import json

    app = build_graph()

    initial_state: AgentState = {
        "user_query": "Can you analyze customer purchasing behavior and predict which customers are likely to churn?",
        "messages": [],
    }

    print("=== Data Scientist Agent ===\n")
    result = app.invoke(initial_state)

    print(f"Intent     : {result.get('intent')} (confidence: {result.get('intent_confidence')})")
    print(f"Tables     : {result.get('relevant_tables')}")
    print(f"\nAnalysis Plan:")
    for step in result.get("analysis_plan", []):
        print(f"  {step['order']}. [{step['step']}] {step['description']}")

    print(f"\nExecution:")
    sql = result.get("sql_results", [])
    print(f"  SQL sonucu  : {len(sql)} satir cekildi")

    profiling = result.get("profiling_report", {})
    if profiling:
        shape = profiling.get("shape", {})
        print(f"  Profiling   : {shape.get('rows')} satir x {shape.get('columns')} kolon")
        recs = profiling.get("recommendations", [])
        if recs:
            print(f"  Oneriler    :")
            for r in recs:
                print(f"    - {r}")

    if result.get("error"):
        print(f"\nHata: {result['error']}")