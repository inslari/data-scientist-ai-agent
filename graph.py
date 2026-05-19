from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from state import AgentState
from nodes.intent_node import intent_node
from nodes.db_context_node import db_context_node
from nodes.planner_node import planner_node
from nodes.execution_node import execution_node
from nodes.feature_plan_node import feature_plan_node
from nodes.feature_engineering_node import feature_engineering_node
from nodes.automl_node import automl_node

def build_graph() -> StateGraph:
    """
    Data Scientist Agent — LangGraph pipeline.

    Aktif pipeline:
        START → intent_node → db_context_node → planner_node
        → execution_node → feature_plan_node → feature_engineering_node
        → automl_node → END

    Sonraki adımlar:
        → evaluation_node → report_node → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("intent_node", intent_node)
    graph.add_node("db_context_node", db_context_node)
    graph.add_node("planner_node", planner_node)
    graph.add_node("execution_node", execution_node)
    graph.add_node("feature_plan_node", feature_plan_node)
    graph.add_node("feature_engineering_node", feature_engineering_node)
    graph.add_node("automl_node", automl_node)

    graph.add_edge(START, "intent_node")
    graph.add_edge("intent_node", "db_context_node")
    graph.add_edge("db_context_node", "planner_node")
    graph.add_edge("planner_node", "execution_node")
    graph.add_edge("execution_node", "feature_plan_node")
    graph.add_edge("feature_plan_node", "feature_engineering_node")
    graph.add_edge("feature_engineering_node", "automl_node")
    graph.add_edge("automl_node", END)

    # Sonraki adımlarda eklenecek:
    # graph.add_node("evaluation_node", evaluation_node)
    # graph.add_node("report_node", report_node)
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
        "user_query": "Bu veri setindeki kişilerin gelir düzeyini tahmin et. Hangi faktörler gelirin 50K üzerinde olmasını etkiliyor?",
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
    else:
        metrics = result.get("metrics", {})
        if metrics:
            print(f"\nModel Sonuçları ({result.get('model_name')}):")
            for k, v in metrics.items():
                if k not in ("per_class_report", "feature_importance", "task"):
                    print(f"  {k}: {v}")
            fi = metrics.get("feature_importance", {})
            if fi:
                print(f"\n  Top 5 Feature Importance:")
                for feat, imp in list(fi.items())[:5]:
                    print(f"    {feat}: {imp:.4f}")

        feature_report = result.get("feature_report", {})
        if feature_report:
            print(f"\nFeature Engineering:")
            print(f"  Hedef değişken       : {feature_report.get('target_name')}")
            print(f"  Final feature sayısı : {feature_report.get('n_features')}")
            print(f"  Eğitim seti          : {feature_report.get('n_train')} satır")
            print(f"  Test seti            : {feature_report.get('n_test')} satır")
            dist = feature_report.get("target_distribution_train", {})
            print(f"  Hedef dağılımı       : {dist}")
            print(f"  Özellikler           : {feature_report.get('final_features')}")