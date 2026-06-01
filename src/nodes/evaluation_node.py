from src.state import AgentState
from src.services.evaluator import evaluate_model

def evaluation_node(state: AgentState) -> AgentState:
    """
    Node 7: Makine Öğrenmesi modeli değerlendirmesi.
    Tüm metrik hesaplama mantığı `src.services.evaluator` içerisine taşındı.
    """

        
    result = evaluate_model(state)
    print(f"[DEBUG EVAL] Keys from evaluator: {list(result.keys())}")
    if result.get("error"):
        return {
            **state,
            "error": result["error"],
            "current_step": "evaluation_node",
        }
        
    return {
        **state,
        "metrics": result.get("metrics", state.get("metrics")),
        "feature_importance": result.get("feature_importance", state.get("feature_importance")),
        "evaluation_report": result.get("evaluation_report", {}),
        "current_step": "evaluation_node",
    }