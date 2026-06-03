from src.state import AgentState
from src.services.feature_builder import build_features

def feature_engineering_node(state: AgentState) -> AgentState:
    """
    Node 5: Makine Öğrenmesi için Veri Hazırlığı.
    Tüm feature engineering mantığı `src.services.feature_builder` içerisine taşındı.
    """
    print("\n========================================\n[NODE START] feature_engineering_node başlıyor...\n========================================")

    print(f"[DEBUG FE] data_path = '{state.get('data_path')}'")
    print(f"[DEBUG FE] sql_results len = {len(state.get('sql_results') or [])}")

    result = build_features(state)
    
    if result.get("error"):
        return {
            **state,
            "error": result["error"],
            "current_step": "feature_engineering_node",
        }
        
    from src.utils.type_utils import clean_np
    return clean_np({
        **state,
        "X_train": result.get("X_train"),
        "X_test": result.get("X_test"),
        "y_train": result.get("y_train"),
        "y_test": result.get("y_test"),
        "feature_names": result.get("feature_names"),
        "target_name": result.get("target_name"),
        "feature_report": result.get("feature_report"),
        "target_definition": result.get("target_definition", state.get("target_definition")),
        "current_step": "feature_engineering_node",
    })