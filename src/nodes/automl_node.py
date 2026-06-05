from ..state import AgentState
from ..services.ml_trainer import train_model

def automl_node(state: AgentState) -> AgentState:
    """
    Node 6: Makine öğrenmesi modeli seçimi ve eğitimi.
    Tüm ML algoritma mantığı `src.services.ml_trainer` içerisine taşındı.
    """
    print("\n========================================\n[NODE START] automl_node başlıyor...\n========================================")
    if state.get("error"):
        print(f"[DEBUG ML] Mevcut hata bulundu, atlanıyor: {state.get('error')}")
        return {**state, "current_step": "automl_node"}
        
    result = train_model(state)
    
    # ml_node sonunda, return'den hemen önce:
    print(f"[DEBUG ML] State keys being returned: {list(result.keys())}")
    print(f"[DEBUG ML] Model name: {result.get('best_model_name', 'YOK')}")
    print(f"[DEBUG ML] Eval report: {bool(result.get('evaluation_report'))}")

    if result.get("error"):
        return {
            **state,
            "error": result["error"],
            "current_step": "automl_node",
        }
        
    from src.utils.type_utils import clean_np
    print(f"[DEBUG FE OUT] domain_leakage_report var mi: {state.get('domain_leakage_report') is not None}")
    return clean_np({
        **state,
        "intent": result.get("intent", state.get("intent")),
        "y_train": result.get("y_train", state.get("y_train")),
        "y_test": result.get("y_test", state.get("y_test")),
        "model_name": result.get("model_name", ""),
        "model_params": result.get("model_params", {}),
        "model_artifact": result.get("model_artifact"),
        "metrics": result.get("metrics", {}),
        "top_models": result.get("top_models", []),
        "current_step": "automl_node",
    })