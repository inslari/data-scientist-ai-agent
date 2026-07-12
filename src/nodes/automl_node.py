from src.core.nodes import node
from ..state import AgentState
from ..services.ml_trainer import train_model

from src.core.logging_config import get_logger
logger = get_logger(__name__)

@node("automl_node")
def automl_node(state: AgentState) -> AgentState:
    """
    Node 6: Makine öğrenmesi modeli seçimi ve eğitimi.
    Tüm ML algoritma mantığı `src.services.ml_trainer` içerisine taşındı.

    Upstream-error short-circuit @node dekoratöründe — burada kontrol gerekmez.
    """
    result = train_model(state)
    
    # ml_node sonunda, return'den hemen önce:
    logger.info(f"[DEBUG ML] State keys being returned: {list(result.keys())}")
    logger.info(f"[DEBUG ML] Model name: {result.get('best_model_name', 'YOK')}")
    logger.info(f"[DEBUG ML] Eval report: {bool(result.get('evaluation_report'))}")

    if result.get("error"):
        return {
            "error": result["error"],
            "current_step": "automl_node",
        }
        
    from src.utils.type_utils import clean_np
    logger.info(f"[DEBUG FE OUT] domain_leakage_report var mi: {state.get('domain_leakage_report') is not None}")
    return clean_np({
        "intent": result.get("intent", state.get("intent")),
        # AutoML label-encode SONRASI y'yi yeniden diske yazar → güncel yolları ilet
        "y_train_path": result.get("y_train_path", state.get("y_train_path")),
        "y_test_path": result.get("y_test_path", state.get("y_test_path")),
        "model_name": result.get("model_name", ""),
        "model_params": result.get("model_params", {}),
        "model_artifact": result.get("model_artifact"),
        "metrics": result.get("metrics", {}),
        "top_models": result.get("top_models", []),
        # Büyük-N guard notu → rapora yansır (sessiz değil).
        "heavy_model_guard": result.get("heavy_model_guard", {}),
        "current_step": "automl_node",
    })