from src.core.nodes import node
from src.state import AgentState
from src.services.feature_builder import build_features

from src.core.logging_config import get_logger
logger = get_logger(__name__)

@node("feature_engineering_node")
def feature_engineering_node(state: AgentState) -> AgentState:
    """
    Node 5: Makine Öğrenmesi için Veri Hazırlığı.
    Tüm feature engineering mantığı `src.services.feature_builder` içerisine taşındı.
    """

    logger.info(f"[DEBUG FE] data_path = '{state.get('data_path')}'")
    logger.info(f"[DEBUG FE] sql_results len = {len(state.get('sql_results') or [])}")

    result = build_features(state)
    
    if result.get("error"):
        return {
            "error": result["error"],
            "current_step": "feature_engineering_node",
        }
        
    from src.utils.type_utils import clean_np

    logger.info(f"[DEBUG FE OUT] domain_leakage_report var mi: {state.get('domain_leakage_report') is not None}")
    return clean_np({
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