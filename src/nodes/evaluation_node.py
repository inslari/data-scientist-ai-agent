from src.core.nodes import node
from src.state import AgentState
from src.services.evaluator import evaluate_model

from src.core.logging_config import get_logger
logger = get_logger(__name__)

@node("evaluation_node")
def evaluation_node(state: AgentState) -> AgentState:
    """
    Node 7: Makine Öğrenmesi modeli değerlendirmesi.
    Tüm metrik hesaplama mantığı `src.services.evaluator` içerisine taşındı.

    Upstream-error short-circuit @node dekoratöründe — burada kontrol gerekmez.
    """
    result = evaluate_model(state)
    logger.info(f"[DEBUG EVAL] Keys from evaluator: {list(result.keys())}")
    if result.get("error"):
        return {
            "error": result["error"],
            "current_step": "evaluation_node",
        }
        
    from src.utils.type_utils import clean_np
    logger.info(f"[DEBUG FE OUT] domain_leakage_report var mi: {state.get('domain_leakage_report') is not None}")
    return clean_np({
        "metrics": result.get("metrics", state.get("metrics")),
        "feature_importance": result.get("feature_importance", state.get("feature_importance")),
        "evaluation_report": result.get("evaluation_report", {}),
        "current_step": "evaluation_node",
    })