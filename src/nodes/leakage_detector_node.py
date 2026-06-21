from src.core.nodes import node
from src.state import AgentState
from src.services.leakage_detector import detect_leakage

from src.core.logging_config import get_logger
logger = get_logger(__name__)


@node("leakage_detector_node")
def leakage_detector_node(state: AgentState) -> AgentState:
    result = detect_leakage(state)
    
    new_state = {
        "domain_leakage_report": result.get("domain_leakage_report", {}),
        "current_step": "leakage_detector_node",
    }
    
    # DEBUG
    logger.info(f"[DEBUG LEAKAGE NODE OUT] writing domain_leakage_report: {new_state.get('domain_leakage_report')}")
    logger.info(f"[DEBUG LEAKAGE NODE OUT] keys in new_state: {list(new_state.keys())}")
    
    return new_state