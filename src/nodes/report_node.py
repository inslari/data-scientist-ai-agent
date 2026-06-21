from src.core.nodes import node
from ..state import AgentState
from ..services.report_generator import generate_report

from src.core.logging_config import get_logger
logger = get_logger(__name__)

@node("report_node", runs_on_error=True)
def report_node(state: AgentState) -> AgentState:
    """
    Node 9: Pipeline'ın tüm çıktılarını LLM ile yorumlayarak
    kapsamlı bir Markdown raporu üretir.
    Prompt mantığı `src.prompts` altına, üretim mantığı `src.services.report_generator` altına taşındı.
    """
    logger.info(f"[DEBUG REPORT] State keys available: {[k for k, v in state.items() if v]}")
    result = generate_report(state)
    
    if result.get("error"):
        return {
            "report": result.get("report", ""),
            "error": result["error"],
            "current_step": "report_node",
        }
        
    return {
        "report": result.get("report", ""),
        "current_step": "report_node",
    }
