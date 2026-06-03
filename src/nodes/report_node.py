from ..state import AgentState
from ..services.report_generator import generate_report

def report_node(state: AgentState) -> AgentState:
    """
    Node 9: Pipeline'ın tüm çıktılarını LLM ile yorumlayarak
    kapsamlı bir Markdown raporu üretir.
    Prompt mantığı `src.prompts` altına, üretim mantığı `src.services.report_generator` altına taşındı.
    """
    print("\n========================================\n[NODE START] report_node başlıyor...\n========================================")
    print(f"[DEBUG REPORT] State keys available: {[k for k, v in state.items() if v]}")
    result = generate_report(state)
    
    if result.get("error"):
        return {
            **state,
            "report": result.get("report", ""),
            "error": result["error"],
            "current_step": "report_node",
        }
        
    return {
        **state,
        "report": result.get("report", ""),
        "current_step": "report_node",
    }
