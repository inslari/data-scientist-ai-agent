from src.state import AgentState
from src.services.leakage_detector import detect_leakage


def leakage_detector_node(state: AgentState) -> AgentState:
    result = detect_leakage(state)
    
    new_state = {
        **state,
        "domain_leakage_report": result.get("domain_leakage_report", {}),
        "current_step": "leakage_detector_node",
    }
    
    # DEBUG
    print(f"[DEBUG LEAKAGE NODE OUT] writing domain_leakage_report: {new_state.get('domain_leakage_report')}")
    print(f"[DEBUG LEAKAGE NODE OUT] keys in new_state: {list(new_state.keys())}")
    
    return new_state