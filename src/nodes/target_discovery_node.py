from src.state import AgentState
from src.services.target_discovery import discover_target

def target_discovery_node(state: AgentState) -> AgentState:
    """
    Node: Hedef değişkeni belirler.
    Mantık src.services.target_discovery içine taşındı.
    """
    print("\n========================================\n[NODE START] target_discovery_node başlıyor...\n========================================")
    result = discover_target(state)
    
    if result.get("error"):
        return {
            **state,
            "error": result["error"],
            "current_step": "target_discovery_node",
        }
        
    return {
        **state,
        "intent": result.get("intent", state.get("intent")),
        "target_definition": result.get("target_definition", state.get("target_definition")),
        "current_step": "target_discovery_node",
    }