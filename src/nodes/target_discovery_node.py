from src.core.nodes import node
from src.state import AgentState
from src.services.target_discovery import discover_target

@node("target_discovery_node")
def target_discovery_node(state: AgentState) -> AgentState:
    """
    Node: Hedef değişkeni belirler.
    Mantık src.services.target_discovery içine taşındı.
    """
    result = discover_target(state)
    
    if result.get("error"):
        return {
            "error": result["error"],
            "current_step": "target_discovery_node",
        }
        
    return {
        "intent": result.get("intent", state.get("intent")),
        "target_definition": result.get("target_definition", state.get("target_definition")),
        "current_step": "target_discovery_node",
    }