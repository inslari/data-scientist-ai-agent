from src.state import AgentState
from src.services.feature_builder import build_features

def feature_engineering_node(state: AgentState) -> AgentState:
    """
    Node 5: Makine Öğrenmesi için Veri Hazırlığı.
    Tüm feature engineering mantığı `src.services.feature_builder` içerisine taşındı.
    """
    print("\n========================================\n[NODE START] feature_engineering_node başlıyor...\n========================================")

    data_path = state.get('data_path')
    print(f"[DEBUG FE] data_path = '{data_path}'")
    print(f"[DEBUG FE] sql_results len = {len(state.get('sql_results') or [])}")

    feature_plan = state.get("feature_plan", {})
    target = feature_plan.get("target", {})
    target_name = target.get("column_name") or state.get("target_definition", {}).get("column_name", "")
    
    # 1. Agent execution
    if data_path and target_name:
        try:
            from src.agents.feature_agent import get_feature_agent
            import json
            
            agent = get_feature_agent(data_path, target_name)
            
            columns = state.get("db_intelligence", {}).get("columns", [])
            cols_info = json.dumps([{
                "name": c.get("column_name"),
                "type": c.get("data_type"),
                "semantic": c.get("semantic_type"),
                "desc": c.get("description")
            } for c in columns], indent=2)
            
            prompt = f"Target Column: {target_name}\n\nAvailable Columns Metadata:\n{cols_info}"
            
            print(f"[feature_eng] Invoking Feature Agent to discover interactions...")
            agent_result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
            
            discovered_math = []
            discovered_drops = []
            
            for msg in reversed(agent_result["messages"]):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call["name"] == "commit_best_features":
                            args = tool_call["args"]
                            discovered_math = args.get("math_interactions", [])
                            discovered_drops = args.get("drop_features", [])
                            break
                if discovered_math or discovered_drops:
                    break
                    
            if discovered_math:
                print(f"[feature_eng] Agent discovered math interactions: {discovered_math}")
                existing_math = feature_plan.get("math_interactions", [])
                feature_plan["math_interactions"] = list(set(existing_math + discovered_math))
                
            if discovered_drops:
                print(f"[feature_eng] Agent suggested dropping: {discovered_drops}")
                existing_drops = feature_plan.get("drop_features", [])
                feature_plan["drop_features"] = list(set(existing_drops + discovered_drops))
                
            state["feature_plan"] = feature_plan
        except Exception as e:
            print(f"[feature_eng] Feature Agent failed, falling back to static plan. Error: {e}")

    # 2. Optimized deterministic pipeline
    result = build_features(state)
    
    if result.get("error"):
        return {
            **state,
            "error": result["error"],
            "current_step": "feature_engineering_node",
        }
        
    from src.utils.type_utils import clean_np
    return clean_np({
        **state,
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