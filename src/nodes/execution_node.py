from src.state import AgentState
from src.services.data_executor import execute_data_operations

async def execution_node(state: AgentState) -> AgentState:
    """
    Node 3: Planı (SQL sorguları dahil) yürütür ve veriyi Pandas DataFrame'e çeker.
    Tüm yürütme mantığı `src.services.data_executor` içerisine taşındı.
    """
    result = await execute_data_operations(state)
    
    if result.get("error"):
        return {
            **state,
            "error": result["error"],
            "current_step": "execution_node",
        }
    
    print(f"[DEBUG EXEC] data_path = '{result.get('data_path')}'")
    print(f"[DEBUG EXEC] sql_results len = {len(result.get('sql_results') or [])}")

    return {
        **state,
        "data_path": result.get("data_path"),        
        "sql_results": result.get("sql_results"),        
        "dataframe": result.get("dataframe"),
        "db_intelligence": result.get("db_intelligence", state.get("db_intelligence")),
        "profiling_report": result.get("profiling_report", state.get("profiling_report")),
        "current_step": "execution_node",
    }