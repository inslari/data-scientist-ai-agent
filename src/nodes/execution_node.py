from langchain_core.runnables import RunnableConfig
from src.state import AgentState
from src.services.data_executor import execute_data_operations
from src.services.target_discovery import infer_task_type

# Görev tipi override'ının çalışabileceği model-eğitme intent'leri.
# deep_analysis / eda / report / sql_query / clustering / anomaly_detection
# (hedefsiz veya analiz akışları) bu sete DAHİL DEĞİL — intent'lerine dokunulmaz.
_TASK_OVERRIDE_INTENTS = {"classification", "regression"}


def _maybe_override_intent(state: AgentState, data_path: str | None) -> str | None:
    """
    Manuel hedef seçiminde görev tipini hedef sütunun verisinden çıkarıp
    intent'i düzeltir. KÖR ezme YOK — yalnızca şu üç koşul birden sağlanırsa:
      1) target_col dolu (manuel seçim)
      2) mevcut intent model-eğitme (classification/regression)
      3) hedeften çıkarılan tip mevcut intent'ten farklı
    Aksi halde None döner (intent'e dokunulmaz).
    """
    target_col = state.get("target_col")
    intent = state.get("intent", "")

    if not target_col or intent not in _TASK_OVERRIDE_INTENTS:
        return None
    if not data_path:
        return None

    try:
        import pandas as pd
        series = pd.read_parquet(data_path, columns=[target_col])[target_col]
    except Exception as e:
        print(f"[execution] Gorev tipi cikarimi atlandi (hedef '{target_col}' okunamadi): {e}")
        return None

    inferred = infer_task_type(series)
    if inferred != intent:
        print(f"[execution] Gorev tipi override: hedef '{target_col}' -> "
              f"intent '{intent}' yerine '{inferred}' (manuel hedef verisinden cikarildi)")
        return inferred

    print(f"[execution] Gorev tipi dogrulandi: hedef '{target_col}' -> '{intent}'")
    return None


async def execution_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Node 3: Planı (SQL sorguları dahil) yürütür ve veriyi Pandas DataFrame'e çeker.
    Tüm yürütme mantığı `src.services.data_executor` içerisine taşındı.
    """
    print("\n========================================\n[NODE START] execution_node başlıyor...\n========================================")
    result = await execute_data_operations(state, config)

    if result.get("error"):
        return {
            **state,
            "error": result["error"],
            "current_step": "execution_node",
        }

    data_path = result.get("data_path")

    new_state = {
        **state,
        "data_path": data_path,
        "sql_results": result.get("sql_results"),
        "dataframe": result.get("dataframe"),
        "db_intelligence": result.get("db_intelligence", state.get("db_intelligence")),
        "profiling_report": result.get("profiling_report", state.get("profiling_report")),
        "current_step": "execution_node",
    }

    # Manuel hedef seçiminde görev tipini hedef verisinden kesinleştir (koşullu).
    overridden_intent = _maybe_override_intent(state, data_path)
    if overridden_intent:
        new_state["intent"] = overridden_intent

    return new_state