from langchain_core.runnables import RunnableConfig

from src.services.viz_data import execute_viz_sql

from ..state import VizState

from src.core.logging_config import get_logger
logger = get_logger(__name__)


async def viz_sql_node(state: VizState, config: RunnableConfig) -> VizState:
    """
    viz_plan_node'un ürettiği SQL'i (state['sql_query_raw']) MCP/data group üzerinde
    çalıştırır. Üretim yapmaz — yürütmeyi viz_data servisine devreder ve sonucu state'e
    yazar. Hata olursa sql_error + retry_count artırır (graph plan_node'a geri döner).
    """
    logger.info("\n[VIZ] viz_sql_node başlıyor...")

    cfg = config.get("configurable", {})
    data_group_id = cfg.get("data_group_id") or state.get("data_group_id")
    user_id = cfg.get("user_id")
    sql = (state.get("sql_query_raw") or "").strip()

    if not data_group_id:
        return {**state, "error": "data_group_id missing", "current_step": "viz_sql_node"}  # type: ignore[return-value]
    if not sql:
        retry_count = state.get("sql_retry_count") or 0
        return {
            **state,
            "sql_error": "No SQL produced by planner",
            "sql_retry_count": retry_count + 1,
            "current_step": "viz_sql_node",
        }  # type: ignore[return-value]

    try:
        columns, rows = await execute_viz_sql(data_group_id, sql, user_id)
    except Exception as e:
        retry_count = state.get("sql_retry_count") or 0
        logger.error(f"[VIZ] SQL execution failed (retry {retry_count}): {e}")
        return {
            **state,
            "sql_error": str(e),
            "sql_retry_count": retry_count + 1,
            "sql_query_raw": sql,
            "current_step": "viz_sql_node",
        }  # type: ignore[return-value]

    results_as_dicts = [dict(zip(columns, row)) for row in rows]
    logger.info(f"[VIZ] SQL ok: {len(rows)} satır, kolonlar={columns}")

    return {
        **state,
        "sql_error":           None,
        "validation_fix_hint": None,
        "sql_query_raw":       sql,
        "sql_query_columns":   columns,
        "sql_query_results":   results_as_dicts,
        "sql_query_warning":   "",
        "ttd_raw_rows":        rows,
        "current_step":        "viz_sql_node",
    }  # type: ignore[return-value]
