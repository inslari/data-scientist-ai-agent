import ast
from langchain_core.runnables import RunnableConfig

from api.core.database import SessionLocal
from api.schemas.data_group import (
    run_data_group_query_logic as run_data_group_query,
    QueryRequest as DGQueryRequest,
)

from ..state import VizState

from src.core.logging_config import get_logger
logger = get_logger(__name__)


def _parse_query_result(raw_txt: str) -> tuple[list[str], list]:
    """
    run_data_group_query_logic'in metin çıktısını (Columns: ... / Results: ...)
    kolon listesi + ham tuple satırlarına çevirir. ttd_sql_node ile aynı ayrıştırma.
    """
    lines = raw_txt.strip().splitlines()
    columns: list[str] = []
    rows: list = []
    for i, line in enumerate(lines):
        if line.startswith("Columns:"):
            columns = [c.strip() for c in line.replace("Columns:", "").split(",")]
        elif line.startswith("Results:") or line.startswith("("):
            data_lines = lines[i + 1:] if line.startswith("Results:") else lines[i:]
            for dl in data_lines:
                dl = dl.strip()
                if not dl or dl.startswith("..."):
                    continue
                if dl.startswith("(") and dl.endswith(")"):
                    try:
                        rows.append(ast.literal_eval(dl))
                    except Exception:
                        pass
            break
    return columns, rows


async def viz_sql_node(state: VizState, config: RunnableConfig) -> VizState:
    """
    viz_plan_node'un ürettiği SQL'i (state['sql_query_raw']) MCP/data group üzerinde
    çalıştırır. Üretim yapmaz — sadece yürütür ve ham satırları toplar. Hata olursa
    sql_error + retry_count artırır (graph plan_node'a geri döner).
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
        db = SessionLocal()
        try:
            res = await run_data_group_query(data_group_id, DGQueryRequest(query=sql), db, user_id)
            if res.get("status") == "error":
                raise Exception(res.get("result"))
            raw_txt = str(res.get("result", "")) if res.get("status") == "success" else ""
            columns, rows = _parse_query_result(raw_txt)
        finally:
            db.close()
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
