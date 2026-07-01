from langchain_core.runnables import RunnableConfig

from src.services.matplotlib_renderer import render_matplotlib

from ..state import VizState

from src.core.logging_config import get_logger
logger = get_logger(__name__)


async def viz_render_node(state: VizState, config: RunnableConfig) -> VizState:
    """
    viz_spec + ham satırları matplotlib_renderer servisine verir → base64 PNG
    (ve basit tipler için recharts payload) üretir. Node ince adaptör; tüm çizim
    mantığı serviste (CLAUDE.md node/service ayrımı).
    """
    logger.info("\n[VIZ] viz_render_node başlıyor...")

    spec    = state.get("viz_spec") or {}
    columns = state.get("sql_query_columns") or []
    rows    = state.get("ttd_raw_rows") or []

    if not spec or not columns or not rows:
        logger.warning("[VIZ] render atlanıyor — spec/kolon/satır eksik")
        return {
            **state,
            "viz_image_b64":    None,
            "viz_chart":        None,
            "viz_render_error": "No data to render",
            "current_step":     "viz_render_node",
        }  # type: ignore[return-value]

    result = render_matplotlib(spec, rows, columns)

    if result.get("error"):
        logger.error(f"[VIZ] render error: {result['error']}")

    updated = dict(state)
    updated["viz_image_b64"]    = result.get("image_b64")
    updated["viz_chart"]        = result.get("chart")
    updated["viz_render_error"] = result.get("error")
    updated["current_step"]     = "viz_render_node"
    updated.pop("ttd_raw_rows", None)  # type: ignore[call-overload]
    return updated  # type: ignore[return-value]
