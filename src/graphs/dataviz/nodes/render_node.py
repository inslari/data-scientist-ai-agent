from langchain_core.runnables import RunnableConfig

from src.services.matplotlib_renderer import render_matplotlib
from src.services.viz_data import to_storable

from ..state import VizState

from src.core.logging_config import get_logger
logger = get_logger(__name__)

async def viz_render_node(state: VizState, config: RunnableConfig) -> VizState:
    """
    viz_spec + ham satırlar + tema → matplotlib_renderer servisine verilir: base64 PNG
    (indirme için) + frontend'in çizdiği payload. Node ince adaptör; tüm çizim mantığı
    serviste (CLAUDE.md node/service ayrımı).

    `viz_chart` payload'ı tema-bağımsızdır (etiketler + değerler + biçim meta'sı): renkler
    istemcide uygulandığı için tema değiştirmek pipeline'ı yeniden çalıştırmayı gerektirmez.
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
            "viz_data":         None,
            "viz_render_error": "No data to render",
            "current_step":     "viz_render_node",
        }  # type: ignore[return-value]

    result = render_matplotlib(
        spec, rows, columns,
        theme=state.get("viz_theme"),
        locale=state.get("locale") or "tr",
    )

    if result.get("error"):
        logger.error(f"[VIZ] render error: {result['error']}")

    updated = dict(state)
    updated["viz_image_b64"]    = result.get("image_b64")
    updated["viz_chart"]        = result.get("chart")
    # Ham sonuç saklanır: tablo görünümü + spec/tema değişiminde LLM'siz yeniden render
    updated["viz_data"]         = to_storable(columns, rows) if result.get("image_b64") else None
    updated["viz_render_error"] = result.get("error")
    updated["current_step"]     = "viz_render_node"
    updated.pop("ttd_raw_rows", None)  # type: ignore[call-overload]
    return updated  # type: ignore[return-value]
