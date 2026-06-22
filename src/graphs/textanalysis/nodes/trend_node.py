from src.core.nodes import node
from src.services.text_analysis_engine import build_time_trend

from src.core.logging_config import get_logger
logger = get_logger(__name__)


@node("ta_trend_node")
def ta_trend_node(state) -> dict:
    """
    Zaman-bazlı trend (deterministik). item_results içindeki timestamp'lerden
    (kolon veya metinden çıkarılan) duygu trendini kurar. Tarih yoksa boş liste.
    """
    item_results = state.get("item_results") or []
    trend = build_time_trend(item_results)
    logger.info(f"[TA] trend: {len(trend)} dönem")
    return {"time_trend": trend}
