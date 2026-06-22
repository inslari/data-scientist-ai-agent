import re

from src.core.nodes import node

from src.core.logging_config import get_logger
logger = get_logger(__name__)


def _clean(t) -> str:
    if not isinstance(t, str):
        return ""
    return re.sub(r"\s+", " ", t).strip()


@node("ta_preprocess_node")
def ta_preprocess_node(state) -> dict:
    """
    Ham metinleri temizler; varsa hizalı zaman damgalarını korur. Boş/çok kısa
    metinleri (ve onlarla hizalı timestamp'leri) birlikte atar.
    """
    raw = state.get("raw_texts") or []
    raw_ts = state.get("raw_timestamps") or []

    texts: list[str] = []
    timestamps: list = []
    for i, item in enumerate(raw):
        s = _clean(item)
        if len(s) < 2:
            continue
        texts.append(s)
        ts = raw_ts[i] if i < len(raw_ts) else None
        timestamps.append(ts if (isinstance(ts, str) and ts.strip()) else None)

    logger.info(f"[TA] preprocess: {len(raw)} ham → {len(texts)} geçerli metin, "
                f"{sum(1 for t in timestamps if t)} kolon-timestamp")

    if not texts:
        return {
            "texts": [],
            "timestamps": [],
            "error": "Analiz edilecek geçerli metin bulunamadı (boş giriş).",
            "error_category": "data",
        }
    return {"texts": texts, "timestamps": timestamps}
