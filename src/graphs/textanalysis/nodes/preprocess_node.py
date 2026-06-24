import re

from src.core.nodes import node
from src.core.config import settings
from src.services.text_analysis_engine import (
    detect_language,
    redact_pii,
    collapse_near_duplicates,
    _norm_for_dedup,
)

from src.core.logging_config import get_logger
logger = get_logger(__name__)


def _clean(t) -> str:
    if not isinstance(t, str):
        return ""
    return re.sub(r"\s+", " ", t).strip()


@node("ta_preprocess_node")
def ta_preprocess_node(state) -> dict:
    """
    Ham metinleri temizler; varsa hizalı zaman damgalarını/puanları korur. Boş/çok
    kısa metinleri (ve onlarla hizalı timestamp/rating'leri) birlikte atar. Ardından
    (ayarlıysa) PII maskeler, yakın-tekrarları birleştirir ve dili tespit eder.
    """
    raw = state.get("raw_texts") or []
    raw_ts = state.get("raw_timestamps") or []
    raw_ratings = state.get("raw_ratings") or []
    raw_labels = state.get("raw_labels") or []

    # 1) Temizle + boş/kısa eleme; timestamp, rating ve etiketi hizalı taşı.
    texts: list[str] = []
    timestamps: list = []
    ratings: list = []
    labels: list = []
    for i, item in enumerate(raw):
        s = _clean(item)
        if len(s) < 2:
            continue
        texts.append(s)
        ts = raw_ts[i] if i < len(raw_ts) else None
        timestamps.append(ts if (isinstance(ts, str) and ts.strip()) else None)
        ratings.append(raw_ratings[i] if i < len(raw_ratings) else None)
        labels.append(raw_labels[i] if i < len(raw_labels) else None)

    # 2) PII maskeleme (LLM'e gönderim/saklama öncesi).
    if settings.TA_REDACT_PII and texts:
        texts = [redact_pii(t) for t in texts]

    # 3) Yakın-tekrar (kopya/spam) birleştirme — sayımları şişirmesin. Timestamp/
    #    rating hizasını korumak için temsilcilerin orijinal indeksini izleriz.
    duplicate_groups: dict = {}
    if settings.TA_DEDUP_NEAR_DUPLICATES and len(texts) > 1:
        kept_texts, duplicate_groups = collapse_near_duplicates(texts)
        if len(kept_texts) < len(texts):
            kept_norms: dict[str, int] = {}
            first_idx_by_norm: dict[str, int] = {}
            for i, t in enumerate(texts):
                norm = _norm_for_dedup(t)
                if norm and norm not in first_idx_by_norm:
                    first_idx_by_norm[norm] = i
            keep_idx = sorted(first_idx_by_norm.values())
            texts = [texts[i] for i in keep_idx]
            timestamps = [timestamps[i] for i in keep_idx]
            ratings = [ratings[i] for i in keep_idx]
            labels = [labels[i] for i in keep_idx]
            logger.info(f"[TA] near-dup: {len(duplicate_groups)} grupta kopya birleştirildi")

    # 4) Dil tespiti (kullanıcı 'auto' verdiyse).
    lang = (state.get("language") or "auto").lower()
    detected = lang if lang in ("tr", "en") else detect_language(texts)

    logger.info(f"[TA] preprocess: {len(raw)} ham → {len(texts)} geçerli metin, "
                f"{sum(1 for t in timestamps if t)} kolon-timestamp, "
                f"{sum(1 for r in ratings if r is not None)} rating, dil={detected}")

    if not texts:
        return {
            "texts": [],
            "timestamps": [],
            "ratings": [],
            "labels": [],
            "error": "Analiz edilecek geçerli metin bulunamadı (boş giriş).",
            "error_category": "data",
        }
    return {
        "texts": texts,
        "timestamps": timestamps,
        "ratings": ratings,
        "labels": labels,
        "detected_language": detected,
        "duplicate_groups": duplicate_groups,
    }
