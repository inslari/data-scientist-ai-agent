from langchain_core.prompts import ChatPromptTemplate

from src.core.nodes import node
from ..state import TimestampDetection
from ..utils import structured_invoke

from src.core.logging_config import get_logger
logger = get_logger(__name__)

DETECT_SYSTEM = """\
Sana bir müşteri geri bildirimi kümesinden ÖRNEK satırlar verilecek. Görevin:
bu metinlerde AÇIK bir tarih/saat (zaman damgası) geçip geçmediğini tespit etmek.

- has_timestamp: Satırların önemli kısmında tarih/saat varsa true, yoksa false.
- example: Varsa, veriden BİREBİR bir tarih örneği (örn. "2026-05-01", "01.05.2026 14:30").
- confidence: 0..1 arası güven.
- note: Türkçe kısa not (format ve nerede göründüğü).

Sadece metinde GERÇEKTEN görünen tarihleri say; tahmin/uydurma yapma.

--- ÖRNEK SATIRLAR ---
{sample}
"""

DETECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", DETECT_SYSTEM),
    ("human", "Bu veride zaman damgası var mı? Yapısal çıktı döndür."),
])


@node("ta_detect_timestamp_node")
def ta_detect_timestamp_node(state) -> dict:
    """
    Agent ile zaman damgası tespiti. İki kaynak:
      1) CSV gibi yapılandırılmış girdiden gelen kolon timestamp'leri (varsa LLM'e
         gerek yok → source="column").
      2) Metin içine gömülü tarihler → LLM örnek satırlara bakıp karar verir
         (source="text"). Yoksa source="none".
    """
    texts: list[str] = state.get("texts") or []
    timestamps: list = state.get("timestamps") or []

    # 1) Kolon timestamp'leri yeterince doluysa LLM'i atla.
    col_filled = sum(1 for t in timestamps if t)
    if texts and col_filled >= max(2, int(0.3 * len(texts))):
        detection = {
            "has_timestamp": True,
            "source": "column",
            "confidence": 1.0,
            "example": next((t for t in timestamps if t), ""),
            "note": f"{col_filled}/{len(texts)} kayıtta tarih kolonu mevcut.",
        }
        logger.info(f"[TA] timestamp: kolon kaynağı ({col_filled}/{len(texts)})")
        return {"timestamp_detection": detection}

    # 2) Metin içi tespit (örneklem üzerinden).
    sample = "\n".join(f"- {t}" for t in texts[:12]) or "(metin yok)"
    try:
        result: TimestampDetection = structured_invoke(DETECT_PROMPT, TimestampDetection, {"sample": sample})
        detection = {
            "has_timestamp": bool(result.has_timestamp),
            "source": "text" if result.has_timestamp else "none",
            "confidence": float(result.confidence or 0.0),
            "example": result.example or "",
            "note": result.note or "",
        }
    except Exception as e:
        logger.error(f"[TA] timestamp tespiti hata: {e}")
        detection = {"has_timestamp": False, "source": "none", "confidence": 0.0, "example": "", "note": ""}

    logger.info(f"[TA] timestamp tespiti: {detection['has_timestamp']} (source={detection['source']})")
    return {"timestamp_detection": detection}
