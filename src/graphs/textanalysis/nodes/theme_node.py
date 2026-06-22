from langchain_core.prompts import ChatPromptTemplate

from src.core.nodes import node
from src.services.text_analysis_engine import (
    cluster_texts,
    build_priority_items,
    theme_sentiment_breakdown,
)
from ..state import ThemeLabel
from ..utils import structured_invoke

from src.core.logging_config import get_logger
logger = get_logger(__name__)

THEME_SYSTEM = """\
Aşağıda bir müşteri geri bildirimi kümesine ait örnek metinler ve bu kümeyi
karakterize eden anahtar terimler var. Bu kümeye Türkçe, kısa ve ayırt edici bir
TEMA adı ver (2-5 kelime, örn. "Teslimat Gecikmesi", "Fiyat Şikayetleri",
"Uygulama Hataları") ve tek cümlelik bir açıklama yaz.

--- Anahtar terimler ---
{top_terms}

--- Örnek metinler ---
{examples}
"""

THEME_PROMPT = ChatPromptTemplate.from_messages([
    ("system", THEME_SYSTEM),
    ("human", "Bu küme için tema adını ve açıklamasını yapısal olarak döndür."),
])


def _label_cluster(top_terms: list[str], example_texts: list[str]) -> ThemeLabel:
    try:
        return structured_invoke(THEME_PROMPT, ThemeLabel, {
            "top_terms": ", ".join(top_terms) or "(belirgin terim yok)",
            "examples": "\n".join(f"- {t}" for t in example_texts[:6]),
        })
    except Exception as e:
        logger.error(f"[TA] tema etiketleme hata: {e}")
        fallback = ", ".join(top_terms[:3]) if top_terms else "Diğer"
        return ThemeLabel(label=fallback.title(), description="")


@node("ta_theme_node")
def ta_theme_node(state) -> dict:
    """
    Hibrit tema çıkarımı: lokal TF-IDF+KMeans kümeleme + LLM ile tema etiketleme.
    Ayrıca şikayet önceliklendirme tablosunu (priority_items) üretir.
    """
    texts: list[str] = state.get("texts") or []
    item_results: list[dict] = state.get("item_results") or []

    clusters = cluster_texts(texts)
    logger.info(f"[TA] theme: {len(clusters)} küme bulundu")

    total = len(texts) or 1
    themes: list[dict] = []
    for c in clusters:
        members = c["member_indices"]
        example_texts = [texts[i] for i in members[:6]]
        label = _label_cluster(c.get("top_terms", []), example_texts)
        themes.append({
            "label": label.label,
            "description": label.description,
            "count": len(members),
            "share": round(len(members) / total, 3),
            "top_terms": c.get("top_terms", []),
            "sentiment_breakdown": theme_sentiment_breakdown(members, item_results),
            "example_texts": example_texts[:3],
        })

    priority_items = build_priority_items(item_results)

    return {"themes": themes, "priority_items": priority_items}
