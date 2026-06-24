from langchain_core.prompts import ChatPromptTemplate

from src.core.nodes import node
from src.services.text_analysis_engine import (
    cluster_texts,
    build_priority_items,
    theme_sentiment_breakdown,
    theme_time_matrix,
)
from ..state import BatchThemeLabel
from ..utils import structured_invoke, language_directive

from src.core.logging_config import get_logger
logger = get_logger(__name__)

THEME_SYSTEM = """\
Aşağıda birden çok müşteri geri bildirimi KÜMESİ var. Her küme için örnek metinler
ve kümeyi karakterize eden anahtar terimler verildi. HER küme için kısa ve ayırt
edici bir TEMA adı (2-5 kelime, örn. "Teslimat Gecikmesi", "Fiyat Şikayetleri",
"Uygulama Hataları") ve tek cümlelik bir açıklama üret.
{lang_directive}
- cluster_id, aşağıdaki kümeyle AYNI olmalı.
- Her küme için MUTLAKA bir kayıt döndür; hiçbirini atlama.

--- KÜMELER ---
{clusters_block}
"""

THEME_PROMPT = ChatPromptTemplate.from_messages([
    ("system", THEME_SYSTEM),
    ("human", "Her küme için tema adı ve açıklamasını yapısal (clusters listesi) döndür."),
])


def _label_clusters(clusters: list[dict], lang_directive: str) -> dict[int, dict]:
    """Tüm kümeleri TEK LLM çağrısında etiketler. {cluster_id: {label, description}}."""
    if not clusters:
        return {}
    blocks = []
    for idx, c in enumerate(clusters):
        terms = ", ".join(c.get("top_terms", [])) or "(belirgin terim yok)"
        examples = "\n".join(f"    - {t}" for t in c.get("_examples", [])[:5])
        blocks.append(f"[Küme {idx}]\n  Anahtar terimler: {terms}\n  Örnekler:\n{examples}")
    clusters_block = "\n\n".join(blocks)

    try:
        result: BatchThemeLabel = structured_invoke(THEME_PROMPT, BatchThemeLabel, {
            "clusters_block": clusters_block,
            "lang_directive": lang_directive,
        })
        out: dict[int, dict] = {}
        for cl in result.clusters:
            if 0 <= cl.cluster_id < len(clusters):
                out[cl.cluster_id] = {"label": cl.label, "description": cl.description}
        return out
    except Exception as e:
        logger.error(f"[TA] batch tema etiketleme hata: {e}")
        return {}


@node("ta_theme_node")
def ta_theme_node(state) -> dict:
    """
    Hibrit tema çıkarımı: lokal embedding/TF-IDF + (HDBSCAN/KMeans) kümeleme +
    LLM ile TEK çağrıda toplu tema etiketleme. Ayrıca şikayet önceliklendirme
    tablosunu (priority_items) ve tema × zaman matrisini (theme_trends) üretir.
    """
    texts: list[str] = state.get("texts") or []
    item_results: list[dict] = state.get("item_results") or []
    lang_directive = language_directive(state.get("detected_language") or "tr")

    clusters = cluster_texts(texts)
    logger.info(f"[TA] theme: {len(clusters)} küme bulundu")

    # Etiketleme için her kümeye örnek metinleri iliştir, tek çağrıda etiketle.
    for c in clusters:
        members = c["member_indices"]
        c["_examples"] = [texts[i] for i in members[:6]]
    labels = _label_clusters(clusters, lang_directive)

    total = len(texts) or 1
    themes: list[dict] = []
    themes_with_members: list[dict] = []
    for idx, c in enumerate(clusters):
        members = c["member_indices"]
        example_texts = c.get("_examples", [])
        lbl = labels.get(idx)
        if lbl:
            label, description = lbl["label"], lbl["description"]
        else:
            # Etiketleme başarısızsa top_terms'ten geri-dönüş ad üret.
            top = c.get("top_terms", [])
            label = (", ".join(top[:3]).title() if top else "Diğer")
            description = ""
        theme = {
            "label": label,
            "description": description,
            "count": len(members),
            "share": round(len(members) / total, 3),
            "top_terms": c.get("top_terms", []),
            "sentiment_breakdown": theme_sentiment_breakdown(members, item_results),
            "example_texts": example_texts[:3],
        }
        themes.append(theme)
        themes_with_members.append({**theme, "member_indices": members})

    priority_items = build_priority_items(item_results)
    theme_trends = theme_time_matrix(themes_with_members, item_results)
    logger.info(f"[TA] theme: {len(theme_trends)} tema için zaman trendi")

    return {"themes": themes, "priority_items": priority_items, "theme_trends": theme_trends}
