import json

from langchain_core.prompts import ChatPromptTemplate

from src.core.nodes import node
from ..utils import _get_llm

from src.core.logging_config import get_logger
logger = get_logger(__name__)

REPORT_SYSTEM = """\
Sen bir müşteri deneyimi analistisin. Aşağıdaki yazılı geri bildirim analizi
sonuçlarından, yöneticiler için Türkçe, markdown formatında net bir rapor yaz.

Raporda şu bölümler olsun:
## Yönetici Özeti
(2-4 cümle: genel duygu durumu, en kritik bulgu)

## Duygu Dağılımı
(olumlu/olumsuz/nötr oranları ve kısa yorum)

## Konu (Aspect) Bazlı Duygu
(aspect_distribution'dan en çok bahsedilen konular ve her birinde olumlu/olumsuz dengesi)

## Öne Çıkan Temalar
(her tema: ad, payı/adedi, baskın duygu, kısa içgörü — madde madde)

## Acil Şikayet ve Talepler
(en kritik {n_priority} maddeyi kısa özetlerle listele; aciliyetini belirt)

## Zaman Trendi
(time_trend doluysa: dönemler arası duygu nasıl değişmiş — iyileşme/kötüleşme;
boşsa "Veride zaman damgası bulunmadığı için zaman trendi üretilemedi" yaz.)

## Öneriler ve Trendler
(eyleme dönük 3-5 öneri; tekrar eden örüntüler/trendler)

Sayıları uydurma; yalnızca verilen verilere dayan. Kısa ve okunur tut.

--- ANALİZ VERİSİ (JSON) ---
{analysis_json}
"""

REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", REPORT_SYSTEM),
    ("human", "Yukarıdaki verilerden markdown raporu üret."),
])


def _compact_payload(state) -> dict:
    """LLM'e gönderilecek kompakt özet (büyük ham metin listesini gönderme)."""
    priority = state.get("priority_items") or []
    return {
        "title": state.get("title"),
        "total_texts": len(state.get("texts") or []),
        "sentiment_distribution": state.get("sentiment_distribution") or {},
        "category_distribution": state.get("category_distribution") or {},
        "urgency_distribution": state.get("urgency_distribution") or {},
        "aspect_distribution": state.get("aspect_distribution") or {},
        "timestamp_detection": state.get("timestamp_detection") or {},
        "time_trend": state.get("time_trend") or [],
        "themes": [
            {k: v for k, v in t.items() if k != "example_texts"}
            for t in (state.get("themes") or [])
        ],
        "top_priority_items": [
            {
                "summary": p.get("summary") or p.get("text", "")[:160],
                "category": p.get("category"),
                "urgency": p.get("urgency"),
                "sentiment": p.get("sentiment"),
            }
            for p in priority[:10]
        ],
    }


@node("ta_report_node", runs_on_error=True)
def ta_report_node(state) -> dict:
    """
    Yönetici özeti + trend raporu (markdown). Upstream hata olsa da çalışır:
    hatayı düzgün bir rapora dönüştürür (pipeline temiz sonlanır).
    """
    if state.get("error"):
        return {
            "report": f"## Analiz Tamamlanamadı\n\nBir hata oluştu:\n\n```\n{state.get('error')}\n```",
        }

    payload = _compact_payload(state)
    try:
        llm = _get_llm(temperature=0.2)
        msg = (REPORT_PROMPT | llm).invoke({
            "analysis_json": json.dumps(payload, ensure_ascii=False, indent=2),
            "n_priority": len(payload["top_priority_items"]),
        })
        report = msg.content if hasattr(msg, "content") else str(msg)
    except Exception as e:
        logger.error(f"[TA] rapor üretimi hata: {e}")
        # LLM başarısız olsa bile deterministik bir özet üret (rapor boş kalmasın).
        report = _fallback_report(payload)

    return {"report": report}


def _fallback_report(payload: dict) -> str:
    lines = [f"# {payload.get('title') or 'Metin Analizi Raporu'}", ""]
    lines.append(f"Toplam **{payload['total_texts']}** geri bildirim analiz edildi.\n")
    lines.append("## Duygu Dağılımı")
    for k, v in (payload.get("sentiment_distribution") or {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("\n## Öne Çıkan Temalar")
    for t in payload.get("themes") or []:
        lines.append(f"- **{t.get('label')}** ({t.get('count')} adet): {t.get('description', '')}")
    lines.append("\n## Acil Maddeler")
    for p in payload.get("top_priority_items") or []:
        lines.append(f"- [{p.get('urgency')}] {p.get('summary')}")
    return "\n".join(lines)
