import json

from langchain_core.prompts import ChatPromptTemplate

from src.core.nodes import node
from ..utils import _get_llm, language_directive, log_token_usage

from src.core.logging_config import get_logger
logger = get_logger(__name__)

# Anlatı (narrative) bölümleri LLM tarafından yazılır; SAYISAL tablolar koddan
# deterministik basılır (halüsinasyon riski düşer). LLM'e yalnızca yorum/öneri
# yazması söylenir, tablo üretmesi istenmez.
REPORT_SYSTEM = """\
Sen bir müşteri deneyimi analistisin. Aşağıdaki yazılı geri bildirim analizi
özet verilerinden, yöneticiler için kısa ve net bir ANLATI yaz (markdown).
{lang_directive}

YALNIZCA şu üç bölümü yaz (sayısal tabloları SEN üretme; onlar ayrıca eklenecek):
## Yönetici Özeti
(2-4 cümle: genel duygu durumu ve en kritik bulgu)

## İçgörüler
(verideki en önemli 3-5 örüntü/çıkarım — temalar, aspect'ler, trend yönü, varsa
puan-duygu tutarsızlığı üzerine madde madde)

## Öneriler
(eyleme dönük 3-5 somut öneri)

Sayıları uydurma; yalnızca verilen verilere dayan. Kısa ve okunur tut.

--- ANALİZ ÖZETİ (JSON) ---
{analysis_json}
"""

REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", REPORT_SYSTEM),
    ("human", "Yukarıdaki verilerden yalnızca anlatı bölümlerini (özet/içgörü/öneri) yaz."),
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
        "emotion_distribution": state.get("emotion_distribution") or {},
        "aspect_distribution": state.get("aspect_distribution") or {},
        "rating_validation": state.get("rating_validation") or {},
        "label_validation": state.get("label_validation") or {},
        "timestamp_detection": state.get("timestamp_detection") or {},
        "time_trend": state.get("time_trend") or [],
        "theme_trends": state.get("theme_trends") or [],
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


# ─────────────────────────────────────────────────────────────────────────────
# Deterministik tablo bölümleri (koddan basılır — sayılar garanti doğru)
# ─────────────────────────────────────────────────────────────────────────────

def _dist_lines(dist: dict) -> list[str]:
    total = sum(dist.values()) or 1
    return [f"- {k}: {v} (%{round(100 * v / total)})"
            for k, v in sorted(dist.items(), key=lambda kv: kv[1], reverse=True)]


def _render_deterministic_sections(p: dict) -> str:
    L: list[str] = []

    L.append("## Duygu Dağılımı")
    L += _dist_lines(p.get("sentiment_distribution") or {}) or ["- (veri yok)"]

    if p.get("emotion_distribution"):
        L.append("\n## Baskın Duygular")
        L += _dist_lines(p["emotion_distribution"])

    if p.get("category_distribution"):
        L.append("\n## Kategori Dağılımı")
        L += _dist_lines(p["category_distribution"])

    asp = p.get("aspect_distribution") or {}
    if asp:
        L.append("\n## Konu (Aspect) Bazlı Duygu")
        L.append("| Konu | Olumlu | Olumsuz | Nötr | Toplam |")
        L.append("|------|-------:|--------:|-----:|-------:|")
        for name, b in asp.items():
            L.append(f"| {name} | {b.get('positive',0)} | {b.get('negative',0)} "
                     f"| {b.get('neutral',0)} | {b.get('total',0)} |")

    rv = p.get("rating_validation") or {}
    if rv:
        agr = rv.get("agreement")
        agr_s = f"%{round(100 * agr)}" if agr is not None else "—"
        L.append("\n## Puan ↔ Duygu Tutarlılığı")
        L.append(f"- Puanlı kayıt: {rv.get('n')} | Ortalama puan: {rv.get('avg_rating')} "
                 f"| Tutarlılık: {agr_s} | Uyumsuzluk: {rv.get('mismatches')}")

    lv = p.get("label_validation") or {}
    if lv:
        acc = lv.get("accuracy")
        acc_s = f"%{round(100 * acc)}" if acc is not None else "—"
        classes = ["positive", "negative", "neutral"]
        conf = lv.get("confusion") or {}
        pc = lv.get("per_class") or {}
        L.append("\n## Model Performansı (Etiketli Karşılaştırma)")
        L.append(f"- Etiketli kayıt: **{lv.get('n')}** | Doğruluk (accuracy): **{acc_s}** "
                 f"| Doğru: {lv.get('correct')} | Hatalı: {lv.get('mismatches')}")
        L.append("\n**Confusion (satır=gerçek etiket, sütun=LLM tahmini):**")
        L.append("| Gerçek ＼ Tahmin | positive | negative | neutral |")
        L.append("|---|---:|---:|---:|")
        for t in classes:
            row = conf.get(t, {})
            L.append(f"| {t} | {row.get('positive', 0)} | {row.get('negative', 0)} | {row.get('neutral', 0)} |")
        L.append("\n**Sınıf bazlı:**")
        L.append("| Sınıf | Precision | Recall | F1 | Destek |")
        L.append("|---|---:|---:|---:|---:|")
        for c in classes:
            m = pc.get(c, {})
            L.append(f"| {c} | {m.get('precision', '—')} | {m.get('recall', '—')} "
                     f"| {m.get('f1', '—')} | {m.get('support', 0)} |")

    themes = p.get("themes") or []
    if themes:
        L.append("\n## Öne Çıkan Temalar")
        L.append("| Tema | Adet | Pay | Baskın Duygu |")
        L.append("|------|-----:|----:|--------------|")
        for t in themes:
            sb = t.get("sentiment_breakdown") or {}
            dom = max(sb, key=sb.get) if sb else "—"
            L.append(f"| {t.get('label')} | {t.get('count')} "
                     f"| %{round(100 * (t.get('share') or 0))} | {dom} |")

    tt = p.get("theme_trends") or []
    if tt:
        L.append("\n## Tema Zaman Eğilimi")
        for t in tt:
            L.append(f"- **{t.get('label')}**: {t.get('direction')} (eğim {t.get('slope')})")

    trend = p.get("time_trend") or []
    if trend:
        L.append("\n## Zaman Trendi")
        L.append("| Dönem | Adet | Ort. Duygu | Δ | Ani Artış |")
        L.append("|-------|-----:|-----------:|---:|:---------:|")
        for d in trend:
            spike = "⚠️" if d.get("spike") else ""
            L.append(f"| {d.get('period')} | {d.get('count')} | {d.get('avg_sentiment')} "
                     f"| {d.get('delta_sentiment', 0)} | {spike} |")
    else:
        L.append("\n## Zaman Trendi")
        L.append("Veride zaman damgası bulunmadığı için zaman trendi üretilemedi.")

    pri = p.get("top_priority_items") or []
    if pri:
        L.append("\n## Acil Şikayet ve Talepler")
        for it in pri:
            L.append(f"- [{it.get('urgency')}] {it.get('summary')} "
                     f"({it.get('category')}, {it.get('sentiment')})")

    return "\n".join(L)


@node("ta_report_node", runs_on_error=True)
def ta_report_node(state) -> dict:
    """
    Hibrit rapor: ANLATI (özet/içgörü/öneri) LLM'den, SAYISAL tablolar deterministik
    koddan. Upstream hata olsa da çalışır (hatayı düzgün rapora dönüştürür).
    """
    if state.get("error"):
        return {
            "report": f"## Analiz Tamamlanamadı\n\nBir hata oluştu:\n\n```\n{state.get('error')}\n```",
        }

    payload = _compact_payload(state)
    facts = _render_deterministic_sections(payload)

    narrative = ""
    try:
        llm = _get_llm(temperature=0.2)
        msg = (REPORT_PROMPT | llm).invoke({
            "analysis_json": json.dumps(payload, ensure_ascii=False, indent=2),
            "lang_directive": language_directive(state.get("detected_language") or "tr"),
        })
        log_token_usage(msg, "report")
        narrative = msg.content if hasattr(msg, "content") else str(msg)
    except Exception as e:
        logger.error(f"[TA] rapor anlatı üretimi hata: {e}")
        narrative = _fallback_narrative(payload)

    title = payload.get("title") or "Metin Analizi Raporu"
    report = f"# {title}\n\n{narrative.strip()}\n\n---\n\n{facts}"
    return {"report": report}


def _fallback_narrative(payload: dict) -> str:
    sd = payload.get("sentiment_distribution") or {}
    dom = max(sd, key=sd.get) if sd else "belirsiz"
    return (f"## Yönetici Özeti\n\nToplam {payload.get('total_texts', 0)} geri bildirim "
            f"analiz edildi; baskın duygu: **{dom}**.\n\n"
            f"## İçgörüler\n\n(Otomatik anlatı üretilemedi; sayısal bulgular aşağıdadır.)\n\n"
            f"## Öneriler\n\n(Acil maddeler ve temalar aşağıdaki tablolarda listelenmiştir.)")
