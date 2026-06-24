import asyncio
import time

from langchain_core.prompts import ChatPromptTemplate

from src.core.nodes import node
from src.core.config import settings
from src.services.text_analysis_engine import (
    distribution,
    canonicalize_aspects,
    rating_sentiment_validation,
    label_sentiment_validation,
)
from ..state import BatchClassification, ItemClassification
from ..utils import structured_invoke, language_directive
from ..cache import get_cached, set_cached

from src.core.logging_config import get_logger
logger = get_logger(__name__)

CLASSIFY_SYSTEM = """\
Sen müşteri geri bildirimi (yorum, şikayet, talep, çağrı dökümü) analiz eden bir uzmansın.
{lang_directive}
Sana numaralı metinlerden oluşan bir parti verilecek. HER metin için tek tek şu alanları üret:

- sentiment: positive | negative | neutral
- sentiment_score: -1.0 (çok olumsuz) ile 1.0 (çok olumlu) arası ondalık
- emotion: baskın duygu (örn. anger, frustration, satisfaction, joy, disappointment, neutral)
- category: complaint (şikayet) | request (talep/istek) | praise (övgü) | question (soru) | other
- urgency: low | medium | high | critical  (eyleme geçilmesi ne kadar acil?)
- summary: Türkçe, tek cümlelik özet
- aspects: metinde geçen konular ve her birine dair duygu, örn.
  [{{"aspect":"teslimat","sentiment":"negative"}}, {{"aspect":"fiyat","sentiment":"neutral"}}]
  (konu yoksa boş liste). Konuları Türkçe ve sade tut (teslimat, fiyat, kalite, müşteri hizmetleri, uygulama...).
{timestamp_instruction}
Kurallar:
- index, aşağıdaki metnin başındaki numarayla AYNI olmalı (0 tabanlı).
- Her metin için MUTLAKA bir kayıt döndür; hiçbirini atlama.
- Öfkeli/tehditkâr/iptal/kayıp niyeti içeren şikayetler için urgency yüksek/kritik olmalı.

--- METİNLER ---
{numbered_texts}
"""

CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CLASSIFY_SYSTEM),
    ("human", "Yukarıdaki her metni sınıflandır ve yapısal çıktı (items listesi) döndür."),
])


def _model_key() -> str:
    # Cache anahtarı gerçek kullanılan modele bağlı: text-analiz modeli değişince
    # eski sınıflandırma cache'i otomatik geçersizleşir.
    return f"{settings.TEXT_ANALYSIS_LLM_PROVIDER}:{settings.TEXT_ANALYSIS_LLM_MODEL}"


_TS_INSTRUCTION = (
    "- timestamp: Eğer metinde AÇIK bir tarih/saat varsa ISO 8601 (YYYY-MM-DD) olarak yaz; "
    "yoksa boş bırak. Tarih UYDURMA.\n"
)


def _to_payload(it: ItemClassification, text: str) -> dict:
    return {
        "text": text,
        "sentiment": (it.sentiment or "neutral").lower(),
        "sentiment_score": float(it.sentiment_score or 0.0),
        "emotion": it.emotion or "neutral",
        "category": (it.category or "other").lower(),
        "urgency": (it.urgency or "low").lower(),
        "summary": it.summary or "",
        "aspects": [
            {"aspect": (a.aspect or "").strip().lower(), "sentiment": (a.sentiment or "neutral").lower()}
            for a in (it.aspects or []) if (a.aspect or "").strip()
        ],
        "timestamp": (it.timestamp or "").strip(),
    }


def _classify_batch_sync(texts: list[str], ts_instruction: str = "", lang_directive: str = "") -> dict[str, dict]:
    """Bir partiyi (benzersiz metinler) sınıflandırır; {text: payload} döndürür."""
    numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))

    # Retry + exponential backoff (rate-limit / geçici LLM hatasına karşı).
    attempts, delay = 3, 2.0
    items: list[ItemClassification] = []
    for attempt in range(attempts):
        try:
            result: BatchClassification = structured_invoke(
                CLASSIFY_PROMPT, BatchClassification,
                {"numbered_texts": numbered, "timestamp_instruction": ts_instruction,
                 "lang_directive": lang_directive},
            )
            items = list(result.items)
            break
        except Exception as e:
            if attempt == attempts - 1:
                logger.error(f"[TA] classify batch kalıcı hata: {e}")
            else:
                time.sleep(delay)
                delay *= 2

    by_local = {it.index: it for it in items if 0 <= it.index < len(texts)}
    out: dict[str, dict] = {}
    for local_i, text in enumerate(texts):
        it = by_local.get(local_i)
        if it is None:
            it = ItemClassification(index=local_i, sentiment="neutral", sentiment_score=0.0,
                                    emotion="neutral", category="other", urgency="low",
                                    summary=text[:120], aspects=[])
        out[text] = _to_payload(it, text)
    return out


def _make_batches(texts: list[str], max_items: int, char_budget: int) -> list[list[str]]:
    """
    Adaptif partileme: metinleri uzunluğa göre paketler. Bir parti en çok
    `char_budget` karakter (≈ /4 token) veya `max_items` öğe taşır. Böylece kısa
    metinler çok, uzun metinler az gruplanır; tek başına bütçeyi aşan uzun bir
    metin kendi partisinde gider (kardeş metinlerce truncate edilmez).
    """
    batches: list[list[str]] = []
    cur: list[str] = []
    cur_chars = 0
    for t in texts:
        tlen = len(t)
        # Mevcut parti doluysa (öğe ya da karakter sınırı) önce onu kapat.
        if cur and (len(cur) >= max_items or cur_chars + tlen > char_budget):
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(t)
        cur_chars += tlen
    if cur:
        batches.append(cur)
    return batches


def _aggregate_aspects(item_results: list[dict]) -> dict:
    agg: dict[str, dict] = {}
    for it in item_results:
        for a in it.get("aspects") or []:
            name = a.get("aspect")
            sent = a.get("sentiment", "neutral")
            if not name:
                continue
            bucket = agg.setdefault(name, {"positive": 0, "negative": 0, "neutral": 0, "total": 0})
            bucket[sent if sent in bucket else "neutral"] += 1
            bucket["total"] += 1
    # En çok bahsedilen ilk 15 aspect (gürültüyü sınırla).
    return dict(sorted(agg.items(), key=lambda kv: kv[1]["total"], reverse=True)[:15])


@node("ta_classify_node")
async def ta_classify_node(state) -> dict:
    """
    Her metin için sentiment + duygu + kategori + aciliyet + aspect üretir.
    Optimizasyonlar:
      - dedupe: aynı metin tek kez sınıflandırılır (çalıştırma içi cache)
      - kalıcı cache: aynı metin+model daha önce işlendiyse LLM'e gidilmez
      - eşzamanlılık: partiler MAX_CONCURRENCY ile paralel, retry/backoff'lu
    """
    texts: list[str] = state.get("texts") or []
    timestamps: list = state.get("timestamps") or []
    ratings: list = state.get("ratings") or []
    labels: list = state.get("labels") or []
    detection: dict = state.get("timestamp_detection") or {}
    lang_directive = language_directive(state.get("detected_language") or "tr")
    model = _model_key()

    # Metin içine gömülü tarih varsa LLM'den çıkarmasını iste; kolon kaynağında
    # veya tarih yoksa boş bırak (token israfı yok).
    ts_instruction = _TS_INSTRUCTION if detection.get("source") == "text" else ""
    # ts_instruction cache anahtarına dahil değil; aynı metin farklı modda
    # çağrılırsa timestamp boş kalabilir — kolon overlay aşağıda bunu telafi eder.

    # 1) Benzersiz metinler.
    unique_texts = list(dict.fromkeys(texts))

    # 2) Kalıcı cache'ten oku.
    cache_kind = "classify_ts" if ts_instruction else "classify"
    cached = get_cached(cache_kind, model, unique_texts)
    misses = [t for t in unique_texts if t not in cached]

    # 3) Cache miss'lerini ADAPTİF partilere böl (uzunluğa göre), eşzamanlı sınıflandır.
    batches = _make_batches(misses, settings.TA_BATCH_MAX_ITEMS, settings.TA_BATCH_CHAR_BUDGET)
    logger.info(f"[TA] classify: {len(texts)} metin, {len(unique_texts)} benzersiz, "
                f"{len(cached)} cache-hit, {len(misses)} metin → {len(batches)} LLM çağrısı "
                f"(adaptif: ≤{settings.TA_BATCH_MAX_ITEMS} öğe / ≤{settings.TA_BATCH_CHAR_BUDGET} karakter, "
                f"eşzamanlı={settings.TA_MAX_CONCURRENCY}, ts={'on' if ts_instruction else 'off'})")

    results_by_text: dict[str, dict] = dict(cached)
    if batches:
        sem = asyncio.Semaphore(settings.TA_MAX_CONCURRENCY)

        async def run_batch(batch: list[str]) -> dict[str, dict]:
            async with sem:
                return await asyncio.to_thread(_classify_batch_sync, batch, ts_instruction, lang_directive)

        for batch_result in await asyncio.gather(*[run_batch(b) for b in batches]):
            results_by_text.update(batch_result)

        # 4) Yeni sonuçları cache'e yaz.
        set_cached(cache_kind, model, {t: results_by_text[t] for t in misses if t in results_by_text})

    # 5) Orijinal sıraya göre item_results kur (dedupe'u geri eşle).
    item_results: list[dict] = []
    for i, text in enumerate(texts):
        payload = dict(results_by_text.get(text) or {
            "text": text, "sentiment": "neutral", "sentiment_score": 0.0,
            "emotion": "neutral", "category": "other", "urgency": "low",
            "summary": "", "aspects": [], "timestamp": "",
        })
        # Kolon timestamp'i varsa metinden çıkarılana göre önceliklidir.
        col_ts = timestamps[i] if i < len(timestamps) else None
        if col_ts:
            payload["timestamp"] = col_ts
        item_results.append({"index": i, **payload})

    aspect_dist = canonicalize_aspects(_aggregate_aspects(item_results))
    rating_validation = rating_sentiment_validation(item_results, ratings) if any(r is not None for r in ratings) else {}
    label_validation = label_sentiment_validation(item_results, labels) if any(l is not None for l in labels) else {}

    return {
        "item_results": item_results,
        "sentiment_distribution": distribution([r["sentiment"] for r in item_results]),
        "category_distribution": distribution([r["category"] for r in item_results]),
        "urgency_distribution": distribution([r["urgency"] for r in item_results]),
        "emotion_distribution": distribution([r.get("emotion") for r in item_results]),
        "aspect_distribution": aspect_dist,
        "rating_validation": rating_validation,
        "label_validation": label_validation,
    }
