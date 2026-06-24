"""
Yazılı geri bildirim analizi — deterministik yardımcı katman (hibrit motorun
"lokal" yarısı).
=======================================================================

Bu serviste LLM YOK. Hibrit yaklaşımın lokal/deterministik parçası burada:
  - metin temizleme/normalize
  - TF-IDF + KMeans ile tema kümeleme (random_state=RANDOM_SEED → tekrarlanabilir)
  - dağılım/agregasyon hesapları (sentiment/kategori/aciliyet sayımları)

LLM çağrıları (sınıflandırma, tema etiketleme, rapor) node'larda yapılır; bu
servis node/service ayrımındaki "service" tarafıdır (bkz. CLAUDE.md). Fonksiyonlar
dataset-agnostik ve dilden bağımsızdır (TF-IDF token bazlı çalışır; Türkçe dahil).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from src.core.config import settings
from src.core.logging_config import get_logger

logger = get_logger(__name__)


# Önceliklendirme için sıralama ağırlıkları (yüksek = daha acil).
URGENCY_ORDER = {"critical": 3, "high": 2, "medium": 1, "low": 0}

# ─────────────────────────────────────────────────────────────────────────────
# Stopword'ler (TF-IDF tema ipuçlarını temizler). Türkçe + İngilizce birleşik.
# TfidfVectorizer'a doğrudan liste olarak verilir; "bir/ve/için" gibi yüksek
# frekanslı ama ayırt edici olmayan kelimeler top_terms'i kirletmesin.
# ─────────────────────────────────────────────────────────────────────────────
TURKISH_STOPWORDS = {
    "acaba", "ама", "ama", "ancak", "artık", "asla", "aslında", "az", "bana", "bazen",
    "bazı", "belki", "ben", "beni", "benim", "beş", "bile", "bir", "birçok", "biri",
    "birkaç", "birşey", "biz", "bize", "bizi", "bizim", "böyle", "böylece", "bu",
    "buna", "bunda", "bundan", "bunu", "bunun", "burada", "bütün", "çok", "çünkü",
    "da", "daha", "dahi", "de", "defa", "değil", "diğer", "diye", "doğru", "elbette",
    "en", "fakat", "falan", "gibi", "göre", "hâlâ", "hangi", "hani", "hatta", "hem",
    "henüz", "hep", "hepsi", "her", "herhangi", "herkes", "hiç", "hiçbir", "için",
    "iki", "ile", "ilgili", "ise", "işte", "kadar", "karşı", "kendi", "ki", "kim",
    "mı", "mu", "mü", "nasıl", "ne", "neden", "nerede", "nereye", "niçin", "niye",
    "o", "olan", "olarak", "oldu", "olduğu", "olur", "ona", "onlar", "onları",
    "onların", "onu", "onun", "orada", "öyle", "pek", "rağmen", "sana", "sanki",
    "şey", "şu", "şuna", "şunu", "sen", "senin", "siz", "sizin", "sonra", "tüm",
    "vb", "ve", "veya", "ya", "yani", "yine",
}

ENGLISH_STOPWORDS = {
    "a", "about", "after", "again", "all", "also", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "but", "by", "can", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "just", "me", "more", "my", "no", "not", "of", "on",
    "one", "only", "or", "other", "our", "out", "she", "so", "some", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this", "to",
    "too", "up", "very", "was", "we", "were", "what", "when", "which", "who", "why",
    "will", "with", "would", "you", "your",
}

# TfidfVectorizer list bekler (frozenset değil).
STOPWORDS = sorted(TURKISH_STOPWORDS | ENGLISH_STOPWORDS)

# Türkçe'ye özgü karakterler — hafif dil tespiti için sinyal.
_TR_CHARS = set("çğıöşüÇĞİÖŞÜ")

# PII desenleri (LLM'e gönderim/saklama öncesi maskeleme).
_PII_PATTERNS = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[email]"),
    (re.compile(r"\bTR\d{2}[\s]?(?:\d{4}[\s]?){5}\d{2}\b", re.IGNORECASE), "[iban]"),
    # 13-19 haneli kart benzeri sayılar (boşluk/tireli gruplar dahil).
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[card]"),
    # TC kimlik benzeri 11 haneli ardışık sayı.
    (re.compile(r"\b\d{11}\b"), "[id]"),
    # Telefon (TR/uluslararası, +90 / 0 ön ekli, 10-11 hane).
    (re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:0?\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"), "[phone]"),
]


def detect_language(texts: list[str], sample: int = 50) -> str:
    """
    Hafif, bağımlılıksız dil tespiti (tr/en). Türkçe'ye özgü karakterler ve
    stopword örtüşmesine bakar. Belirsizse "tr" döner (ürünün varsayılan dili).
    langdetect gibi ağır bir bağımlılık gerektirmez.
    """
    blob = " ".join((t or "") for t in (texts or [])[:sample]).lower()
    if not blob.strip():
        return "tr"
    if any(ch in _TR_CHARS for ch in blob):
        return "tr"
    tokens = re.findall(r"[a-zçğıöşü]+", blob)
    if not tokens:
        return "tr"
    tr_hits = sum(1 for tok in tokens if tok in TURKISH_STOPWORDS)
    en_hits = sum(1 for tok in tokens if tok in ENGLISH_STOPWORDS)
    return "en" if en_hits > tr_hits else "tr"


def redact_pii(text: str) -> str:
    """Metindeki e-posta/telefon/IBAN/kart/kimlik gibi PII'leri maskeler."""
    if not isinstance(text, str) or not text:
        return text
    out = text
    for pattern, repl in _PII_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def _norm_for_dedup(text: str) -> str:
    """Yakın-tekrar tespiti için agresif normalize: küçük harf, noktalama/boşluk sadeleştirme.

    Not: Türkçe "İ".lower() birleşik nokta (U+0307) üretir; "GECİKTİ" ile "gecikti"
    eşleşsin diye bu birleşik noktayı temizleriz (Türkçe casing tuzağı).
    """
    s = (text or "").lower().replace("̇", "")
    s = re.sub(r"[^\wçğıöşü]+", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def clean_texts(texts: list[str]) -> list[str]:
    """Boşlukları sadeleştirir, boş/çok kısa metinleri atar, sırayı korur."""
    cleaned: list[str] = []
    for t in texts or []:
        if not isinstance(t, str):
            continue
        s = re.sub(r"\s+", " ", t).strip()
        if len(s) >= 2:
            cleaned.append(s)
    return cleaned


def _suggest_cluster_count(n_texts: int) -> int:
    """Veri büyüklüğüne göre makul küme sayısı (2..8)."""
    if n_texts < 6:
        return min(2, n_texts)
    if n_texts < 15:
        return 3
    if n_texts < 40:
        return 4
    if n_texts < 100:
        return 5
    return min(8, n_texts // 20)


def _embed_texts(texts: list[str]):
    """
    Metinleri dense semantik vektörlere çevirir (get_embeddings fabrikası;
    sağlayıcı env'den — openai/ollama/google). İçerik-adresli cache kullanır:
    aynı metin+model daha önce embed edildiyse yeniden çağrı yapılmaz. Cosine'a
    yakın KMeans için L2 normalize edilir. Sağlayıcı/anahtar yoksa veya hata
    olursa None döner → çağıran TF-IDF'e düşer (zarif degradasyon korunur).
    """
    try:
        import numpy as np
        from src.core.embeddings import get_embeddings
        from src.graphs.textanalysis.cache import get_cached, set_cached

        model = f"{settings.EMBEDDING_PROVIDER}:{settings.EMBEDDING_MODEL}"
        unique = list(dict.fromkeys(texts))
        cached = get_cached("embed", model, unique)
        misses = [t for t in unique if t not in cached]

        vec_by_text: dict = dict(cached)
        if misses:
            emb = get_embeddings()
            vectors = emb.embed_documents(misses)
            if len(vectors) != len(misses):
                return None
            new_items = {t: list(map(float, v)) for t, v in zip(misses, vectors)}
            vec_by_text.update(new_items)
            set_cached("embed", model, new_items)

        arr = np.asarray([vec_by_text[t] for t in texts], dtype="float32")
        if arr.ndim != 2 or arr.shape[0] != len(texts):
            return None
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        logger.info(f"[TA] embedding ile kümeleme: {arr.shape[0]}x{arr.shape[1]} "
                    f"({model}, {len(misses)} yeni / {len(cached)} cache-hit)")
        return arr / norms
    except Exception as e:  # pragma: no cover - sağlayıcı/anahtar yokluğunda
        logger.warning(f"[TA] embedding alınamadı ({type(e).__name__}: {e}); TF-IDF'e düşülüyor")
        return None


def _best_k_by_silhouette(X, k_min: int = 2, k_max: int = 8) -> int:
    """
    En iyi küme sayısını silhouette skoruyla seçer. X seyrek (TF-IDF) ya da
    dense (embedding) olabilir. Büyük veride maliyeti sınırlamak için örneklenmez
    ama k aralığı dardır. Hata/dejenere durumda sezgisele döner.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        n = X.shape[0]
        kmax = min(k_max, n - 1)
        best_k, best_score = _suggest_cluster_count(n), -1.0
        for k in range(k_min, kmax + 1):
            km = KMeans(n_clusters=k, random_state=settings.RANDOM_SEED, n_init=10)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score, best_k = score, k
        logger.info(f"[TA] silhouette ile k={best_k} seçildi (skor={best_score:.3f})")
        return max(2, min(best_k, n))
    except Exception as e:  # pragma: no cover
        logger.warning(f"[TA] silhouette başarısız ({e}); sezgisel k kullanılıyor")
        return max(2, min(_suggest_cluster_count(X.shape[0]), X.shape[0]))


def _hdbscan_labels(X):
    """
    HDBSCAN ile kümele (dense embedding üzerinde). Gürültü noktaları -1 etiketi
    alır (temaya dahil edilmez). Çoğunluk gürültü olur ya da tek küme çıkarsa
    None döner → çağıran KMeans'e düşer. min_cluster_size veri büyüklüğüyle ölçeklenir.
    """
    try:
        import numpy as np
        from sklearn.cluster import HDBSCAN

        n = X.shape[0]
        Xd = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        min_size = max(3, n // 25)
        labels = HDBSCAN(min_cluster_size=min_size, metric="euclidean").fit_predict(Xd)
        uniq = set(int(l) for l in labels if l != -1)
        noise = int((labels == -1).sum())
        # Hiç küme yok ya da neredeyse her şey gürültü → KMeans daha uygun.
        if len(uniq) < 2 or noise > 0.6 * n:
            logger.info(f"[TA] HDBSCAN dejenere ({len(uniq)} küme, {noise} gürültü); KMeans'e düşülüyor")
            return None
        logger.info(f"[TA] HDBSCAN ile {len(uniq)} küme, {noise} gürültü noktası")
        return labels
    except Exception as e:  # pragma: no cover
        logger.warning(f"[TA] HDBSCAN başarısız ({e}); KMeans'e düşülüyor")
        return None


def cluster_texts(texts: list[str], n_clusters: Optional[int] = None) -> list[dict]:
    """
    Metinleri temalara kümeler. Hibrit: önce semantik embedding (get_embeddings)
    denenir; alınamazsa TF-IDF'e düşer. Her iki durumda da tema etiketi ipucu
    (top_terms) üye metinlerin ortalama TF-IDF'inden türetilir.

    Returns: küme listesi; her eleman:
        {"cluster_id": int, "member_indices": [int...], "top_terms": [str...]}
    member_indices, `texts` listesindeki orijinal indekslerdir.
    Çok az metin (<3) varsa tek küme döner (kümeleme anlamsız).
    """
    n = len(texts)
    if n == 0:
        return []
    if n < 3:
        return [{"cluster_id": 0, "member_indices": list(range(n)), "top_terms": []}]

    # Ağır importları fonksiyon içinde tut (worker başlangıcını yavaşlatmasın).
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    # TF-IDF her zaman hesaplanır: embedding olsa bile top_terms (etiket ipucu)
    # için gereklidir; embedding yoksa kümeleme matrisi olarak da kullanılır.
    # Stopword'ler temizlenir; min_df küçük veride 1, yeterli veride 2 (gürültü/
    # nadir terim eler). Çok dilli stopword listesi (TR+EN) verilir.
    min_df = 2 if n >= 10 else 1
    vectorizer = TfidfVectorizer(
        max_features=2000, ngram_range=(1, 2), min_df=min_df,
        sublinear_tf=True, stop_words=STOPWORDS,
    )
    def _fit_tfidf(stops, mdf):
        v = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=mdf,
                            sublinear_tf=True, stop_words=stops)
        m = v.fit_transform(texts)
        return m, v.get_feature_names_out()

    try:
        tfidf, terms = _fit_tfidf(STOPWORDS, min_df)
    except ValueError:
        # Stopword+min_df=2 sözlüğü boşalttıysa daha gevşek ayarla bir kez dene.
        try:
            tfidf, terms = _fit_tfidf(None, 1)
        except ValueError:
            tfidf, terms = None, []

    # Hibrit: önce semantik embedding, yoksa TF-IDF.
    embeddings = _embed_texts(texts)
    if embeddings is not None:
        X = embeddings
    elif tfidf is not None:
        X = tfidf
    else:
        # Ne embedding ne TF-IDF — kümeleme yapılamaz.
        return [{"cluster_id": 0, "member_indices": list(range(n)), "top_terms": []}]

    # Etiketleme: embedding varsa ve HDBSCAN açıksa onu dene (gürültü kümesi +
    # değişken yoğunluk); başarısız/dejenere olursa KMeans'e zarifçe düş.
    labels = None
    if n_clusters is None and embeddings is not None and settings.TA_USE_HDBSCAN and n >= 15:
        labels = _hdbscan_labels(X)

    if labels is None:
        # Küme sayısı: kullanıcı verdiyse onu kullan; yoksa yeterli veri varsa
        # silhouette skoruyla otomatik seç, küçük veride sezgisele düş.
        if n_clusters:
            k = max(2, min(n_clusters, n))
        elif n >= 8:
            k = _best_k_by_silhouette(X)
        else:
            k = max(2, min(_suggest_cluster_count(n), n))
        km = KMeans(n_clusters=k, random_state=settings.RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X)

    clusters: list[dict] = []
    for cid in sorted(set(int(l) for l in labels)):
        if cid == -1:
            continue  # HDBSCAN gürültü noktaları (kümesiz) — temaya dahil edilmez.
        member_indices = [i for i, lab in enumerate(labels) if int(lab) == cid]
        if not member_indices:
            continue
        # Etiket ipucu: üye metinlerin ortalama TF-IDF'inde en ağırlıklı terimler.
        top_terms: list[str] = []
        if tfidf is not None and len(terms):
            mean_vec = np.asarray(tfidf[member_indices].mean(axis=0)).ravel()
            top_idx = mean_vec.argsort()[::-1][:6]
            top_terms = [terms[j] for j in top_idx if mean_vec[j] > 0]
        clusters.append({
            "cluster_id": cid,
            "member_indices": member_indices,
            "top_terms": top_terms,
        })

    clusters.sort(key=lambda c: len(c["member_indices"]), reverse=True)
    return clusters


def distribution(values: list[str]) -> dict:
    """Bir kategorik listenin sayım dağılımını (label -> count) döndürür."""
    return dict(Counter(v for v in values if v))


def build_priority_items(item_results: list[dict], limit: int = 20) -> list[dict]:
    """
    Şikayet önceliklendirme tablosu: aciliyet (critical→low) ve olumsuz duygu
    şiddetine göre sıralı, en kritik geri bildirimler.
    """
    def sort_key(it: dict):
        urg = URGENCY_ORDER.get((it.get("urgency") or "low").lower(), 0)
        # Daha olumsuz sentiment_score önce gelsin (-1 en olumsuz).
        neg = -float(it.get("sentiment_score") or 0.0)
        return (urg, neg)

    # Sadece eyleme dönük olanlar: şikayet/talep veya yüksek aciliyet.
    actionable = [
        it for it in item_results
        if (it.get("category") in ("complaint", "request"))
        or URGENCY_ORDER.get((it.get("urgency") or "low").lower(), 0) >= 2
    ]
    actionable.sort(key=sort_key, reverse=True)
    return actionable[:limit]


def build_time_trend(item_results: list[dict], min_dated: int = 3) -> list[dict]:
    """
    Zaman damgalı geri bildirimlerden duygu trendini kurar. Tarih aralığına göre
    granülerlik otomatik seçilir (gün / hafta / ay). Yeterli tarihli kayıt
    (min_dated) yoksa boş liste döner.

    Returns: [{period, count, avg_sentiment, positive, negative, neutral}]
    """
    import pandas as pd

    rows = [
        {
            "ts": it.get("timestamp"),
            "score": float(it.get("sentiment_score") or 0.0),
            "sentiment": it.get("sentiment") or "neutral",
        }
        for it in item_results
        if it.get("timestamp")
    ]
    if len(rows) < min_dated:
        return []

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])
    if len(df) < min_dated:
        return []

    span_days = (df["ts"].max() - df["ts"].min()).days
    freq = "D" if span_days <= 60 else ("W" if span_days <= 730 else "MS")

    trend: list[dict] = []
    for period, grp in df.set_index("ts").groupby(pd.Grouper(freq=freq)):
        if len(grp) == 0:
            continue
        vc = grp["sentiment"].value_counts().to_dict()
        trend.append({
            "period": period.strftime("%Y-%m-%d"),
            "count": int(len(grp)),
            "avg_sentiment": round(float(grp["score"].mean()), 3),
            "positive": int(vc.get("positive", 0)),
            "negative": int(vc.get("negative", 0)),
            "neutral": int(vc.get("neutral", 0)),
        })

    return _augment_trend_stats(trend)


def _augment_trend_stats(trend: list[dict], ma_window: int = 3) -> list[dict]:
    """
    Trend dönemlerine türev istatistikler ekler:
      - ma_sentiment: son `ma_window` dönemin hareketli ortalaması
      - delta_sentiment: bir önceki döneme göre duygu değişimi
      - spike: hacim (count) son ortalamanın ~2 katından fazlaysa ani-artış işareti
    """
    if not trend:
        return trend
    scores = [t["avg_sentiment"] for t in trend]
    counts = [t["count"] for t in trend]
    for i, t in enumerate(trend):
        window = scores[max(0, i - ma_window + 1): i + 1]
        t["ma_sentiment"] = round(sum(window) / len(window), 3)
        t["delta_sentiment"] = round(scores[i] - scores[i - 1], 3) if i > 0 else 0.0
        prev_counts = counts[max(0, i - ma_window): i]
        avg_prev = (sum(prev_counts) / len(prev_counts)) if prev_counts else 0.0
        t["spike"] = bool(avg_prev > 0 and counts[i] >= 2 * avg_prev)
    return trend


def theme_sentiment_breakdown(member_indices: list[int], item_results: list[dict]) -> dict:
    """Bir temanın içindeki sentiment dağılımı (rapor/görselleştirme için)."""
    sentiments = []
    for i in member_indices:
        if 0 <= i < len(item_results):
            sentiments.append(item_results[i].get("sentiment"))
    return distribution(sentiments)


def collapse_near_duplicates(texts: list[str]) -> tuple[list[str], dict[str, int]]:
    """
    Yakın-tekrarları (kopya/spam: yalnızca büyük-küçük harf, noktalama, boşluk
    farkıyla aynı olan metinler) birleştirir. İlk görülen temsilci olarak tutulur;
    sayımları şişirmesin diye sonrakiler atılır.

    Returns: (benzersiz_metinler, {temsilci_metin: kaç_kopya_birleşti}).
    Deterministik (embedding/LLM yok), ucuz ve dilden bağımsızdır.
    """
    seen: dict[str, str] = {}      # norm -> temsilci ham metin
    dup_counts: dict[str, int] = {}
    out: list[str] = []
    for t in texts or []:
        norm = _norm_for_dedup(t)
        if not norm:
            continue
        if norm in seen:
            rep = seen[norm]
            dup_counts[rep] = dup_counts.get(rep, 1) + 1
            continue
        seen[norm] = t
        out.append(t)
    return out, {k: v for k, v in dup_counts.items() if v > 1}


def canonicalize_aspects(aspect_distribution: dict, threshold: Optional[float] = None) -> dict:
    """
    Aspect adlarını embedding kosinüs benzerliğiyle kanonik forma indirger
    (eşanlamlı birleştirme: "kargo"/"teslimat"/"teslim" → en sık geçen tek ad).
    En sık aspect'ler çekirdek alınır; benzerleri onlara eklenir. Embedding
    alınamazsa girdi olduğu gibi döner (zarif degradasyon).
    """
    if not aspect_distribution or len(aspect_distribution) < 2:
        return aspect_distribution
    thr = settings.TA_ASPECT_MERGE_THRESHOLD if threshold is None else threshold
    try:
        import numpy as np
    except Exception:
        return aspect_distribution

    names = list(aspect_distribution.keys())
    vecs = _embed_texts(names)
    if vecs is None:
        return aspect_distribution  # vektör yoksa birleştirme yapma.

    # En çok bahsedilenden aza doğru; çekirdek temsilcilere benzerleri ekle.
    order = sorted(range(len(names)), key=lambda i: aspect_distribution[names[i]].get("total", 0), reverse=True)
    canon_idx: list[int] = []
    assign: dict[int, int] = {}
    for i in order:
        best_c, best_sim = None, thr
        for c in canon_idx:
            sim = float(np.dot(vecs[i], vecs[c]))  # L2 normalize edilmiş → kosinüs
            if sim >= best_sim:
                best_c, best_sim = c, sim
        if best_c is None:
            canon_idx.append(i)
            assign[i] = i
        else:
            assign[i] = best_c

    merged: dict[str, dict] = {}
    for i, name in enumerate(names):
        c = assign.get(i, i)
        canon_name = names[c]
        bucket = merged.setdefault(canon_name, {"positive": 0, "negative": 0, "neutral": 0, "total": 0})
        for key in ("positive", "negative", "neutral", "total"):
            bucket[key] += int(aspect_distribution[name].get(key, 0))

    merged = dict(sorted(merged.items(), key=lambda kv: kv[1]["total"], reverse=True)[:15])
    if len(merged) < len(aspect_distribution):
        logger.info(f"[TA] aspect kanonikleştirme: {len(aspect_distribution)} → {len(merged)} aspect")
    return merged


# Dinamik duygu-etiketi sözlüğü (ground-truth kolonu farklı isimlendirilebilir:
# Olumlu/Olumsuz, positive/negative, iyi/kötü, pozitif/negatif ...). Eşanlamlıları
# kanonik positive/negative/neutral'a indirger. Bilinmeyen/sayısal değer → None.
_LABEL_POSITIVE = {"positive", "pos", "olumlu", "pozitif", "iyi", "good", "memnun",
                   "beğendim", "begendim", "olumlu yorum", "positif"}
_LABEL_NEGATIVE = {"negative", "neg", "olumsuz", "negatif", "kötü", "kotu", "bad",
                   "şikayet", "sikayet", "memnuniyetsiz", "kötü yorum"}
_LABEL_NEUTRAL = {"neutral", "neu", "nötr", "notr", "noter", "orta", "kararsız",
                  "kararsiz", "mixed", "karışık", "ne iyi ne kötü"}


def normalize_sentiment_label(value) -> Optional[str]:
    """
    Serbest etiketi (CSV ground-truth kolonu) kanonik positive/negative/neutral'a
    çevirir. Türkçe/İngilizce eşanlamlıları ve önek eşleşmesini destekler. Saf
    sayısal değer (puan) veya tanınmayan → None (puan ayrı doğrulanır).
    """
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s or s.replace(".", "").replace(",", "").isdigit():
        return None
    if s in _LABEL_POSITIVE:
        return "positive"
    if s in _LABEL_NEGATIVE:
        return "negative"
    if s in _LABEL_NEUTRAL:
        return "neutral"
    # Önek eşleşmesi ("olumlu yorum", "negative review" ...).
    for tok in _LABEL_NEGATIVE:
        if s.startswith(tok):
            return "negative"
    for tok in _LABEL_POSITIVE:
        if s.startswith(tok):
            return "positive"
    for tok in _LABEL_NEUTRAL:
        if s.startswith(tok):
            return "neutral"
    return None


def label_sentiment_validation(item_results: list[dict], labels: list) -> dict:
    """
    LLM tahmini ↔ veri etiketi (ground-truth) KARŞILAŞTIRMALI performansı. Etiket
    kolonu farklı isimlendirilse/sözcüklense de normalize edilir. Doğruluk,
    confusion matrisi ve sınıf-bazlı precision/recall/f1 döner.

    Returns: {n, accuracy, correct, mismatches, confusion, per_class, labels_seen, note} | {}
    """
    classes = ["positive", "negative", "neutral"]
    pairs: list[tuple[str, str]] = []
    for i, it in enumerate(item_results):
        true = normalize_sentiment_label(labels[i] if i < len(labels) else None)
        if true is None:
            continue
        pred = (it.get("sentiment") or "neutral").lower()
        pairs.append((pred if pred in classes else "neutral", true))
    if len(pairs) < 3:
        return {}

    confusion = {t: {p: 0 for p in classes} for t in classes}
    correct = 0
    for pred, true in pairs:
        confusion[true][pred] += 1
        if pred == true:
            correct += 1
    n = len(pairs)

    per_class: dict[str, dict] = {}
    for c in classes:
        tp = confusion[c][c]
        fp = sum(confusion[t][c] for t in classes if t != c)
        fn = sum(confusion[c][p] for p in classes if p != c)
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
        per_class[c] = {
            "precision": round(prec, 3) if prec is not None else None,
            "recall": round(rec, 3) if rec is not None else None,
            "f1": round(f1, 3) if f1 is not None else None,
            "support": sum(confusion[c].values()),
        }

    return {
        "n": n,
        "accuracy": round(correct / n, 3),
        "correct": correct,
        "mismatches": n - correct,
        "confusion": confusion,
        "per_class": per_class,
        "labels_seen": sorted({t for _, t in pairs}),
        "note": "LLM tahmini ↔ veri etiketi (ground-truth) karşılaştırması.",
    }


def rating_sentiment_validation(item_results: list[dict], ratings: list) -> dict:
    """
    LLM duygusu ile gerçek (CSV) puan arasındaki tutarlılığı ölçer — kalibrasyon/
    güven sinyali. Puan 1..5 (veya 1..10) varsayılır; ortalamanın üstü→olumlu,
    altı→olumsuz beklenir. Yeterli puanlı kayıt yoksa boş döner.

    Returns: {n, agreement, mismatches, avg_rating, note} | {}
    """
    pairs = []
    for i, it in enumerate(item_results):
        r = ratings[i] if i < len(ratings) else None
        if r is None:
            continue
        try:
            rv = float(r)
        except (TypeError, ValueError):
            continue
        pairs.append((rv, (it.get("sentiment") or "neutral").lower()))
    if len(pairs) < 3:
        return {}

    rmax = max(rv for rv, _ in pairs)
    mid = (rmax / 2.0) if rmax > 1 else 0.5
    agree = mism = 0
    for rv, sent in pairs:
        expected = "positive" if rv > mid else ("negative" if rv < mid else "neutral")
        if sent == "neutral" or expected == "neutral":
            continue
        if sent == expected:
            agree += 1
        else:
            mism += 1
    total = agree + mism
    return {
        "n": len(pairs),
        "agreement": round(agree / total, 3) if total else None,
        "mismatches": mism,
        "avg_rating": round(sum(rv for rv, _ in pairs) / len(pairs), 2),
        "note": "LLM duygusu ile kullanıcı puanı tutarlılığı (yüksek = LLM puanla uyumlu).",
    }


def theme_time_matrix(themes: list[dict], item_results: list[dict], min_dated: int = 3) -> list[dict]:
    """
    Tema × zaman: her temanın dönem-dönem hacmini ve yönünü (yükselen/düşen)
    çıkarır. Yön, dönem indeksine karşı hacmin doğrusal eğiminden belirlenir.
    item_results'taki member_indices ve timestamp'leri kullanır.

    Returns: [{label, direction, slope, periods:[{period,count,avg_sentiment}]}]
    Tarihli kayıt yetersizse boş liste.
    """
    import pandas as pd

    dated = [it for it in item_results if it.get("timestamp")]
    if len(dated) < min_dated or not themes:
        return []

    # Tüm tarihli kayıtlar için ortak granülerlik belirle.
    all_ts = pd.to_datetime([it["timestamp"] for it in dated], errors="coerce")
    all_ts = all_ts.dropna()
    if len(all_ts) < min_dated:
        return []
    span_days = (all_ts.max() - all_ts.min()).days
    freq = "D" if span_days <= 60 else ("W" if span_days <= 730 else "MS")

    out: list[dict] = []
    for th in themes:
        members = th.get("member_indices") or []
        rows = []
        for i in members:
            if 0 <= i < len(item_results):
                it = item_results[i]
                if it.get("timestamp"):
                    rows.append({"ts": it["timestamp"],
                                 "score": float(it.get("sentiment_score") or 0.0)})
        if len(rows) < min_dated:
            continue
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df = df.dropna(subset=["ts"])
        if len(df) < min_dated:
            continue
        periods = []
        for period, grp in df.set_index("ts").groupby(pd.Grouper(freq=freq)):
            if len(grp) == 0:
                continue
            periods.append({
                "period": period.strftime("%Y-%m-%d"),
                "count": int(len(grp)),
                "avg_sentiment": round(float(grp["score"].mean()), 3),
            })
        if len(periods) < 2:
            continue
        counts = [p["count"] for p in periods]
        slope = _linear_slope(counts)
        direction = "yükselen" if slope > 0.15 else ("düşen" if slope < -0.15 else "sabit")
        out.append({
            "label": th.get("label"),
            "direction": direction,
            "slope": round(slope, 3),
            "periods": periods,
        })
    return out


def _linear_slope(ys: list[float]) -> float:
    """En küçük kareler eğimi (x = 0..n-1). Trend yönü için basit gösterge."""
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / denom
