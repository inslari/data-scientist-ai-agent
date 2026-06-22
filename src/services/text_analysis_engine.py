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
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    try:
        tfidf = vectorizer.fit_transform(texts)
        terms = vectorizer.get_feature_names_out()
    except ValueError:
        tfidf = None
        terms = []

    # Hibrit: önce semantik embedding, yoksa TF-IDF.
    embeddings = _embed_texts(texts)
    if embeddings is not None:
        X = embeddings
    elif tfidf is not None:
        X = tfidf
    else:
        # Ne embedding ne TF-IDF — kümeleme yapılamaz.
        return [{"cluster_id": 0, "member_indices": list(range(n)), "top_terms": []}]

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
    for cid in range(k):
        member_indices = [i for i, lab in enumerate(labels) if lab == cid]
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
    return trend


def theme_sentiment_breakdown(member_indices: list[int], item_results: list[dict]) -> dict:
    """Bir temanın içindeki sentiment dağılımı (rapor/görselleştirme için)."""
    sentiments = []
    for i in member_indices:
        if 0 <= i < len(item_results):
            sentiments.append(item_results[i].get("sentiment"))
    return distribution(sentiments)
