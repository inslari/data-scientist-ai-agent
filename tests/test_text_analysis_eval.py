"""
Text-analysis — roadmap özelliklerinin deterministik eval/test seti.

Yeni eklenen (LLM'siz) yetenekleri doğrular: dil tespiti, PII redaksiyon, yakın-
tekrar birleştirme, puan-duygu doğrulama, trend istatistikleri, tema×zaman matrisi
ve aspect kanonikleştirme (embedding yoksa identity).

Çalıştırma (LOKAL sqlite sandbox — asla remote app DB değil; bkz. CLAUDE.md):
    API_DATABASE_URL=sqlite:////tmp/ta_eval.db uv run pytest tests/test_text_analysis_eval.py
"""
import os
os.environ.setdefault("API_DATABASE_URL", "sqlite:////tmp/ta_eval.db")
import api.core.database  # noqa: E402,F401

assert api.core.database.SQLALCHEMY_DATABASE_URL.startswith("sqlite"), (
    "Refusing to run: app DB is not a local sqlite sandbox "
    f"({api.core.database.SQLALCHEMY_DATABASE_URL.split('://')[0]})"
)


# ─────────────────────────── dil tespiti ──────────────────────────────────────

def test_detect_language_tr_via_chars():
    from src.services.text_analysis_engine import detect_language
    assert detect_language(["kargo çok geç geldi", "ürün güzel"]) == "tr"


def test_detect_language_en():
    from src.services.text_analysis_engine import detect_language
    assert detect_language(["the delivery was very late", "i love this product"]) == "en"


def test_detect_language_empty_defaults_tr():
    from src.services.text_analysis_engine import detect_language
    assert detect_language([]) == "tr"


# ─────────────────────────── PII redaksiyon ───────────────────────────────────

def test_redact_pii_masks_email_and_phone():
    from src.services.text_analysis_engine import redact_pii
    out = redact_pii("Bana ali@example.com adresinden veya 0532 123 45 67 ulaşın")
    assert "ali@example.com" not in out
    assert "[email]" in out
    assert "[phone]" in out


def test_redact_pii_masks_iban():
    from src.services.text_analysis_engine import redact_pii
    out = redact_pii("IBAN: TR33 0006 1005 1978 6457 8413 26 numaram")
    assert "[iban]" in out


# ─────────────────────────── yakın-tekrar ─────────────────────────────────────

def test_collapse_near_duplicates_merges_case_punct_variants():
    from src.services.text_analysis_engine import collapse_near_duplicates
    texts = ["Kargo gecikti!", "kargo gecikti", "KARGO  GECİKTİ.", "ürün harika"]
    kept, groups = collapse_near_duplicates(texts)
    assert len(kept) == 2
    assert any(v >= 3 for v in groups.values())


# ─────────────────────────── puan-duygu doğrulama ─────────────────────────────

def test_rating_validation_agreement():
    from src.services.text_analysis_engine import rating_sentiment_validation
    items = [
        {"sentiment": "positive"}, {"sentiment": "positive"},
        {"sentiment": "negative"}, {"sentiment": "negative"},
    ]
    ratings = [5, 4, 1, 2]  # rmax=5, mid=2.5 → ilk ikisi pozitif, son ikisi negatif beklenir
    rv = rating_sentiment_validation(items, ratings)
    assert rv["n"] == 4
    assert rv["agreement"] == 1.0
    assert rv["mismatches"] == 0


def test_rating_validation_detects_mismatch():
    from src.services.text_analysis_engine import rating_sentiment_validation
    items = [{"sentiment": "positive"}, {"sentiment": "positive"}, {"sentiment": "negative"}]
    ratings = [1, 5, 5]  # ilk kayıt uyumsuz (pozitif ama puan düşük)
    rv = rating_sentiment_validation(items, ratings)
    assert rv["mismatches"] >= 1


def test_rating_validation_empty_when_too_few():
    from src.services.text_analysis_engine import rating_sentiment_validation
    assert rating_sentiment_validation([{"sentiment": "positive"}], [5]) == {}


# ─────────────────────── adaptif partileme ───────────────────────────────────

def test_make_batches_packs_short_texts_up_to_max_items():
    from src.graphs.textanalysis.nodes.classify_node import _make_batches
    texts = ["kısa"] * 50  # her biri 4 karakter → karakter bütçesi bol
    batches = _make_batches(texts, max_items=24, char_budget=6000)
    assert all(len(b) <= 24 for b in batches)
    assert sum(len(b) for b in batches) == 50
    assert len(batches) == 3  # 24 + 24 + 2


def test_make_batches_long_text_gets_own_batch():
    from src.graphs.textanalysis.nodes.classify_node import _make_batches
    long_t = "x" * 5000
    texts = ["a", "b", long_t, "c"]
    batches = _make_batches(texts, max_items=24, char_budget=2000)
    # uzun metin kardeşleriyle aynı partiye sıkışmamalı
    assert any(b == [long_t] for b in batches)
    assert sum(len(b) for b in batches) == 4


def test_make_batches_char_budget_splits():
    from src.graphs.textanalysis.nodes.classify_node import _make_batches
    texts = ["y" * 600] * 10  # her biri 600 karakter, bütçe 2000 → ~3'lü gruplar
    batches = _make_batches(texts, max_items=24, char_budget=2000)
    assert all(sum(len(t) for t in b) <= 2000 or len(b) == 1 for b in batches)
    assert sum(len(b) for b in batches) == 10


# ─────────────────────── dinamik etiket + performans ─────────────────────────

def test_normalize_sentiment_label_dynamic_vocab():
    from src.services.text_analysis_engine import normalize_sentiment_label as n
    assert n("Olumlu") == "positive"
    assert n("OLUMSUZ") == "negative"
    assert n("Nötr") == "neutral"
    assert n("positive") == "positive"
    assert n("negatif yorum") == "negative"  # önek eşleşmesi
    assert n("iyi") == "positive"
    assert n("5") is None      # saf sayı → puan, etiket değil
    assert n("") is None
    assert n("blah") is None


def test_label_sentiment_validation_accuracy_and_confusion():
    from src.services.text_analysis_engine import label_sentiment_validation
    items = [
        {"sentiment": "positive"}, {"sentiment": "positive"},
        {"sentiment": "negative"}, {"sentiment": "negative"},
        {"sentiment": "positive"},  # bu yanlış (gerçek negatif)
    ]
    labels = ["Olumlu", "Olumlu", "Olumsuz", "Olumsuz", "Olumsuz"]  # farklı isimlendirme
    lv = label_sentiment_validation(items, labels)
    assert lv["n"] == 5
    assert lv["correct"] == 4
    assert lv["mismatches"] == 1
    assert lv["accuracy"] == 0.8
    # confusion[gerçek][tahmin]: negatif→pozitif 1 hata
    assert lv["confusion"]["negative"]["positive"] == 1
    assert lv["per_class"]["positive"]["support"] == 2
    assert set(lv["labels_seen"]) == {"positive", "negative"}


def test_label_validation_empty_when_unlabeled():
    from src.services.text_analysis_engine import label_sentiment_validation
    items = [{"sentiment": "positive"}, {"sentiment": "negative"}, {"sentiment": "neutral"}]
    assert label_sentiment_validation(items, [None, "5", "blah"]) == {}


# ─────────────────────────── trend istatistikleri ────────────────────────────

def test_trend_has_ma_delta_spike():
    from src.services.text_analysis_engine import build_time_trend
    items = (
        [{"timestamp": "2026-01-01", "sentiment": "negative", "sentiment_score": -0.5}] * 2
        + [{"timestamp": "2026-01-02", "sentiment": "positive", "sentiment_score": 0.5}] * 8
    )
    trend = build_time_trend(items, min_dated=2)
    assert trend
    for p in trend:
        assert {"ma_sentiment", "delta_sentiment", "spike"} <= set(p)
    # İkinci dönem hacmi ilkinin ~4 katı → spike işaretli olmalı.
    assert any(p["spike"] for p in trend)


# ─────────────────────────── tema × zaman matrisi ────────────────────────────

def test_theme_time_matrix_direction():
    from src.services.text_analysis_engine import theme_time_matrix
    item_results = [
        {"timestamp": "2026-01-01", "sentiment_score": -0.5},
        {"timestamp": "2026-01-02", "sentiment_score": -0.5},
        {"timestamp": "2026-01-03", "sentiment_score": -0.5},
        {"timestamp": "2026-01-03", "sentiment_score": -0.5},
    ]
    themes = [{"label": "Kargo", "member_indices": [0, 1, 2, 3]}]
    out = theme_time_matrix(themes, item_results, min_dated=3)
    assert out
    assert out[0]["label"] == "Kargo"
    assert out[0]["direction"] in ("yükselen", "düşen", "sabit")
    assert out[0]["periods"]


# ─────────────────────────── aspect kanonikleştirme ──────────────────────────

def test_canonicalize_aspects_identity_without_embeddings(monkeypatch):
    import src.services.text_analysis_engine as eng
    monkeypatch.setattr(eng, "_embed_texts", lambda names: None)  # embedding kapalı
    dist = {"kargo": {"positive": 1, "negative": 2, "neutral": 0, "total": 3},
            "fiyat": {"positive": 0, "negative": 1, "neutral": 0, "total": 1}}
    assert eng.canonicalize_aspects(dist) == dist  # birleştirme yok → identity


def test_canonicalize_aspects_merges_synonyms(monkeypatch):
    import numpy as np
    import src.services.text_analysis_engine as eng
    # "kargo" ve "teslimat" neredeyse aynı vektör → birleşmeli; "fiyat" ayrı.
    vecs = {
        "kargo": np.array([1.0, 0.0]),
        "teslimat": np.array([0.99, 0.01]),
        "fiyat": np.array([0.0, 1.0]),
    }
    monkeypatch.setattr(eng, "_embed_texts",
                        lambda names: np.array([vecs[n] for n in names]))
    dist = {
        "kargo": {"positive": 1, "negative": 3, "neutral": 0, "total": 4},
        "teslimat": {"positive": 0, "negative": 2, "neutral": 1, "total": 3},
        "fiyat": {"positive": 0, "negative": 1, "neutral": 0, "total": 1},
    }
    out = eng.canonicalize_aspects(dist, threshold=0.8)
    assert "fiyat" in out
    assert len(out) == 2  # kargo+teslimat → tek kanonik ad
    canon = max(out, key=lambda k: out[k]["total"])
    assert out[canon]["total"] == 7  # 4 + 3 birleşti


# ─────────────────────────── preprocess entegrasyonu ─────────────────────────

def test_preprocess_redacts_and_dedups_and_aligns():
    from src.graphs.textanalysis.nodes.preprocess_node import ta_preprocess_node
    state = {
        "raw_texts": ["Mail: a@b.com kargo gecikti", "mail: a@b.com KARGO gecikti!", "ürün güzel"],
        "raw_timestamps": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "raw_ratings": [1, 1, 5],
        "language": "auto",
    }
    out = ta_preprocess_node(state)
    # PII maskelendi
    assert all("a@b.com" not in t for t in out["texts"])
    # near-dup birleşti (ilk iki metin) → 2 metin kaldı
    assert len(out["texts"]) == 2
    # tarih/puan hizası korundu
    assert len(out["timestamps"]) == len(out["texts"])
    assert len(out["ratings"]) == len(out["texts"])
    assert out["detected_language"] == "tr"
