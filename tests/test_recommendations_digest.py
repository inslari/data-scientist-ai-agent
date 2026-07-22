"""#11(a): öneriler digest'inden importance KESİLDİ — yapısal (LLM'siz) doğrulama.

ÖZEL ZORLUK: öneri prose'u LLM ile üretilir, offline harness onu render etmez → "ihlal
gitti mi" GÖZLE görülemez. Bu yüzden testi KAYNAĞA vururuz: recs-scoped _build_digest
çıktısında importance bölümü/etiketleri BULUNMAMALI. Bu "ihlal görünmüyor"dan güçlüdür —
ihlalin KAYNAĞININ kesildiğini kanıtlar.

DÜRÜST SINIR: bu testin yeşil olması ihlalin bittiğini GARANTİ ETMEZ, yalnız importance
kaynağını kapatır. detect_speculation (⊇ detect_causal_language, fail-closed) hâlâ son
savunma hattıdır (bkz. build_recommendations_section yorumu, #11-b TODO).
"""
from src.core.report_schema import Evidence, ReportSection
from src.services.report_sections import _build_digest

_RECS_EXCLUDE = frozenset({"features"})
_RECS_NOTE = frozenset({"error_analysis"})


def _sections():
    return [
        ReportSection(id="performance", title="Model Performansı", status="ok",
                      headline="ROC 0.85", evidence=[Evidence("ROC-AUC", 0.85, "test_set")]),
        ReportSection(id="features", title="En Etkili Özellikler", status="ok",
                      headline="En etkili: credit_score",
                      evidence=[Evidence("credit_score — tahmine katkı payı", 0.32, "model_internal"),
                                Evidence("age — tahmine katkı payı", 0.11, "model_internal")]),
        ReportSection(id="error_analysis", title="Modelin Nerede Yanıldığı", status="ok",
                      headline="Hatalar yaşlı müşterilerde yoğun",
                      evidence=[Evidence("ortalama yaş (hatalı vakalar)", 45, "test_set")]),
        ReportSection(id="assessment", title="Güçlü ve Zayıf Yönler", status="ok",
                      headline="1 risk", evidence=[Evidence("Azınlık recall", 0.62, "test_set")]),
    ]


def test_recs_digest_importance_kesildi():
    """RECS digest'inde importance bölümü VE etiketleri YOK (kaynak kesildi)."""
    recs = _build_digest(_sections(), "tr", exclude_ids=_RECS_EXCLUDE, causal_note_ids=_RECS_NOTE)
    assert "En Etkili Özellikler" not in recs
    assert "tahmine katkı payı" not in recs
    assert "credit_score" not in recs


def test_full_digest_importance_gorur_hedefli_dislama():
    """HEDEFLİ olmalı: dışlamasız (summary/assessment) digest importance'ı GÖRÜR."""
    full = _build_digest(_sections(), "tr")
    assert "En Etkili Özellikler" in full
    assert "tahmine katkı payı" in full


def test_error_analysis_kalir_ama_nedensellik_damgali():
    """error_analysis öneri girdisinde KALIR (aksiyon alınabilir) + 'sebep değil' damgası taşır."""
    recs = _build_digest(_sections(), "tr", exclude_ids=_RECS_EXCLUDE, causal_note_ids=_RECS_NOTE)
    assert "Modelin Nerede Yanıldığı" in recs                 # kaldı
    assert "NEDENİNİ değil" in recs                           # damga taşındı
    # damga başlığın hemen ardında (o bölüme ait)
    idx_title = recs.index("Modelin Nerede Yanıldığı")
    idx_note = recs.index("NEDENİNİ değil")
    assert 0 < (idx_note - idx_title) < 120


def test_diger_aksiyon_bulgulari_kalir():
    """Öneri girdisinde performans/assessment gibi aksiyon bulguları KALIR — yalnız importance gider."""
    recs = _build_digest(_sections(), "tr", exclude_ids=_RECS_EXCLUDE, causal_note_ids=_RECS_NOTE)
    assert "Model Performansı" in recs and "Güçlü ve Zayıf Yönler" in recs


def test_en_damga_dili_dogru():
    """EN digest'te damga İngilizce (TR sızıntısı yok)."""
    recs = _build_digest(_sections(), "en", exclude_ids=_RECS_EXCLUDE, causal_note_ids=_RECS_NOTE)
    assert "not WHY" in recs and "NEDENİNİ" not in recs
