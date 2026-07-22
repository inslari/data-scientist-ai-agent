"""Sentez bölümleri (Özet / Güçlü-Zayıf / Öneriler) + Veri Kalitesi + Genelleme.

Sentezin can alıcı riski: KENDİ KANITI YOK, üstteki bölümlere dayanıyor. Üç risk:
  1. UYDURMA: sentez üstte olmayan yeni bir sayı yazabilir → detect_fabrication,
     izinli-sayı kümesi kurulmuş bölümlerden gelir (yapısal).
  2. "GÜVENİLİR" mayını: Güçlü/Zayıf "model güvenilir" demeye davet eder →
     detect_overstatement burada da; ölçüm kararlılığı ≠ model güvenilirliği.
  3. SPEKÜLASYON: Öneriler "şunu yaparsanız iyileşir" nedenselliğine açık →
     detect_speculation; öneri kanıta bağlı olmalı.

Ayrıca Veri Kalitesi ve Genelleme KANIT bölümleridir (kendi verileri var, sentez
değil) — Genelleme "güvenilir mayını"nın yapısal panzehiri: overfit verdict'i KOD
tarafından yazılır, LLM'in hükmüne bırakılmaz.
"""
import pytest

from src.core.report_schema import (
    ORDER_DATA_QUALITY, ORDER_GENERALIZATION, ORDER_SUMMARY, ORDER_RECOMMENDATIONS,
    Evidence, ReportSection, render_section,
)
from src.services.report_sections import (
    ORDER_ASSESSMENT,
    build_data_quality_section, build_generalization_section,
    build_summary_section, build_assessment_section, build_recommendations_section,
    collect_allowed_numbers, detect_fabrication, _build_digest,
)


# ═══ FABRICATION (yapısal anti-uydurma) ══════════════════════════════════════

def _built():
    """Kurulmuş kanıt bölümleri — sentezin dayanacağı 'üst'."""
    return [
        ReportSection(id="performance", title="Model Performansı", evidence=[
            Evidence(label="ROC-AUC", value=0.922, provenance="test_set"),
            Evidence(label="Doğruluk", value=0.8705, provenance="test_set"),
        ]),
        ReportSection(id="data_relations", title="Veri İlişkileri", evidence=[
            Evidence(label="x ↔ y", value=0.61, provenance="train_only", note="7 çift redundant"),
        ]),
    ]


def test_izinli_kume_kurulmus_bolumlerden_geliyor():
    allowed = collect_allowed_numbers(_built())
    assert "0.922" in allowed and "0.8705" in allowed and "0.61" in allowed
    # yüzde formu da meşru (0.922 → 92.2)
    assert "92.2" in allowed and "87.05" in allowed
    # note içindeki sayı da izinli (7 çift)
    assert "7" in allowed


def test_ustte_olmayan_sayi_UYDURMA_sayilir():
    allowed = collect_allowed_numbers(_built())
    # 0.95 hiçbir bölümde yok → uydurma
    assert detect_fabrication("Model 0.95 skorla mükemmel.", allowed) == ["0.95"]
    # 0.922 var → temiz
    assert detect_fabrication("Ayırt etme gücü 0.922.", allowed) == []


def test_yuzde_formu_uydurma_DEGIL():
    """0.8705 → '%87.05' meşru gösterim (Adım 4'te öğrenildi), uydurma değil."""
    allowed = collect_allowed_numbers(_built())
    assert detect_fabrication("Doğruluk %87.05.", allowed) == []
    assert detect_fabrication("Doğruluk yaklaşık %87.", allowed) == []  # 87 de var


def test_yuvarlama_toleransi_sinirlari():
    """Meşru yuvarlama temiz; gerçek değerin yuvarlaması OLMAYAN sayı uydurma.

    0.922 → 0.92 / %92 / (tam sayı yok). 0.8705 → 0.87 / %87.
    Ama 0.93, 0.95, %88 hiçbir üst değerin yuvarlaması değil → yakalanır.
    """
    allowed = collect_allowed_numbers(_built())
    for meşru in ("0.92", "%92", "0.87"):
        assert detect_fabrication(f"değer {meşru}", allowed) == [], f"meşru yuvarlama reddedildi: {meşru}"
    for uydurma in ("0.93", "0.95", "%88"):
        assert detect_fabrication(f"değer {uydurma}", allowed), f"uydurma kaçtı: {uydurma}"


def test_turkce_ondalik_virgul_uydurma_sayilmiyor():
    """LLM Türkçe prose'da '0,922' / '%92,2' yazar — virgül ondalık, binlik değil.

    Ayrım olmadan '92,2' → '922' olur, izinli '92.2' bulunmaz, doğru bir yüzde
    UYDURMA sanılırdı. '6,508' (binlik) ise silinip '6508' kalmalı.
    """
    allowed = collect_allowed_numbers(_built())
    # 0.922 ve 0.8705 üstte var → Türkçe ondalık formları temiz
    assert detect_fabrication("ayırt etme gücü 0,922", allowed) == []
    assert detect_fabrication("doğruluk %92,2", allowed) == []   # 0.922 yüzde formu
    assert detect_fabrication("doğruluk %87,05", allowed) == []  # 0.8705 yüzde formu
    # gerçek uydurma Türkçe virgülle de yakalanmalı
    assert detect_fabrication("skor 0,95", allowed) == ["0,95"]


def test_kucuk_tamsayilar_serbest():
    """'3 öneri', '2 risk' gibi sayaçlar uydurma sayılmaz."""
    allowed = collect_allowed_numbers(_built())
    assert detect_fabrication("3 güçlü yön ve 2 risk var.", allowed) == []


def test_digest_ust_bolumleri_iceriyor_ham_state_DEGIL():
    digest = _build_digest(_built(), "tr")
    assert "Model Performansı" in digest and "ROC-AUC" in digest
    assert "0.922" in digest
    # provenance etiketi digest'te (sentez kaynağı bilsin)
    assert "test seti" in digest


def test_not_computed_bolum_digestte_sebebiyle():
    secs = [ReportSection(id="fairness", title="Adalet", status="not_computed",
                          not_computed_reason="segment kolonu yok")]
    digest = _build_digest(secs, "tr")
    assert "segment kolonu yok" in digest


# ═══ VERİ KALİTESİ (kanıt bölümü) ════════════════════════════════════════════

def _dq_state(**over):
    st = {
        "intent": "classification",
        "profiling_report": {"shape": {"rows": 32561, "columns": 15},
                             "summary": {"missing_rate": 0.02}},
        "feature_report": {"n_train": 26029, "n_test": 6508, "n_features": 50},
        "evaluation_report": {},
    }
    st.update(over)
    return st


def test_veri_kalitesi_order_ve_kanit():
    sec = build_data_quality_section(_dq_state(), "tr")
    assert sec.order == ORDER_DATA_QUALITY and sec.status == "ok"
    labels = " ".join(e.label for e in sec.evidence)
    assert "boyut" in labels.lower() and "Modele giren" in labels


def test_karantina_GORUNUR_sebeple():
    """Kaç satır neden düşürüldü — sessiz düşme yok."""
    st = _dq_state(
        data_quality_status="degraded",
        data_quality_report={"quarantine": {"rows": 120, "total": 32561, "mismatch_rate": 0.004}},
    )
    sec = build_data_quality_section(st, "tr")
    q = next(e for e in sec.evidence if "arantina" in e.label)
    assert q.value == 120 and "120" in q.note
    # degraded → disclaimer
    assert "kısmen düşük" in sec.disclaimer


def test_ornekleme_gorunur():
    st = _dq_state(feature_report={"n_train": 100000, "n_test": 25000, "n_features": 40,
                                   "sampling": {"sampled": True, "sampled_rows": 100000,
                                                "original_rows": 500000, "cap": 100000}})
    sec = build_data_quality_section(st, "tr")
    assert any("örneklendi" in e.label.lower() for e in sec.evidence)


def test_profil_yoksa_gorunur_sebep():
    sec = build_data_quality_section({"intent": "classification", "evaluation_report": {}}, "tr")
    assert sec.status == "not_computed" and sec.not_computed_reason


# ═══ GENELLEME (güvenilir mayınının yapısal panzehiri) ═══════════════════════

def _gen_state(status="normal", gap=0.008, **ovf_extra):
    ovf = {"gap": gap, "status": status, "display_label": "CV (ROC-AUC)",
           "ci_lower": 0.91, "ci_upper": 0.93}
    ovf.update(ovf_extra)
    return {"intent": "classification",
            "evaluation_report": {"overfitting": ovf,
                                  "calibration": {"brier_score": 0.09}}}


def test_genelleme_verdict_KOD_tarafindan_deterministik():
    """'model güvenilir mi' LLM'e sorulmaz — CV−Test farkı deterministik yorumlanır."""
    sec = build_generalization_section(_gen_state("normal"), "tr")
    assert sec.order == ORDER_GENERALIZATION and sec.status == "ok"
    gen = next(e for e in sec.evidence if "genelleme" in e.label.lower())
    assert gen.value == 0.008
    assert "neredeyse test kadar iyi" in gen.note  # normal verdict metni


def test_genelleme_overfit_ve_inconclusive_ayri_metin():
    ovf = build_generalization_section(_gen_state("overfitting_detected", gap=0.15), "tr")
    inc = build_generalization_section(
        _gen_state("inconclusive", gap=0.05, n_test_positives=12), "tr")
    ovf_note = next(e for e in ovf.evidence if "genelleme" in e.label.lower()).note
    inc_note = next(e for e in inc.evidence if "genelleme" in e.label.lower()).note
    assert "belirgin düşüş" in ovf_note
    assert "kalın değil" in inc_note and "12 pozitif" in inc_note


def test_kalibrasyon_brier_kanit():
    sec = build_generalization_section(_gen_state(), "tr")
    assert any("Brier" in e.label for e in sec.evidence)


def test_genelleme_yoksa_gorunur_sebep():
    sec = build_generalization_section({"intent": "classification", "evaluation_report": {}}, "tr")
    assert sec.status == "not_computed"


# ═══ SENTEZ BÖLÜM İSKELETLERİ ════════════════════════════════════════════════

def test_sentez_bolumleri_order_ve_kanitsiz():
    """Özet/Öneriler'in kendi kanıtı YOK (headline/explanation sonradan LLM'den).
    DEĞERLENDİRME (assessment) İSTİSNA: Faz 2 Grup B'de zayıf yön/risk listesi
    DETERMİNİSTİK kanıt olarak taşınıyor (kod üretir, LLM anlatır) — koşmayan
    kontroller (fairness vb.) rule 3 gereği not-run kanıtı olarak her zaman görünür."""
    su = build_summary_section({"report_language": "tr"})
    ass = build_assessment_section({"report_language": "tr"})
    rec = build_recommendations_section({"report_language": "tr"})
    assert su.order == ORDER_SUMMARY
    assert ass.order == ORDER_ASSESSMENT
    assert rec.order == ORDER_RECOMMENDATIONS
    # Özet + Öneriler: kanıtsız (saf sentez)
    for s in (su, rec):
        assert s.evidence == [] and s.status == "ok"
    # Assessment: deterministik risk listesi taşır (en azından fairness not-run) — kanıtsız DEĞİL
    assert ass.status == "ok"
    assert any("fairness" in str(e.label).lower() for e in ass.evidence), \
        "assessment fairness not-run kanıtını taşımalı (rule 3)"


def test_ozet_en_ustte_oneriler_sonda():
    """Yerleşim: Özet(10) tepe, Değerlendirme(85), Öneriler(90) son."""
    assert ORDER_SUMMARY < ORDER_DATA_QUALITY < ORDER_ASSESSMENT < ORDER_RECOMMENDATIONS


# ═══ SENTEZ GUARD KAYITLARI ══════════════════════════════════════════════════

def test_sentez_dusunce_BOS_BASLIK_birakmiyor(monkeypatch):
    """Sentezin kanıtı yok — prose düşerse '## Özet' boş başlık kalırdı. Sebep yaz."""
    import src.services.report_sections as RS

    class _FakeLLM:
        def with_structured_output(self, schema):
            return self
        def invoke(self, msgs):
            # Her seferinde UYDURMA sayı → guard iki denemede de reddeder
            return {"headline": "Skor 0.999 mükemmel.", "explanation": "Model 0.999 aldı."}

    monkeypatch.setattr(RS, "get_llm", lambda **k: _FakeLLM())
    built = [ReportSection(id="performance", title="Perf",
                           evidence=[Evidence(label="ROC", value=0.922, provenance="test_set")])]
    sec = RS.build_synthesis_prose(RS.build_summary_section({"report_language": "tr"}),
                                   built, {"report_language": "tr", "user_query": "?"}, "tr")
    assert sec.headline == "" and sec.explanation == ""
    assert sec.status == "not_computed" and sec.not_computed_reason
    md = render_section(sec, "tr")
    assert "## Özet" in md
    assert "\n## \n" not in md and not md.rstrip().endswith("## ")
    # Boş başlık artığı YOK: başlığın altında sebep var
    assert "üretilemedi" in md


def test_sentez_basari_yolunda_prose_yaziliyor(monkeypatch):
    import src.services.report_sections as RS

    class _FakeLLM:
        def with_structured_output(self, schema):
            return self
        def invoke(self, msgs):
            return {"headline": "Model güçlü ayrım yapıyor.",
                    "explanation": "Ayırt etme gücü 0.922 ile yüksek."}

    monkeypatch.setattr(RS, "get_llm", lambda **k: _FakeLLM())
    built = [ReportSection(id="performance", title="Perf",
                           evidence=[Evidence(label="ROC", value=0.922, provenance="test_set")])]
    sec = RS.build_synthesis_prose(RS.build_summary_section({"report_language": "tr"}),
                                   built, {"report_language": "tr", "user_query": "?"}, "tr")
    assert sec.status == "ok"
    assert "0.922" in sec.explanation and sec.headline


def test_sentez_guard_haritasi():
    from src.services.report_sections import (
        _SYNTHESIS_GUARDS, detect_causal_language, detect_overstatement,
        detect_speculation,
    )

    def _fns(sid):
        return [fn for fn, _suffix in _SYNTHESIS_GUARDS.get(sid, [])]

    # HER sentez bölümü nedensellik (ya da onu İÇEREN spekülasyon) guard'ı taşımalı.
    # Gerçek koşuda Özet hiç dil-guard'ı almıyordu → "significantly influence churn"
    # ve "likely ... play a crucial role" kaçtı. Bu testin varlık sebebi o regresyon.
    assert detect_causal_language in _fns("summary")
    # Güçlü/Zayıf → nedensellik + abartma (güvenilir mayını)
    assert detect_causal_language in _fns("assessment")
    assert detect_overstatement in _fns("assessment")
    # Öneriler → spekülasyon (nedenselliği içerir; en sıkı yer)
    assert detect_speculation in _fns("recommendations")


def test_sentez_ozet_nedensel_dili_yakalar():
    """Özet bölümünde nedensel dil artık guard'a takılır (eskiden hiç kontrol edilmiyordu)."""
    from src.services.report_sections import _SYNTHESIS_GUARDS, detect_causal_language
    guards = _SYNTHESIS_GUARDS["summary"]
    prose = "These factors significantly influence churn and play a crucial role."
    hits = []
    for fn, _suffix in guards:
        hits += fn(prose)
    assert hits, "Özet guard'ı nedensel dili yakalamalı"


# ═══════════════════════════════════════════════════════════════════════════
#  ADIM 8 — SAYI FORMATI: prose'da oran=%, skor=ondalık; kanıt katmanı kesin
# ═══════════════════════════════════════════════════════════════════════════

def test_oran_metrikleri_yuzde_skorlar_ondalik_digestte():
    """LLM'in gördüğü digest: accuracy/recall YÜZDE, ROC-AUC/MCC ONDALIK."""
    from src.services.report_sections import _build_digest
    secs = [ReportSection(id="performance", title="Model Performansı", evidence=[
        Evidence(label="Doğruluk", value=0.8385, provenance="test_set"),
        Evidence(label="ROC-AUC", value=0.922, provenance="test_set"),
        Evidence(label="Recall", value=0.64, provenance="test_set"),
        Evidence(label="MCC", value=0.62, provenance="test_set"),
    ])]
    digest = _build_digest(secs, "en")
    assert "83.9%" in digest, "oran metriği (accuracy) yüzde olmalı"
    assert "64%" in digest, "oran metriği (recall) yüzde olmalı"
    assert "0.922" in digest and "0.62" in digest, "skorlar (ROC-AUC/MCC) ondalık kalmalı"
    assert "0.8385" not in digest, "oran prose'da ham ondalık GÖSTERİLMEMELİ"


def test_kanit_katmani_ondalik_gosterir_yuzde_degil():
    """Kanıt katmanı (render_evidence) skoru ONDALIK gösterir, YÜZDE değil.

    Faz 3 #4 ondalık disiplini: skorlar 3 haneye yuvarlanır (0.8385 → 0.838).
    Bu, digest'in yüzde göstermesinden (test_..._digestte) ayrı bir katman —
    kanıt hâlâ ondalık, yüzdeye çevrilmez. (Model-karşılaştırma bölümü 4 hane
    tutar; o değeri STRING olarak verir, bu genel kurala girmez.)
    """
    from src.core.report_schema import render_evidence, _fmt_value
    line = render_evidence(Evidence(label="Doğruluk", value=0.8385, provenance="test_set"), "en")
    assert _fmt_value(0.8385) in line, "kanıt katmanı skoru ondalık (3 hane) göstermeli"
    assert "%" not in line, "kanıt katmanı yüzdeye çevrilmemeli"


def test_yuzde_formu_uydurma_sayilmaz():
    """Digest yüzde gösterince (%86 / 86.4% / 83.9%) fabrication guard tetiklenmez."""
    from src.services.report_sections import collect_allowed_numbers, detect_fabrication
    secs = [ReportSection(id="performance", title="Perf", evidence=[
        Evidence(label="Doğruluk", value=0.8385, provenance="test_set"),
    ])]
    allowed = collect_allowed_numbers(secs)
    for meşru in ("83.9%", "%83,9", "yaklaşık %84", "84%"):
        assert detect_fabrication(meşru, allowed) == [], f"{meşru} meşru yüzde formu"


def test_rate_metrik_siniflandirici():
    """Oran metrikleri True, skor/oran-oranı False."""
    from src.services.report_sections import _is_rate_metric
    for rate in ("Accuracy", "Doğruluk", "Recall (churn)", "Precision",
                 "yanlış alarm oranı (FPR)", "kaçırma oranı (FNR)", "Hatalı tahmin oranı"):
        assert _is_rate_metric(rate), f"{rate} oran metriği olmalı"
    for score in ("ROC-AUC", "PR-AUC", "MCC", "F1", "seçim oranı oranı (DI)",
                  "verideki gerçek oran oranı", "yanlış alarm farkı", "ayrışma büyüklüğü"):
        assert not _is_rate_metric(score), f"{score} skor/oran-oranı olmalı (ondalık)"


# ═══ ADIM 9 — "YENİDEN İFADE EDİLMİŞ GÜÇLÜ YÖN" (Güçlü/Zayıf yapısal koruması) ═══
# Bu haftanın üçüncü prompt-yetmez dersi: LLM zayıflık listesine güçlü yön yazıyor.
# Dedektör YAPISAL karşılıktır. En kritik kısım OLUMSUZLAMA yönü: "overfit riski var"
# gerçek zayıflıktır; "overfit yok" değildir.

# YAKALANMALI — güçlü yön risk/zayıflık gibi sunulmuş (iki dil)
_RESTATED_FIRE = [
    "Zayıf yön: model iyi genelliyor.",
    "Risk: overfit görülmedi.",
    "Zayıf yönler: yüksek doğruluk ve dengeli sınıflar.",
    "Zayıf yön / risk: overfit yok, model sağlam.",
    "A key weakness is the strong, stable performance.",
    "Risk: the model shows no overfitting.",
    "Weakness: high discrimination and good generalization.",
]

# YAKALANMAMALI — gerçek zayıflık, "risk yok" beyanı, ya da çerçeve yok (iki dil)
_RESTATED_CLEAN = [
    "Zayıf yön: overfit riski var.",                     # gerçek zayıflık (olumsuzlama tuzağı)
    "Zayıf yön: azınlık sınıfta recall düşük.",
    "Weakness: recall on the minority class is low.",
    "Risk: aşırı öğrenme tespit edildi, fark yüksek.",
    "Overfitting is present; the gap is wide.",
    "Güçlü yön: model iyi genelliyor ve overfit yok.",   # çerçeve YOK (güçlü yön bölümü)
    "The model generalizes well and shows no overfitting.",
    "Güçlü yönler: yüksek skor, dengeli sınıflar.",
    "Belirgin bir risk tespit edilmedi; model iyi genelliyor.",  # "risk yok" beyanı
    "No significant risk was detected; the model generalizes well.",
    "Model dengeli ama azınlık sınıfta zayıf.",          # ödünleme (gerçek olumsuz var)
    "**Zayıf yönler / riskler:** Bu çalışmada belirgin bir risk tespit edilmedi.",  # deterministik satır
    "**Weaknesses / risks:** No significant risk was detected in this run.",
]


@pytest.mark.parametrize("prose", _RESTATED_FIRE)
def test_restated_strength_yakalaniyor(prose):
    from src.services.report_sections import detect_restated_strength
    assert detect_restated_strength(prose), f"güçlü-yön-risk-gibi yakalanmalı: {prose!r}"


@pytest.mark.parametrize("prose", _RESTATED_CLEAN)
def test_restated_strength_yanlis_alarm_vermiyor(prose):
    from src.services.report_sections import detect_restated_strength
    assert detect_restated_strength(prose) == [], f"YANLIŞ ALARM: {prose!r}"


def test_restated_strength_guard_haritasina_bagli():
    """Güçlü/Zayıf bölümü artık detektörü guard olarak taşımalı (yapısal boşluk kapandı)."""
    from src.services.report_sections import _SYNTHESIS_GUARDS, detect_restated_strength
    fns = [fn for fn, _ in _SYNTHESIS_GUARDS["assessment"]]
    assert detect_restated_strength in fns


def test_risk_yoksa_deterministik_satir_KOD_tarafindan_basiliyor(monkeypatch):
    """Gerçek risk yoksa zayıflık satırı LLM'e BIRAKILMAZ — kod deterministik basar."""
    import src.services.report_sections as RS
    from src.services.report_sections import (
        build_assessment_section, build_synthesis_prose, detect_restated_strength)

    class _Fake:
        def with_structured_output(self, s): return self
        def invoke(self, m):
            return {"headline": "The model discriminates strongly.",
                    "explanation": "It reaches ROC-AUC 0.94 on the held-out test set."}
    monkeypatch.setattr(RS, "get_llm", lambda **k: _Fake())

    state = {"report_language": "en", "user_query": "?", "evaluation_report": {
        "overfitting": {"status": "normal"}, "target_lock_leakage": [],
        "feature_importance_summary": {"leakage_suspects": []}}}
    built = [ReportSection(id="performance", title="Perf", evidence=[
        Evidence(label="ROC-AUC", value=0.94, provenance="test_set", sample_size=5000)])]
    sec = build_synthesis_prose(build_assessment_section(state), built, state, "en")
    assert sec.status == "ok"
    assert "No significant risk was detected" in sec.explanation
    # Eklenen deterministik satır kendi guard'ını tetiklememeli
    assert detect_restated_strength(sec.explanation) == []


def test_gercek_risk_varsa_no_risk_satiri_BASILMAZ():
    """İnce/güvenilmez kanıt (gerçek sınır) varsa deterministik 'risk yok' tetiklenmez."""
    from src.services.report_sections import _assessment_no_genuine_risk
    thin = [ReportSection(id="fairness", title="Fair", evidence=[
        Evidence(label="FPR (grup X)", value=0.4, provenance="test_set", sample_size=8)])]  # n<30
    state = {"evaluation_report": {"overfitting": {"status": "normal"},
             "target_lock_leakage": [], "feature_importance_summary": {"leakage_suspects": []}}}
    assert _assessment_no_genuine_risk(thin, state) is False


def test_no_risk_tarama_sizinti_ve_overfiti_gorur():
    """Deterministik tarama muhafazakâr: sızıntı ya da overfit varsa 'risk yok' DEMEZ."""
    from src.services.report_sections import _assessment_no_genuine_risk
    clean = [ReportSection(id="performance", title="P", evidence=[
        Evidence(label="ROC-AUC", value=0.94, provenance="test_set", sample_size=5000)])]
    base = {"overfitting": {"status": "normal"}, "target_lock_leakage": [],
            "feature_importance_summary": {"leakage_suspects": []}}
    assert _assessment_no_genuine_risk(clean, {"evaluation_report": base}) is True
    # overfit → risk
    ov = {**base, "overfitting": {"status": "mild_overfitting"}}
    assert _assessment_no_genuine_risk(clean, {"evaluation_report": ov}) is False
    # target-lock sızıntısı → risk
    tl = {**base, "target_lock_leakage": [{"feature": "x", "confidence": "high"}]}
    assert _assessment_no_genuine_risk(clean, {"evaluation_report": tl}) is False
    # importance-suspect → risk
    su = {**base, "feature_importance_summary": {"leakage_suspects": ["x"]}}
    assert _assessment_no_genuine_risk(clean, {"evaluation_report": su}) is False
