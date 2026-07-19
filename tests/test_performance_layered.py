"""Model Performansı bölümü — katmanlı yapı testleri.

Bu bölümün ölçülmüş risk sınıfları:
  1. Eksik metrik 0 sayılır  → metrics_unavailable sessizce düşerse rapor
     "ölçmedik"i "iyi"ye çevirir.
  2. per_class oranı sayısız → "class 1 recall 0.52" okuyucuya hiçbir şey demez.
  3. Çok sınıflıda GA anahtarı kaymış (test_metrics 'roc_auc_ovr' / CI 'roc_auc')
     → aralık sessizce basılmaz, metrik olduğundan güvenilir görünür.
  4. LLM hüküm verir ("üretime hazır") → kanıtın taşımadığı iddia.
"""
import pytest

from src.core.report_schema import ORDER_PERFORMANCE, render_section, render_table
from src.services.report_sections import (
    build_performance_section,
    detect_overstatement,
)


# ── Fixture'lar ──────────────────────────────────────────────────────────────

def _binary_state(**over):
    """Dengesiz ikili sınıflandırma: 5,000 negatif / 1,700 pozitif."""
    ev = {
        "task": "binary_classification",
        "test_metrics": {
            "accuracy": 0.88, "f1_weighted": 0.87, "f1_macro": 0.79,
            "precision_weighted": 0.87, "recall_weighted": 0.88, "mcc": 0.62,
            "roc_auc": 0.922, "pr_auc": 0.78, "pr_auc_baseline": 0.2537,
        },
        "confusion_matrix": {"matrix": [[4700, 300], [816, 884]], "labels": [0, 1]},
        "per_class_metrics": {
            "0": {"precision": 0.852, "recall": 0.94, "f1": 0.894, "support": 5000},
            "1": {"precision": 0.7466, "recall": 0.52, "f1": 0.613, "support": 1700},
        },
        "confidence_intervals": {
            "roc_auc": {"mean": 0.922, "lower": 0.914, "upper": 0.930},
            "accuracy": {"mean": 0.88, "lower": 0.872, "upper": 0.888},
        },
    }
    ev.update(over.pop("evaluation_report", {}))
    state = {"intent": "classification", "evaluation_report": ev}
    state.update(over)
    return state


# ── 1. Temel yapı ────────────────────────────────────────────────────────────

def test_bolum_dogru_siraya_oturuyor():
    """order verilmezse blob'un içine (50) düşer — fairness'ta tam bu olmuştu."""
    sec = build_performance_section(_binary_state(), "tr")
    assert sec.order == ORDER_PERFORMANCE
    assert sec.order < 50, "performans blob'un ARKASINA düşüyor"


def test_metrikler_kanit_olarak_test_setinden_geliyor():
    sec = build_performance_section(_binary_state(), "tr")
    assert sec.status == "ok"
    assert all(e.provenance == "test_set" for e in sec.evidence)
    labels = " ".join(e.label for e in sec.evidence)
    assert "ROC-AUC" in labels and "Doğruluk" in labels


def test_her_metrik_duz_dilli_aciklama_tasiyor():
    """Metrik adı tek başına okuyucuya hiçbir şey söylemez."""
    sec = build_performance_section(_binary_state(), "tr")
    metrik_kanitlari = [e for e in sec.evidence if "ROC-AUC" in e.label or "Doğruluk" in e.label]
    assert metrik_kanitlari
    for e in metrik_kanitlari:
        assert len(e.note) > 40, f"'{e.label}' açıklamasız kalmış"


# ── 2. Güven aralığı eşleşmesi ───────────────────────────────────────────────

def test_roc_araligi_metrigin_yanina_basiliyor():
    sec = build_performance_section(_binary_state(), "tr")
    roc = next(e for e in sec.evidence if "ROC-AUC" in e.label)
    assert roc.uncertainty == {"lower": 0.914, "upper": 0.930, "method": "bootstrap"}
    assert roc.reliable is True


def test_cok_sinifli_roc_ovr_araligi_KAYMADAN_eslesiyor():
    """test_metrics 'roc_auc_ovr' yazar, bootstrap 'roc_auc' tutar.

    Eşleştirilmezse aralık sessizce düşer: metrik görünür, belirsizliği görünmez.
    """
    st = _binary_state()
    ev = st["evaluation_report"]
    ev["test_metrics"].pop("roc_auc")
    ev["test_metrics"]["roc_auc_ovr"] = 0.83
    ev["confidence_intervals"] = {"roc_auc": {"mean": 0.83, "lower": 0.81, "upper": 0.85}}

    sec = build_performance_section(st, "tr")
    ovr = next(e for e in sec.evidence if "ROC-AUC" in e.label)
    assert ovr.value == 0.83
    assert ovr.uncertainty is not None, "OVR aralığı SESSİZCE düştü"
    assert ovr.uncertainty["lower"] == 0.81


def test_birimli_metrige_oran_esigi_UYGULANMIYOR():
    """RMSE 12,500 için 'aralık 0.30'dan geniş → güvenilmez' saçmalık olurdu."""
    st = {
        "intent": "regression",
        "evaluation_report": {
            "test_metrics": {"r2": 0.81, "rmse": 12500.0, "mae": 9000.0},
            "confidence_intervals": {},
        },
        "feature_report": {"n_test": 4000},
    }
    sec = build_performance_section(st, "tr")
    rmse = next(e for e in sec.evidence if "RMSE" in e.label)
    assert rmse.uncertainty is None
    assert rmse.reliable is True, "birimli metrik oran eşiğine takılmış"


# ── 3. PR-AUC taban çizgisi (hüküm değil, referans) ──────────────────────────

def test_pr_auc_taban_cizgisiyle_birlikte_veriliyor():
    """0.78 tek başına okunamaz: pozitif sınıf %25 ise anlamı bambaşkadır."""
    sec = build_performance_section(_binary_state(), "tr")
    pr = next(e for e in sec.evidence if "PR-AUC" in e.label)
    assert "0.2537" in pr.reference
    assert "rastgele" in pr.reference.lower()


def test_pr_auc_baseline_kendi_basina_metrik_olarak_BASILMIYOR():
    """Taban çizgisi bir başarı ölçüsü değil, referanstır."""
    sec = build_performance_section(_binary_state(), "tr")
    assert not any("baseline" in e.label.lower() for e in sec.evidence)


# ── 4. Confusion matrix okunabilirliği ───────────────────────────────────────

def test_ikili_matris_hucreleri_ADLANDIRILIYOR():
    """'[[4700, 300], [816, 884]]' okuyucuya hiçbir şey söylemez."""
    sec = build_performance_section(_binary_state(), "tr")
    md = render_table(sec.table)
    assert "yanlış alarm" in md and "kaçırılan" in md
    assert "4,700" in md and "300" in md and "816" in md and "884" in md
    # 0/1 -> hayır/evet: "Gerçekte: 1" okuyucuya hiçbir şey söylemez
    assert "Gerçekte: hayır" in md and "Model: evet dedi" in md


def test_anlamli_etiketlere_isim_UYDURULMUYOR():
    """'churn'/'stay' zaten okunabilir — onları 'evet/hayır'a çevirmek YANLIŞ olurdu."""
    st = _binary_state()
    st["evaluation_report"]["confusion_matrix"] = {
        "matrix": [[900, 100], [50, 200]], "labels": ["stay", "churn"],
    }
    st["evaluation_report"]["per_class_metrics"] = {
        "stay": {"precision": 0.95, "recall": 0.9, "f1": 0.92, "support": 1000},
        "churn": {"precision": 0.67, "recall": 0.8, "f1": 0.73, "support": 250},
    }
    sec = build_performance_section(st, "tr")
    md = render_section(sec, "tr")
    assert "Model: churn dedi" in md and "Gerçekte: stay" in md
    assert "Model: evet dedi" not in md, "anlamlı etikete isim uydurulmuş"


def test_ucten_fazla_sinifta_etiket_cevrilmiyor():
    """Çeviri sözleşmesi YALNIZCA {0,1} için geçerli — 0/1/2'de 1='evet' anlamsız."""
    st = _binary_state()
    st["evaluation_report"]["confusion_matrix"] = {
        "matrix": [[9, 1, 0], [1, 8, 1], [0, 2, 8]], "labels": [0, 1, 2],
    }
    st["evaluation_report"]["per_class_metrics"] = {}
    sec = build_performance_section(st, "tr")
    md = render_table(sec.table)
    assert "Model: 1 dedi" in md
    assert "evet" not in md


def test_matris_sayilari_UYDURULMUYOR():
    """Tablodaki her sayı confusion_matrix'ten AYNEN gelmeli."""
    st = _binary_state()
    st["evaluation_report"]["confusion_matrix"]["matrix"] = [[1, 2], [3, 4]]
    sec = build_performance_section(st, "tr")
    md = render_table(sec.table)
    for n in ("1", "2", "3", "4"):
        assert n in md


def test_cok_sinifli_matris_kosegeni_isaretliyor():
    st = _binary_state()
    st["evaluation_report"]["confusion_matrix"] = {
        "matrix": [[50, 3, 2], [4, 40, 6], [1, 7, 30]], "labels": ["A", "B", "C"],
    }
    st["evaluation_report"]["per_class_metrics"] = {}
    sec = build_performance_section(st, "tr")
    md = render_table(sec.table)
    assert "✅ 50" in md and "✅ 40" in md and "✅ 30" in md
    assert "karıştırıldığını" in sec.table["caption"]


def test_bozuk_matris_coktürmüyor_sessizce_atlaniyor():
    """labels ile matrix uyuşmuyorsa tablo kurulmaz — yarım tablo basmaktansa yok."""
    st = _binary_state()
    st["evaluation_report"]["confusion_matrix"] = {"matrix": [[1, 2]], "labels": [0, 1, 2]}
    sec = build_performance_section(st, "tr")
    assert sec.table is None
    assert sec.status == "ok", "metrikler hâlâ raporlanmalı"


# ── 5. per_class → iş dili ───────────────────────────────────────────────────

def test_recall_SAYIYA_cevriliyor():
    """'recall 0.52' yerine 'gerçekte 1 olan 1,700 kayıttan 884'ünü yakaladı'."""
    sec = build_performance_section(_binary_state(), "tr")
    pc = next(s for s in sec.subsections if s.id == "performance_per_class")
    recall_1 = next(e for e in pc.evidence if "'evet' sınıfı" in e.label and "yakalama" in e.label)
    assert recall_1.value == 0.52
    assert "1,700" in recall_1.note and "884" in recall_1.note
    assert "'evet'" in recall_1.note, "kanıt ham etiketi kullanıyor, tabloyla uyuşmuyor"


def test_precision_SAYIYA_cevriliyor():
    sec = build_performance_section(_binary_state(), "tr")
    pc = next(s for s in sec.subsections if s.id == "performance_per_class")
    prec_1 = next(e for e in pc.evidence if "'evet' sınıfı" in e.label and "isabet" in e.label)
    # modelin '1' dediği = sütun toplamı = 300 + 884 = 1,184
    assert "1,184" in prec_1.note and "884" in prec_1.note


def test_per_class_sayilari_MATRISTEN_geliyor_carpimla_uretilmiyor():
    """recall x support yuvarlama gürültüsü üretir; kanıt katmanı bunu kaldırmaz.

    0.52 * 1700 = 884.0 ama 0.5203 * 1700 = 884.5 → '884' mü '885' mi?
    Matristen okunursa soru hiç doğmaz.
    """
    st = _binary_state()
    st["evaluation_report"]["per_class_metrics"]["1"]["recall"] = 0.5203  # kasıtlı tutarsız
    sec = build_performance_section(st, "tr")
    pc = next(s for s in sec.subsections if s.id == "performance_per_class")
    recall_1 = next(e for e in pc.evidence if "'evet' sınıfı" in e.label and "yakalama" in e.label)
    assert "884" in recall_1.note, "sayı matristen değil çarpımdan üretilmiş"
    assert "885" not in recall_1.note


def test_ince_sinif_ORNEKLEMLE_damgalaniyor():
    """3 kayıttan 2'si = '%67 yakalama' — tek başına yargıya dayanak OLAMAZ."""
    st = _binary_state()
    st["evaluation_report"]["confusion_matrix"] = {"matrix": [[5000, 10], [1, 2]], "labels": [0, 1]}
    st["evaluation_report"]["per_class_metrics"] = {
        "0": {"precision": 0.99, "recall": 0.998, "f1": 0.99, "support": 5010},
        "1": {"precision": 0.1667, "recall": 0.6667, "f1": 0.26, "support": 3},
    }
    sec = build_performance_section(st, "tr")
    pc = next(s for s in sec.subsections if s.id == "performance_per_class")
    recall_1 = next(e for e in pc.evidence if "'evet' sınıfı" in e.label and "yakalama" in e.label)
    assert recall_1.sample_size == 3
    assert recall_1.reliable is False, "3 kayıtlık oran güvenilir sayılıyor!"
    assert "Ölçüm ince" in render_section(pc, "tr")


def test_matris_yoksa_per_class_alt_bolumu_ACILMIYOR():
    """Sayıya çeviremeyeceksek çıplak oran basmanın değeri yok."""
    st = _binary_state()
    st["evaluation_report"].pop("confusion_matrix")
    sec = build_performance_section(st, "tr")
    assert not any(s.id == "performance_per_class" for s in sec.subsections)


# ── 6. Sessiz sıfır yasağı ───────────────────────────────────────────────────

def test_uretilemeyen_metrikler_GORUNUR_sebeple_geliyor():
    """Olasılık üretilemediyse ROC/PR-AUC YOKTUR — 0 değil, yok."""
    st = _binary_state()
    ev = st["evaluation_report"]
    for k in ("roc_auc", "pr_auc", "pr_auc_baseline"):
        ev["test_metrics"].pop(k, None)
    ev["confidence_intervals"] = {}
    ev["metrics_unavailable"] = {
        "reason": "predict_proba_failed",
        "detail": "AttributeError: 'SVC' object has no attribute 'predict_proba'",
        "affected": ["roc_auc", "pr_auc", "log_loss", "calibration"],
    }
    sec = build_performance_section(st, "tr")

    assert not any("ROC-AUC" in e.label for e in sec.evidence), "olmayan metrik basılmış"
    sub = next(s for s in sec.subsections if s.id == "performance_unavailable")
    assert len(sub.evidence) == 4
    assert all(e.value is None for e in sub.evidence)
    assert all("predict_proba" in e.note for e in sub.evidence)

    md = render_section(sub, "tr")
    assert "iyi olduğu anlamına" in md, "eksik metrik iyi habere çevrilebilir"
    assert "0.000" not in md


def test_test_metrigi_yoksa_SEBEP_zorunlu():
    st = {"intent": "classification", "evaluation_report": {}}
    sec = build_performance_section(st, "tr")
    assert sec.status == "not_computed"
    assert sec.not_computed_reason
    assert "ölçülemedi" in sec.not_computed_reason


def test_denetimsiz_analizde_TANIMSIZ_deniyor_eksik_degil():
    """Clustering'de 'test metrikleri üretilmedi' YANLIŞ çerçeve — soru tanımsız."""
    sec = build_performance_section({"intent": "clustering", "evaluation_report": {}}, "tr")
    assert sec.status == "not_applicable"
    assert "tanımlı değil" in sec.not_computed_reason
    assert "üretilmedi" not in sec.not_computed_reason


# ── 7. Model karşılaştırması ─────────────────────────────────────────────────

def test_diger_modeller_CV_etiketiyle_geliyor():
    st = _binary_state(
        model_name="lightgbm",
        metrics={"all_model_scores": {"lightgbm": 0.92, "logistic": 0.86, "rf": 0.90}},
    )
    sec = build_performance_section(st, "tr")
    cmp = next(s for s in sec.subsections if s.id == "performance_model_comparison")
    assert [e.value for e in cmp.evidence] == [0.92, 0.90, 0.86], "skora göre sıralı değil"
    assert "← seçilen" in cmp.evidence[0].label
    assert all(e.provenance == "cv" for e in cmp.evidence), "CV skoru test seti sanılıyor"


# ── 8. Abartma dedektörü ─────────────────────────────────────────────────────

@pytest.mark.parametrize("cumle", [
    "Model üretime hazır durumda.",
    "Bu skorlarla model canlıya alınabilir.",
    "The model is production-ready.",
    "Model kusursuz çalışıyor.",
    "Tahminler hatasız.",
    "Model her zaman doğru sonuç veriyor.",
    "Sonuçlar garanti altındadır.",
    "Model tamamen güvenilir.",
    "The model is fully reliable.",
    # GERÇEK LLM ÇIKTISI (ölçüldü): dar kapsam bunu geçirmişti.
    # "güvenilir" bu projenin mayını — uydurulan status rozeti de tam buydu.
    "Modelin Performansı Yüksek ve Güvenilir",
    "Modelin genel performansı güvenilir bir düzeyde.",
    "The model is reliable at this level.",
    "Model güvenilirdir.",
])
def test_hukum_yakalaniyor(cumle):
    assert detect_overstatement(cumle), f"kaçtı: {cumle}"


def test_olcum_kararliligi_ile_model_guvenilirligi_AYRILIYOR():
    """Dar aralık ÖLÇÜMÜN kararlı olduğunu söyler, MODELİN güvenilir olduğunu değil.

    Vasat bir modelin kararlı ölçümü hâlâ vasattır — iki iddia karıştırılamaz.
    """
    assert detect_overstatement("Skor tekrarlı örneklemede kararlı; band dar.") == []
    assert detect_overstatement("The score is stable across resamples.") == []
    assert detect_overstatement("Ölçüm güvenilir bir aralıkta.") != []


def test_guvenilirlik_skoru_kelimesi_yanlis_alarm_vermiyor():
    """'güvenilirlik' (skor adı) != 'güvenilir' (hüküm) — \\b eki ayırmalı."""
    assert detect_overstatement("Güvenilirlik skoru 70/100 olarak hesaplandı.") == []


@pytest.mark.parametrize("cumle", [
    # OLUMSUZLANMIŞ — istediğimiz çerçevenin ta kendisi
    "Bu skor modelin üretime hazır olduğunu göstermez.",
    "Model kusursuz değildir; hataların yoğunlaştığı alanlar var.",
    "These numbers do not make the model production-ready.",
    "The model is not fully reliable on the smaller class.",
    # Skoru niteleyen sıfatlar YASAK DEĞİL (blob'da da serbest — tutarlılık)
    "Model güçlü bir ayırt etme gücü gösteriyor (ROC-AUC 0.922).",
    "0.922 yüksek bir ayırt etme gücüdür.",
    "Model, gerçek pozitiflerin yaklaşık yarısını yakalıyor.",
])
def test_yanlis_alarm_yok(cumle):
    assert detect_overstatement(cumle) == [], f"yanlış alarm: {cumle}"


def test_dedektor_performans_bolumune_bagli():
    from src.services.report_sections import _PROSE_GUARDS
    assert _PROSE_GUARDS["performance"][0] is detect_overstatement


# ── 9. LLM sınırı (yapısal) ──────────────────────────────────────────────────

def test_llm_semasi_tabloyu_ICERMIYOR():
    """Tablo kod tarafından kurulur — LLM hücre uyduramaz çünkü alanı yok."""
    from src.core.report_schema import LLM_SECTION_SCHEMA
    assert set(LLM_SECTION_SCHEMA["properties"]) == {"headline", "explanation"}


def test_render_tam_bolumu_basiyor():
    sec = build_performance_section(_binary_state(), "tr")
    md = render_section(sec, "tr")
    assert "## Model Performansı" in md
    assert "yanlış alarm" in md          # tablo
    assert "0.922" in md                 # metrik
    assert "%95 aralık" in md            # belirsizlik
    assert "Sınıf başına başarı" in md   # alt bölüm
    assert "\n## \n" not in md and not md.rstrip().endswith("## ")
