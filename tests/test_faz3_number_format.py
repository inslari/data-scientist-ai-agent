"""Faz 3 #4 — ondalık disiplini (BÜYÜKLÜK-tabanlı, dinamik; kolon/veri seti yaması YOK).

Kural:
  • |v| >= 1000  → binlik ayraç + 2 hane (para / hedef-birimli büyük değer)
  • 1 <= |v| <1000 → 3 hane, sondaki sıfırlar atılır
  • |v| < 1       → 3 hane (0-1 skorları)
İSTİSNA: Model-karşılaştırma bölümü 3.–4. hanenin iki modeli AYIRDIĞI tek yerdir;
değeri 4-haneli STRING olarak verir → genel kurala tabi olmaz.

Standalone + pytest uyumlu: `uv run pytest tests/test_faz3_number_format.py -q`
"""
import pytest

from src.core.report_schema import Evidence, ReportSection, _fmt_value, render_evidence, render_section
from src.services.report_sections import build_performance_section


# ── _fmt_value büyüklük kuralı (uydurma değerler → dinamiklik kanıtı) ─────────

@pytest.mark.parametrize("value,expected", [
    (91571.9019, "91,571.90"),     # para / büyük değer → binlik ayraç, 2 hane
    (1000.0, "1,000.00"),          # eşik tam üstü
    (12345.678, "12,345.68"),      # yuvarlama 2 haneye
    (123.4, "123.4"),              # <1000: 3 hane, sondaki sıfır atılır
    (91.5, "91.5"),                # 1-1000: 3 hane, sondaki sıfır atılır
    (1.2503, "1.25"),              # Cohen's d / oran-oranı
    (0.8535, "0.854"),             # skor → 3 hane (float yuvarlaması yukarı)
    (0.922, "0.922"),              # zaten 3 hane
    (0.2537, "0.254"),             # 3 haneye
    (0.5, "0.5"),                  # sondaki sıfırlar atılır
    (-0.0062, "-0.006"),           # negatif gap
])
def test_fmt_value_buyukluk_kurali(value, expected):
    assert _fmt_value(value) == expected


def test_fmt_value_int_binlik_ayrac():
    assert _fmt_value(6700) == "6,700"
    assert _fmt_value(1700) == "1,700"


def test_fmt_value_onceden_bicimlenmis_string_aynen_basar():
    """Model-karşılaştırma bölümü değeri STRING verir → _fmt_value dokunmaz."""
    assert _fmt_value("0.8581") == "0.8581"
    assert _fmt_value("0.8569") == "0.8569"


def test_render_evidence_skor_uc_hane():
    """Kanıt satırı skoru 3 haneye yuvarlar (91571.9019 belasının skor tarafı)."""
    line = render_evidence(Evidence(label="ROC-AUC", value=0.85351, provenance="test_set"), "en")
    assert "0.854" in line and "0.85351" not in line


def test_render_evidence_para_binlik_ayrac():
    line = render_evidence(Evidence(label="Ortalama tutar", value=91571.9019, provenance="test_set"), "tr")
    assert "91,571.90" in line and "91571.9019" not in line


# ── Model-karşılaştırma İSTİSNASI: 4 hane KORUNUR (bölüm-tipine özel) ─────────

def _perf_state_with_models(scores: dict, chosen: str):
    return {
        "intent": "classification",
        "model_name": chosen,
        "metrics": {"all_model_scores": scores},
        "evaluation_report": {
            "task": "binary_classification",
            "test_metrics": {"accuracy": 0.88, "f1_weighted": 0.87, "roc_auc": 0.922, "mcc": 0.62},
            "confusion_matrix": {"matrix": [[4700, 300], [816, 884]], "labels": [0, 1]},
            "per_class_metrics": {
                "0": {"precision": 0.85, "recall": 0.94, "f1": 0.89, "support": 5000},
                "1": {"precision": 0.74, "recall": 0.52, "f1": 0.61, "support": 1700},
            },
            "confidence_intervals": {"roc_auc": {"lower": 0.914, "upper": 0.930}},
        },
    }


def _model_comparison_sub(lang="tr"):
    # yakın iki model: yalnız 4. hane ayırır (0.8581 vs 0.8569)
    st = _perf_state_with_models({"oof_blend": 0.8581, "stacking": 0.8569, "lightgbm": 0.8544}, "oof_blend")
    sec = build_performance_section(st, lang)
    return next(s for s in sec.subsections if s.id == "performance_model_comparison")


def test_model_karsilastirma_dort_hane_korunur():
    """0.8581 vs 0.8569 — 3 haneye yuvarlansaydı ikisi de 0.858 olurdu; seçim gerekçesi kaybolurdu."""
    md = render_section(_model_comparison_sub("tr"), "tr")
    # her iki değer 4 haneyle ve BİRBİRİNDEN AYRI görünür (3 haneye inseydi ikisi de "0.858")
    assert "0.8581" in md and "0.8569" in md and "0.8544" in md, md


def test_model_karsilastirma_tekrar_not_kalkti_aciklama_bir_kez():
    """Faz 3 #3: her satırdaki tekrar not gitti; açıklama bir kez başta."""
    sub = _model_comparison_sub("en")
    assert all(not e.note for e in sub.evidence), "satır notu hâlâ tekrar ediyor"
    assert "cross-validation" in (sub.explanation or "").lower(), "tek-sefer açıklama eksik"


def test_model_karsilastirma_secilen_isaretli():
    sub = _model_comparison_sub("en")
    chosen = [e for e in sub.evidence if "selected" in e.label]
    assert len(chosen) == 1 and "oof_blend" in chosen[0].label


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
