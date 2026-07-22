"""Faz 3 #1/#2/#7 — jargon düz-dil glossları, MCC ölçek uçları, editoryal fluff guard.

  #1: GA yöntemi (bootstrap/wilson/fisher-z) ve OOF terimi çıplak basılıyordu → gloss.
  #2: MCC ölçeğinin İKİ UCU (0=rastgele, 1=kusursuz) — ara-değer yorumu YOK (dinamik değil,
      ama örnek-değer yorumu da yok: 0.5 civarı yorumu kopukluk yaratır).
  #7: veri/model'i niteleyen boş övgü İSİM TAMLAMASI olarak yasak; tek-kelime meşru
      istatistik terimleri ("robust standard errors") yakalanmaz.

Standalone + pytest: `uv run pytest tests/test_faz3_jargon_gloss.py -q`
"""
import pytest

from src.core.report_schema import Evidence, render_evidence
from src.services.report_sections import build_performance_section, detect_speculation


# ── #1 method gloss (render_evidence, TR + EN) ───────────────────────────────

def _ci_line(method, lang):
    ev = Evidence(label="X", value=0.9, provenance="test_set",
                  uncertainty={"lower": 0.85, "upper": 0.95, "method": method})
    return render_evidence(ev, lang)


@pytest.mark.parametrize("lang,needle", [
    ("tr", "yeniden örnekle"),        # bootstrap glossu TR (örnekle-nerek)
    ("en", "resampling"),             # bootstrap glossu EN
])
def test_bootstrap_gloss(lang, needle):
    line = _ci_line("bootstrap", lang)
    assert "bootstrap" in line and needle in line


@pytest.mark.parametrize("lang,needle", [
    ("tr", "küçük örneklem"),
    ("en", "small samples"),
])
def test_wilson_gloss(lang, needle):
    line = _ci_line("wilson", lang)
    assert "Wilson" in line and needle in line


@pytest.mark.parametrize("lang,needle", [
    ("tr", "korelasyon"),
    ("en", "correlation"),
])
def test_fisher_z_gloss(lang, needle):
    line = _ci_line("fisher-z", lang)
    assert "Fisher-z" in line and needle in line


def test_bilinmeyen_method_aynen_basar():
    """Haritada olmayan yöntem KIRILMADAN çıplak adıyla basılır (geriye uyum)."""
    assert "jackknife" in _ci_line("jackknife", "en")


# ── #2 MCC ölçek uçları (iki uç var, ara-değer yorumu YOK) ────────────────────

def _perf_state():
    return {
        "intent": "classification",
        "evaluation_report": {
            "task": "binary_classification",
            "test_metrics": {"accuracy": 0.88, "f1_weighted": 0.87, "roc_auc": 0.922, "mcc": 0.52},
            "confusion_matrix": {"matrix": [[4700, 300], [816, 884]], "labels": [0, 1]},
            "per_class_metrics": {
                "0": {"precision": 0.85, "recall": 0.94, "f1": 0.89, "support": 5000},
                "1": {"precision": 0.74, "recall": 0.52, "f1": 0.61, "support": 1700},
            },
            "confidence_intervals": {"roc_auc": {"lower": 0.914, "upper": 0.930}},
        },
    }


def _mcc_note(lang):
    sec = build_performance_section(_perf_state(), lang)
    mcc = next(e for e in sec.evidence if "MCC" in e.label)
    return mcc.note


def test_mcc_iki_uc_tanimli_tr():
    note = _mcc_note("tr")
    assert "0 rastgele" in note and "1 kusursuz" in note


def test_mcc_iki_uc_tanimli_en():
    note = _mcc_note("en")
    assert "0 means no better than guessing" in note and "1 is perfect" in note


def test_mcc_ornek_deger_yorumu_yok():
    """'0.5 civarı orta-güçlü' gibi örnek-değer yorumu KOYMA — başka raporun 0.15'inde kopar."""
    for lang in ("tr", "en"):
        note = _mcc_note(lang).lower()
        assert "0.5 civar" not in note and "around 0.5" not in note


# ── #1 OOF açılımı (eşik bölümü, TR + EN) ─────────────────────────────────────

def _oof_state():
    st = _perf_state()
    st["evaluation_report"]["threshold"] = {
        "value": 0.5682, "provenance": {"source": "oof_cv_f1max", "n": 1000},
    }
    return st


def test_oof_acilir_tr():
    from src.core.report_schema import render_section
    md = render_section(build_performance_section(_oof_state(), "tr"), "tr")
    assert "katman-dışı" in md and "OOF" in md


def test_oof_acilir_en():
    from src.core.report_schema import render_section
    md = render_section(build_performance_section(_oof_state(), "en"), "en")
    assert "out-of-fold" in md and "OOF" in md


# ── #7 editoryal fluff — isim tamlaması yasak, tek-kelime meşru serbest ────────

@pytest.mark.parametrize("fluff", [
    "This is a robust dataset with strong signal.",
    "We ran a comprehensive analysis of the churn drivers.",
    "The model is powerful and state-of-the-art.",
    "Zengin bir veri üzerinde kapsamlı analiz yapıldı.",
    "Sağlam veri ve güçlü model kullanıldı.",
])
def test_editoryal_fluff_yakalanir(fluff):
    assert detect_speculation(fluff), f"fluff kaçtı: {fluff}"


@pytest.mark.parametrize("legit", [
    "We report robust standard errors for each coefficient.",   # istatistik terimi
    "The leakage layer runs a comprehensive scan across folds.",  # 'scan' veri-kelimesi değil
    "The dataset has 6,700 rows and 12 columns.",               # düz betimleme
    "Veri setinde 6.700 satır var.",                            # düz betimleme, sıfat yok
    "Model doğruluğu %88.",                                     # övgü yok
])
def test_mesru_kullanim_yakalanmaz(legit):
    assert detect_speculation(legit) == [], f"meşru kullanım yanlış yakalandı: {legit}"


def test_olumsuzlanmis_fluff_temiz():
    """'robust DEĞİL' gerçek bir eleştiridir → fluff sayılmaz (olumsuzlama penceresi)."""
    assert detect_speculation("This is not a robust dataset.") == []


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
