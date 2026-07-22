"""Faz 1 mekanik bug'ların bölüm-seviyesi guard'ları (mock state, hızlı — model eğitimi yok).

#2 leakage kontrol sayısı tek kaynaktan (intro = gövde alt bölüm sayısı, ayrışamaz).
#3 generalization gap Evidence'ında yanlış-etiketli CI YOK.
#4 ratio etiketi üretim metadata'sından; metadata yoksa belirsiz ad TAHMİN edilmez (ham).
"""
import sys
import pytest

from src.services.report_sections import (
    build_leakage_section, build_generalization_section, build_features_section,
)
from src.core.report_schema import render_section


# ── #2 leakage kontrol sayısı ────────────────────────────────────────────────
def _leak_state(domain=False):
    st = {"intent": "classification",
          "evaluation_report": {"feature_importance_summary": {"leakage_suspects": []},
                                "target_lock_leakage": []}}
    if domain:
        st["domain_leakage_report"] = {"concerns": [], "overall_assessment": ""}
    return st


@pytest.mark.parametrize("lang", ["tr", "en"])
def test_leakage_count_matches_subsections_importance_only(lang):
    """domain koşmadı → 2 dedektör (dilim kilidi + baskınlık). intro '2', gövde 2 alt bölüm."""
    sec = build_leakage_section(_leak_state(domain=False), lang)
    txt = render_section(sec, lang)
    assert ("2 independent check" in txt) if lang == "en" else ("2 bağımsız kontrol" in txt), txt
    # gövdedeki sızıntı alt bölümü sayısı da 2 (leakage_target_lock + leakage_importance)
    leak_subs = [s for s in sec.subsections if str(getattr(s, "id", "")).startswith("leakage_")]
    assert len(leak_subs) == 2, [s.id for s in leak_subs]
    # "üç/three" ARTIK YOK
    assert "three independent" not in txt.lower() and "üç bağımsız" not in txt.lower()


@pytest.mark.parametrize("lang", ["tr", "en"])
def test_leakage_count_three_when_domain_ran(lang):
    """domain koştu → 3 dedektör. intro '3', gövde 3 alt bölüm. Sayı DİNAMİK."""
    sec = build_leakage_section(_leak_state(domain=True), lang)
    txt = render_section(sec, lang)
    assert ("3 independent check" in txt) if lang == "en" else ("3 bağımsız kontrol" in txt), txt
    leak_subs = [s for s in sec.subsections if str(getattr(s, "id", "")).startswith("leakage_")]
    assert len(leak_subs) == 3, [s.id for s in leak_subs]


# ── #3 generalization gap: yanlış-etiketli CI yok ────────────────────────────
@pytest.mark.parametrize("lang", ["tr", "en"])
def test_generalization_gap_has_no_misattributed_ci(lang):
    """ovf.ci_lower/upper (TEST metriğinin CI'ı) gap Evidence'ına iliştirilmemeli."""
    st = {"intent": "classification",
          "evaluation_report": {"overfitting": {
              "gap": -0.0062, "status": "normal", "display_label": "CV",
              "ci_lower": 0.8412, "ci_upper": 0.8811}}}
    sec = build_generalization_section(st, lang)
    gap_ev = next(e for e in sec.evidence if "gap" in e.label.lower() or "fark" in e.label.lower())
    assert gap_ev.uncertainty is None, f"gap'e yanlış CI iliştirilmiş: {gap_ev.uncertainty}"
    txt = render_section(sec, lang)
    # gap satırında "%95 aralık / 95% interval" görünmemeli
    for line in txt.splitlines():
        if "gap" in line.lower() or "fark" in line.lower():
            assert "interval" not in line.lower() and "aralık" not in line.lower(), line


# ── #4 ratio etiketi metadata'dan (tutarlı, belirsizse ham) ──────────────────
def _feat_state(ratio_meta):
    return {"intent": "classification",
            "feature_report": {"ratio_feature_meta": ratio_meta},
            "evaluation_report": {
                "shap_importance": {"top5_shap_features": {"ratio_age_products_number": 30.0},
                                    "all_shap_values": {}, "n_samples_used": 200},
                "feature_importance_summary": {
                    "top5_features": {"ratio_age_products_number": 30.0},
                    "top5_total_contribution_pct": 30.0,
                    "insignificant_features_count": 0, "leakage_suspects": []}}}


def test_features_ratio_label_from_metadata():
    """metadata varsa doğru etiket ('age to products-number ratio'), yanlış parse YOK."""
    st = _feat_state({"ratio_age_products_number": ["age", "products_number"]})
    txt = render_section(build_features_section(st, "en"), "en")
    assert "age to products-number ratio" in txt, txt
    assert "age products to number ratio" not in txt   # eski string-parse bug'ı


def test_features_ratio_label_ambiguous_without_metadata_shows_raw():
    """metadata YOK + belirsiz (>2 token) → ham ad; sessiz yanlış etiket YOK."""
    st = _feat_state({})   # metadata yok
    txt = render_section(build_features_section(st, "en"), "en")
    assert "ratio_age_products_number" in txt          # ham ad görünür
    assert "age products to number ratio" not in txt   # yanlış tahmin ASLA


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
