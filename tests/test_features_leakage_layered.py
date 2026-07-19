"""Etkili Özellikler + Sızıntı & Riskler — katmanlı yapı testleri.

Ölçülmüş / kullanıcı-belirtilmiş risk sınıfları:
  1. SHAP (test_set) ile feature_importance (model_internal) KARIŞIRSA provenance
     yalan söyler — okuyucu iki farklı ölçüyü tek sanır.
  2. "X önemli ÇÜNKÜ hedefi belirliyor" → önem sıralaması tanımlar, sebebini değil
     (nedensellik dersinin türevi).
  3. Sızıntıda üç dedektör (domain / target_lock / importance) karışırsa "aynı şey
     üç kez mi" sanılır — kalem bazında ayrışmalı.
  4. purity_gain'in DOĞRU okunuşu: ham oran değil, genel oranın ÜSTÜNE eklenen —
     Wilson düzeltmesiyle eski sessiz yanlış-alarm kapandı.
  5. Flag varsa NEDEN görünür; flag yoksa "temiz" AÇIKÇA — "koştu ama temiz" ile
     "hiç koşmadı" farklı, ikisi de sessiz boşluk bırakmamalı.
"""
import pytest

from src.core.report_schema import (
    ORDER_FEATURES, ORDER_LEAKAGE_RISK, PROVENANCE_LABEL, render_section,
)
from src.services.report_sections import (
    build_features_section, build_leakage_section, detect_speculation,
)


# ═══ ETKİLİ ÖZELLİKLER ═══════════════════════════════════════════════════════

def _feat_state(**over):
    ev = {
        "shap_importance": {
            "top5_shap_features": {"egitim": 32.5, "yas": 24.1, "capital_gain": 18.0},
            "all_shap_values": {}, "n_samples_used": 200,
        },
        "feature_importance_summary": {
            "top5_features": {"capital_gain": 41.2, "egitim": 28.0, "yas": 12.5},
            "top5_total_contribution_pct": 81.7,
            "insignificant_features_count": 14,
            "leakage_suspects": [],
        },
    }
    ev.update(over.pop("evaluation_report", {}))
    st = {"intent": "classification", "evaluation_report": ev}
    st.update(over)
    return st


def test_features_order():
    assert build_features_section(_feat_state(), "tr").order == ORDER_FEATURES


def test_shap_test_set_importance_model_internal_KARISMIYOR():
    """Kullanıcı kararı: SHAP test_set, feature_importance model_internal."""
    sec = build_features_section(_feat_state(), "tr")
    # Üst kanıt = SHAP = test_set
    assert sec.evidence, "SHAP kanıtı yok"
    assert all(e.provenance == "test_set" for e in sec.evidence)

    mi = next(s for s in sec.subsections if s.id == "features_model_internal")
    assert all(e.provenance == "model_internal" for e in mi.evidence)
    # İkisi AYRI bölümde — tek listede karışmıyor
    assert not any(e.provenance == "model_internal" for e in sec.evidence)


def test_provenance_etiketleri_raporda_dogru_basiliyor():
    sec = build_features_section(_feat_state(), "tr")
    md = render_section(sec, "tr")
    assert PROVENANCE_LABEL["tr"]["test_set"] in md
    assert PROVENANCE_LABEL["tr"]["model_internal"] in md


def test_her_feature_kaniti_ONEM_DEGIL_SEBEP_damgasi_tasiyor():
    """Damga: 'katkı ölçüsüdür, hedefi belirlediğini göstermez'."""
    sec = build_features_section(_feat_state(), "tr")
    for e in sec.evidence:
        assert "BELİRLEDİĞİNİ göstermez" in e.note


def test_bolum_uyarisi_nedensellik_reddediyor():
    sec = build_features_section(_feat_state(), "tr")
    assert "hedefi neyin belirlediğini değil" in sec.disclaimer


def test_shap_sayilari_UYDURULMUYOR():
    sec = build_features_section(_feat_state(), "tr")
    vals = {e.value for e in sec.evidence}
    assert vals == {32.5, 24.1, 18.0}


def test_hic_onem_yoksa_gorunur_sebep():
    st = _feat_state(evaluation_report={"shap_importance": {}, "feature_importance_summary": {}})
    sec = build_features_section(st, "tr")
    assert sec.status == "not_computed"
    assert sec.not_computed_reason


def test_shap_yoksa_importance_UST_DUZEYE_cikiyor():
    """SHAP üretilmeyen modellerde (ensemble) importance BİRİNCİL kanıt olmalı.

    Ölçüldü: adult'ta ensemble → SHAP boş → üst düzey kanıt boşalıyor →
    build_section_prose zemin bulamayıp yorumu SESSİZCE düşürüyordu. Regresyon testi.
    """
    st = _feat_state(evaluation_report={
        "shap_importance": {},
        "feature_importance_summary": {
            "top5_features": {"a": 50.0, "b": 30.0}, "top5_total_contribution_pct": 80.0,
            "insignificant_features_count": 3, "leakage_suspects": [],
        },
    })
    sec = build_features_section(st, "tr")
    assert sec.status == "ok"
    # Üst düzey kanıt DOLU (prose zemini var) ve model_internal provenance'lı
    assert len(sec.evidence) == 2, "SHAP yokken importance üste çıkmamış — prose boş düşer"
    assert all(e.provenance == "model_internal" for e in sec.evidence)
    # Tek ölçü var → ayrı alt bölüm açılmaz (aynı şeyi iki kez basma)
    assert not any(s.id == "features_model_internal" for s in sec.subsections)


def test_shap_varken_importance_ALT_bolumde_kalir():
    """İki ölçü de varsa SHAP üstte, importance ayrı alt bölümde — karışmasın."""
    sec = build_features_section(_feat_state(), "tr")
    assert all(e.provenance == "test_set" for e in sec.evidence)
    assert any(s.id == "features_model_internal" for s in sec.subsections)


# ── "önemli çünkü" yasağı ─────────────────────────────────────────────────────

@pytest.mark.parametrize("cumle", [
    "Eğitim en önemli özellik çünkü geliri belirliyor.",
    "Bu alan yüksek katkı sağlıyor, sebebi müşterinin yaşıdır.",
    "Capital gain drives the prediction because it determines income.",
    "Yaş önemli çünkü modeli etkiliyor.",
])
def test_onem_sebep_spekulasyonu_yakalaniyor(cumle):
    assert detect_speculation(cumle), f"kaçtı: {cumle}"


@pytest.mark.parametrize("cumle", [
    "Eğitim, modelin kararında en yüksek katkıyı taşıyan alandır.",
    "Bu sıralama bu modele özeldir; başka bir model farklı sıralayabilir.",
    "Yaş alanı yüksek önem taşıyor ancak bunun sebebi ölçülmemiştir.",
    "SHAP ve iç önem sıralaması bu alanda ayrışıyor.",
])
def test_tanimlayici_onem_dili_yanlis_alarm_vermiyor(cumle):
    assert detect_speculation(cumle) == [], f"yanlış alarm: {cumle}"


def test_features_guard_bagli():
    from src.services.report_sections import _PROSE_GUARDS
    assert _PROSE_GUARDS["features"][0] is detect_speculation


# ═══ SIZINTI & RİSKLER ═══════════════════════════════════════════════════════

def _leak_state(domain=None, target_lock=None, suspects=None, domain_key=True):
    ev = {
        "feature_importance_summary": {
            "top5_features": {"a": 40.0}, "leakage_suspects": suspects or [],
        },
    }
    if target_lock is not None:
        ev["target_lock_leakage"] = target_lock
    st = {"intent": "classification", "evaluation_report": ev}
    if domain_key:
        st["domain_leakage_report"] = {"concerns": domain or [], "overall_assessment": "Belirgin bir sorun yok."}
    return st


def test_leakage_order():
    assert build_leakage_section(_leak_state(), "tr").order == ORDER_LEAKAGE_RISK


def test_hicbir_kontrol_kosmadiysa_not_computed():
    """feature_importance yoksa VE domain anahtarı yoksa → hiç koşmadı."""
    sec = build_leakage_section({"intent": "classification", "evaluation_report": {}}, "tr")
    assert sec.status == "not_computed"
    assert sec.not_computed_reason


def test_kostu_ama_temiz_ACIKCA_soyleniyor():
    """SESSİZ BOŞLUK YOK: kontroller koştu, işaret yok → 'temiz' net yazılmalı."""
    sec = build_leakage_section(_leak_state(), "tr")
    assert sec.status == "ok"
    assert "belirtisi yok" in sec.headline.lower()
    # Kaç kontrolün koştuğu görünür
    assert "kontrol" in sec.headline.lower()
    # "kesin değil" dürüstlüğü
    assert "kesin olmadığı" in sec.explanation


def test_temiz_ile_kosmadi_AYRI_seyler():
    """'koştu ama temiz' (ok + belirti yok) ≠ 'hiç koşmadı' (not_computed)."""
    temiz = build_leakage_section(_leak_state(), "tr")
    kosmadi = build_leakage_section({"intent": "classification", "evaluation_report": {}}, "tr")
    assert temiz.status == "ok" and kosmadi.status == "not_computed"


# ── target_lock: purity_gain doğru sunumu (bu haftanın düzeltmesi) ───────────

def _target_lock_flag(gain=0.62, base=0.24, conf="high"):
    return {
        "feature": "sure_gun", "importance": 0.55, "bin": "(0, 30]",
        "purity": 0.86, "bin_size": 140, "confidence": conf, "local_jump_ratio": 3.1,
        "purity_gain": gain, "baseline": base, "locked_class": "1",
        "reason": "detektörün ham cümlesi",
    }


def test_purity_gain_GENEL_ORANA_gore_sunuluyor_ham_oran_degil():
    """purity_gain'in DOĞRU okunuşu: ham %86 değil, %24'ün üstüne eklenen kazanç.

    Eski sessiz yanlış-alarm 'bu aralıkta %99 aynı sonuç' derdi — dengesiz veride
    ZATEN öyle. Doğru sunum genel oranı da gösterir.
    """
    sec = build_leakage_section(_leak_state(target_lock=[_target_lock_flag()]), "tr")
    tl = next(s for s in sec.subsections if s.id == "leakage_target_lock")
    note = tl.evidence[0].note
    assert "24.0%" in note or "%24" in note, "genel oran (baseline) görünmüyor"
    assert "kazanç 0.62" in note, "kazanç değeri görünmüyor"
    # Ham saflık (0.86) tek başına 'kesinlik' diye sunulmamalı — kazanç öne çıkmalı
    assert "kilitleniyor" in note


def test_target_lock_high_ani_kopus_low_kademeli_AYRIT_ediliyor():
    """high = olası sızıntı, low = gerçek sinyal — dedektörün yargısı korunmalı."""
    high = build_leakage_section(_leak_state(target_lock=[_target_lock_flag(conf="high")]), "tr")
    low = build_leakage_section(_leak_state(target_lock=[_target_lock_flag(conf="low")]), "tr")
    h_note = next(s for s in high.subsections if s.id == "leakage_target_lock").evidence[0].note
    l_note = next(s for s in low.subsections if s.id == "leakage_target_lock").evidence[0].note
    assert "olası sızıntı" in h_note
    assert "gerçek sinyal" in l_note


def test_target_lock_test_set_provenanceli():
    sec = build_leakage_section(_leak_state(target_lock=[_target_lock_flag()]), "tr")
    tl = next(s for s in sec.subsections if s.id == "leakage_target_lock")
    assert all(e.provenance == "test_set" for e in tl.evidence)


def test_target_lock_kostu_ama_temiz():
    """target_lock boş ama importance koştu → 'kontrol yapıldı, işaret yok'."""
    sec = build_leakage_section(_leak_state(target_lock=[]), "tr")
    tl = next(s for s in sec.subsections if s.id == "leakage_target_lock")
    assert "işaretlenen alan yok" in tl.headline


# ── üç dedektör kalem bazında ayrışıyor ──────────────────────────────────────

def test_uc_dedektor_AYRI_alt_bolumlerde():
    """domain / target_lock / importance karışmaz — her biri kendi yöntemiyle."""
    st = _leak_state(
        domain=[{"column_name": "gelir_sonraki", "relationship_type": "temporal",
                 "reasoning": "hedefin geleceğinden", "confidence": "high"}],
        target_lock=[_target_lock_flag()],
        suspects=[{"column": "id_kopya", "reason": "tek başına %85 importance", "severity": "high"}],
    )
    sec = build_leakage_section(st, "tr")
    ids = {s.id for s in sec.subsections}
    assert ids == {"leakage_domain", "leakage_target_lock", "leakage_importance"}
    # Her biri kendi provenance'ında
    dom = next(s for s in sec.subsections if s.id == "leakage_domain")
    imp = next(s for s in sec.subsections if s.id == "leakage_importance")
    assert dom.evidence[0].provenance == "schema_semantic"
    assert imp.evidence[0].provenance == "model_internal"


def test_domain_high_confidence_otomatik_drop_gorunur():
    """high-confidence domain leakage modelden atıldı — bu görünmeli."""
    st = _leak_state(domain=[
        {"column_name": "gelir_kopya", "relationship_type": "duplicate",
         "reasoning": "hedefin kopyası", "confidence": "high"}])
    sec = build_leakage_section(st, "tr")
    dom = next(s for s in sec.subsections if s.id == "leakage_domain")
    assert "otomatik" in dom.headline.lower()
    assert "gerçek performans" in dom.explanation


def test_flag_varsa_ust_baslik_sayiyi_soyluyor():
    st = _leak_state(target_lock=[_target_lock_flag()],
                     suspects=[{"column": "x", "reason": "r", "severity": "high"}])
    sec = build_leakage_section(st, "tr")
    assert "2 işaret" in sec.headline
    assert "FARKLI" in sec.explanation  # aynı alan birden çok kontrolde çıkabilir


def test_domain_kostu_ama_temiz_overall_assessment_gosteriliyor():
    sec = build_leakage_section(_leak_state(), "tr")  # domain_key=True, boş concerns
    dom = next(s for s in sec.subsections if s.id == "leakage_domain")
    assert "şüpheli alan bulunmadı" in dom.headline
    assert "Belirgin bir sorun yok" in dom.explanation


def test_domain_kosmadiysa_domain_alt_bolumu_YOK():
    """domain_leakage_report anahtarı hiç yoksa o kontrol koşmamış — alt bölüm açılmaz."""
    st = _leak_state(target_lock=[], domain_key=False)
    sec = build_leakage_section(st, "tr")
    assert not any(s.id == "leakage_domain" for s in sec.subsections)
    # ama importance koştu → target_lock alt bölümü var
    assert any(s.id == "leakage_target_lock" for s in sec.subsections)


def test_leakage_render_bozulmadan_basiliyor():
    st = _leak_state(target_lock=[_target_lock_flag()],
                     domain=[{"column_name": "c", "relationship_type": "t",
                              "reasoning": "r", "confidence": "low"}])
    md = render_section(build_leakage_section(st, "tr"), "tr")
    assert "## Sızıntı ve Riskler" in md
    assert "Dilim kilidi" in md
    assert "\n## \n" not in md and not md.rstrip().endswith("## ")
