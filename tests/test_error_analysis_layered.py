"""
Hata Analizi bölümü — DEĞİŞMEZ testleri.

İki sessiz hatayı çitler:
  1. Değerler ÖLÇEKLİ sunuluyordu ("0.3682"), ham değil ("42 saat").
  2. numeric_cols[:15] — ilk 15 kolon taranıp "en çok ayrışan" diye sunuluyordu.

Ve bir çerçeveleme kuralını: bölüm NEREDE yanıldığını gösterir, NEDEN'ini iddia etmez.

Çalıştırma:
    uv run pytest tests/test_error_analysis_layered.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.report_schema import ORDER_ERROR_ANALYSIS, _fmt_value, render_section
from src.services.evaluator import _analyze_errors
from src.services.report_sections import (
    ERRORS_VISIBLE_TOP,
    build_error_analysis_section,
    detect_causal_language,
    detect_speculation,
)


def _scenario(n=200):
    """Hatalar 'haftalik_saat' yüksek olanlarda yoğunlaşıyor.
    X_test ÖLÇEKLİ (modelin gördüğü), raw_test HAM (kullanıcının anladığı)."""
    rng = np.random.default_rng(0)
    saat_ham = np.concatenate([np.full(n // 2, 17.0), np.full(n // 2, 42.0)])
    saat_ham = saat_ham + rng.normal(0, 0.5, n)
    y_true = np.zeros(n, dtype=int)
    y_pred = np.zeros(n, dtype=int)
    y_pred[n // 2:] = 1                    # ikinci yarı (42 saatliler) hatalı

    raw_test = pd.DataFrame({"haftalik_saat": saat_ham})
    # Ölçekli hâli — StandardScaler'ın yaptığı
    X_test = pd.DataFrame({
        "haftalik_saat": (saat_ham - saat_ham.mean()) / saat_ham.std()
    })
    return y_true, y_pred, X_test, raw_test


# ═══════════════════════════════════════════════════════════════════════════
#  BUG 1 — ölçekli değer "değer" diye sunuluyordu
# ═══════════════════════════════════════════════════════════════════════════

def test_ham_deger_tasiyicidan_geliyor_olcekli_degil():
    """'Hatalı vakalarda 42, doğrularda 17' — kanıt katmanının varlık sebebi.

    X_test ölçeklidir (0.99 / -0.99); ham değer YALNIZCA taşıyıcıdan gelebilir.
    """
    y_true, y_pred, X_test, raw_test = _scenario()
    out = _analyze_errors(y_true, y_pred, X_test, ["haftalik_saat"], raw_test=raw_test)
    info = out["feature_distribution_diff"]["haftalik_saat"]

    assert info["raw_available"] is True
    assert info["mean_in_errors_raw"] == pytest.approx(42, abs=0.5), \
        "hatalı vakaların HAM ortalaması ~42 olmalı"
    assert info["mean_in_correct_raw"] == pytest.approx(17, abs=0.5), \
        "doğru vakaların HAM ortalaması ~17 olmalı"
    # Ölçekli değerler hâlâ duruyor ama ham olanlardan FARKLI — karıştırılmamalı.
    assert abs(info["mean_in_errors"]) < 2, "ölçekli değer ham değerle karışmış"


def test_tasiyici_yoksa_ham_degil_diye_gorunur_sebep():
    """Taşıyıcı yoksa ölçekli sayı 'gerçek değer' gibi sunulmaz — sebep taşınır."""
    y_true, y_pred, X_test, _ = _scenario()
    out = _analyze_errors(y_true, y_pred, X_test, ["haftalik_saat"], raw_test=None)
    info = out["feature_distribution_diff"]["haftalik_saat"]

    assert info["raw_available"] is False
    assert info["raw_unavailable_reason"], "ham değer yoksa SEBEP zorunlu"
    assert "mean_in_errors_raw" not in info


def test_turetilmis_feature_ham_karsiligi_yok_diye_isaretlenir():
    """One-hot dummy split SONRASI üretilir → taşıyıcıda yoktur. Sessizce ölçekli
    değer sunmak yerine görünür sebep taşınmalı."""
    y_true, y_pred, X_test, raw_test = _scenario()
    X_test = X_test.assign(workclass_Private=np.concatenate([np.zeros(100), np.ones(100)]))
    out = _analyze_errors(y_true, y_pred, X_test, list(X_test.columns), raw_test=raw_test)
    d = out["feature_distribution_diff"]

    assert d["haftalik_saat"]["raw_available"] is True
    assert d["workclass_Private"]["raw_available"] is False
    assert d["workclass_Private"]["raw_unavailable_reason"]


def test_bolumde_ham_deger_gosteriliyor():
    """Rapor metninde HAM ortalamalar görünmeli — ölçekli olanlar değil."""
    y_true, y_pred, X_test, raw_test = _scenario()
    err = _analyze_errors(y_true, y_pred, X_test, ["haftalik_saat"], raw_test=raw_test)
    info = err["feature_distribution_diff"]["haftalik_saat"]
    sec = build_error_analysis_section({"evaluation_report": {"error_analysis": err}}, "tr")
    md = render_section(sec, "tr")

    assert "Hatalı vakalarda ortalama" in md
    # Faz 3 ondalık disiplini: metin HAM değeri gösterir ama `_fmt_value` ile BİÇİMLENDİRİR
    # (magnitude-tabanlı; 41.9747 → "41.975"). İddia "ham mı ölçekli mi", "kaç ondalık" değil →
    # assert biçimlendirilmiş ham değere bağlanır (ham repr'e değil; o bayat sözleşmeydi).
    assert _fmt_value(info["mean_in_errors_raw"]) in md, "ham hata ortalaması metinde yok"
    assert _fmt_value(info["mean_in_correct_raw"]) in md, "ham doğru ortalaması metinde yok"
    # Ölçekli değer kullanıcıya "değer" diye SUNULMAMALI (asıl korunan davranış).
    assert str(info["mean_in_errors"]) not in md
    assert _fmt_value(info["mean_in_errors"]) not in md


# ═══════════════════════════════════════════════════════════════════════════
#  BUG 2 — numeric_cols[:15] sessiz kırpması
# ═══════════════════════════════════════════════════════════════════════════

def test_tum_sayisal_kolonlar_taraniyor_ilk_15_degil():
    """20 kolonun 20.'sinde güçlü ayrışma var. Eski kod ilk 15'e bakıp onu
    KAÇIRIYOR ve kalanı 'en çok ayrışan' diye sunuyordu."""
    n = 200
    rng = np.random.default_rng(1)
    y_true = np.zeros(n, dtype=int)
    y_pred = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])

    cols = {f"gurultu_{i}": rng.normal(size=n) for i in range(19)}
    # 20. kolon: hatalarda belirgin şekilde farklı
    cols["gec_gelen_sinyal"] = np.concatenate([np.zeros(n // 2), np.full(n // 2, 5.0)])
    X_test = pd.DataFrame(cols)
    assert X_test.columns.get_loc("gec_gelen_sinyal") >= 15, "test 15. sıradan sonra olmalı"

    out = _analyze_errors(y_true, y_pred, X_test, list(X_test.columns))
    assert "gec_gelen_sinyal" in out["feature_distribution_diff"], \
        "15. sıradan sonraki güçlü sinyal kaçırılmış — sessiz kırpma geri gelmiş"
    assert out["n_features_examined"] == 20


def test_tarama_kapsami_rapora_yaziliyor():
    """'En çok ayrışan' iddiasının KAPSAMI görünür olmalı."""
    n = 200
    rng = np.random.default_rng(2)
    y_true = np.zeros(n, dtype=int)
    y_pred = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])
    X_test = pd.DataFrame({f"c{i}": rng.normal(size=n) for i in range(8)})
    X_test["ayrisan"] = np.concatenate([np.zeros(n // 2), np.full(n // 2, 4.0)])

    err = _analyze_errors(y_true, y_pred, X_test, list(X_test.columns))
    sec = build_error_analysis_section({"evaluation_report": {"error_analysis": err}}, "tr")
    scope = next(s for s in sec.subsections if s.id == "error_analysis_scope")
    assert "9" in scope.headline, "taranan alan sayısı yazılmalı"


# ═══════════════════════════════════════════════════════════════════════════
#  ÇERÇEVELEME — NEREDE, NEDEN değil
# ═══════════════════════════════════════════════════════════════════════════

def test_nerede_degil_neden_cercevesi_bolum_basinda_bir_kez():
    """BOILERPLATE TEKRARI YASAĞI: 'nerede/neden' çerçevesi + Cohen's d ölçeği
    bölüm başında BİR KEZ (disclaimer), her alanın kanıt satırında DEĞİL.

    Satırda yalnız DEĞİŞEN bilgi kalır: hatalı/doğru ortalamalar (ya da ham-yok)."""
    y_true, y_pred, X_test, raw_test = _scenario()
    err = _analyze_errors(y_true, y_pred, X_test, ["haftalik_saat"], raw_test=raw_test)
    sec = build_error_analysis_section({"evaluation_report": {"error_analysis": err}}, "tr")
    # Çerçeve + ölçek bölüm başında BİR KEZ:
    assert "sebebini açıklamaz" in sec.disclaimer      # nerede/neden çerçevesi
    assert "Cohen's d" in sec.disclaimer                # büyüklük ölçeği (0.2/0.5/0.8)
    feat_ev = [e for e in sec.evidence if "ayrışma" in e.label]
    assert feat_ev
    # Satır başına TEKRAR YOK — satırda yalnız değişen bilgi:
    for e in feat_ev:
        assert "NEDENİNİ değil" not in (e.note or ""), "satırda 'nerede/neden' tekrarı"
        assert "Cohen's d" not in (e.note or ""), "satırda Cohen's d ölçeği tekrarı"


def test_bolum_uyarisi_kod_tarafindan_basiliyor():
    y_true, y_pred, X_test, raw_test = _scenario()
    err = _analyze_errors(y_true, y_pred, X_test, ["haftalik_saat"], raw_test=raw_test)
    sec = build_error_analysis_section({"evaluation_report": {"error_analysis": err}}, "tr")
    assert "sebebini açıklamaz" in sec.disclaimer
    md = render_section(sec, "tr")
    assert md.index("nerede yoğunlaştığını") < md.index("Kanıt")


def test_spekulasyon_dedektoru_neden_iddiasini_yakaliyor():
    assert detect_speculation("Model yaşlılarda yanılıyor çünkü veri az.")
    assert detect_speculation("Bunun sebebi eksik veridir.")
    assert detect_speculation("Muhtemelen dengesiz sınıf dağılımından.")
    assert detect_speculation("Görünüşe göre model bu grubu tanımıyor.")
    assert detect_speculation("The model fails because the data is sparse.")
    assert detect_speculation("This suggests that the feature is noisy.")
    assert detect_speculation("The reason is class imbalance.")
    # Nedensel kalıplar da bu bölümde yasak
    assert detect_speculation("Yüksek çalışma saati hataya yol açıyor.")


def test_spekulasyon_dedektoru_tanimlayici_dile_yanlis_alarm_vermiyor():
    for temiz in (
        "Hatalı tahminler haftalık çalışma saati yüksek olan kayıtlarda yoğunlaşıyor.",
        "Hatalı vakalarda ortalama 42, doğru vakalarda 17.",
        "Model, emin olduğu 415 tahminde yanılmış.",
        "Errors concentrate in records with higher weekly hours.",
        "The average is 42 in wrong cases and 17 in correct ones.",
    ):
        assert detect_speculation(temiz) == [], f"yanlış alarm: {temiz!r}"


def test_dedektor_OLUMSUZLANAN_kullanimi_cezalandirmiyor():
    """DOĞRU çerçeveyi yazan cümle ceza almamalı.

    Gerçek olay: LLM "...hangi alanlarda yoğunlaştığını göstermektedir, ancak
    nedenlerini AÇIKLAMAMAKTADIR" yazdı — istediğimiz cümlenin ta kendisi — ve
    dedektör 'nedenlerini' gördüğü için iki denemede de reddetti; iyi bir açıklama
    düşürüldü. Olumsuzlama penceresi bunu önler.
    """
    for temiz in (
        "Bu veriler hataların hangi alanlarda yoğunlaştığını gösterir, ancak nedenlerini açıklamamaktadır.",
        "Hatanın sebebi bu analizde ölçülmedi.",
        "Bu, bir alanın diğerine sebep olduğunu göstermez.",
        "İki alanın birlikte hareket etmesi nedensellik anlamına gelmez.",
        "Bu bulgu, çalışma saatinin hataya yol açtığını göstermez.",
        "This does not show that one field causes the other.",
        "These numbers are not evidence of the reason.",
    ):
        assert detect_speculation(temiz) == [], f"olumsuzlanan kullanım cezalandırıldı: {temiz!r}"
        assert detect_causal_language(temiz) == [], f"olumsuzlanan kullanım cezalandırıldı: {temiz!r}"


def test_olumsuzlama_gercek_iddiayi_aklamiyor():
    """Pencere dar: cümlenin ilerisindeki alakasız bir 'değil' gerçek nedensel
    iddiayı temizleyemez."""
    assert detect_speculation("Yaş, hataya sebep oluyor.")
    assert detect_causal_language("Yaş, gelire sebep oluyor.")
    assert detect_speculation("Model yanılıyor çünkü veri az.")


# ═══════════════════════════════════════════════════════════════════════════
#  KATMANLAMA / DURUM
# ═══════════════════════════════════════════════════════════════════════════

def test_ilk_5_gorunur_kalani_katlanir():
    n = 200
    rng = np.random.default_rng(3)
    y_true = np.zeros(n, dtype=int)
    y_pred = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])
    cols = {}
    for i in range(9):
        cols[f"a{i}"] = np.concatenate([np.zeros(n // 2), np.full(n // 2, 3.0 - i * 0.2)])
    X_test = pd.DataFrame(cols)

    err = _analyze_errors(y_true, y_pred, X_test, list(X_test.columns))
    toplam = len(err["feature_distribution_diff"])
    assert toplam > ERRORS_VISIBLE_TOP

    sec = build_error_analysis_section({"evaluation_report": {"error_analysis": err}}, "tr")
    gorunur = [e for e in sec.evidence if "ayrışma" in e.label]
    katlanan = next(s for s in sec.subsections if s.id == "error_analysis_all")
    assert len(gorunur) == ERRORS_VISIBLE_TOP
    assert len(gorunur) + len(katlanan.evidence) == toplam, "hiçbir alan kırpılmamalı"


# ── Ham değer merceği ────────────────────────────────────────────────────────
# Ölçülen desen (adult census): en çok ayrışan alanların HEPSİ türetilmiş olabilir
# (one-hot / count_* / grpstd_*) → "hatalı vakalarda 42, doğrularda 17" cümlesi
# görünür top-5'te hiç kurulmaz. Çözüm sıralamayı ham-öncelikli yapmak DEĞİL
# (en güçlü ayrıştırıcıyı gizlerdi); aynı alanlara ikinci bir mercek açmak.

def _karisik_senaryo(n=200):
    """3 türetilmiş alan güçlü ayrışıyor, 2 ham alan daha zayıf → ham'lar top-5 dışı."""
    rng = np.random.default_rng(7)
    y_true = np.zeros(n, dtype=int)
    y_pred = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])

    yas = np.concatenate([rng.normal(35, 4, n // 2), rng.normal(44, 4, n // 2)])
    saat = np.concatenate([rng.normal(38, 3, n // 2), rng.normal(44, 3, n // 2)])
    cols = {}
    for i in range(5):  # türetilmişler: daha güçlü ayrışma
        cols[f"count_x{i}"] = np.concatenate([np.zeros(n // 2), np.full(n // 2, 5.0)])
    cols["yas"], cols["saat"] = yas, saat
    X_test = pd.DataFrame(cols)
    raw_test = pd.DataFrame({"yas": yas, "saat": saat})  # yalnız ham alanların karşılığı var
    return y_true, y_pred, X_test, raw_test


def test_ham_mercek_top5_disina_dusen_alanlari_erisilebilir_kiliyor():
    y_true, y_pred, X_test, raw_test = _karisik_senaryo()
    err = _analyze_errors(y_true, y_pred, X_test, list(X_test.columns), raw_test=raw_test)
    sec = build_error_analysis_section({"evaluation_report": {"error_analysis": err}}, "tr")

    gorunur = [e for e in sec.evidence if "ayrışma" in e.label]
    assert not any("Hatalı vakalarda ortalama" in e.note for e in gorunur), (
        "fixture kurulumu bozuk: ham alan zaten top-5'te, mercek sınanmıyor"
    )

    mercek = next(s for s in sec.subsections if s.id == "error_analysis_raw")
    etiketler = " ".join(e.label for e in mercek.evidence)
    assert "yas" in etiketler.lower() and "saat" in etiketler.lower()
    assert all("Hatalı vakalarda ortalama" in e.note for e in mercek.evidence)


def test_ham_mercek_SIRALAMAYI_degistirmiyor():
    """En güçlü ayrıştırıcı görünür kalmalı — mercek yalnızca yeniden gruplar."""
    y_true, y_pred, X_test, raw_test = _karisik_senaryo()
    err_ham = _analyze_errors(y_true, y_pred, X_test, list(X_test.columns), raw_test=raw_test)
    err_hamsiz = _analyze_errors(y_true, y_pred, X_test, list(X_test.columns))

    sira_ham = [e.label for e in build_error_analysis_section(
        {"evaluation_report": {"error_analysis": err_ham}}, "tr").evidence]
    sira_hamsiz = [e.label for e in build_error_analysis_section(
        {"evaluation_report": {"error_analysis": err_hamsiz}}, "tr").evidence]
    assert sira_ham == sira_hamsiz, "ham varlığı görünür sıralamayı kaydırmış"


def test_ham_mercek_ALT_KUME_oldugunu_soyluyor():
    """Aynı sayıyı iki yerde görmek 'iki ayrı bulgu' sanılabilir — açıkça reddet."""
    y_true, y_pred, X_test, raw_test = _karisik_senaryo()
    err = _analyze_errors(y_true, y_pred, X_test, list(X_test.columns), raw_test=raw_test)
    sec = build_error_analysis_section({"evaluation_report": {"error_analysis": err}}, "tr")
    mercek = next(s for s in sec.subsections if s.id == "error_analysis_raw")
    assert "ALT KÜMESİDİR" in mercek.explanation
    assert "yeni bir ölçüm değil" in mercek.explanation


def test_ham_alan_yoksa_mercek_ACILMIYOR():
    """Hepsi türetilmişse boş bir 'gerçek birimler' bölümü açmanın anlamı yok."""
    y_true, y_pred, X_test, _ = _karisik_senaryo()
    err = _analyze_errors(y_true, y_pred, X_test, list(X_test.columns))  # raw_test yok
    sec = build_error_analysis_section({"evaluation_report": {"error_analysis": err}}, "tr")
    assert not any(s.id == "error_analysis_raw" for s in sec.subsections)


def test_hic_hata_yoksa_gorunur_sebep():
    y_true = np.zeros(50, dtype=int)
    err = _analyze_errors(y_true, y_true.copy(), pd.DataFrame({"x": np.arange(50.0)}), ["x"])
    sec = build_error_analysis_section({"evaluation_report": {"error_analysis": err}}, "tr")
    assert sec.status == "not_applicable"
    assert sec.not_computed_reason


def test_hata_analizi_yoksa_gorunur_sebep():
    sec = build_error_analysis_section({}, "tr")
    assert sec.status == "not_computed"
    assert sec.not_computed_reason


def test_bolum_kendi_order_ini_tasir():
    assert build_error_analysis_section({}, "tr").order == ORDER_ERROR_ANALYSIS
