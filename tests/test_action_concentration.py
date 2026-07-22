"""Aksiyon Yoğunlaşması (D3) — top-decile grup dağılımı.

- grup→decile rep_ratio; yön base_rate ile (beklenen konsantrasyon FLAG'lenmez).
- işaret-uyuşmazlığı + CI kapısı; kapsama boşluğu warn adayı, over-konsantrasyon info.
- çift-taraflı n-guard; nedensellik yasağı; kolon seçici demografi-agnostik.
"""
import numpy as np
import pandas as pd
import pytest

from src.services.action_concentration import compute_action_concentration, PROPORTIONALITY_BAND
from src.services.segment_carrier import select_concentration_columns


def _build(specs):
    """specs: her satır için (group, y, in_decile). prob'u decile satırları tepeye
    gelecek şekilde kurar (deterministik, gürültüsüz). n %10 = decile satır sayısı olmalı."""
    n = len(specs)
    groups = [s[0] for s in specs]
    y = np.array([s[1] for s in specs])
    prob = np.zeros(n, dtype=float)
    dec = [i for i, s in enumerate(specs) if s[2]]
    non = [i for i, s in enumerate(specs) if not s[2]]
    assert len(dec) == max(1, round(0.10 * n)), (len(dec), round(0.10 * n))
    for r, i in enumerate(dec):
        prob[i] = 0.99 - r * 1e-4
    for r, i in enumerate(non):
        prob[i] = 0.50 - r * 1e-4
    return prob, pd.DataFrame({"g": groups}), y


def _grp(specs, value):
    prob, raw, y = _build(specs)
    ac = compute_action_concentration(prob, raw, ["g"], y)
    return ac, ac["columns"]["g"]["groups"].get(value)


# ── Kolon seçici: demografi-agnostik ─────────────────────────────────────────

def test_selector_demografi_agnostik_kredi_kolonu_secer():
    df = pd.DataFrame({"loan_grade": ["A", "B", "C"] * 10,
                       "person_home_ownership": ["RENT", "OWN"] * 15})
    cols, skipped = select_concentration_columns(df)
    assert "loan_grade" in cols and "person_home_ownership" in cols   # demografik olmasa da


def test_selector_turev_ve_yuksek_kardinalite_eler():
    df = pd.DataFrame({"ratio_x_y": [0.1] * 30, "customer_id": range(30),
                       "loan_grade": (["A", "B", "C"] * 10)})
    cols, skipped = select_concentration_columns(df)
    assert cols == ["loan_grade"]                     # türev + ID (yüksek kard.) elendi


# ── Yön netliği: beklenen konsantrasyon FLAG'lenmez ──────────────────────────

def test_beklenen_konsantrasyon_flaglenmez():
    """Grade D/E vakası: rep yüksek AMA base_rate de yüksek → iki gözlem hizalı → flag YOK."""
    # 200 satır, n_top=20. G: 40 satır, 18'i decile'da, 36'sı pozitif (base 0.9).
    specs = [("G", 1, True)] * 18 + [("G", 1, False)] * 18 + [("G", 0, False)] * 4
    specs += [("A", 0, True)] * 2 + [("A", 0, False)] * 158       # A: düşük risk, decile'ı doldurur
    ac, g = _grp(specs, "G")
    assert g["rep_ratio"] >= PROPORTIONALITY_BAND[1]              # over-temsil
    assert g["base_rate_ratio"] >= PROPORTIONALITY_BAND[1]        # yüksek taban riski
    assert g["kind"] == "expected"
    assert g["flag_severity"] is None                            # KUSUR DEĞİL


def test_over_konsantrasyon_dusuk_taban_info():
    """Proxy vakası: rep yüksek ama base düşük → beklenmedik → info (warn değil)."""
    # Q: 40 satır, 15'i decile'da, sadece 4'ü pozitif (base 0.10, düşük).
    specs = [("Q", 1, True)] * 4 + [("Q", 0, True)] * 11 + [("Q", 0, False)] * 25
    specs += [("A", 1, True)] * 5 + [("A", 1, False)] * 35 + [("A", 0, False)] * 120
    ac, g = _grp(specs, "Q")
    # ÜÇ KOŞULUN KESİŞİMİ (tek koşul değil): rep band-dışı + base ≤ 1 + rep CI 1.0'ı dışlar.
    assert g["rep_ratio"] >= PROPORTIONALITY_BAND[1]              # (1) over-temsil
    assert g["base_rate_ratio"] <= 1.0                           # (2) taban riski düşük
    assert g["rep_ci"]["lower"] > 1.0                            # (3) CI kapısı: yığılma gerçek
    assert g["kind"] == "over_unexpected" and g["flag_severity"] == "info"


def test_kapsama_boslugu_warn():
    """Riskli grup decile'da yok + ikisi de CI-anlamlı → kapsama boşluğu WARN."""
    # P: 40 satır, 30'u pozitif (base 0.75), HİÇBİRİ decile'da değil.
    specs = [("P", 1, False)] * 30 + [("P", 0, False)] * 10
    specs += [("A", 1, True)] * 20 + [("A", 0, False)] * 140     # A decile'ı doldurur
    ac, g = _grp(specs, "P")
    # DÖRT KOŞULUN KESİŞİMİ: rep band-altı + base band-üstü + rep CI ve base CI ikisi de anlamlı.
    assert g["rep_ratio"] <= PROPORTIONALITY_BAND[0]             # (1) az/hiç temsil
    assert g["base_rate_ratio"] >= PROPORTIONALITY_BAND[1]       # (2) yüksek taban riski
    assert g["rep_ci"]["upper"] < 1.0                           # (3) rep CI 1.0'ı dışlar (gerçek under)
    assert g["base_rate_ci"]["lower"] > ac["prevalence"]        # (4) base CI prevalansı dışlar (warn kapısı)
    assert g["kind"] == "coverage_gap" and g["flag_severity"] == "warn"


def test_orantili_grup_flaglenmez():
    """rep ≈ 1 → orantılı → flag yok."""
    # M: 100 satır (pay 0.5), decile'ın 10'u (pay 0.5) → rep 1.0.
    specs = [("M", 0, True)] * 10 + [("M", 0, False)] * 90
    specs += [("N", 0, True)] * 10 + [("N", 0, False)] * 90
    ac, g = _grp(specs, "M")
    assert g["rep_ratio"] == pytest.approx(1.0, abs=0.01)
    assert g["flag_severity"] is None


# ── Çift-taraflı n-guard ─────────────────────────────────────────────────────

def test_ince_grup_olculmez_gorunur_sebep():
    """pop_count < min_n → ölçülmez, skipped_groups'ta görünür (sessiz eleme yok)."""
    specs = [("tiny", 1, True)] * 2 + [("big", 0, True)] * 18 + [("big", 0, False)] * 180
    prob, raw, y = _build(specs)
    ac = compute_action_concentration(prob, raw, ["g"], y)
    sk = ac["columns"]["g"]["skipped_groups"]
    assert "tiny" in sk and sk["tiny"]["code"] == "ac_thin_group"
    assert "tiny" not in ac["columns"]["g"]["groups"]


def test_ince_decile_analiz_yapilmaz():
    """n_top < min_n → tüm analiz not_computed (analiz-düzeyi guard)."""
    prob = np.linspace(0.9, 0.1, 30)                            # n=30 → n_top=3 < 5
    raw = pd.DataFrame({"g": ["a", "b"] * 15})
    ac = compute_action_concentration(prob, raw, ["g"], (np.arange(30) % 2))
    assert ac["status"] == "not_computed" and ac["reason"]["code"] == "ac_thin_decile"


def test_hizalama_bozuksa_reddedilir():
    prob = np.array([0.9, 0.8, 0.7])
    raw = pd.DataFrame({"g": ["a", "b"]})                        # 2 ≠ 3
    ac = compute_action_concentration(prob, raw, ["g"], np.array([1, 0, 1]))
    assert ac["status"] == "not_computed" and ac["reason"]["code"] == "ac_misaligned"
