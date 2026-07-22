"""#10: binary is_zero YASAĞI (+ coarsening istisnası, Faz 0 invariant çözümü) ve
near-dup tavsiyesinin ham↔türetilmiş ayrımı.

Bu testler "neden bazı is_zero var, bazıları yok" sorusunu KALICI cevaplar:
  - 2-değerli kolon → is_zero = orijinalin birebir kopyası (corr −1.0) → ÜRETİLMEZ.
  - 3+ değerli kolon → is_zero bir COARSENING (hangi sıfır-dışı değer olduğunu atar),
    ayrı sinyal taşıyabilir → KORUNUR.
  - binary bir kolon oran paydası (force_zero) OLSA BİLE üretilmez: Faz 0 invariant
    "sıfır bilgisi MEVCUT olsun" der, binary'de bilgi zaten korunan orijinalde tam.
"""
import numpy as np
import pandas as pd

from src.services.feature_builder import _build_zero_missing_flags
from src.services.report_sections import build_relations_section


def _flags(df: pd.DataFrame, force=None) -> list[str]:
    _, added = _build_zero_missing_flags(
        df.copy(), set(df.columns), "classification", False, force_zero_cols=force)
    return added


# ── binary is_zero yasağı + coarsening istisnası ─────────────────────────────

def test_binary_is_zero_uretilmez_ama_cokdegerli_korunur():
    df = pd.DataFrame({
        "credit_card": np.array([0] * 150 + [1] * 350),                       # binary (%30 sıfır)
        "products": np.array([0] * 100 + [1] * 150 + [2] * 150 + [3] * 100),  # 4-değerli (%20 sıfır)
    })
    added = _flags(df)
    assert "is_zero_credit_card" not in added, "binary → garantili kopya, üretilmemeli"
    assert "is_zero_products" in added, "3+ değerli → coarsening ayrı sinyal, korunmalı"


def test_binary_invariant_force_da_uretmez():
    """binary kolon oran paydası (force_zero) olsa BİLE is_zero üretilmez (invariant orijinalle sağlanır)."""
    df = pd.DataFrame({
        "active_member": np.array([0] * 160 + [1] * 340),          # binary
        "amount": np.concatenate([np.zeros(60), np.arange(1, 441)]),  # çok-değerli
    })
    added = _flags(df, force={"active_member", "amount"})
    assert "is_zero_active_member" not in added, "force OLSA BİLE binary → üretme (invariant orijinalde)"
    assert "is_zero_amount" in added, "çok-değerli payda → invariant force ile üretilir"


def test_iki_degerli_sifirsiz_zaten_elenir():
    """{0 içermeyen} 2-değerli kolon nunique kapısına ulaşmadan s.all()/not s.any() ile elenir."""
    df = pd.DataFrame({
        "flag_1_2": np.array([1] * 200 + [2] * 300),               # 2-değerli, sıfır YOK
        "amount": np.concatenate([np.zeros(50), np.arange(1, 451)]),
    })
    added = _flags(df)
    assert "is_zero_flag_1_2" not in added                          # is_zero hep-sıfır → zaten yok
    assert "is_zero_amount" in added


def test_sabit_kolon_is_zero_uretmez():
    """nunique=1 sabit kolon: nunique kapısına ulaşmaz (s.all()/not s.any() eler) — sessiz kayıp yok."""
    df = pd.DataFrame({
        "const0": np.zeros(300, dtype=int),                        # sabit 0 → s.all()
        "amount": np.concatenate([np.zeros(40), np.arange(1, 261)]),
    })
    added = _flags(df)
    assert "is_zero_const0" not in added
    assert "is_zero_amount" in added


# ── near-dup: ham↔ham vs türetilmiş tavsiye ayrımı ───────────────────────────

def _relations_state(lang="tr"):
    return {
        "report_language": lang,
        "relations_report": {
            "generated_from": "train_only", "n_rows": 1000, "target": "churn",
            "target_relations": [{"feature": "age", "score": 0.3, "ci": None}],
            "top_pairs": [],
            "redundant_pairs": [
                {"a": "balance", "b": "estimated_salary", "corr": 0.98},   # ham↔ham
                {"a": "ratio_balance_age", "b": "credit_score", "corr": -0.97},  # türetilmiş dahil
            ],
        },
    }


def _redundant_subsection(sec):
    return next(s for s in sec.subsections if s.id == "data_relations_redundant")


def test_neardup_ham_cift_veri_toplama_tavsiyesi():
    red = _redundant_subsection(build_relations_section(_relations_state("tr"), "tr"))
    raw_ev = red.evidence[0]        # balance ↔ estimated_salary (ikisi de ham)
    assert "veri toplama" in raw_ev.note and "özellik üretim" not in raw_ev.note


def test_neardup_turetilmis_cift_feature_engineering_notu():
    red = _redundant_subsection(build_relations_section(_relations_state("tr"), "tr"))
    der_ev = red.evidence[1]        # ratio_balance_age ↔ credit_score (biri türev)
    assert "özellik üretiminde" in der_ev.note and "DEĞİL" in der_ev.note
    # eski yanlış tavsiye ("veri toplama fırsatı") türetilmiş çiftte OLMAMALI
    assert "yalnız birini toplamak" not in der_ev.note


def test_neardup_en_dili_dogru_tr_sizmaz():
    red = _redundant_subsection(build_relations_section(_relations_state("en"), "en"))
    assert "feature generation" in red.evidence[1].note                  # türetilmiş
    assert "collecting only one suffices" in red.evidence[0].note        # ham
    for e in red.evidence:
        assert "özellik" not in e.note and "veri toplama" not in e.note   # TR sızıntısı yok
