"""Faz 2 Grup C: eşik provenance + takas (F1 düz-aralık) + top-decile lift.

- Takas aralığı OOF'tan SEÇİLİR (yapısal: fonksiyon y_test almaz — Faz 2.5 deseni).
- Top-decile lift EŞİKTEN BAĞIMSIZ; tavan = min(1/decile, 1/prevalence).
- Rare-prevalence'te tavan 10× (1/decile), sade 1/prevalence (=581×) ulaşılamaz.
"""
import inspect
import numpy as np
import pytest

from src.services.ml_trainer import _oof_f1_range, _select_oof_threshold
from src.services.evaluator import _top_decile_lift


# ── Takas aralığı OOF'tan (SEÇİM), yapısal ──────────────────────────────────
def test_oof_f1_range_flat_band_ordered():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(800), np.ones(200)].astype(int)
    p = np.clip(np.where(y == 1, rng.normal(0.65, 0.2, 1000), rng.normal(0.35, 0.2, 1000)), 0, 1)
    chosen, lo, hi, max_f1 = _oof_f1_range(y, p)
    assert 0.0 < lo <= chosen <= hi < 1.0, (lo, chosen, hi)   # düşük eşik=yüksek recall ucu
    assert 0.0 < max_f1 <= 1.0


def test_tradeoff_selection_takes_no_test_data():
    """Yapısal (Faz 2.5 deseni): aralık SEÇİMİ test verisi ALMAZ → sızıntı imkânsız.
    ÖLÇÜM evaluator'da test'te (ayrı). _select_oof_threshold provenance'ı OOF aralığını taşır."""
    params = set(inspect.signature(_oof_f1_range).parameters)
    assert not any("test" in p for p in params), params
    # provenance düz-aralık uçlarını (OOF'tan) taşımalı → evaluator test'te ölçsün
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    import pandas as pd
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(700), np.ones(300)].astype(int)
    X = pd.DataFrame({"f0": y + rng.normal(0, 1, 1000), "f1": rng.normal(0, 1, 1000)})
    _, prov = _select_oof_threshold(LogisticRegression(max_iter=500), X, y,
                                    StratifiedKFold(3, shuffle=True, random_state=42), is_binary=True)
    assert "flat_lo_thr" in prov and "flat_hi_thr" in prov and prov["flat_lo_thr"] <= prov["flat_hi_thr"]


# ── Top-decile lift + tavan formülü ─────────────────────────────────────────
def test_lift_baseline_and_ceiling_churn_like():
    """Prevalence %20 → tavan min(10, 5)=~4.9×. Rastgele sıralama ~ baseline 1.0×."""
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(800), np.ones(200)].astype(int)
    p = rng.random(1000)   # bilgisiz sıralama → lift ~ 1
    out = _top_decile_lift(y, p)
    assert out["baseline"] == 1.0
    assert out["ceiling"] == pytest.approx(5.0, abs=0.05)   # 1/0.20 = 5
    assert 0.5 <= out["lift"] <= 1.8   # rastgele → baseline civarı


def test_lift_rare_prevalence_ceiling_is_10x_not_1_over_prevalence():
    """Rare-fraud %0.172 → tavan min(1/decile=10, 1/prev=581)=10×. Sade 1/prev YANLIŞ
    (pozitiflerin tümü en riskli %10'a sığar → capture 1.0'da tıkanır)."""
    n = 100000
    n_pos = 172   # %0.172
    y = np.r_[np.zeros(n - n_pos), np.ones(n_pos)].astype(int)
    p = rng = np.random.default_rng(3).random(n)
    out = _top_decile_lift(y, p)
    assert out["ceiling"] == 10.0, out["ceiling"]   # min(10, 581) = 10, 581 DEĞİL


def test_lift_perfect_ranking_hits_ceiling():
    """Mükemmel sıralama → lift = tavan."""
    y = np.r_[np.zeros(800), np.ones(200)].astype(int)
    p = y.astype(float) + np.linspace(0, 0.01, len(y))   # pozitifler en üstte
    out = _top_decile_lift(y, p)
    assert out["lift"] == pytest.approx(out["ceiling"], abs=0.05)


def test_lift_positive_is_class_one():
    """positive=1: en riskli %10 = en yüksek olasılık; capture pozitif(1) sınıfına göre."""
    y = np.r_[np.zeros(900), np.ones(100)].astype(int)
    p = y.astype(float)   # class1 prob=1
    out = _top_decile_lift(y, p)
    assert out["total_positives"] == 100 and out["positives_in_top"] == 100


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
