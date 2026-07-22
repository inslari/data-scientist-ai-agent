"""Grup A (Faz 2): sınıf dengesi kademesi + baseline'lar + ilgi sınıfı kararlılığı.

- Kademe (balanced/imbalanced/severe) üniform-pay türevli eşikten — sihirli 0.4 yok, multi-class'ta çalışır.
- İlgi sınıfı POZİTİF-ÖNCELİKLİ → 50/50'ye yakın veride örneklemden BAĞIMSIZ (kararlı, split çevirse de sabit).
- Brier baseline BINARY'ye özel (p(1-p)); multi-class'ta hesaplanmaz (guard).
"""
import numpy as np
import pytest

from src.services.evaluator import _compute_class_balance, _evaluate_classification


def _y(counts: dict) -> np.ndarray:
    return np.array([lab for lab, n in counts.items() for _ in range(n)])


# ── Kademe: üniform-pay türevli eşik ─────────────────────────────────────────
def test_level_balanced_binary():
    cb = _compute_class_balance(_y({0: 700, 1: 300}))   # 30/70 → 0.30 > 0.25 (0.5×0.5)
    assert cb["level"] == "balanced", cb


def test_level_imbalanced_binary_churn():
    cb = _compute_class_balance(_y({0: 7965, 1: 2035}))  # churn %20.35 < 0.25
    assert cb["level"] == "imbalanced"
    assert cb["minority_class"] == 1 and cb["minority_rate"] == pytest.approx(0.2035, abs=1e-3)


def test_level_severe_binary_fraud():
    cb = _compute_class_balance(_y({0: 99828, 1: 172}))  # fraud %0.172 < 0.025 (0.5/20)
    assert cb["level"] == "severe"


def test_level_threshold_is_uniform_derived_multiclass():
    # 3-sınıf: uniform=0.333, imbalanced eşiği 0.167, severe eşiği 0.0167. Sihirli sabit yok.
    cb_imb = _compute_class_balance(_y({0: 500, 1: 400, 2: 100}))   # minority 0.10 < 0.167
    assert cb_imb["level"] == "imbalanced" and cb_imb["n_classes"] == 3
    cb_bal = _compute_class_balance(_y({0: 400, 1: 350, 2: 250}))   # minority 0.25 > 0.167
    assert cb_bal["level"] == "balanced"


# ── İlgi sınıfı kararlılığı (pozitif-öncelikli) ──────────────────────────────
@pytest.mark.parametrize("minority_side", [(499, 501), (501, 499)])
def test_interest_class_stable_near_5050(minority_side):
    """50/50'ye yakın: azınlık taraf çevrilse bile ilgi sınıfı = pozitif (label 1) SABİT.
    Azınlık-tabanlı olsaydı split'e göre değişirdi (Faz 0.5 kararlılık ilkesi)."""
    n0, n1 = minority_side
    cb = _compute_class_balance(_y({0: n0, 1: n1}))
    assert cb["positive_class"] == 1
    assert cb["interest_classes"][0] == 1, "ilgi sınıfı pozitif olmalı (örneklemden bağımsız)"


def test_interest_shows_both_when_positive_not_minority():
    """Pozitif (1) çoğunluksa: ilgi = [pozitif, azınlık] — sessizce birini seçme."""
    cb = _compute_class_balance(_y({0: 200, 1: 800}))  # pozitif=1 çoğunluk, azınlık=0
    assert cb["positive_class"] == 1 and cb["minority_class"] == 0
    assert set(cb["interest_classes"]) == {0, 1}


def test_interest_single_when_positive_is_minority():
    cb = _compute_class_balance(_y({0: 800, 1: 200}))  # churn tipi: pozitif=azınlık=1
    assert cb["interest_classes"] == [1]


# ── Baseline'lar + multi-class Brier guard (evaluator uçtan) ──────────────────
def _fit_eval(y_counts: dict, seed=42):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    y = _y(y_counts)
    n = len(y)
    # Sınıfa göre ayrışan basit sinyal (model predict_proba üretebilsin). Evaluator
    # DataFrame bekler (feature importance/raw_test .iloc kullanır) → DataFrame ver.
    X = pd.DataFrame({"f0": y + rng.normal(0, 1.0, n), "f1": rng.normal(0, 1, n)})
    idx = rng.permutation(n)
    cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
    ytr, yte = y[tr], y[te]
    m = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    return _evaluate_classification(m, Xtr, ytr, Xte, yte, ["f0", "f1"])


def test_binary_baselines_present():
    ev = _fit_eval({0: 800, 1: 200})
    assert ev["class_balance"]["level"] == "imbalanced"
    # accuracy baseline = majority_rate
    assert ev["test_metrics"]["accuracy_baseline"] == pytest.approx(ev["class_balance"]["majority_rate"])
    # Brier baseline = p(1-p) (binary)
    cal = ev.get("calibration") or {}
    assert cal.get("brier_baseline") is not None


def test_multiclass_brier_guard_no_baseline():
    ev = _fit_eval({0: 500, 1: 400, 2: 300})
    assert ev["class_balance"]["n_classes"] == 3
    assert ev["class_balance"]["positive_class"] is None       # multi-class: tek pozitif yok
    # BINARY-özel Brier baseline ÇOK SINIFLIDA hesaplanmaz (calibration bloğu is_binary guard'lı)
    assert (ev.get("calibration") or {}).get("brier_baseline") is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
