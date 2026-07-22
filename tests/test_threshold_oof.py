"""Faz 2.5: sınıflandırma eşiği OOF F1-max ile LEAKAGE-FREE seçiliyor.

Kök bug: eşik test setinde optimize ediliyordu (ml_trainer + evaluator ikisi de
precision_recall_curve(y_test,...)) → eşik-bağımlı tüm metrikler iyimser (winsor vakası).

Beş şart:
1. Fallback ASLA sızıntıya düşmesin — OOF yoksa 0.5 (yapısal, y_test hiç görülmez).
2. y_pred_train de OOF eşiğini kullansın (çift-yanlış yok).
3. Eşik provenance'ı (değer+kaynak+n) kaydedilsin.
5. ROC/PR-AUC eşikten bağımsız — eşik değişince değişmemeli.
"""
import numpy as np
import pytest

from src.services.ml_trainer import _oof_f1_range, _select_oof_threshold


def test_f1_optimal_threshold_range():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(800), np.ones(200)].astype(int)
    p = np.clip(np.where(y == 1, rng.normal(0.7, 0.15, 1000), rng.normal(0.3, 0.15, 1000)), 0, 1)
    thr, lo, hi, max_f1 = _oof_f1_range(y, p)   # Grup C: _f1_optimal_threshold → _oof_f1_range
    assert 0.0 < thr < 1.0 and lo <= thr <= hi


def test_oof_threshold_binary_provenance():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    import pandas as pd
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(700), np.ones(300)].astype(int)
    X = pd.DataFrame({"f0": y + rng.normal(0, 1, 1000), "f1": rng.normal(0, 1, 1000)})
    cv = StratifiedKFold(3, shuffle=True, random_state=42)
    thr, prov = _select_oof_threshold(LogisticRegression(max_iter=500), X, y, cv, is_binary=True)
    assert 0.0 < thr < 1.0
    assert prov["source"] == "oof_cv_f1max" and prov["n"] == 1000
    assert prov["reused_blend_oof"] is False


def test_oof_threshold_reuses_precomputed_blend_oof():
    """Blend seçildiyse harmanlanmış OOF REUSE — cross_val_predict koşmaz (ek maliyet ~0)."""
    y = np.r_[np.zeros(700), np.ones(300)].astype(int)
    oof = np.where(y == 1, 0.72, 0.28)   # ayrışan OOF olasılığı
    thr, prov = _select_oof_threshold(None, None, y, None, is_binary=True, precomputed_oof=oof)
    assert 0.0 < thr < 1.0
    assert prov["source"] == "oof_cv_f1max" and prov["reused_blend_oof"] is True


def test_multiclass_returns_default_half():
    thr, prov = _select_oof_threshold(None, None, np.array([0, 1, 2, 0, 1, 2]), None, is_binary=False)
    assert thr == 0.5 and prov["source"] == "default_0.5" and prov["reason"] == "multiclass"


def test_fallback_never_leaks_on_failure():
    """OOF üretilemezse (model clone/CV patlar) → 0.5, DEFAULT. TEST-optimize ASLA.
    Yapısal kanıt: _select_oof_threshold y_test parametresi ALMAZ — sızıntı imkânsız."""
    import inspect
    from sklearn.model_selection import StratifiedKFold
    class _Boom:   # clone/fit patlatan sahte model
        def get_params(self, deep=True): return {}
        def set_params(self, **kw): return self
        def fit(self, *a, **k): raise RuntimeError("boom")
    thr, prov = _select_oof_threshold(_Boom(), np.zeros((10, 2)), np.r_[np.zeros(5), np.ones(5)].astype(int),
                                      StratifiedKFold(2), is_binary=True)
    assert thr == 0.5 and prov["source"] == "default_0.5" and "reason" in prov
    # Yapısal: fonksiyon imzasında y_test/test YOK
    params = set(inspect.signature(_select_oof_threshold).parameters)
    assert not any("test" in p for p in params), f"threshold seçimi test verisi almamalı: {params}"


def test_evaluator_consumes_threshold_no_test_selection():
    """evaluator eşiği TÜKETİR, y_test'ten HESAPLAMAZ: verilen eşik y_pred'i belirler."""
    import inspect
    from src.services.evaluator import _evaluate_classification
    sig = inspect.signature(_evaluate_classification)
    assert "threshold" in sig.parameters and "threshold_provenance" in sig.parameters
    # Kaynak: test-optimize precision_recall_curve(y_test) eşik-SEÇİMİ kalmadı
    src = inspect.getsource(_evaluate_classification)
    # PR-curve yalnız GÖSTERİM için (y_test ile) olabilir; ama argmax(f1)+pr_thresholds SEÇİMİ olmamalı
    assert "pr_thresholds[best_pr_idx]" not in src, "test-optimize eşik seçimi hala var"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
