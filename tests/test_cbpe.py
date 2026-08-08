"""Katman 2 Parça 4 — CBPE çekirdeği (saf, DB'siz). ROC-AUC formülü İNCE → bilinen cevaplara +
sklearn'e karşı çitlenir (sessizce yanlış olabilir)."""
import numpy as np
from sklearn.metrics import roc_auc_score

from src.services.monitoring.cbpe import _expected_accuracy, _expected_auc, estimate_performance


def test_auc_perfect_separation_near_one():
    probs = [0.99] * 500 + [0.01] * 500                    # kalibre + neredeyse ayrık
    assert _expected_auc(probs) > 0.98


def test_auc_random_is_half():
    assert abs(_expected_auc([0.5] * 1000) - 0.5) < 1e-9   # tümü 0.5 → tam 0.5


def test_auc_single_class_expectation_none():
    assert _expected_auc([0.0] * 100) is None              # Σp≈0 → denom≤0 → tanımsız


def test_auc_matches_sklearn_on_calibrated_data():
    # y_i ~ Bernoulli(p_i) → İNŞAEN kalibre. Beklenen AUC ≈ gerçekleşen (sklearn), n büyük.
    rng = np.random.default_rng(7)
    p = rng.uniform(0.02, 0.98, 8000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    est = _expected_auc(list(p))
    actual = roc_auc_score(y, p)
    assert abs(est - actual) < 0.02, f"CBPE AUC {est:.4f} vs sklearn {actual:.4f}"


def test_expected_accuracy_known_case():
    # eşik 0.5: p=0.9(pred1,P=0.9) · p=0.8(pred1,P=0.8) · p=0.2(pred0,P=0.8) · p=0.1(pred0,P=0.9)
    acc = _expected_accuracy([0.9, 0.8, 0.2, 0.1], 0.5)
    assert abs(acc - (0.9 + 0.8 + 0.8 + 0.9) / 4) < 1e-9


def test_estimate_performance_shape():
    out = estimate_performance([0.9, 0.1, 0.8, 0.2] * 100, 0.5)
    assert set(out) == {"roc_auc", "accuracy"}
    assert 0.0 <= out["roc_auc"] <= 1.0 and 0.0 <= out["accuracy"] <= 1.0


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-q"]))
