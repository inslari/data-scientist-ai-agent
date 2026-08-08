"""Katman 2 — gerçekleşen performans çekirdeği (SAF, DB yok).

Doğrular: ROC-AUC bilinen-cevap + sklearn çapraz-kontrol (bağlı/bağsız) · accuracy tip-toleranslı
eşitlik ("1"==1) · pozitif-oran + seçilim-yanlılığı sapması (deterministik kanıt) · pozitif-sınıf
yokken roc None ama accuracy dolu (bastırma değil) · regresyon rmse/mae.
"""
import math

import pytest

from src.services.monitoring.realized import (
    _roc_auc, compute_realized_metrics,
)


def test_roc_auc_known_answer():
    # Mükemmel ayrım: tüm pozitiflerin skoru negatiflerden büyük → AUC=1.0
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert _roc_auc(labels, scores) == 1.0
    # Ters → 0.0
    assert _roc_auc([1, 1, 0, 0], scores) == 0.0
    # Tek sınıf → None (tanımsız)
    assert _roc_auc([1, 1, 1], [0.3, 0.6, 0.9]) is None


def test_roc_auc_ties_match_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics")
    labels = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
    scores = [0.5, 0.5, 0.3, 0.7, 0.7, 0.2, 0.9, 0.5, 0.6, 0.4]   # bağlar dahil (0.5 ×3, 0.7 ×2)
    ours = _roc_auc(labels, scores)
    ref = sklearn.roc_auc_score(labels, scores)
    assert abs(ours - ref) < 1e-9, f"ours={ours} sklearn={ref}"


def test_accuracy_type_tolerant_equality():
    # prediction int, actual_label CSV'den string → "1"==1 sayılmalı
    m = compute_realized_metrics(
        "binary", preds=[1, 0, 1, 0], actuals=["1", "0", "0", "0"],
        probs=[0.9, 0.1, 0.8, 0.2], positive_label=1,
        baseline={"accuracy": 0.75, "roc_auc": 0.9, "positive_rate": 0.5})
    assert m["accuracy"] == 0.75          # 4'ten 3 doğru (3. satır pred=1/actual=0 yanlış)
    assert m["baseline_accuracy"] == 0.75


def test_positive_rate_deviation_selection_bias():
    # Etiketli alt-küme %75 pozitif ama baseline prevalansı %30 → sansür yanlılığı ÖLÇÜLÜR
    m = compute_realized_metrics(
        "binary", preds=[1, 1, 1, 0], actuals=[1, 1, 1, 0],
        probs=[0.9, 0.8, 0.7, 0.2], positive_label=1,
        baseline={"accuracy": 0.8, "roc_auc": 0.85, "positive_rate": 0.30})
    assert m["labeled_positive_rate"] == 0.75
    assert m["baseline_positive_rate"] == 0.30
    assert m["positive_rate_deviation"] == 0.45      # deterministik kanıt (kod basar, dashboard değil)
    assert m["assumes"] == "labels_missing_at_random"
    assert m["roc_auc"] == 1.0                        # pozitifler negatiften yüksek


def test_no_positive_label_roc_suppressed_accuracy_kept():
    # pozitif sınıf yok (multiclass / baseline'da yok) → roc None AMA accuracy hesaplanır (bastırma değil)
    m = compute_realized_metrics(
        "multiclass", preds=["a", "b", "c"], actuals=["a", "b", "x"],
        probs=[{"a": 0.9}, {"b": 0.8}, {"c": 0.6}], positive_label=None,
        baseline={"accuracy": 0.7})
    assert m["accuracy"] == round(2 / 3, 6)
    assert m["roc_auc"] is None
    assert m["labeled_positive_rate"] is None


def test_regression_rmse_mae():
    m = compute_realized_metrics(
        "regression", preds=[2.0, 4.0, 6.0], actuals=[1.0, 4.0, 9.0],
        probs=[None, None, None], positive_label=None,
        baseline={"rmse": 2.0, "mae": 1.5})
    # errors: -1, 0, +3 → rmse=sqrt((1+0+9)/3)=sqrt(3.333..), mae=(1+0+3)/3 (6 haneye yuvarlı)
    assert abs(m["rmse"] - math.sqrt(10 / 3)) < 1e-5
    assert abs(m["mae"] - 4 / 3) < 1e-5
    assert m["baseline_rmse"] == 2.0
    assert m["labeled_n"] == 3


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-s"]))
