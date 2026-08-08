"""Katman 2 Parça 4 — CBPE (Confidence-Based Performance Estimation) çekirdeği. SAF (DB yok, testli).

Kalibre olasılıklardan GERÇEKLEŞEN etiket OLMADAN beklenen performansı tahmin eder. NannyML DEĞİL
(2 metrik için ağır bağımlılık); ~30 satır deterministik formül. İki metrik: roc_auc + accuracy.

⚠️ BİLİNEN SINIR (kod + şema + dashboard "tahmini" damgası): CBPE **veri/covariate drift** altında
çalışır, **concept drift** altında ÇALIŞMAZ — ve concept drift olduğunda tahmin SİSTEMATİK İYİMSER
sapar (baseline'a yakın kalır), yani doğru ölçümün EN ÇOK gerektiği anda yanıltır. Bu yüzden
`cbpe_estimate` daima `"assumes": "covariate_drift_only"` taşır ve gerçekleşen-performans (actual_label
backfill) geldiğinde CBPE'nin yerini alır. Ön koşul: KALİBRASYON (kapı aggregator'da, is_well_calibrated).
"""
from __future__ import annotations

from itertools import groupby


def _expected_accuracy(probs: list, threshold: float) -> float:
    """Eşik t'de beklenen accuracy: pred_i = p_i≥t → P(doğru_i) = p_i (pred=1) ya da 1−p_i (pred=0)."""
    return sum((p if p >= threshold else 1.0 - p) for p in probs) / len(probs)


def _expected_auc(probs: list) -> "float | None":
    """Beklenen ROC-AUC (kalibre p hem skor hem etiket olasılığı):
        AUC = Σ_{i≠j} p_i(1−p_j)·[1 eğer p_i>p_j, 0.5 eşit] / (Σp_i·Σ(1−p_j) − Σ p_i(1−p_i)).
    O(n log n): artan sırada tarama; her grup için strictly-smaller katkısı + grup-içi bağlar (0.5)."""
    sum_p = sum(probs)
    sum_q = sum(1.0 - p for p in probs)
    denom = sum_p * sum_q - sum(p * (1.0 - p) for p in probs)
    if denom <= 0:                                    # tek sınıf beklentisi (Σp≈0 ya da Σq≈0) → tanımsız
        return None
    num = 0.0
    seen_q = 0.0                                       # Σ(1−p_j) strictly-smaller p_j için
    for val, grp in groupby(sorted(probs)):
        g = sum(1 for _ in grp)
        num += g * val * seen_q                         # strictly-greater çiftler
        num += 0.5 * val * (1.0 - val) * g * (g - 1)    # grup-içi bağlar (p_i==p_j)
        seen_q += g * (1.0 - val)
    return num / denom


def estimate_performance(probs: list, threshold: float) -> dict:
    """Binary kalibre pozitif-olasılıklardan {roc_auc, accuracy} tahmini. probs = prod pencere."""
    return {"roc_auc": _expected_auc(probs), "accuracy": _expected_accuracy(probs, threshold)}
