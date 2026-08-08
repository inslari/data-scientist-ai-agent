"""Katman 2 — GERÇEKLEŞEN performans çekirdeği. SAF fonksiyonlar (DB yok → tek başına testli).

``actual_label`` backfill'i geldiğinde CBPE'nin (tahmini performans) yerini alır: burası TAHMİN değil
GERÇEK etiketle ölçülen accuracy/roc_auc (sınıflandırma) ya da rmse/mae (regresyon). CBPE'nin kendi
doğrulaması — dashboard "tahmini 0.81 / gerçekleşen 0.79" der.

⚠️ SEÇİLİM YANLILIĞI (kod + şema + dashboard, CBPE'nin concept-drift damgasıyla AYNI disiplin):
etiketler rastgele gelmez. Genelde "olayı gerçekleşen" önce etiketlenir (churn edenler belli olur,
etmeyenler sansürlü bekler) → erken/düşük-kapsam etiket kümesi pozitif-ağırlıklı olabilir → gerçekleşen
metrik YANLI çıkar. İki savunma: (1) ``assumes="labels_missing_at_random"`` damgası; (2) DETERMİNİSTİK
KANIT — ``labeled_positive_rate`` vs ``baseline_positive_rate`` (``positive_rate_deviation``): etiketli
alt-kümenin pozitif oranı baseline prevalansından belirgin sapıyorsa sansür yanlılığı VARSAYIM değil
ÖLÇÜLMÜŞ olur. Dashboard sapmayı GÖSTERİR, hesabı burada yapar ("deterministik olan kod tarafından basılır").
"""
from __future__ import annotations

import math


def _eq(a, b) -> bool:
    """Etiket eşitliği tip-toleranslı: prediction/actual_label JSON'dan gelir, "1"==1 olmalı
    (kullanıcı CSV'si string, model classes_ int olabilir). str-normalize karşılaştırma."""
    return str(a) == str(b)


def _roc_auc(labels_bin: list, scores: list) -> "float | None":
    """GERÇEK ROC-AUC (bağ-farkında, Mann-Whitney U = ortalama-rank formülü, sklearn ile birebir).

    labels_bin: 0/1 (1 = pozitif sınıf); scores: pozitif-sınıf olasılığı (hizalı). Tek sınıf → None."""
    n = len(labels_bin)
    n_pos = sum(labels_bin)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None                                     # tek sınıf → AUC tanımsız
    order = sorted(range(n), key=lambda i: scores[i])   # skora göre artan
    ranks = [0.0] * n
    i = 0
    while i < n:                                         # bağları ortalama-rank ile çöz
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0                  # 1-tabanlı ortalama rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_ranks_pos = sum(ranks[i] for i in range(n) if labels_bin[i] == 1)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _regression(preds, actuals, baseline) -> dict:
    pairs = []
    for p, a in zip(preds, actuals):
        try:
            pairs.append((float(p), float(a)))
        except (TypeError, ValueError):
            continue
    m = len(pairs)
    if m == 0:
        return {"labeled_n": len(actuals), "rmse": None, "mae": None,
                "baseline_rmse": (baseline or {}).get("rmse"),
                "assumes": "labels_missing_at_random"}
    rmse = math.sqrt(sum((p - a) ** 2 for p, a in pairs) / m)
    mae = sum(abs(p - a) for p, a in pairs) / m
    return {
        "labeled_n": len(actuals),
        "rmse": round(rmse, 6), "mae": round(mae, 6),
        "baseline_rmse": (baseline or {}).get("rmse"),
        "baseline_mae": (baseline or {}).get("mae"),
        "assumes": "labels_missing_at_random",
    }


def compute_realized_metrics(task: str, preds: list, actuals: list, probs: list,
                             positive_label, baseline: "dict | None") -> dict:
    """Etiketli pencere satırlarından gerçekleşen metrik. Hizalı listeler (yalnız actual_label DOLU satırlar).

    task: "binary" | "multiclass" | "regression". Regresyon → rmse/mae. Sınıflandırma → accuracy (+
    binary'de roc_auc, pozitif oran, sapma). ``positive_label`` YOKSA (baseline'da yok / regresyon) →
    roc_auc + pozitif-oran alanları None; accuracy YİNE hesaplanır (bkz. gate: bu bir bastırma DEĞİL —
    accuracy geçerli, yalnız roc_auc None; "status yalan söyleyemez" → gate="ok" kalır)."""
    if task == "regression":
        return _regression(preds, actuals, baseline)

    n = len(actuals)
    baseline = baseline or {}
    correct = sum(1 for p, a in zip(preds, actuals) if _eq(p, a))
    out = {
        "labeled_n": n,
        "accuracy": round(correct / n, 6) if n else None,
        "baseline_accuracy": baseline.get("accuracy"),
        "roc_auc": None,
        "baseline_roc_auc": baseline.get("roc_auc"),
        "labeled_positive_rate": None,
        "baseline_positive_rate": None,
        "positive_rate_deviation": None,
        "assumes": "labels_missing_at_random",
    }
    if positive_label is None:                          # multiclass / pozitif sınıf bilinmiyor
        return out

    labels_bin = [1 if _eq(a, positive_label) else 0 for a in actuals]
    lpr = sum(labels_bin) / n if n else None
    bpr = baseline.get("positive_rate")
    out["labeled_positive_rate"] = round(lpr, 6) if lpr is not None else None
    out["baseline_positive_rate"] = bpr
    out["positive_rate_deviation"] = (round(lpr - bpr, 6)
                                      if (lpr is not None and bpr is not None) else None)
    if task == "binary":
        scores = [p if isinstance(p, (int, float)) else None for p in probs]
        if all(s is not None for s in scores):          # her satırda pozitif-olasılık şart
            auc = _roc_auc(labels_bin, scores)
            out["roc_auc"] = round(auc, 6) if auc is not None else None
    return out
