from __future__ import annotations
from typing import Any


def evaluate_quality_gate(state: dict) -> dict:
    """
    Modelin sonucunu değerlendirip kullanıcıya nasıl sunulacağına karar verir.
    
    Status seviyeleri:
      pass  - Model güvenilir, normal raporla
      warn  - Çalışıyor ama dikkat edilmesi gereken durumlar var
      fail  - Yetersiz, üretime alınmamalı
    """
    evaluation_report = state.get("evaluation_report") or {}
    test_metrics      = evaluation_report.get("test_metrics") or {}
    overfitting       = evaluation_report.get("overfitting") or {}
    reliability       = evaluation_report.get("reliability_score") or {}
    fi_summary        = evaluation_report.get("feature_importance_summary") or {}
    domain_leakage    = state.get("domain_leakage_report") or {}
    task              = evaluation_report.get("task", "")

    failures: list[dict] = []
    warnings: list[dict] = []
    checks_performed: list[str] = []

    # ── FAIL kontrolleri ──────────────────────────────
    
    # Düşük performans (classification)
    if "classification" in task:
        checks_performed.append("classification_performance")
        auc = test_metrics.get("roc_auc", test_metrics.get("roc_auc_ovr", 0))
        if auc < 0.60:
            failures.append({
                "code": "low_auc",
                "message": f"AUC çok düşük ({auc:.3f}). Model rastgele tahminden zar zor iyi.",
                "value": round(auc, 3),
            })

    # Düşük performans (regression)
    if "regression" in task:
        checks_performed.append("regression_performance")
        r2 = test_metrics.get("r2", 0)
        if r2 < 0.30:
            failures.append({
                "code": "low_r2",
                "message": f"R² çok düşük ({r2:.3f}). Model hedefi açıklayamıyor.",
                "value": round(r2, 3),
            })

    # Ciddi overfitting
    checks_performed.append("overfitting")
    gap = overfitting.get("gap", 0) or 0
    if gap > 0.20:
        failures.append({
            "code": "severe_overfitting",
            "message": f"Ciddi overfitting (train-test farkı: {gap:.3f}). Model eğitim verisini ezberlemiş.",
            "value": round(gap, 3),
        })

    # ── WARN kontrolleri ─────────────────────────────
    
    # Hafif overfitting
    if 0.10 < gap <= 0.20:
        warnings.append({
            "code": "mild_overfitting",
            "message": f"Hafif overfitting (fark: {gap:.3f}). Modelin genelleme kalitesini kontrol edin.",
            "value": round(gap, 3),
        })

    # Domain leakage (high confidence)
    checks_performed.append("domain_leakage")
    concerns = domain_leakage.get("concerns", []) or []
    high_concerns = [c for c in concerns if c.get("confidence") == "high"]
    if high_concerns:
        cols = [c["column_name"] for c in high_concerns]
        warnings.append({
            "code": "domain_leakage_high",
            "message": f"Yüksek şüpheli leakage: {', '.join(cols)}. Tahminler güvenilir olmayabilir.",
            "details": high_concerns,
        })

    # Importance-based leakage (Adım 2'den)
    checks_performed.append("importance_dominance")
    importance_suspects = fi_summary.get("leakage_suspects", []) or []
    if importance_suspects:
        warnings.append({
            "code": "importance_dominance",
            "message": f"{len(importance_suspects)} feature anormal yüksek importance'a sahip.",
            "details": importance_suspects,
        })

    # Düşük reliability
    checks_performed.append("reliability_score")
    rel_score = reliability.get("score", 100) or 100
    if rel_score < 50 and not failures:  # FAIL durumunda gereksiz
        warnings.append({
            "code": "low_reliability",
            "message": f"Güvenilirlik skoru düşük ({rel_score}/100).",
            "value": rel_score,
        })

    # Yüksek MAPE (regression)
    if "regression" in task:
        mape = test_metrics.get("mape", 0)
        if mape > 60 and not failures:
            warnings.append({
                "code": "high_mape",
                "message": f"MAPE yüksek (%{mape:.1f}). Tahmin hataları büyük olabilir.",
                "value": round(mape, 1),
            })

    # ── Sonuç ─────────────────────────────────────────
    
    if failures:
        status = "fail"
        severity = "critical"
        summary = (
            f"Model yetersiz — {len(failures)} kritik sorun tespit edildi. "
            "Üretime alınması önerilmez."
        )
    elif warnings:
        status = "warn"
        severity = "caution"
        summary = (
            f"Model çalışıyor ancak {len(warnings)} dikkat edilmesi gereken durum var. "
            "Sonuçları kullanmadan önce kontrol edin."
        )
    else:
        status = "pass"
        severity = "ok"
        summary = "Model güvenilir, üretime alınabilir."

    print(f"\n[quality_gate] STATUS: {status.upper()}")
    print(f"[quality_gate] {summary}")
    if failures:
        for f in failures:
            print(f"  [FAIL] {f['message']}")
    if warnings:
        for w in warnings:
            print(f"  [WARN] {w['message']}")

    return {
        "status": status,
        "severity": severity,
        "summary": summary,
        "failures": failures,
        "warnings": warnings,
        "checks_performed": checks_performed,
    }