from __future__ import annotations
from typing import Any

from src.core.logging_config import get_logger
logger = get_logger(__name__)


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

    # Domain leakage (high confidence) — yalnızca MODELDE KALAN kolonlar için uyar.
    # Detector'ın high-confidence suspect'leri feature_builder tarafından zaten
    # düşürülmüş olabilir (feature_report.leakage_dropped'a kaydedilir). Düşürülen
    # kolon modele hiç girmediğinden onun için "tahminler güvenilir olmayabilir"
    # demek yanıltıcı (örn. RowNumber/CustomerId/Surname gibi tanımlayıcılar drop
    # edilip yine de uyarı veriyordu). Drop edilmişleri uyarıdan çıkar.
    checks_performed.append("domain_leakage")
    feature_report = state.get("feature_report") or {}
    dropped_cols = {
        d.get("column") for d in (feature_report.get("leakage_dropped") or [])
        if isinstance(d, dict)
    }
    concerns = domain_leakage.get("concerns", []) or []
    high_concerns = [c for c in concerns if c.get("confidence") == "high"]
    high_in_model = [c for c in high_concerns if c.get("column_name") not in dropped_cols]
    if high_in_model:
        cols = [c["column_name"] for c in high_in_model]
        warnings.append({
            "code": "domain_leakage_high",
            "message": f"Yüksek şüpheli leakage: {', '.join(cols)}. Tahminler güvenilir olmayabilir.",
            "details": high_in_model,
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

    # Target-lock (censorship/survival) leakage
    # Target-lock (veri sızıntısı) kontrolü
    checks_performed.append("target_lock_leakage")
    target_lock = evaluation_report.get("target_lock_leakage", []) or []
    high_lock = [c for c in target_lock if c.get("confidence") == "high"]
    low_lock = [c for c in target_lock if c.get("confidence") == "low"]

    # Mesajlar task tipine göre dallanır: regresyonda "varyansı dar bir aralıkta
    # neredeyse sabitliyor" dili, classification'da mevcut "sonucu belirliyor" dili.
    is_regression = "regression" in task

    if high_lock:
        feats = ", ".join(c["feature"] for c in high_lock)
        if is_regression:
            high_message = (
                f"Olası veri sızıntısı: {feats}. "
                f"Bu özellikler belirli aralıklarda hedefi dar bir bantta neredeyse tam belirliyor "
                f"(hedefin varyansı çöküyor) — yani model 'tahmin' etmek yerine cevabı doğrudan "
                f"görüyor olabilir. Gerçek kullanımda bu bilgi elde olmayabilir; bu özelliklerin "
                f"çıkarılması önerilir."
            )
        else:
            high_message = (
                f"Olası veri sızıntısı: {feats}. "
                f"Bu özellikler belirli aralıklarda sonucu neredeyse kesin belirliyor — "
                f"yani model 'tahmin' etmek yerine cevabı doğrudan görüyor olabilir. "
                f"Gerçek kullanımda bu bilgi elde olmayabilir; bu özelliklerin çıkarılması önerilir."
            )
        warnings.append({
            "code": "target_lock_leakage_high",
            "message": high_message,
            "details": high_lock,
        })
    if low_lock:
        feats = ", ".join(c["feature"] for c in low_lock)
        if is_regression:
            low_message = (
                f"Güçlü belirleyici özellik: {feats}. "
                f"Hedefle güçlü ama tutarlı bir ilişkisi var (belirli aralıklarda varyansını "
                f"kademeli azaltıyor) — bu genelde normaldir, sızıntı değil gerçek bir sinyaldir. "
                f"Yine de gerçek kullanımda bu bilginin elde olacağından emin olun."
            )
        else:
            low_message = (
                f"Güçlü belirleyici özellik: {feats}. "
                f"Sonuçla güçlü ama tutarlı bir ilişkisi var — bu genelde normaldir, "
                f"sızıntı değil gerçek bir sinyaldir. Yine de gerçek kullanımda bu bilginin "
                f"elde olacağından emin olun."
            )
        warnings.append({
            "code": "target_lock_signal_low",
            "message": low_message,
            "details": low_lock,
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

    # ── Manuel hedef seçimi sanity-check (öneri DEĞİL, bariz hata WARN'ı) ──
    # Yalnızca kullanıcı UI'dan hedefi el ile seçtiyse (target_col dolu) çalışır.
    # Engelleyici değil — pipeline çalışmaya devam eder, sadece WARN gösterilir.
    # profile_table'dan gelen hazır veriyi kullanır, yeniden hesaplama yok.
    target_col = state.get("target_col")
    if target_col:
        checks_performed.append("target_sanity")
        profiling_report = state.get("profiling_report") or {}
        target_profile = (profiling_report.get("columns") or {}).get(target_col) or {}
        unique_count = target_profile.get("unique_count")
        unique_rate  = target_profile.get("unique_rate")
        null_rate    = target_profile.get("null_rate")

        # 1) Sıfır varyans — tek değer
        if unique_count is not None and unique_count <= 1:
            warnings.append({
                "code": "target_zero_variance",
                "message": (
                    f"Seçilen hedef '{target_col}' tek bir değerden oluşuyor (sıfır varyans) — "
                    f"modellenecek bir değişkenlik yok."
                ),
                "value": unique_count,
            })
        # 2) ID-vari — kardinalite ≈ satır sayısı (eşik feature_builder ile aynı: 0.95)
        elif unique_rate is not None and unique_rate > 0.95:
            warnings.append({
                "code": "target_id_like",
                "message": (
                    f"Seçilen hedef '{target_col}' neredeyse tüm satırlarda benzersiz "
                    f"(%{unique_rate*100:.0f}) — bir kimlik (ID) sütunu olabilir, "
                    f"modelleme anlamlı olmayabilir."
                ),
                "value": round(unique_rate, 4),
            })

        # 3) Yüksek null — null_rate > %50
        if null_rate is not None and null_rate > 0.50:
            warnings.append({
                "code": "target_high_null",
                "message": (
                    f"Seçilen hedef '{target_col}' satırların %{null_rate*100:.0f}'inde boş — "
                    f"yüksek eksik oranı modellemeyi güvenilmez kılabilir."
                ),
                "value": round(null_rate, 4),
            })

    # ── Sonuç ─────────────────────────────────────────

    if failures:
        status = "fail"
        severity = "critical"
        summary = (
            f"Model yetersiz — {len(failures)} kritik sorun tespit edildi. "
            "Sonuçlar bu haliyle güvenilir değil; sorunlar giderilmeden kullanılmamalı."
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
        summary = (
            "Model güçlü performans gösteriyor. Dağıtımdan önce bağımsız "
            "doğrulama ve sürekli izleme önerilir."
        )

    logger.info(f"\n[quality_gate] STATUS: {status.upper()}")
    logger.info(f"[quality_gate] {summary}")
    if failures:
        for f in failures:
            logger.error(f"  [FAIL] {f['message']}")
    if warnings:
        for w in warnings:
            logger.info(f"  [WARN] {w['message']}")

    return {
        "status": status,
        "severity": severity,
        "summary": summary,
        "failures": failures,
        "warnings": warnings,
        "checks_performed": checks_performed,
    }