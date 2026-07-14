from __future__ import annotations
from typing import Any

from src.core.logging_config import get_logger
from src.core.report_i18n import normalize_lang, R
logger = get_logger(__name__)


def evaluate_quality_gate(state: dict) -> dict:
    """
    Modelin sonucunu değerlendirip kullanıcıya nasıl sunulacağına karar verir.
    
    Status seviyeleri:
      pass  - Model güvenilir, normal raporla
      warn  - Çalışıyor ama dikkat edilmesi gereken durumlar var
      fail  - Yetersiz, üretime alınmamalı
    """
    # Rapor diliyle aynı: mesajlar/summary çıktıya verbatim girdiği için lokalize edilir.
    lang = normalize_lang(state.get("report_language"))

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
        # METRİK YOK ≠ METRİK 0. Eskiden eksik ROC `0` sayılıyordu ve model mükemmel olsa bile
        # "AUC 0.000 → çok düşük" FAIL'i basılıyordu. Ölçemediğimiz şey hakkında hüküm vermeyiz:
        # metrik yoksa kontrolü ATLA ve bunu WARN olarak GÖRÜNÜR kıl (sessizce geçme de yok).
        auc = test_metrics.get("roc_auc", test_metrics.get("roc_auc_ovr"))
        if auc is None:
            warnings.append({
                "code": "auc_unavailable",
                "message": R("auc_unavailable", lang),
                "value": None,
            })
        elif auc < 0.60:
            failures.append({
                "code": "low_auc",
                "message": R("low_auc", lang, a=auc),
                "value": round(auc, 3),
            })

    # Düşük performans (regression)
    if "regression" in task:
        checks_performed.append("regression_performance")
        r2 = test_metrics.get("r2", 0)
        if r2 < 0.30:
            failures.append({
                "code": "low_r2",
                "message": R("low_r2", lang, r=r2),
                "value": round(r2, 3),
            })

    # Ciddi overfitting
    # Olcum belirsizse (CV skoru test skorunun bootstrap GA'sinin icinde — az pozitifli test seti)
    # gap hicbir sey kanitlamiyor; gurultuyu FAIL/WARN'a cevirmek yanlis alarmdir. gap=0 → sessiz.
    checks_performed.append("overfitting")
    gap = 0 if overfitting.get("status") == "inconclusive" else (overfitting.get("gap", 0) or 0)
    if gap > 0.20:
        failures.append({
            "code": "severe_overfitting",
            "message": R("severe_overfitting", lang, g=gap),
            "value": round(gap, 3),
        })

    # ── WARN kontrolleri ─────────────────────────────
    
    # Hafif overfitting
    if 0.10 < gap <= 0.20:
        warnings.append({
            "code": "mild_overfitting",
            "message": R("mild_overfitting", lang, g=gap),
            "value": round(gap, 3),
        })

    # Domain leakage (high confidence) — yalnızca MODELDE GERÇEKTEN KALAN kolonlar için uyar.
    # Detector'ın high-confidence suspect'i feature_builder tarafından düşürülmüş olabilir;
    # düşen kolon modele hiç girmediğinden onun için "tahminler güvenilir olmayabilir" demek
    # yanıltıcıdır (örn. customer_id/RowNumber/Surname gibi tanımlayıcılar drop edilip yine de
    # uyarı veriyordu).
    #
    # ASIL ÖLÇÜT: feature_report.final_features = modele giren gerçek kolon listesi
    # (X_train.columns). Bir kolon; identifier/timestamp erken-drop'u, critic drop'u,
    # kardinalite/korelasyon drop'u VB. herhangi bir yolla modelden çıkmış olabilir ve
    # bunların hepsi leakage_dropped'a yazılmaz (örn. id/ts drop'u yalnızca log'a yazar).
    # Bu yüzden tek güvenilir sinyal final_features ÜYELİĞİdir. leakage_dropped sadece
    # final_features mevcut değilse (eski koşu) geriye-uyumlu fallback olarak kullanılır.
    checks_performed.append("domain_leakage")
    feature_report = state.get("feature_report") or {}
    dropped_cols = {
        d.get("column") for d in (feature_report.get("leakage_dropped") or [])
        if isinstance(d, dict)
    }
    final_features = feature_report.get("final_features") or []
    final_features_low = {str(f).lower() for f in final_features}

    def _is_in_model(col: str | None) -> bool:
        """Kolon modele gerçekten girdi mi? final_features varsa ona göre (case-insensitive),
        yoksa leakage_dropped fallback'ine göre karar verir."""
        if col is None:
            return False
        if final_features_low:
            return str(col).lower() in final_features_low
        return col not in dropped_cols

    concerns = domain_leakage.get("concerns", []) or []
    high_concerns = [c for c in concerns if c.get("confidence") == "high"]
    high_in_model = [c for c in high_concerns if _is_in_model(c.get("column_name"))]
    if high_in_model:
        cols = [c["column_name"] for c in high_in_model]
        warnings.append({
            "code": "domain_leakage_high",
            "message": R("domain_leakage_high", lang, cols=", ".join(cols)),
            "details": high_in_model,
        })

    # Importance-based leakage (Adım 2'den)
    checks_performed.append("importance_dominance")
    importance_suspects = fi_summary.get("leakage_suspects", []) or []
    if importance_suspects:
        warnings.append({
            "code": "importance_dominance",
            "message": R("importance_dominance", lang, n=len(importance_suspects)),
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
        high_message = R(
            "target_lock_high_reg" if is_regression else "target_lock_high_clf",
            lang, f=feats,
        )
        warnings.append({
            "code": "target_lock_leakage_high",
            "message": high_message,
            "details": high_lock,
        })
    if low_lock:
        feats = ", ".join(c["feature"] for c in low_lock)
        low_message = R(
            "target_lock_low_reg" if is_regression else "target_lock_low_clf",
            lang, f=feats,
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
            "message": R("low_reliability", lang, s=rel_score),
            "value": rel_score,
        })

    # Yüksek MAPE (regression)
    if "regression" in task:
        mape = test_metrics.get("mape", 0)
        if mape > 60 and not failures:
            warnings.append({
                "code": "high_mape",
                "message": R("high_mape", lang, m=mape),
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
                "message": R("target_zero_variance", lang, c=target_col),
                "value": unique_count,
            })
        # 2) ID-vari — kardinalite ≈ satır sayısı (eşik feature_builder ile aynı: 0.95)
        elif unique_rate is not None and unique_rate > 0.95:
            warnings.append({
                "code": "target_id_like",
                "message": R("target_id_like", lang, c=target_col, p=unique_rate * 100),
                "value": round(unique_rate, 4),
            })

        # 3) Yüksek null — null_rate > %50
        if null_rate is not None and null_rate > 0.50:
            warnings.append({
                "code": "target_high_null",
                "message": R("target_high_null", lang, c=target_col, p=null_rate * 100),
                "value": round(null_rate, 4),
            })

    # ── Sonuç ─────────────────────────────────────────

    if failures:
        status = "fail"
        severity = "critical"
        summary = R("summary_fail", lang, n=len(failures))
    elif warnings:
        status = "warn"
        severity = "caution"
        summary = R("summary_warn", lang, n=len(warnings))
    else:
        status = "pass"
        severity = "ok"
        summary = R("summary_pass", lang)

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