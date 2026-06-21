from __future__ import annotations

from typing import Any

from src.core.logging_config import get_logger
logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────
# Severity eşikleri (gerçek hayat etkisine göre)
#   8-10 : Model kullanılamaz seviyede, mutlaka çözülmeli
#   5-7  : Çözülebilir orta düzey problem
#   1-4  : Var ama kritik değil, retry'a değmeyebilir
# ─────────────────────────────────────────────────────────────
MIN_ACTION_SEVERITY = 5      # Bu severity altında retry yapma
MIN_IMPROVEMENT     = 0.02   # İki tur arası bu kadar iyileşme yoksa dur


# ─────────────────────────────────────────────────────────────
# Yardımcı: birincil metrik (karşılaştırma için)
# ─────────────────────────────────────────────────────────────
def _primary_metric(task: str, test_metrics: dict) -> float:
    """Classification -> roc_auc, regression -> r2. İkisi de 'yüksek = iyi'."""
    if "classification" in task:
        return test_metrics.get(
            "roc_auc",
            test_metrics.get("roc_auc_ovr", test_metrics.get("accuracy", 0.0)),
        ) or 0.0
    return test_metrics.get("r2", 0.0) or 0.0


# ─────────────────────────────────────────────────────────────
# Teşhis fonksiyonları — her biri severity'li bir dict döner (veya None)
# ─────────────────────────────────────────────────────────────
def _diagnose_overfitting(overfitting: dict) -> dict | None:
    gap = overfitting.get("gap", 0) or 0
    if gap > 0.30:
        sev = 10
    elif gap > 0.20:
        sev = 7
    elif gap > 0.10:
        sev = 4
    else:
        return None
    return {
        "type": "overfitting",
        "severity": sev,
        "value": round(gap, 4),
        "action": "automl_node",
        "feedback": (
            f"TESHIS: Overfitting (train-test farki {gap:.3f}). "
            "EYLEM: Modeli basitlestir. max_depth dusur (3-5 araligi), "
            "n_estimators azalt, ridge/lasso gibi regularize modelleri oncele, "
            "Optuna search space'ini daha konservatif tut."
        ),
        "secondary_note": (
            f"Overfitting da mevcut (fark {gap:.3f}) — feature sayisini "
            "dusuk tut, karmasik etkilesimlerden kacin."
        ),
    }


def _diagnose_weak_features(fi_sum: dict) -> dict | None:
    top5_pct   = fi_sum.get("top5_total_contribution_pct", 100) or 100
    insig      = fi_sum.get("insignificant_features_count", 0) or 0
    total      = fi_sum.get("total_features", 0) or 0
    insig_ratio = (insig / total) if total else 0

    if top5_pct < 30 and insig_ratio > 0.7:
        sev = 9   # model rastgele feature'lara tutunuyor
    elif top5_pct < 50:
        sev = 6
    else:
        return None
    return {
        "type": "weak_features",
        "severity": sev,
        "value": round(top5_pct, 1),
        "action": "feature_plan_node",
        "feedback": (
            f"TESHIS: Zayif feature seti (top5 toplam katki %{top5_pct:.1f}, "
            f"{insig} onemsiz feature). "
            "EYLEM: Yeni feature uret — math interaction (ratio, product), "
            "polynomial feature, aggregation dene. Onceki plani TEKRARLAMA, "
            "farkli bir feature stratejisi kullan."
        ),
        "secondary_note": (
            f"Feature seti zayif (top5 %{top5_pct:.1f}) — "
            "daha aciklayici feature'lar gerekli."
        ),
    }


def _diagnose_low_performance(task: str, test_metrics: dict, fi_sum: dict) -> dict | None:
    """
    Classification: AUC + F1 + accuracy birlikte değerlendir.
    AUC iyi ama F1 çok düşükse → metrik tutarsızlığı (imbalance kalıbı).
    """
    healthy_features = (fi_sum.get("top5_total_contribution_pct", 0) or 0) >= 50

    if "classification" in task:
        auc = test_metrics.get(
            "roc_auc", test_metrics.get("roc_auc_ovr", 0)
        ) or 0
        f1  = test_metrics.get(
            "f1_weighted", test_metrics.get("f1", test_metrics.get("f1_score", 0))
        ) or 0
        acc = test_metrics.get("accuracy", 0) or 0

        if auc < 0.55 or f1 < 0.35:
            sev = 8           # gerçekten kötü
        elif auc < 0.65 or f1 < 0.50:
            sev = 6
        elif (auc - f1) > 0.20 and f1 < 0.60:
            sev = 5           # AUC iyi ama F1 çok düşük → imbalance kalıbı
        else:
            return None

        problems = []
        if auc < 0.65: problems.append(f"AUC={auc:.3f}")
        if f1  < 0.50: problems.append(f"F1={f1:.3f}")
        if acc < 0.60: problems.append(f"accuracy={acc:.2f}")
        if not problems:
            problems.append(f"AUC={auc:.2f} ama F1={f1:.2f}")
        problem_str = ", ".join(problems)

        if healthy_features:
            feedback = (
                f"TESHIS: Performans dengesiz ({problem_str}). Feature dagilimi "
                "saglikli ama metrikler tutarsiz — class imbalance veya task "
                "mantiksizligi olasiligi var. "
                "EYLEM: Sinif dengesini kontrol et (SMOTE, class_weight). "
                "Iyilesme olmazsa veri bu task icin uygun olmayabilir."
            )
        else:
            feedback = (
                f"TESHIS: Performans dusuk ({problem_str}). "
                "EYLEM: Feature engineering'i guclendir, yeni feature'lar uret."
            )
        return {
            "type": "low_performance",
            "severity": sev,
            "value": {"auc": round(auc, 4), "f1": round(f1, 4), "acc": round(acc, 4)},
            "action": "feature_plan_node",
            "feedback": feedback,
            "secondary_note": f"Metrikler tutarsiz ({problem_str}).",
        }

    # Regression
    score = _primary_metric(task, test_metrics)
    if score < 0.20:
        sev = 8
    elif score < 0.40:
        sev = 5
    else:
        return None
    if healthy_features:
        feedback = (
            f"TESHIS: Performans dusuk (R2={score:.3f}) ama feature dagilimi "
            "saglikli. Veri bu hedefi aciklamiyor olabilir. "
            "EYLEM: Domain-spesifik turetilmis feature dene; faydasi yoksa "
            "bu task icin veri yetersiz olabilir."
        )
    else:
        feedback = (
            f"TESHIS: Performans dusuk (R2={score:.3f}). "
            "EYLEM: Feature engineering'i guclendir, yeni feature'lar uret."
        )
    return {
        "type": "low_performance",
        "severity": sev,
        "value": round(score, 4),
        "action": "feature_plan_node",
        "feedback": feedback,
        "secondary_note": f"Performans dusuk (R2={score:.3f}).",
    }



def _diagnose_class_imbalance(task: str, evaluation_report: dict) -> dict | None:
    """Classification + birden fazla sınıfta düşük recall."""
    if "classification" not in task:
        return None

    # Doğru key: per_class_metrics
    per_class = (
        evaluation_report.get("per_class_metrics")
        or evaluation_report.get("per_class_report")
        or evaluation_report.get("per_class")
        or {}
    )
    if not per_class:
        return None

    # Düşük recall'lı sınıfları topla
    failing = []
    min_recall = 1.0
    for cls, metrics in per_class.items():
        if not isinstance(metrics, dict):
            continue
        rec = metrics.get("recall")
        if rec is None:
            continue
        if rec < min_recall:
            min_recall = rec
        if rec < 0.30:
            failing.append((cls, rec))

    if min_recall >= 0.40:
        return None

    n_classes  = len(per_class)
    fail_ratio = (len(failing) / n_classes) if n_classes else 0

    # Severity: ne kadar düşük + kaç sınıfta yaygın
    if min_recall < 0.10 and fail_ratio >= 0.30:
        sev = 9                    # birden fazla sınıf neredeyse görünmez
    elif min_recall < 0.20:
        sev = 7
    elif min_recall < 0.30:
        sev = 5
    else:
        return None                # 0.30-0.40 arası action threshold altı

    failing_str = ", ".join(
        f"sinif {c} (recall={r:.2f})" for c, r in sorted(failing, key=lambda x: x[1])[:3]
    )
    return {
        "type": "class_imbalance",
        "severity": sev,
        "value": round(min_recall, 3),
        "action": "feature_plan_node",
        "feedback": (
            f"TESHIS: {len(failing)}/{n_classes} sinifta dusuk recall. "
            f"En kotuler: {failing_str}. "
            "EYLEM: SMOTE/oversampling oranini artir, class_weight='balanced' "
            "kullan. Cok dusuk support'lu siniflarda iyilestirme sinirli kalir — "
            "veri yetersiz olabilir."
        ),
        "secondary_note": (
            f"{len(failing)} sinifta dusuk recall (en kotusu {min_recall:.2f})."
        ),
    }


# ─────────────────────────────────────────────────────────────
# Ana fonksiyon
# ─────────────────────────────────────────────────────────────
def advise(state: dict) -> dict:
    """
    Severity-tabanlı teşhis + retry kararı + spesifik feedback üretir.

    Dönüş:
        {
          "should_retry": bool,
          "next_node": "automl_node"|"feature_plan_node"|"report_node",
          "stop_reason": str | None,
          "diagnoses": [...],            # bulunan tüm teşhisler
          "primary_diagnosis": dict|None,
          "feedback": [str],            # LLM'e spesifik talimatlar
          "current_score": float,       # bu turun birincil metriği
          "primary_type": str|None,     # oscillation tespiti için
        }
    """
    evaluation_report = state.get("evaluation_report") or {}
    task              = evaluation_report.get("task", "")
    test_metrics      = evaluation_report.get("test_metrics") or {}
    overfitting       = evaluation_report.get("overfitting") or {}
    fi_sum            = evaluation_report.get("feature_importance_summary") or {}

    retry_count   = state.get("retry_count", 0)
    max_retries   = state.get("max_retries", 3)
    retry_history = state.get("retry_history") or []

    current_score = _primary_metric(task, test_metrics)

    # ── 1) Tüm teşhisleri topla ────────────────────────────
    raw = [
        _diagnose_overfitting(overfitting),
        _diagnose_weak_features(fi_sum),
        _diagnose_low_performance(task, test_metrics, fi_sum),
        _diagnose_class_imbalance(task, evaluation_report),
    ]
    diagnoses = [d for d in raw if d is not None]
    diagnoses.sort(key=lambda d: d["severity"], reverse=True)

    logger.info(f"\n[critic_advisor] Birincil metrik: {current_score:.4f}")
    if diagnoses:
        for d in diagnoses:
            logger.info(f"[critic_advisor]   teshis: {d['type']} (severity={d['severity']}, deger={d['value']})")
    else:
        logger.info(f"[critic_advisor]   teshis yok — model temiz")

    # ── 2) Aksiyon gerektiren teşhis var mı? ───────────────
    actionable = [d for d in diagnoses if d["severity"] >= MIN_ACTION_SEVERITY]

    if not actionable:
        return {
            "should_retry": False,
            "next_node": "report_node",
            "stop_reason": "Model yeterli — kritik sorun yok",
            "diagnoses": diagnoses,
            "primary_diagnosis": None,
            "feedback": [],
            "current_score": current_score,
            "primary_type": None,
        }

    primary = actionable[0]

    # ── 3) Retry bütçesi doldu mu? ─────────────────────────
    if retry_count >= max_retries:
        return {
            "should_retry": False,
            "next_node": "report_node",
            "stop_reason": f"Max retry ({max_retries}) asildi",
            "diagnoses": diagnoses,
            "primary_diagnosis": primary,
            "feedback": [],
            "current_score": current_score,
            "primary_type": primary["type"],
        }

    # ── 4) İyileşme var mı? (önceki tura göre) ─────────────
    if retry_history:
        prev_score = retry_history[-1].get("score", 0)
        improvement = current_score - prev_score
        if retry_count > 0 and improvement < MIN_IMPROVEMENT:
            return {
                "should_retry": False,
                "next_node": "report_node",
                "stop_reason": (
                    f"Iyilesme yok (onceki {prev_score:.4f} -> "
                    f"simdi {current_score:.4f}, fark {improvement:+.4f})"
                ),
                "diagnoses": diagnoses,
                "primary_diagnosis": primary,
                "feedback": [],
                "current_score": current_score,
                "primary_type": primary["type"],
            }

    # ── 5) Oscillation: aynı teşhis üst üste tekrarlıyor mu? ─
    if len(retry_history) >= 2:
        last_two = [h.get("primary_type") for h in retry_history[-2:]]
        if last_two == [primary["type"], primary["type"]]:
            return {
                "should_retry": False,
                "next_node": "report_node",
                "stop_reason": (
                    f"Ayni teshis ({primary['type']}) tekrarlaniyor, "
                    "duzeltme ise yaramiyor"
                ),
                "diagnoses": diagnoses,
                "primary_diagnosis": primary,
                "feedback": [],
                "current_score": current_score,
                "primary_type": primary["type"],
            }

    # ── 6) Retry kararı — primary feedback + ikincil uyarılar ─
    feedback = [primary["feedback"]]
    for d in actionable[1:]:
        note = d.get("secondary_note")
        if note:
            feedback.append(f"EK UYARI: {note}")

    return {
        "should_retry": True,
        "next_node": primary["action"],
        "stop_reason": None,
        "diagnoses": diagnoses,
        "primary_diagnosis": primary,
        "feedback": feedback,
        "current_score": current_score,
        "primary_type": primary["type"],
    }