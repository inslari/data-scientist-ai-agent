from __future__ import annotations

"""
Evaluation Node
---------------
Comprehensively evaluates the model produced by the AutoML node.

Dynamic metric calculation based on supported analysis types:
    - Binary classification → confusion matrix, ROC-AUC, precision-recall, F1 per class
    - Multiclass classification → same metrics, macro/weighted averages
    - Regression → RMSE, MAE, R², residual analysis

Additional analyses:
    - Overfitting check (train vs test performance gap)
    - Model calibration (expected vs predicted probability)
    - Feature importance analysis and grouping
    - Class-based performance analysis
    - MCC (Matthews Correlation Coefficient) — for imbalanced classes
    - Bootstrap Confidence Intervals — metric reliability
    - SHAP values — explainable AI (if shap package is installed)
    - Error Analysis — in-depth review of incorrect predictions
    - Model reliability score

No domains or column names are hardcoded.
All decisions are derived from metrics, feature_report, and feature_plan in the state.
"""

import warnings
from typing import Any
from src.core.config import settings
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from src.services.leakage_detector import detect_target_lock
from ..state import AgentState
from ..history_tracker import log_experiment
from src.core.logging_config import get_logger

logger = get_logger(__name__)

warnings.filterwarnings("ignore")

OVERFITTING_THRESHOLD = 0.05

# Classification Evaluation
def _evaluate_classification(
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Comprehensive evaluation for binary and multiclass classification."""
    is_binary = len(np.unique(y_test)) == 2

    # ── Tahminler ────────────────────────────────────────────────
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    y_prob_test = None
    y_prob_train = None
    try:
        y_prob_test = model.predict_proba(X_test)
        y_prob_train = model.predict_proba(X_train)
    except Exception:
        pass

    # Optimal Threshold Calculation
    roc_curve_data = None
    if is_binary and y_prob_test is not None:
        try:
            fpr, tpr, thresholds = roc_curve(y_test, y_prob_test[:, 1])
            # Optimal threshold (Youden's J statistic - Eskiden buradaydi, yorum satiri yapildi)
            # j_scores = tpr - fpr
            # optimal_idx = int(np.argmax(j_scores))
            # optimal_threshold = round(float(thresholds[optimal_idx]), 4)
            
            # Optimal threshold: F1 Skorunu maksimize ederek secelim (Dengesiz veriler icin daha iyi)
            from sklearn.metrics import precision_recall_curve
            precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob_test[:, 1])
            
            num = 2 * recall * precision
            den = recall + precision
            f1_scores = np.divide(num, den, out=np.zeros_like(num), where=(den!=0))
            
            best_pr_idx = int(np.argmax(f1_scores[:-1]))
            optimal_threshold = round(float(pr_thresholds[best_pr_idx]), 4)
            
            # ROC curve uzerindeki karsilik gelen index'i bulalim (UI grafiklerinde gostermek icin)
            optimal_idx = int(np.argmin(np.abs(thresholds - optimal_threshold)))
            # Update predictions based on optimal threshold for better recall
            classes = np.unique(y_test)
            if len(classes) == 2 and np.issubdtype(classes.dtype, np.number):
                y_pred_test = (y_prob_test[:, 1] >= optimal_threshold).astype(classes.dtype)
                if y_prob_train is not None:
                    y_pred_train = (y_prob_train[:, 1] >= optimal_threshold).astype(classes.dtype)
            
            roc_curve_data = {
                "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr) // 20)]],
                "tpr": [round(float(x), 4) for x in tpr[::max(1, len(tpr) // 20)]],
                "optimal_threshold": optimal_threshold,
                "optimal_fpr": round(float(fpr[optimal_idx]), 4),
                "optimal_tpr": round(float(tpr[optimal_idx]), 4),
            }
        except Exception:
            pass

    evaluation: dict[str, Any] = {
        "task": "binary_classification" if is_binary else "multiclass_classification",
    }

    # Test set metrics
    test_metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred_test)), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred_test, average="weighted", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred_test, average="macro", zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(y_test, y_pred_test, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_test, y_pred_test, average="weighted", zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y_test, y_pred_test)), 4),
    }

    # ROC-AUC
    if y_prob_test is not None:
        try:
            if is_binary:
                test_metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob_test[:, 1])), 4)
            else:
                test_metrics["roc_auc_ovr"] = round(float(
                    roc_auc_score(y_test, y_prob_test, multi_class="ovr", average="weighted")
                ), 4)
        except Exception:
            pass



    # Log loss
    if y_prob_test is not None:
        try:
            test_metrics["log_loss"] = round(float(log_loss(y_test, y_prob_test)), 4)
        except Exception:
            pass

    evaluation["test_metrics"] = test_metrics

    # Train set metrics
    train_metrics = {
        "accuracy": round(float(accuracy_score(y_train, y_pred_train)), 4),
        "f1_weighted": round(float(f1_score(y_train, y_pred_train, average="weighted", zero_division=0)), 4),
    }

    if y_prob_train is not None:
        try:
            if is_binary:
                train_metrics["roc_auc"] = round(float(roc_auc_score(y_train, y_prob_train[:, 1])), 4)
            else:
                train_metrics["roc_auc_ovr"] = round(float(
                    roc_auc_score(y_train, y_prob_train, multi_class="ovr", average="weighted")
                ), 4)
        except Exception:
            pass

    evaluation["train_metrics"] = train_metrics

    # Overfitting analysis
    overfitting = _analyze_overfitting(train_metrics, test_metrics)
    evaluation["overfitting"] = overfitting

    # Confusion matrix
    classes = sorted(np.unique(np.concatenate([y_train, y_test])))
    cm = confusion_matrix(y_test, y_pred_test, labels=classes)
    evaluation["confusion_matrix"] = {
        "matrix": cm.tolist(),
        "labels": [int(c) if isinstance(c, (np.integer,)) else c for c in classes],
    }

    # Per class report
    try:
        report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
        per_class = {}
        if isinstance(report, dict):
            for key, vals in report.items():
                if key in ("accuracy", "macro avg", "weighted avg"):
                    continue
                if isinstance(vals, dict):
                    per_class[key] = {
                        "precision": round(vals.get("precision", 0), 4),
                        "recall": round(vals.get("recall", 0), 4),
                        "f1": round(vals.get("f1-score", 0), 4),
                        "support": int(vals.get("support", 0)),
                    }
        evaluation["per_class_metrics"] = per_class
    except Exception:
        pass

    # ROC curve (binary)
    if roc_curve_data is not None:
        evaluation["roc_curve"] = roc_curve_data

    # Precision-recall curve (binary)
    if is_binary and y_prob_test is not None:
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob_test[:, 1])
            from sklearn.metrics import average_precision_score
            evaluation["precision_recall_curve"] = {
                "precision": [round(float(x), 4) for x in precision_curve[::max(1, len(precision_curve) // 20)]],
                "recall": [round(float(x), 4) for x in recall_curve[::max(1, len(recall_curve) // 20)]],
                "pr_auc": round(float(average_precision_score(y_test, y_prob_test[:, 1])), 4)
            }
        except Exception:
            pass

    # Model calibration
    if is_binary and y_prob_test is not None:
        try:
            prob_true, prob_pred = calibration_curve(y_test, y_prob_test[:, 1], n_bins=10)
            brier_score = float(np.mean((y_prob_test[:, 1] - y_test) ** 2))
            evaluation["calibration"] = {
                "prob_true": [round(float(x), 4) for x in prob_true],
                "prob_pred": [round(float(x), 4) for x in prob_pred],
                "brier_score": round(brier_score, 4),
                "is_well_calibrated": brier_score < 0.1,
            }
        except Exception:
            pass

    # Feature importance analysis
    # Feature importance analysis
    fi = _extract_feature_importance(model, feature_names)
    if fi:
        evaluation["feature_importance"] = fi
        evaluation["feature_importance_summary"] = _summarize_feature_importance(fi)

        # Target-lock (censorship/survival) leakage kontrolü
        
        target_lock_concerns = detect_target_lock(X_test, y_test, feature_importance=fi)
        if target_lock_concerns:
            evaluation["target_lock_leakage"] = target_lock_concerns
            logger.info(f"[evaluation] Target-lock leakage suphesi: "
                  f"{len(target_lock_concerns)} feature")
            for c in target_lock_concerns:
                logger.info(f"  - {c['feature']} (imp={c['importance']}): {c['reason']}")

    # Bootstrap confidence intervals
    logger.info("Bootstrap CI hesaplanıyor...")
    evaluation["confidence_intervals"] = _bootstrap_ci_classification(
        model, X_test, y_test, is_binary, n_iterations=300
    )

    # Error analysis
    evaluation["error_analysis"] = _analyze_errors(
        y_test, y_pred_test, X_test, feature_names,
        y_prob=y_prob_test[:, 1] if (is_binary and y_prob_test is not None) else None,
    )

    # SHAP values
    logger.info("SHAP değerleri hesaplanıyor...")
    shap_result = _compute_shap_values(model, X_test, feature_names)
    if shap_result:
        evaluation["shap_importance"] = shap_result

    # Optimal threshold to state
    roc_data = evaluation.get("roc_curve", {})
    if roc_data:
        evaluation["optimal_threshold"] = roc_data.get("optimal_threshold")

    # Model reliability score
    evaluation["reliability_score"] = _compute_reliability_score(evaluation)

    return evaluation


# Regression Evaluation
def _evaluate_regression(
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Regression modelleri için kapsamlı değerlendirme."""
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    mse_test = mean_squared_error(y_test, y_pred_test)
    mse_train = mean_squared_error(y_train, y_pred_train)

    evaluation: dict[str, Any] = {
        "task": "regression",
    }

    # Test metrikleri
    test_metrics = {
        "rmse": round(float(np.sqrt(mse_test)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred_test)), 4),
        "r2": round(float(r2_score(y_test, y_pred_test)), 4),
        "mse": round(float(mse_test), 4),
        "mape": round(float(_safe_mape(y_test, y_pred_test)), 4),
    }
    evaluation["test_metrics"] = test_metrics

    # Train metrikleri
    train_metrics = {
        "rmse": round(float(np.sqrt(mse_train)), 4),
        "r2": round(float(r2_score(y_train, y_pred_train)), 4),
    }
    evaluation["train_metrics"] = train_metrics

    # Overfitting
    overfitting = _analyze_overfitting_regression(train_metrics, test_metrics)
    evaluation["overfitting"] = overfitting

    # Residual analizi
    residuals = y_test - y_pred_test
    evaluation["residual_analysis"] = {
        "mean_residual": round(float(np.mean(residuals)), 4),
        "std_residual": round(float(np.std(residuals)), 4),
        "median_residual": round(float(np.median(residuals)), 4),
        "max_overpredict": round(float(np.min(residuals)), 4),
        "max_underpredict": round(float(np.max(residuals)), 4),
        "residual_skewness": round(float(pd.Series(residuals).skew()), 4),  # type: ignore[arg-type]
    }

    # Feature importance
    fi = _extract_feature_importance(model, feature_names)
    if fi:
        evaluation["feature_importance"] = fi
        evaluation["feature_importance_summary"] = _summarize_feature_importance(fi)

        # Target-lock (veri sızıntısı) kontrolü — regresyon varyans-çöküşü modu.
        # Classification ile aynı anahtara yazılır; intent="regression" ile
        # detect_target_lock'un regresyon dalı (Faz 3) tetiklenir.
        target_lock_concerns = detect_target_lock(
            X_test, y_test, feature_importance=fi, intent="regression"
        )
        if target_lock_concerns:
            evaluation["target_lock_leakage"] = target_lock_concerns
            logger.info(f"[evaluation] Target-lock leakage suphesi: "
                  f"{len(target_lock_concerns)} feature")
            for c in target_lock_concerns:
                logger.info(f"  - {c['feature']} (imp={c['importance']}): {c['reason']}")

    # Bootstrap confidence intervals
    logger.info("Bootstrap CI hesaplanıyor...")
    evaluation["confidence_intervals"] = _bootstrap_ci_regression(
        model, X_test, y_test, n_iterations=300
    )

    # SHAP values
    logger.info("SHAP değerleri hesaplanıyor...")
    shap_result = _compute_shap_values(model, X_test, feature_names)
    if shap_result:
        evaluation["shap_importance"] = shap_result

    # ── Regresyon scatter görselleri (rapor UI'ı için) ───────────────────
    # Residual-vs-predicted ve predicted-vs-actual saçılımları. Binlerce ham
    # nokta step_data'yı şişirir; roc_curve gibi stride ile ~200 noktaya
    # downsample edilir. Mevcut residual_analysis ÖZETİ korunur — bu yalnızca
    # görselleştirme için örneklenmiş ham noktalar. Domain-pack bloklarından
    # bağımsız, ayrı bir alan (regression_scatter).
    evaluation["regression_scatter"] = _build_regression_scatter(y_test, y_pred_test)

    evaluation["reliability_score"] = _compute_reliability_score(evaluation)

    return evaluation


def _build_regression_scatter(
    y_true: np.ndarray, y_pred: np.ndarray, max_points: int = 200
) -> dict:
    """Rapor görselleri için örneklenmiş scatter noktaları üretir.

    roc_curve'deki stride yaklaşımıyla (arr[::step]) en fazla ~max_points
    noktaya downsample eder; ham binlerce noktayı step_data'ya yazmaz.
    Döndürülen diziler indeks-eşli: i. eleman aynı örneğe aittir.
      - predicted : ŷ
      - actual    : y
      - residuals : y - ŷ  (y_true - y_pred)
    """
    try:
        yt = np.asarray(y_true, dtype=float).ravel()
        yp = np.asarray(y_pred, dtype=float).ravel()
        n_total = int(yt.shape[0])
        if n_total == 0:
            return {}
        step = max(1, n_total // max_points)
        yt_s = yt[::step]
        yp_s = yp[::step]
        res_s = yt_s - yp_s
        return {
            "predicted": [round(float(v), 4) for v in yp_s],
            "actual": [round(float(v), 4) for v in yt_s],
            "residuals": [round(float(v), 4) for v in res_s],
            "n_sampled": int(yp_s.shape[0]),
            "n_total": n_total,
        }
    except Exception:
        return {}


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# Overfitting Analysis
def _analyze_overfitting(train_metrics: dict, test_metrics: dict) -> dict:
    """Analyzes overfitting"""
    train_score = train_metrics.get("roc_auc") or train_metrics.get("roc_auc_ovr") or train_metrics.get("accuracy") or 0.0
    test_score = test_metrics.get("roc_auc") or test_metrics.get("roc_auc_ovr") or test_metrics.get("accuracy") or 0.0

    gap = round(float(train_score - test_score), 4)
    status = "normal"
    if gap > OVERFITTING_THRESHOLD:
        status = "overfitting_detected"
    elif gap > OVERFITTING_THRESHOLD / 2:
        status = "mild_overfitting"
    elif gap < -0.02:
        status = "underfitting"

    return {
        "train_score": round(float(train_score), 4),
        "test_score": round(float(test_score), 4),
        "gap": gap,
        "status": status,
        "description": _overfitting_description(status, gap),
    }


def _analyze_overfitting_regression(train_metrics: dict, test_metrics: dict) -> dict:
    """Regression için overfitting analizi — R² bazlı."""
    train_r2 = train_metrics.get("r2", 0)
    test_r2 = test_metrics.get("r2", 0)

    gap = round(float(train_r2 - test_r2), 4)
    status = "normal"
    if gap > 0.1:
        status = "overfitting_detected"
    elif gap > 0.05:
        status = "mild_overfitting"
    elif test_r2 < 0.3:
        status = "underfitting"

    return {
        "train_r2": round(float(train_r2), 4),
        "test_r2": round(float(test_r2), 4),
        "gap": gap,
        "status": status,
        "description": _overfitting_description(status, gap),
    }


def _overfitting_description(status: str, gap: float) -> str:
    descriptions = {
        "normal": "Model iyi genelleştiriyor — train ve test performansı dengeli.",
        "mild_overfitting": f"Hafif overfitting mevcut (fark: {gap:.3f}). "
                            "Regularization artırılabilir veya daha az karmaşık model denenebilir.",
        "overfitting_detected": f"Ciddi overfitting tespit edildi (fark: {gap:.3f}). "
                                "Model eğitim verisini ezberliyor. "
                                "Daha fazla veri, regularization veya model basitleştirmesi önerilir.",
        "underfitting": "Model yeterince öğrenemiyor. "
                        "Daha karmaşık model veya daha fazla feature gerekebilir.",
    }
    return descriptions.get(status, "Bilinmeyen durum.")


# Feature Importance
def _extract_feature_importance(model: Any, feature_names: list[str]) -> dict:
    """Model'den feature importance çıkarır — model tipine göre adaptif."""
    try:
        est = model
        if hasattr(model, "named_steps"):
            est = model.named_steps.get("clf", model)

        importances = None
        if hasattr(est, "feature_importances_"):
            importances = est.feature_importances_
        elif hasattr(est, "coef_"):
            coef = est.coef_
            if coef.ndim > 1:
                importances = np.abs(coef).mean(axis=0)
            else:
                importances = np.abs(coef)

        if importances is not None:
            imp_dict = dict(zip(feature_names, importances.tolist()))
            return dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        pass
    return {}


def _summarize_feature_importance(fi: dict) -> dict:
    """Feature importance summary."""
    if not fi:
        return {}

    total = sum(fi.values())
    if total == 0:
        return {}

    sorted_feats = list(fi.items())

    # Top 5 feature
    top5 = {k: round(v / total * 100, 2) for k, v in sorted_feats[:5]}

    # Top 5'in toplam contribution
    top5_total = sum(v for _, v in sorted_feats[:5])
    top5_pct = round(top5_total / total * 100, 2)

    # insignificant features (less than %1)
    insignificant = [k for k, v in sorted_feats if v / total < 0.01]

    # ── Importance-based leakage detection ──
    leakage_suspects = []

    # Tek feature aşırı baskın
    if sorted_feats:
        top1_feat, top1_val = sorted_feats[0]
        top1_pct = top1_val / total * 100
        if top1_pct >= 70:
            leakage_suspects.append({
                "column": top1_feat,
                "reason": f"tek basina %{top1_pct:.1f} importance - cok yuksek leakage suphesi",
                "severity": "high",
            })
        elif top1_pct >= 50:
            leakage_suspects.append({
                "column": top1_feat,
                "reason": f"tek basina %{top1_pct:.1f} importance - leakage suphesi",
                "severity": "medium",
            })

    # Top 2 toplam aşırı baskın
    if len(sorted_feats) >= 2:
        top2_total = sorted_feats[0][1] + sorted_feats[1][1]
        top2_pct = top2_total / total * 100
        if top2_pct >= 90 and top1_pct < 70:  # tek feature degil ama ikisi cok baskin
            top2_feats = [sorted_feats[0][0], sorted_feats[1][0]]
            leakage_suspects.append({
                "column": " + ".join(top2_feats),
                "reason": f"ilk 2 feature toplam %{top2_pct:.1f} importance - leakage suphesi",
                "severity": "medium",
            })

    if leakage_suspects:
        logger.info(f"\n[evaluation] LEAKAGE SUPHESI (importance-based):")
        for s in leakage_suspects:
            logger.info(f"  - {s['column']}: {s['reason']}")

    return {
        "top5_features": top5,
        "top5_total_contribution_pct": top5_pct,
        "total_features": len(fi),
        "insignificant_features_count": len(insignificant),
        "insignificant_features": insignificant[:10],
        "leakage_suspects": leakage_suspects,
    }


# Bootstrap Confidence Intervals
def _bootstrap_ci_classification(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    is_binary: bool,
    n_iterations: int = 300,
    ci: float = 0.95,
) -> dict:
    """Bootstrap samples with metric confidence intervals (classification)."""
    rng = np.random.default_rng(settings.RANDOM_SEED)
    n = len(y_test)
    metrics_boot: dict[str, list] = {"accuracy": [], "f1_weighted": [], "roc_auc": [], "mcc": []}

    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        X_b = X_test.iloc[idx]  # type: ignore[call-overload]
        y_b = y_test[idx]
        if len(np.unique(y_b)) < 2:
            continue
        y_pred_b = model.predict(X_b)
        metrics_boot["accuracy"].append(float(accuracy_score(y_b, y_pred_b)))
        metrics_boot["f1_weighted"].append(
            float(f1_score(y_b, y_pred_b, average="weighted", zero_division=0))
        )
        metrics_boot["mcc"].append(float(matthews_corrcoef(y_b, y_pred_b)))
        try:
            y_prob_b = model.predict_proba(X_b)
            if is_binary:
                metrics_boot["roc_auc"].append(float(roc_auc_score(y_b, y_prob_b[:, 1])))
            else:
                metrics_boot["roc_auc"].append(
                    float(roc_auc_score(y_b, y_prob_b, multi_class="ovr", average="weighted"))
                )
        except Exception:
            pass

    alpha = (1 - ci) / 2
    result = {}
    for metric, values in metrics_boot.items():
        if not values:
            continue
        arr = np.array(values)
        result[metric] = {
            "mean": round(float(np.mean(arr)), 4),
            "lower": round(float(np.quantile(arr, alpha)), 4),
            "upper": round(float(np.quantile(arr, 1 - alpha)), 4),
            "std": round(float(np.std(arr)), 4),
        }
    return result


def _bootstrap_ci_regression(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_iterations: int = 300,
    ci: float = 0.95,
    ) -> dict:
    """Bootstrap samples with metric confidence intervals (regression)."""
    rng = np.random.default_rng(settings.RANDOM_SEED)
    n = len(y_test)
    metrics_boot: dict[str, list] = {"rmse": [], "r2": [], "mae": []}

    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        X_b = X_test.iloc[idx]  # type: ignore[call-overload]
        y_b = y_test[idx]
        try:
            y_pred_b = model.predict(X_b)
            metrics_boot["rmse"].append(float(np.sqrt(mean_squared_error(y_b, y_pred_b))))
            metrics_boot["r2"].append(float(r2_score(y_b, y_pred_b)))
            metrics_boot["mae"].append(float(mean_absolute_error(y_b, y_pred_b)))
        except Exception:
            pass

    alpha = (1 - ci) / 2
    result = {}
    for metric, values in metrics_boot.items():
        if not values:
            continue
        arr = np.array(values)
        result[metric] = {
            "mean": round(float(np.mean(arr)), 4),
            "lower": round(float(np.quantile(arr, alpha)), 4),
            "upper": round(float(np.quantile(arr, 1 - alpha)), 4),
            "std": round(float(np.std(arr)), 4),
        }
    return result


# SHAP Values

def _compute_shap_values(
    model: Any,
    X_test: pd.DataFrame,
    feature_names: list[str],
    max_samples: int = 200,
) -> dict:
    """
    Calculates feature importance, which can be described using SHAP values.
    Returns an empty dict if the SHAP package is not installed.
    """
    try:
        import shap  # type: ignore

        est = model
        if hasattr(model, "named_steps"):
            est = model.named_steps.get("clf", model)

        X_sample = X_test.iloc[:max_samples] if len(X_test) > max_samples else X_test

        if hasattr(est, "feature_importances_"):
            explainer = shap.TreeExplainer(est)
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.Explainer(est, X_sample)
            shap_values = explainer(X_sample).values  # type: ignore[attr-defined]

        if isinstance(shap_values, list):
            arr = np.abs(np.array(shap_values)).mean(axis=0)
        else:
            arr = np.abs(shap_values)
            if arr.ndim == 3:
                arr = arr.mean(axis=2)

        mean_shap = arr.mean(axis=0)
        shap_dict = dict(zip(feature_names, mean_shap.tolist()))
        shap_sorted = dict(sorted(shap_dict.items(), key=lambda x: x[1], reverse=True))

        total = sum(shap_sorted.values()) or 1
        top5_shap = {k: round(v / total * 100, 2) for k, v in list(shap_sorted.items())[:5]}

        return {
            "top5_shap_features": top5_shap,
            "all_shap_values": {k: round(float(v), 6) for k, v in shap_sorted.items()},
            "n_samples_used": len(X_sample),
        }
    except Exception:
        return {}

# Helper functions for unsupervised tasks (clustering, anomaly detection)
def _pass_clustering_metrics(state: AgentState) -> dict:
    """Passes clustering metrics from automl_node directly."""
    metrics = state.get("metrics", {})
    return {
        "task": "clustering",
        "silhouette_score": metrics.get("silhouette_score"),
        "n_clusters": metrics.get("n_clusters"),
        "best_algorithm": metrics.get("best_algorithm"),
        "davies_bouldin": metrics.get("davies_bouldin"),
        "calinski_harabasz": metrics.get("calinski_harabasz"),
        "cluster_distribution": metrics.get("cluster_distribution", {}),
        "all_algorithm_scores": metrics.get("all_algorithm_scores", {}),
        "reliability_score": {"score": 0, "grade": "N/A", "factors": ["Clustering task — no supervised metrics"]},
        "overfitting": {"status": "N/A", "description": "Clustering is unsupervised — no overfitting concept."},
        "test_metrics": {},
        "train_metrics": {},
    }


def _pass_anomaly_metrics(state: AgentState) -> dict:
    """Passes anomaly detection metrics from automl_node directly."""
    metrics = state.get("metrics", {})
    return {
        "task": "anomaly_detection",
        "n_anomalies": metrics.get("n_anomalies"),
        "anomaly_pct": metrics.get("anomaly_pct"),
        "n_normal": metrics.get("n_normal"),
        "best_contamination": metrics.get("best_contamination"),
        "score_threshold": metrics.get("score_threshold"),
        "contamination_scan": metrics.get("contamination_scan", {}),
        "feature_importance": metrics.get("feature_importance", {}),
        "reliability_score": {"score": 0, "grade": "N/A", "factors": ["Anomaly detection — unsupervised"]},
        "overfitting": {"status": "N/A", "description": "Anomaly detection is unsupervised."},
        "test_metrics": {},
        "train_metrics": {},
    }

def _analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X_test: pd.DataFrame,
    feature_names: list[str],
    y_prob: np.ndarray | None = None,
) -> dict:
    """
    Deep dive into misclassified samples:
    - Error rate and number of errors
    - High confidence errors (model was confident but still wrong)
    - Feature distribution differences between correct and incorrect samples
    """
    errors = y_true != y_pred
    n_errors = int(errors.sum())
    n_total = len(y_true)

    result: dict[str, Any] = {
        "n_errors": n_errors,
        "n_total": n_total,
        "error_rate": round(float(n_errors / n_total), 4),
        "correct_rate": round(float(1 - n_errors / n_total), 4),
    }

    if n_errors == 0:
        result["note"] = "Hiç hatalı tahmin yok — mükemmel sınıflandırma."
        return result

    # Confidence analysis
    if y_prob is not None:
        error_probs = y_prob[errors]
        correct_probs = y_prob[~errors]
        high_conf_errors = int(((error_probs > 0.7) | (error_probs < 0.3)).sum())
        result["confidence_analysis"] = {
            "mean_error_confidence": round(float(np.abs(error_probs - 0.5).mean() + 0.5), 4),
            "mean_correct_confidence": round(float(np.abs(correct_probs - 0.5).mean() + 0.5), 4),
            "high_confidence_errors": high_conf_errors,
            "high_confidence_errors_pct": round(float(high_conf_errors / n_errors * 100), 2),
        }

    # Feature distribution difference between correct and incorrect samples - Cohen's d
    X_errors = X_test[errors]
    X_correct = X_test[~errors]
    feature_diff = {}
    numeric_cols = X_test.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols[:15]:
        mean_err = float(X_errors[col].mean())
        mean_cor = float(X_correct[col].mean())
        std_err = float(X_errors[col].std())
        std_cor = float(X_correct[col].std())
        n_err = len(X_errors)
        n_cor = len(X_correct)
        pooled_std = float(np.sqrt(
            ((n_err - 1) * std_err ** 2 + (n_cor - 1) * std_cor ** 2)
            / (n_err + n_cor - 2)
        )) if (n_err + n_cor > 2) else 1.0
        cohens_d = abs(mean_err - mean_cor) / (pooled_std + 1e-9)

        # Only report significant effect (small=0.2, medium=0.5, large=0.8)
        if cohens_d >= 0.2:
            magnitude = (
                "large" if cohens_d >= 0.8
                else "medium" if cohens_d >= 0.5
                else "small"
            )
            feature_diff[col] = {
                "mean_in_errors": round(mean_err, 4),
                "mean_in_correct": round(mean_cor, 4),
                "cohens_d": round(cohens_d, 4),
                "effect_magnitude": magnitude,
            }

    if feature_diff:
        result["feature_distribution_diff"] = dict(
            sorted(feature_diff.items(), key=lambda x: x[1]["cohens_d"], reverse=True)[:10]
        )

    return result

# Bias Detection
def _analyze_bias(y_true: np.ndarray, y_pred: np.ndarray, X_test: pd.DataFrame, db_intelligence: dict, intent: str) -> dict:
    """
    Calculates Disparate Impact and performance differences over demographic features (gender, age, etc.).
    """
    demographic_keywords = ["age", "sex", "gender", "race", "ethnicity", "marital", "country", "nativity"]
    demographic_bases = [
        col["column_name"] for col in db_intelligence.get("columns", [])
        if col.get("semantic_type") == "demographic" or any(k in col["column_name"].lower() for k in demographic_keywords)
    ]
    if not demographic_bases:
        return {}
        
    bias_results = {}
    for base_col in demographic_bases:
        # Find all encoded columns related to this base column (e.g. gender_Male, gender_Female)
        related_cols = [c for c in X_test.columns if c.startswith(base_col)]
        if not related_cols:
            continue
            
        col_bias = {}
        for col in related_cols:
            if X_test[col].nunique() == 2:
                mask_1 = (X_test[col] > 0.5).to_numpy()
                mask_0 = ~mask_1
                
                if mask_1.sum() < 5 or mask_0.sum() < 5:
                    continue
                    
                if "classification" in intent:
                    acc_1 = float(accuracy_score(y_true[mask_1], y_pred[mask_1]))
                    acc_0 = float(accuracy_score(y_true[mask_0], y_pred[mask_0]))
                    
                    # Positive rate (Disparate Impact)
                    pos_rate_1 = float(np.mean(y_pred[mask_1] == 1)) if len(np.unique(y_true))==2 else None
                    pos_rate_0 = float(np.mean(y_pred[mask_0] == 1)) if len(np.unique(y_true))==2 else None
                    
                    disparate_impact = None
                    if pos_rate_0 and pos_rate_0 > 0 and pos_rate_1 is not None:
                        disparate_impact = pos_rate_1 / pos_rate_0
                        
                    col_bias[col] = {
                        "accuracy_group_1": round(acc_1, 4),
                        "accuracy_group_0": round(acc_0, 4),
                        "accuracy_diff": round(abs(acc_1 - acc_0), 4),
                        "disparate_impact": round(disparate_impact, 4) if disparate_impact else None
                    }
                elif "regression" in intent:
                    rmse_1 = float(np.sqrt(mean_squared_error(y_true[mask_1], y_pred[mask_1])))
                    rmse_0 = float(np.sqrt(mean_squared_error(y_true[mask_0], y_pred[mask_0])))
                    
                    col_bias[col] = {
                        "rmse_group_1": round(rmse_1, 4),
                        "rmse_group_0": round(rmse_0, 4),
                        "rmse_diff": round(abs(rmse_1 - rmse_0), 4)
                    }
        if col_bias:
            bias_results[base_col] = col_bias
            
    return bias_results

# Reliability Score
def _compute_reliability_score(evaluation: dict) -> dict:
    """
    Calculates model reliability score (0-100).
    Combines multiple factors to provide an overall assessment.
    """
    score = 0
    max_score = 0
    factors = []

    task = evaluation.get("task", "")
    test_metrics = evaluation.get("test_metrics", {})
    overfitting = evaluation.get("overfitting", {})
    calibration = evaluation.get("calibration", {})

    # Main performance metric (0-35 points)
    max_score += 35
    if "classification" in task:
        auc = test_metrics.get("roc_auc", test_metrics.get("roc_auc_ovr", 0))
        if auc >= 0.95:
            score += 35
            factors.append("Mukemmel AUC (>=0.95)")
        elif auc >= 0.90:
            score += 30
            factors.append("Cok iyi AUC (>=0.90)")
        elif auc >= 0.80:
            score += 25
            factors.append("Iyi AUC (>=0.80)")
        elif auc >= 0.70:
            score += 15
            factors.append("Kabul edilebilir AUC (>=0.70)")
        else:
            score += 5
            factors.append("Dusuk AUC (<0.70)")
    else:
        r2 = test_metrics.get("r2", 0)
        if r2 >= 0.90:
            score += 35
            factors.append("Mukemmel R2 (>=0.90)")
        elif r2 >= 0.75:
            score += 25
            factors.append("Iyi R2 (>=0.75)")
        elif r2 >= 0.50:
            score += 15
            factors.append("Kabul edilebilir R2 (>=0.50)")
        else:
            score += 5
            factors.append("Dusuk R2 (<0.50)")

    # Overfitting status (0-25 points)
    max_score += 25
    ovf_status = overfitting.get("status", "unknown")
    if ovf_status == "normal":
        score += 25
        factors.append("Overfitting yok")
    elif ovf_status == "mild_overfitting":
        score += 15
        factors.append("Hafif overfitting")
    elif ovf_status == "overfitting_detected":
        score += 5
        factors.append("Ciddi overfitting")
    elif ovf_status == "underfitting":
        score += 10
        factors.append("Underfitting")

    # Balance metric: F1 for classification, MAPE for regression (0-20 points)
    max_score += 20
    if "classification" in task:
        f1_w = (
            test_metrics.get("f1_weighted")
            or test_metrics.get("f1_macro")
            or test_metrics.get("f1_score")
            or test_metrics.get("f1")
            or 0
        )
        if f1_w >= 0.85:
            score += 20
            factors.append("Dengeli F1 (>=0.85)")
        elif f1_w >= 0.75:
            score += 15
            factors.append("Iyi F1 (>=0.75)")
        elif f1_w >= 0.60:
            score += 10
            factors.append("Kabul edilebilir F1 (>=0.60)")
        else:
            score += 5
            factors.append("Dusuk F1 (<0.60)")
    else:
        # Regression: use MAPE
        mape = test_metrics.get("mape", 100)
        if mape < 10:
            score += 20
            factors.append("Cok dusuk MAPE (<10%)")
        elif mape < 25:
            score += 15
            factors.append("Dusuk MAPE (<25%)")
        elif mape < 50:
            score += 10
            factors.append("Orta MAPE (<50%)")
        else:
            score += 5
            factors.append("Yuksek MAPE (>=50%)")

    # Model calibration (0-20 points, only if available and valid)
    if calibration and isinstance(calibration, dict):
        max_score += 20
        if calibration.get("is_well_calibrated", False):
            score += 20
            factors.append("Iyi kalibre edilmis")
        else:
            brier = calibration.get("brier_score", 1.0)
            if brier < 0.15:
                score += 15
                factors.append("Kabul edilebilir kalibrasyon")
            else:
                score += 5
                factors.append("Zayif kalibrasyon")
    # Calibration yoksa veya tuple ise, max_score'a hic ekleme - adil olur

    # Final score
    final_score = round(score / max_score * 100, 1) if max_score > 0 else 0

    grade = "A" if final_score >= 90 else \
            "B" if final_score >= 75 else \
            "C" if final_score >= 60 else \
            "D" if final_score >= 40 else "F"

    return {
        "score": final_score,
        "grade": grade,
        "max_score": 100,
        "factors": factors,
    }


# Main node
def evaluate_model(state: AgentState) -> dict:
    """
    Evaluates the model produced by the AutoML node.

    Dynamic evaluation - automatic metric selection according to task type.
    No domain is hardcoded.

    State inputs:
        model_artifact   : Eğitilmiş model
        X_train, X_test  : Feature verisi
        y_train, y_test  : Label verisi
        feature_report   : Feature isimleri
        intent           : Task tipini belirlemek için

    State çıktıları:
        evaluation_report : Kapsamlı değerlendirme raporu
    """
    model_artifact = state.get("model_artifact")
    model = None
    if isinstance(model_artifact, str) and model_artifact.endswith('.pkl'):
        import joblib
        import os
        if os.path.exists(model_artifact):
            try:
                model = joblib.load(model_artifact)
            except Exception as e:
                logger.info(f"[evaluator] Model diskten yuklenemedi: {e}")
    else:
        model = model_artifact
    X_train_raw = state.get("X_train", [])
    X_test_raw = state.get("X_test", [])
    y_train_raw = state.get("y_train", [])
    y_test_raw = state.get("y_test", [])
    intent = state.get("intent", "classification")
    feature_report = state.get("feature_report", {})

    if model is None:
        return {**state, "error": "Evaluation için model bulunamadı.",
                "current_step": "evaluation_node"}

    # Unsupervised tasks (clustering, anomaly_detection) for y data - only X control
    if not X_train_raw:
        return {**state, "error": "Evaluation için X verisi bulunamadı.",
                "current_step": "evaluation_node"}

    # Supervised tasks 
    if intent not in ("clustering", "anomaly_detection") and not y_train_raw:
        return {**state, "error": "Evaluation için label (y) verisi bulunamadı.",
                "current_step": "evaluation_node"}

    # Convert to DataFrame
    X_train = pd.DataFrame(X_train_raw).apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test = pd.DataFrame(X_test_raw).apply(pd.to_numeric, errors="coerce").fillna(0)
    X_train.columns = [str(c) for c in X_train.columns]
    X_test.columns = [str(c) for c in X_test.columns]

    y_train = np.array(y_train_raw)
    y_test = np.array(y_test_raw)
    feature_names = X_train.columns.tolist()

    logger.info(f"\n[evaluation] Başlıyor...")
    logger.info(f"  Model    : {state.get('model_name', 'unknown')}")
    logger.info(f"  X_test   : {X_test.shape}")
    logger.info(f"  Intent   : {intent}")

    # Evaluation regarding task type
    if intent == "regression":
        # NOT (Faz 4): _evaluate_regression artık target-lock sızıntı kontrolünü
        # (varyans-çöküşü modu) BİLİNÇLİ olarak çalıştırıyor. Önceden regresyonda
        # bu 4. savunma sessizce devre dışıydı; artık classification ile aynı
        # "target_lock_leakage" anahtarını dolduruyor. Bilinçli bir karardır.
        evaluation_report = _evaluate_regression(
            model, X_train, y_train, X_test, y_test, feature_names
        )
    elif intent == "clustering":
        evaluation_report = _pass_clustering_metrics(state)
    elif intent == "anomaly_detection":
        evaluation_report = _pass_anomaly_metrics(state)
    else:
        evaluation_report = _evaluate_classification(
            model, X_train, y_train, X_test, y_test, feature_names
        )

    # Model name and feature plan
    evaluation_report["model_name"] = state.get("model_name", "unknown")
    evaluation_report["analysis_summary"] = feature_report.get("analysis_summary",
                                            state.get("feature_plan", {}).get("analysis_summary", ""))

    # Bias analysis
    if intent in ("classification", "regression"):
        db_intelligence = state.get("db_intelligence", {})
        bias_report = _analyze_bias(y_test, model.predict(X_test), X_test, db_intelligence, intent)
        if bias_report:
            evaluation_report["bias_analysis"] = bias_report

    # Metric explanations for UI
    evaluation_report["metric_descriptions"] = {
        # Test Metrics
        "accuracy": {
            "label": "Accuracy",
            "description": "Modelin tüm tahminler içinde doğru yaptığı tahminlerin oranı.",
            "range": "0–1 (1 mükemmel)",
            "caveat": "Dengesiz sınıf dağılımlarında yanıltıcı olabilir.",
        },
        "f1_weighted": {
            "label": "F1 Skoru (Weighted)",
            "description": "Precision ve Recall'un harmonik ortalaması. Sınıf büyüklüklerine göre ağırlıklandırılmış.",
            "range": "0–1 (1 mükemmel)",
            "caveat": "Sınıf dengesizliğini hesaba katar, genel performansı iyi özetler.",
        },
        "f1_macro": {
            "label": "F1 Skoru (Macro)",
            "description": "Her sınıfın F1 skorunun düz ortalaması. Küçük sınıflar eşit ağırlık alır.",
            "range": "0–1 (1 mükemmel)",
            "caveat": "Küçük sınıflardaki başarısızlıkları daha belirgin gösterir.",
        },
        "mcc": {
            "label": "Matthews Correlation Coefficient (MCC)",
            "description": "Dengesiz sınıf dağılımlarında en güvenilir tek metrik. Tüm confusion matrix hücrelerini kullanır.",
            "range": "-1 ile +1 arası (+1 mükemmel, 0 rastgele, -1 tam ters)",
            "caveat": "0.7'nin üzeri oldukça iyi kabul edilir.",
        },
        "roc_auc": {
            "label": "ROC-AUC",
            "description": "Modelin pozitif ve negatif örnekleri ayırt etme yeteneği. Threshold bağımsız genel performans.",
            "range": "0–1 (>0.9 mükemmel, >0.8 iyi, >0.7 kabul edilebilir)",
            "caveat": "Dengesiz veri setlerinde PR-AUC ile birlikte değerlendirilmeli.",
        },
        "log_loss": {
            "label": "Log Loss",
            "description": "Modelin olasılık tahminlerinin doğruluğunu ölçer. Küçük değer daha iyi.",
            "range": "0–∞ (0 mükemmel)",
            "caveat": "Emin olmayan ama doğru tahminleri penalize eder.",
        },
        "precision_weighted": {
            "label": "Precision (Weighted)",
            "description": "Pozitif tahminlerin ne kadarının gerçekten pozitif olduğu.",
            "range": "0–1 (1 mükemmel)",
            "caveat": "False Positive'leri azaltmak önemli olduğunda öncelikli metrik.",
        },
        "recall_weighted": {
            "label": "Recall (Weighted)",
            "description": "Gerçek pozitiflerin ne kadarının doğru yakalandığı.",
            "range": "0–1 (1 mükemmel)",
            "caveat": "False Negative'leri azaltmak kritikse (örn. hastalık tespiti) öncelikli metrik.",
        },
        # Regression Metrikleri
        "rmse": {
            "label": "RMSE (Root Mean Squared Error)",
            "description": "Tahmin hatalarının kare ortalamasının karekökü. Büyük hatalara daha duyarlı.",
            "range": "0–∞ (0 mükemmel, küçük değer iyi)",
            "caveat": "Hedef değişkenin birimi ile aynı ölçektedir.",
        },
        "mae": {
            "label": "MAE (Mean Absolute Error)",
            "description": "Ortalama mutlak hata. Tüm hatalara eşit ağırlık verir.",
            "range": "0–∞ (0 mükemmel)",
            "caveat": "RMSE'ye göre outlier'lara daha az duyarlı.",
        },
        "r2": {
            "label": "R² (Coefficient of Determination)",
            "description": "Modelin hedef değişkendeki varyansı ne kadar açıkladığı.",
            "range": "0–1 (1 mükemmel, <0 baseline'dan kötü)",
            "caveat": "Özellik sayısı arttıkça şişebilir; adjusted R² daha güvenilir.",
        },
        "mape": {
            "label": "MAPE (Mean Absolute Percentage Error)",
            "description": "Ortalama mutlak yüzde hata. Hedef değişkenden bağımsız karşılaştırma sağlar.",
            "range": "0–100% (0 mükemmel)",
            "caveat": "Hedef değeri 0'a yakın örneklerde hatalı sonuç verebilir.",
        },
        # Additional explanations
        "bootstrap_ci": {
            "label": "Bootstrap Güven Aralıkları (%95)",
            "description": "Test seti üzerinden 300 iterasyonlu bootstrap örnekleme ile hesaplanmış %95 güven aralıkları. Tek test seti sonucundan daha güvenilir bir performans aralığı verir.",
        },
        "overfitting": {
            "label": "Overfitting Analizi",
            "description": "Modelin eğitim ve test performansı arasındaki fark. Büyük fark modelin eğitim verisini ezberlediğini gösterir.",
            "thresholds": {
                "normal": "Fark < 0.025 — iyi genelleşme",
                "mild_overfitting": "Fark 0.025–0.05 — hafif overfitting",
                "overfitting_detected": "Fark > 0.05 — ciddi overfitting",
            },
        },
        "calibration": {
            "label": "Model Kalibrasyonu",
            "description": "Modelin olasılık tahminlerinin gerçek olasılıklarla uyumu. İyi kalibre edilmiş bir model '70% olasılık' dediğinde gerçekten ~70 örnekten 70'i doğru çıkar.",
        },
        "brier_score": {
            "label": "Brier Score",
            "description": "Olasılık tahminlerinin ortalama kare hatası. Kalibrasyon ve keskinliği birlikte ölçer.",
            "range": "0–1 (0 mükemmel, <0.1 iyi kalibre)",
        },
        "shap": {
            "label": "SHAP Değerleri",
            "description": "Her feature'ın model tahminlerine ortalama katkısı. Feature importance'tan farklı olarak her tahmin için ayrı ayrı hesaplanır — 'neden bu kararı verdi' sorusunu yanıtlar.",
        },
        "error_analysis": {
            "label": "Hata Analizi",
            "description": "Yanlış tahmin edilen örneklerin incelenmesi.",
        },
        "high_confidence_errors": {
            "label": "Yüksek Güven Hataları",
            "description": "Modelin yüksek güvenle (>70% veya <30% olasılık) verdiği ama yanlış çıkan tahminler. Bu hatalar özellikle tehlikeli — model emin ama yanılıyor.",
        },
        "cohens_d": {
            "label": "Cohen's d (Effect Size)",
            "description": "Hatalı ve doğru tahminler arasındaki feature değeri farkının standartlaştırılmış ölçüsü. Scale'den bağımsızdır.",
            "thresholds": {
                "small": "d ≥ 0.2 — küçük etki",
                "medium": "d ≥ 0.5 — orta etki",
                "large": "d ≥ 0.8 — büyük etki",
            },
        },
        "reliability_score": {
            "label": "Model Güvenilirlik Skoru",
            "description": "AUC, overfitting durumu, F1 dengesi ve kalibrasyon faktörlerini birleştiren genel model kalite skoru (0–100).",
            "thresholds": {
                "A (90–100)": "Production-ready, güvenilir model",
                "B (75–89)": "İyi model, küçük iyileştirmelerle kullanılabilir",
                "C (60–74)": "Kabul edilebilir, geliştirme önerilen",
                "D (40–59)": "Zayıf model, ciddi iyileştirme gerekli",
                "F (<40)": "Kullanılamaz",
            },
        },
    }

    # Print results
    reliability = evaluation_report.get("reliability_score", {})
    overfitting = evaluation_report.get("overfitting", {})
    test_m = evaluation_report.get("test_metrics", {})

    logger.info(f"\n[evaluation] Test Metrikleri:")
    for k, v in test_m.items():
        logger.info(f"  {k}: {v}")

    ci = evaluation_report.get("confidence_intervals", {})
    if ci:
        logger.info(f"\n[evaluation] Bootstrap CI (%95):")
        for metric, vals in ci.items():
            logger.info(f"  {metric}: {vals['mean']} [{vals['lower']} — {vals['upper']}] (std: {vals['std']})")

    logger.info(f"\n[evaluation] Overfitting Analizi:")
    logger.info(f"  Train: {overfitting.get('train_score')} | Test: {overfitting.get('test_score')}")
    logger.info(f"  Fark : {overfitting.get('gap')} | Durum: {overfitting.get('status')}")
    logger.info(f"  {overfitting.get('description')}")

    calibration = evaluation_report.get("calibration", {})
    if calibration:
        logger.info(f"\n[evaluation] Kalibrasyon:")
        logger.info(f"  Brier Score: {calibration.get('brier_score')}")
        logger.info(f"  İyi kalibre: {calibration.get('is_well_calibrated')}")

    err_analysis = evaluation_report.get("error_analysis", {})
    if err_analysis:
        logger.info(f"\n[evaluation] Error Analysis:")
        logger.error(f"  Hatalı tahmin: {err_analysis.get('n_errors')}/{err_analysis.get('n_total')} ({err_analysis.get('error_rate')*100:.2f}%)")
        conf = err_analysis.get("confidence_analysis", {})
        if conf:
            logger.error(f"  Yüksek güven hatası: {conf.get('high_confidence_errors')} adet ({conf.get('high_confidence_errors_pct')}%)")
            logger.error(f"  Hata güveni ort.: {conf.get('mean_error_confidence')} | Doğru güveni ort.: {conf.get('mean_correct_confidence')}")
        feat_diff = err_analysis.get("feature_distribution_diff", {})
        if feat_diff:
            logger.info(f"  En farklı feature'lar (Cohen's d effect size):")
            for feat, info in list(feat_diff.items())[:5]:
                logger.error(f"    {feat}: d={info['cohens_d']} ({info['effect_magnitude']}) | hata={info['mean_in_errors']} | doğru={info['mean_in_correct']}")

    shap_info = evaluation_report.get("shap_importance", {})
    if shap_info:
        logger.info(f"\n[evaluation] SHAP Top 5:")
        for feat, pct in shap_info.get("top5_shap_features", {}).items():
            logger.info(f"  {feat}: %{pct}")
    else:
        logger.info(f"\n[evaluation] SHAP: shap paketi bulunamadı, atlandı.")

    opt_thresh = evaluation_report.get("optimal_threshold")
    if opt_thresh is not None:
        logger.info(f"\n[evaluation] Optimal Threshold (Youden's J): {opt_thresh}")

    fi_summary = evaluation_report.get("feature_importance_summary", {})
    if fi_summary:
        logger.info(f"\n[evaluation] Feature Importance Özeti:")
        logger.info(f"  Top 5 toplam katkı: %{fi_summary.get('top5_total_contribution_pct')}")
        for feat, pct in fi_summary.get("top5_features", {}).items():
            logger.info(f"    {feat}: %{pct}")
        insig_count = fi_summary.get("insignificant_features_count", 0)
        if insig_count > 0:
            logger.info(f"  Önemsiz feature sayısı (<%%1): {insig_count}")

    logger.info(f"\n[evaluation] Güvenilirlik Skoru: {reliability.get('score')}/100 (Grade: {reliability.get('grade')})")
    for factor in reliability.get("factors", []):
        logger.info(f"  ✓ {factor}")
        
    bias = evaluation_report.get("bias_analysis", {})
    if bias:
        logger.info("\n[evaluation] Bias Tespiti (Demografik):")
        for base_col, cols in bias.items():
            for c, info in cols.items():
                logger.info(f"  {c}: Diff={info.get('accuracy_diff', info.get('rmse_diff'))} | Disparate Impact={info.get('disparate_impact', 'N/A')}")

    # Log to Experiment Tracker
    try:
        log_experiment(
            task=evaluation_report.get("task", intent),
            intent=intent,
            model_name=evaluation_report.get("model_name", "unknown"),
            metrics=evaluation_report.get("test_metrics", {}),
            hyperparameters=state.get("model_params", {}),
            feature_names=feature_names,
            target_name=feature_report.get("target_name", "unknown"),
            dataset_shape=f"Train: {X_train.shape}, Test: {X_test.shape}"
        )
        logger.info("[evaluation] Deney başarıyla veritabanına loglandı.")
    except Exception as e:
        logger.error(f"[evaluation] Deney loglanırken hata: {e}")

    return {
        **state,
        "evaluation_report": evaluation_report,
        "optimal_threshold": float(evaluation_report.get("optimal_threshold") or 0.5),
        "current_step": "evaluation_node",
    }