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

from src.state import AgentState

warnings.filterwarnings("ignore")

OVERFITTING_THRESHOLD = 0.05  # Train-test farkı bu kadardan fazlaysa uyar


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

    evaluation: dict[str, Any] = {
        "task": "binary_classification" if is_binary else "multiclass_classification",
    }

    # ── Test seti metrikleri ─────────────────────────────────────
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

    # ── Train seti metrikleri (overfitting kontrolü) ─────────────
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

    # ── Overfitting analizi ──────────────────────────────────────
    overfitting = _analyze_overfitting(train_metrics, test_metrics)
    evaluation["overfitting"] = overfitting

    # ── Confusion matrix ─────────────────────────────────────────
    classes = sorted(np.unique(np.concatenate([y_train, y_test])))
    cm = confusion_matrix(y_test, y_pred_test, labels=classes)
    evaluation["confusion_matrix"] = {
        "matrix": cm.tolist(),
        "labels": [int(c) if isinstance(c, (np.integer,)) else c for c in classes],
    }

    # ── Sınıf bazlı rapor ────────────────────────────────────────
    try:
        report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
        per_class = {}
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

    # ── ROC curve verileri (binary) ──────────────────────────────
    if is_binary and y_prob_test is not None:
        try:
            fpr, tpr, thresholds = roc_curve(y_test, y_prob_test[:, 1])
            # Optimal threshold (Youden's J statistic)
            j_scores = tpr - fpr
            optimal_idx = int(np.argmax(j_scores))
            optimal_threshold = round(float(thresholds[optimal_idx]), 4)

            evaluation["roc_curve"] = {
                "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr) // 20)]],
                "tpr": [round(float(x), 4) for x in tpr[::max(1, len(tpr) // 20)]],
                "optimal_threshold": optimal_threshold,
                "optimal_fpr": round(float(fpr[optimal_idx]), 4),
                "optimal_tpr": round(float(tpr[optimal_idx]), 4),
            }
        except Exception:
            pass

    # ── Precision-recall curve (binary) ──────────────────────────
    if is_binary and y_prob_test is not None:
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob_test[:, 1])
            evaluation["precision_recall_curve"] = {
                "precision": [round(float(x), 4) for x in precision_curve[::max(1, len(precision_curve) // 20)]],
                "recall": [round(float(x), 4) for x in recall_curve[::max(1, len(recall_curve) // 20)]],
            }
        except Exception:
            pass

    # ── Model kalibrasyonu ───────────────────────────────────────
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

    # ── Feature importance analizi ───────────────────────────────
    fi = _extract_feature_importance(model, feature_names)
    if fi:
        evaluation["feature_importance"] = fi
        evaluation["feature_importance_summary"] = _summarize_feature_importance(fi)

    # ── Bootstrap confidence intervals ───────────────────────────
    print("  Bootstrap CI hesaplanıyor...")
    evaluation["confidence_intervals"] = _bootstrap_ci_classification(
        model, X_test, y_test, is_binary, n_iterations=300
    )

    # ── Error analizi ────────────────────────────────────────────
    evaluation["error_analysis"] = _analyze_errors(
        y_test, y_pred_test, X_test, feature_names,
        y_prob=y_prob_test[:, 1] if (is_binary and y_prob_test is not None) else None,
    )

    # ── SHAP değerleri ───────────────────────────────────────────
    print("  SHAP değerleri hesaplanıyor...")
    shap_result = _compute_shap_values(model, X_test, feature_names)
    if shap_result:
        evaluation["shap_importance"] = shap_result

    # ── Optimal threshold state'e taşı ───────────────────────────
    roc_data = evaluation.get("roc_curve", {})
    if roc_data:
        evaluation["optimal_threshold"] = roc_data.get("optimal_threshold")

    # ── Model güvenilirlik skoru ─────────────────────────────────
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
        "residual_skewness": round(float(pd.Series(residuals).skew()), 4),
    }

    # Feature importance
    fi = _extract_feature_importance(model, feature_names)
    if fi:
        evaluation["feature_importance"] = fi
        evaluation["feature_importance_summary"] = _summarize_feature_importance(fi)

    # ── Bootstrap confidence intervals ───────────────────────────
    print("  Bootstrap CI hesaplanıyor...")
    evaluation["confidence_intervals"] = _bootstrap_ci_regression(
        model, X_test, y_test, n_iterations=300
    )

    # ── SHAP değerleri ───────────────────────────────────────────
    print("  SHAP değerleri hesaplanıyor...")
    shap_result = _compute_shap_values(model, X_test, feature_names)
    if shap_result:
        evaluation["shap_importance"] = shap_result

    evaluation["reliability_score"] = _compute_reliability_score(evaluation)

    return evaluation


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# Overfitting Analysis

def _analyze_overfitting(train_metrics: dict, test_metrics: dict) -> dict:
    """Train vs test metrik farkını analiz eder."""
    # Ana metrik: roc_auc varsa onu kullan, yoksa accuracy
    train_score = train_metrics.get("roc_auc",
                  train_metrics.get("roc_auc_ovr",
                  train_metrics.get("accuracy", 0)))
    test_score = test_metrics.get("roc_auc",
                 test_metrics.get("roc_auc_ovr",
                 test_metrics.get("accuracy", 0)))

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
    """Feature importance özetini çıkarır."""
    if not fi:
        return {}

    total = sum(fi.values())
    if total == 0:
        return {}

    sorted_feats = list(fi.items())

    # Top 5 feature ve oranları
    top5 = {k: round(v / total * 100, 2) for k, v in sorted_feats[:5]}

    # Top 5'in toplam katkısı
    top5_total = sum(v for _, v in sorted_feats[:5])
    top5_pct = round(top5_total / total * 100, 2)

    # Önemsiz feature'lar (katkısı %1'den az)
    insignificant = [k for k, v in sorted_feats if v / total < 0.01]

    return {
        "top5_features": top5,
        "top5_total_contribution_pct": top5_pct,
        "total_features": len(fi),
        "insignificant_features_count": len(insignificant),
        "insignificant_features": insignificant[:10],  # ilk 10'unu göster
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
    """Bootstrap örnekleme ile metrik güven aralıkları hesaplar (classification)."""
    rng = np.random.default_rng(42)
    n = len(y_test)
    metrics_boot: dict[str, list] = {"accuracy": [], "f1_weighted": [], "roc_auc": [], "mcc": []}

    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        X_b = X_test.iloc[idx]
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
    """Bootstrap örnekleme ile metrik güven aralıkları hesaplar (regression)."""
    rng = np.random.default_rng(42)
    n = len(y_test)
    metrics_boot: dict[str, list] = {"rmse": [], "r2": [], "mae": []}

    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        X_b = X_test.iloc[idx]
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
    SHAP değerleri ile açıklanabilir feature importance hesaplar.
    shap paketi kurulu değilse sessizce boş dict döner.
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
            shap_values = explainer(X_sample).values

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
    """Clustering için automl_node'dan gelen metrikleri doğrudan döndürür."""
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
    """Anomaly detection için automl_node'dan gelen metrikleri döndürür."""
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
    Yanlış sınıflandırılan örneklerin derinlemesine analizi:
    - Hata oranı ve sayısı
    - Yüksek güven hataları (modelin emin olduğu ama yanlış tahminler)
    - Hatalı vs doğru örnekler arası feature dağılımı farkı
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

    # Confidence analizi — modelin emin olduğu ama yanıldığı örnekler
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

    # Feature dağılımı farkı: hatalı vs doğru örnekler — Cohen's d ile
    # (scaled feature'larda yüzde farkı anlamsız olur, effect size scale-bağımsızdır)
    X_errors = X_test[errors]
    X_correct = X_test[~errors]
    feature_diff = {}
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols[:15]:
        mean_err = float(X_errors[col].mean())
        mean_cor = float(X_correct[col].mean())
        # Pooled standart sapma
        std_err = float(X_errors[col].std())
        std_cor = float(X_correct[col].std())
        n_err = len(X_errors)
        n_cor = len(X_correct)
        pooled_std = float(np.sqrt(
            ((n_err - 1) * std_err ** 2 + (n_cor - 1) * std_cor ** 2)
            / (n_err + n_cor - 2)
        )) if (n_err + n_cor > 2) else 1.0
        cohens_d = abs(mean_err - mean_cor) / (pooled_std + 1e-9)

        # Yalnızca anlamlı etkiyi raporla (small=0.2, medium=0.5, large=0.8)
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


# Reliability Score

def _compute_reliability_score(evaluation: dict) -> dict:
    """
    Model güvenilirlik skoru hesaplar (0-100).
    Birden fazla faktörü birleştirerek genel bir değerlendirme yapar.
    """
    score = 0
    max_score = 0
    factors = []

    task = evaluation.get("task", "")
    test_metrics = evaluation.get("test_metrics", {})
    overfitting = evaluation.get("overfitting", {})
    calibration = evaluation.get("calibration", {})

    # 1. Ana performans metriği (0-35 puan)
    max_score += 35
    if "classification" in task:
        auc = test_metrics.get("roc_auc", test_metrics.get("roc_auc_ovr", 0))
        if auc >= 0.95:
            score += 35
            factors.append("Mükemmel AUC (≥0.95)")
        elif auc >= 0.90:
            score += 30
            factors.append("Çok iyi AUC (≥0.90)")
        elif auc >= 0.80:
            score += 25
            factors.append("İyi AUC (≥0.80)")
        elif auc >= 0.70:
            score += 15
            factors.append("Kabul edilebilir AUC (≥0.70)")
        else:
            score += 5
            factors.append("Düşük AUC (<0.70)")
    else:
        r2 = test_metrics.get("r2", 0)
        if r2 >= 0.90:
            score += 35
            factors.append("Mükemmel R² (≥0.90)")
        elif r2 >= 0.75:
            score += 25
            factors.append("İyi R² (≥0.75)")
        elif r2 >= 0.50:
            score += 15
            factors.append("Kabul edilebilir R² (≥0.50)")
        else:
            score += 5
            factors.append("Düşük R² (<0.50)")

    # 2. Overfitting durumu (0-25 puan)
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

    # 3. F1 dengesi (precision-recall balance) (0-20 puan)
    max_score += 20
    f1_w = test_metrics.get("f1_weighted", 0)
    if f1_w >= 0.85:
        score += 20
        factors.append("Dengeli F1 (≥0.85)")
    elif f1_w >= 0.75:
        score += 15
        factors.append("İyi F1 (≥0.75)")
    elif f1_w >= 0.60:
        score += 10
        factors.append("Kabul edilebilir F1 (≥0.60)")
    else:
        score += 5
        factors.append("Düşük F1 (<0.60)")

    # 4. Model kalibrasyonu (0-20 puan)
    max_score += 20
    if calibration:
        if calibration.get("is_well_calibrated", False):
            score += 20
            factors.append("İyi kalibre edilmiş")
        else:
            brier = calibration.get("brier_score", 1.0)
            if brier < 0.15:
                score += 15
                factors.append("Kabul edilebilir kalibrasyon")
            else:
                score += 5
                factors.append("Zayıf kalibrasyon")
    else:
        score += 10
        factors.append("Kalibrasyon bilgisi yok")

    # Final skor
    final_score = round(score / max_score * 100, 1)

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


# ─────────────────────────────────────────────────────────────────
# MAIN NODE
# ─────────────────────────────────────────────────────────────────

def evaluation_node(state: AgentState) -> AgentState:
    """
    Node 8: AutoML node'unun ürettiği modeli kapsamlı olarak değerlendirir.

    Dinamik değerlendirme — task tipine göre otomatik metrik seçimi.
    Hiçbir domain hardcode edilmemiştir.

    State girdileri:
        model_artifact   : Eğitilmiş model
        X_train, X_test  : Feature verisi
        y_train, y_test  : Label verisi
        feature_report   : Feature isimleri
        intent           : Task tipini belirlemek için

    State çıktıları:
        evaluation_report : Kapsamlı değerlendirme raporu
    """
    model = state.get("model_artifact")
    X_train_raw = state.get("X_train", [])
    X_test_raw = state.get("X_test", [])
    y_train_raw = state.get("y_train", [])
    y_test_raw = state.get("y_test", [])
    intent = state.get("intent", "classification")
    feature_report = state.get("feature_report", {})

    if model is None:
        return {**state, "error": "Evaluation için model bulunamadı.",
                "current_step": "evaluation_node"}

    # Unsupervised task'lar (clustering, anomaly_detection) için y verisi olmaz — sadece X kontrol et
    if not X_train_raw:
        return {**state, "error": "Evaluation için X verisi bulunamadı.",
                "current_step": "evaluation_node"}

    # Supervised task'lar için y zorunlu
    if intent not in ("clustering", "anomaly_detection") and not y_train_raw:
        return {**state, "error": "Evaluation için label (y) verisi bulunamadı.",
                "current_step": "evaluation_node"}

    # DataFrame'e dönüştür
    X_train = pd.DataFrame(X_train_raw).apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test = pd.DataFrame(X_test_raw).apply(pd.to_numeric, errors="coerce").fillna(0)
    X_train.columns = [str(c) for c in X_train.columns]
    X_test.columns = [str(c) for c in X_test.columns]

    y_train = np.array(y_train_raw)
    y_test = np.array(y_test_raw)
    feature_names = X_train.columns.tolist()

    print(f"\n[evaluation] Başlıyor...")
    print(f"  Model    : {state.get('model_name', 'unknown')}")
    print(f"  X_test   : {X_test.shape}")
    print(f"  Intent   : {intent}")

    # Task tipine göre evaluation
    if intent == "regression":
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

    # Model adı ve feature_plan bilgisini ekle
    evaluation_report["model_name"] = state.get("model_name", "unknown")
    evaluation_report["analysis_summary"] = feature_report.get("analysis_summary",
                                            state.get("feature_plan", {}).get("analysis_summary", ""))

    # ── UI için metrik açıklamaları ──────────────────────────────
    # Arayüz bu dict'i alıp tooltip / info-card olarak gösterebilir.
    evaluation_report["metric_descriptions"] = {
        # ── Test Metrikleri ────────────────────────────────────────
        "accuracy": {
            "label": "Doğruluk (Accuracy)",
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
        # ── Regression Metrikleri ──────────────────────────────────
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
        # ── Bölüm Açıklamaları ─────────────────────────────────────
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

    # ── Print sonuçlar ───────────────────────────────────────────
    reliability = evaluation_report.get("reliability_score", {})
    overfitting = evaluation_report.get("overfitting", {})
    test_m = evaluation_report.get("test_metrics", {})

    print(f"\n[evaluation] Test Metrikleri:")
    for k, v in test_m.items():
        print(f"  {k}: {v}")

    ci = evaluation_report.get("confidence_intervals", {})
    if ci:
        print(f"\n[evaluation] Bootstrap CI (%95):")
        for metric, vals in ci.items():
            print(f"  {metric}: {vals['mean']} [{vals['lower']} — {vals['upper']}] (std: {vals['std']})")

    print(f"\n[evaluation] Overfitting Analizi:")
    print(f"  Train: {overfitting.get('train_score')} | Test: {overfitting.get('test_score')}")
    print(f"  Fark : {overfitting.get('gap')} | Durum: {overfitting.get('status')}")
    print(f"  {overfitting.get('description')}")

    calibration = evaluation_report.get("calibration", {})
    if calibration:
        print(f"\n[evaluation] Kalibrasyon:")
        print(f"  Brier Score: {calibration.get('brier_score')}")
        print(f"  İyi kalibre: {calibration.get('is_well_calibrated')}")

    err_analysis = evaluation_report.get("error_analysis", {})
    if err_analysis:
        print(f"\n[evaluation] Error Analysis:")
        print(f"  Hatalı tahmin: {err_analysis.get('n_errors')}/{err_analysis.get('n_total')} ({err_analysis.get('error_rate')*100:.2f}%)")
        conf = err_analysis.get("confidence_analysis", {})
        if conf:
            print(f"  Yüksek güven hatası: {conf.get('high_confidence_errors')} adet ({conf.get('high_confidence_errors_pct')}%)")
            print(f"  Hata güveni ort.: {conf.get('mean_error_confidence')} | Doğru güveni ort.: {conf.get('mean_correct_confidence')}")
        feat_diff = err_analysis.get("feature_distribution_diff", {})
        if feat_diff:
            print(f"  En farklı feature'lar (Cohen's d effect size):")
            for feat, info in list(feat_diff.items())[:5]:
                print(f"    {feat}: d={info['cohens_d']} ({info['effect_magnitude']}) | hata={info['mean_in_errors']} | doğru={info['mean_in_correct']}")

    shap_info = evaluation_report.get("shap_importance", {})
    if shap_info:
        print(f"\n[evaluation] SHAP Top 5:")
        for feat, pct in shap_info.get("top5_shap_features", {}).items():
            print(f"  {feat}: %{pct}")
    else:
        print(f"\n[evaluation] SHAP: shap paketi bulunamadı, atlandı.")

    opt_thresh = evaluation_report.get("optimal_threshold")
    if opt_thresh is not None:
        print(f"\n[evaluation] Optimal Threshold (Youden's J): {opt_thresh}")

    fi_summary = evaluation_report.get("feature_importance_summary", {})
    if fi_summary:
        print(f"\n[evaluation] Feature Importance Özeti:")
        print(f"  Top 5 toplam katkı: %{fi_summary.get('top5_total_contribution_pct')}")
        for feat, pct in fi_summary.get("top5_features", {}).items():
            print(f"    {feat}: %{pct}")
        insig_count = fi_summary.get("insignificant_features_count", 0)
        if insig_count > 0:
            print(f"  Önemsiz feature sayısı (<%%1): {insig_count}")

    print(f"\n[evaluation] Güvenilirlik Skoru: {reliability.get('score')}/100 (Grade: {reliability.get('grade')})")
    for factor in reliability.get("factors", []):
        print(f"  ✓ {factor}")

    return {
        **state,
        "evaluation_report": evaluation_report,
        "optimal_threshold": evaluation_report.get("optimal_threshold"),
        "current_step": "evaluation_node",
    }