from __future__ import annotations

"""
AutoML Node
-----------
Intelligent automatic model selection and training based on intent.

Supported tasks:
    - classification    : Binary/multiclass — 6 model, 5-fold CV, ROC-AUC
    - regression        : Numerical prediction — 5 model, 5-fold CV, R²
    - clustering        : KMeans + DBSCAN — optimal k, silhouette score
    - anomaly_detection : IsolationForest — contamination otomatik

No domain or column names are hardcoded.
"""

import time
import warnings
from typing import Any
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.state import AgentState

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────
# MODEL KATALOGLARI
# ─────────────────────────────────────────────────────────────────

def _get_classification_models() -> dict:
    models = {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            eval_metric="logloss", verbosity=0, n_jobs=-1,
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(
            n_estimators=200, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
        )
    except ImportError:
        pass
    return models


def _get_regression_models() -> dict:
    models = {
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "ridge": Pipeline([("scaler", StandardScaler()), ("reg", Ridge())]),
        "lasso": Pipeline([("scaler", StandardScaler()), ("reg", Lasso(max_iter=5000))]),
    }
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = XGBRegressor(
            n_estimators=200, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        models["lightgbm"] = LGBMRegressor(
            n_estimators=200, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
        )
    except ImportError:
        pass
    return models


# ─────────────────────────────────────────────────────────────────
# HELPERS — feature importance
# ─────────────────────────────────────────────────────────────────

def _extract_feature_importance(model: Any, feature_names: list[str]) -> dict:
    try:
        est = model
        if hasattr(model, "named_steps"):
            est = list(model.named_steps.values())[-1]
        if hasattr(est, "feature_importances_"):
            imp = dict(zip(feature_names, est.feature_importances_.tolist()))
            return dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(est, "coef_"):
            coef = est.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            imp = dict(zip(feature_names, np.abs(coef).tolist()))
            return dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        pass
    return {}


def _prep_data(X_train_raw, X_test_raw, y_train_raw, y_test_raw):
    """DataFrame'e dönüştür ve temizle."""
    X_train = pd.DataFrame(X_train_raw)
    X_test = pd.DataFrame(X_test_raw)
    X_train.columns = [str(c) for c in X_train.columns]
    X_test.columns = [str(c) for c in X_test.columns]
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train = np.array(y_train_raw)
    y_test = np.array(y_test_raw)
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────
# OPTUNA TUNING
# ─────────────────────────────────────────────────────────────────

def _tune_model_classification(model_name: str, X_train: pd.DataFrame, y_train: np.ndarray, is_binary: bool) -> Any:
    """Tunes the selected model using Optuna."""
    print(f"\n[automl] {model_name} için Optuna hiperparametre optimizasyonu başlıyor (15 sn limiti)...")
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scoring = "roc_auc" if is_binary else "roc_auc_ovr_weighted"

    def objective(trial):
        params = {}
        if model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            }
            model = RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)
        elif model_name == "extra_trees":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            }
            model = ExtraTreesClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)
        elif model_name == "gradient_boosting":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
            }
            model = GradientBoostingClassifier(**params, random_state=RANDOM_STATE)
        elif model_name == "xgboost":
            from xgboost import XGBClassifier
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
            }
            model = XGBClassifier(**params, random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0, n_jobs=-1)
        elif model_name == "lightgbm":
            from lightgbm import LGBMClassifier
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
            }
            model = LGBMClassifier(**params, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
        else:
            raise optuna.exceptions.TrialPruned()

        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    try:
        study.optimize(objective, n_trials=20, timeout=15)
        best_params = study.best_params
        print(f"[automl] Optuna tamamlandı. En iyi parametreler: {best_params}")
        
        # Instantiate the best model
        if model_name == "random_forest":
            return RandomForestClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
        elif model_name == "extra_trees":
            return ExtraTreesClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
        elif model_name == "gradient_boosting":
            return GradientBoostingClassifier(**best_params, random_state=RANDOM_STATE)
        elif model_name == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(**best_params, random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0, n_jobs=-1)
        elif model_name == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**best_params, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    except Exception as e:
        print(f"[automl] Optuna tuning başarısız oldu, default modele dönülüyor: {e}")
        
    return None

def _tune_model_regression(model_name: str, X_train: pd.DataFrame, y_train: np.ndarray) -> Any:
    """Tunes the selected regression model using Optuna."""
    print(f"\n[automl] {model_name} için Optuna hiperparametre optimizasyonu başlıyor (15 sn limiti)...")
    
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {}
        if model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            }
            model = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
        elif model_name == "extra_trees":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            }
            model = ExtraTreesRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
        elif model_name == "gradient_boosting":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
            }
            model = GradientBoostingRegressor(**params, random_state=RANDOM_STATE)
        elif model_name == "xgboost":
            from xgboost import XGBRegressor
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
            }
            model = XGBRegressor(**params, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
        elif model_name == "lightgbm":
            from lightgbm import LGBMRegressor
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
            }
            model = LGBMRegressor(**params, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
        else:
            raise optuna.exceptions.TrialPruned()

        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    try:
        study.optimize(objective, n_trials=20, timeout=15)
        best_params = study.best_params
        print(f"[automl] Optuna tamamlandı. En iyi parametreler: {best_params}")
        
        if model_name == "random_forest":
            return RandomForestRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
        elif model_name == "extra_trees":
            return ExtraTreesRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
        elif model_name == "gradient_boosting":
            return GradientBoostingRegressor(**best_params, random_state=RANDOM_STATE)
        elif model_name == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(**best_params, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
        elif model_name == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(**best_params, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    except Exception as e:
        print(f"[automl] Optuna tuning başarısız oldu, default modele dönülüyor: {e}")
        
    return None




# ─────────────────────────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────────────────────────

def _run_classification(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
    feature_names: list[str],
) -> tuple[str, Any, dict]:
    """6 model, 5-fold stratified CV, ROC-AUC bazlı seçim."""
    is_binary = len(np.unique(y_train)) == 2
    models = _get_classification_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, model in models.items():
        start = time.time()
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv,
                                     scoring="roc_auc" if is_binary else "roc_auc_ovr_weighted",
                                     n_jobs=-1)
            elapsed = time.time() - start
            results[name] = {"score": scores.mean(), "std": scores.std(), "time": elapsed}
            print(f"[automl] {name:25s} ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f} ({elapsed:.1f}sn)")
        except Exception as e:
            print(f"[automl] {name:25s} HATA: {e}")

    best_name = max(results, key=lambda k: results[k]["score"])
    print(f"\n[automl] Baseline en iyi model: {best_name} (ROC-AUC: {results[best_name]['score']:.4f})")
    
    # Optuna tuning
    tuned_model = _tune_model_classification(best_name, X_train, y_train, is_binary)
    best_model = tuned_model if tuned_model is not None else models[best_name]

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_prob = None
    try:
        y_prob = best_model.predict_proba(X_test)
    except Exception:
        pass

    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "task": "binary_classification" if is_binary else "multiclass_classification",
        "all_model_scores": {k: round(v["score"], 4) for k, v in results.items()},
        "cv_roc_auc": round(float(results[best_name]["score"]), 4),
        "cv_roc_auc_std": round(float(results[best_name]["std"]), 4),
    }

    if y_prob is not None:
        try:
            if is_binary:
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob[:, 1])), 4)
            else:
                metrics["roc_auc_ovr"] = round(float(
                    roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
                ), 4)
        except Exception:
            pass

    metrics["feature_importance"] = _extract_feature_importance(best_model, feature_names)
    return best_name, best_model, metrics


# ─────────────────────────────────────────────────────────────────
# REGRESSION
# ─────────────────────────────────────────────────────────────────

def _run_regression(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
    feature_names: list[str],
) -> tuple[str, Any, dict]:
    """6 model, 5-fold CV, R² bazlı seçim."""
    models = _get_regression_models()
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, model in models.items():
        start = time.time()
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv,
                                     scoring="r2", n_jobs=-1)
            elapsed = time.time() - start
            results[name] = {"score": scores.mean(), "std": scores.std(), "time": elapsed}
            print(f"[automl] {name:25s} R²: {scores.mean():.4f} ± {scores.std():.4f} ({elapsed:.1f}sn)")
        except Exception as e:
            print(f"[automl] {name:25s} HATA: {e}")

    best_name = max(results, key=lambda k: results[k]["score"])
    print(f"\n[automl] Baseline en iyi model: {best_name} (R²: {results[best_name]['score']:.4f})")
    
    # Optuna tuning
    tuned_model = _tune_model_regression(best_name, X_train, y_train)
    best_model = tuned_model if tuned_model is not None else models[best_name]

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    metrics: dict[str, Any] = {
        "rmse": round(float(np.sqrt(mse)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "mse": round(float(mse), 4),
        "task": "regression",
        "all_model_scores": {k: round(v["score"], 4) for k, v in results.items()},
        "cv_r2": round(float(results[best_name]["score"]), 4),
        "cv_r2_std": round(float(results[best_name]["std"]), 4),
    }

    metrics["feature_importance"] = _extract_feature_importance(best_model, feature_names)
    return best_name, best_model, metrics



# CLUSTERING
def _run_clustering(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list[str],
) -> tuple[str, Any, dict]:
    """
    KMeans (k=2..8) + DBSCAN dener.
    Silhouette score ile en iyi algoritma ve parametreyi seçer.
    """
    X_all = pd.concat([X_train, X_test], ignore_index=True).values
    results = {}

    # KMeans — k=2..8
    print("[automl] KMeans deniyor (k=2..8)...")
    best_k = 2
    best_kmeans_score = -1

    for k in range(2, 9):
        try:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = km.fit_predict(X_all)
            if len(np.unique(labels)) < 2:
                continue
            sil = silhouette_score(X_all, labels)
            if sil > best_kmeans_score:
                best_kmeans_score = sil
                best_k = k
            print(f"[automl]   k={k}: silhouette={sil:.4f}")
        except Exception:
            continue

    km_best = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    km_labels = km_best.fit_predict(X_all)
    results["kmeans"] = {
        "model": km_best,
        "labels": km_labels,
        "silhouette": best_kmeans_score,
        "n_clusters": best_k,
        "params": f"k={best_k}",
    }
    print(f"[automl] KMeans best: k={best_k}, silhouette={best_kmeans_score:.4f}")

    # DBSCAN
    print("[automl] DBSCAN deniyor...")
    try:
        db = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
        db_labels = db.fit_predict(X_all)
        n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        noise_pct = round(float((db_labels == -1).mean() * 100), 2)

        if n_clusters_db >= 2:
            mask = db_labels != -1
            db_sil = silhouette_score(X_all[mask], db_labels[mask]) if mask.sum() > 1 else -1
        else:
            db_sil = -1

        results["dbscan"] = {
            "model": db,
            "labels": db_labels,
            "silhouette": db_sil,
            "n_clusters": n_clusters_db,
            "params": f"eps=0.5, min_samples=5, noise={noise_pct}%",
        }
        print(f"[automl] DBSCAN: clusters={n_clusters_db}, silhouette={db_sil:.4f}, noise={noise_pct}%")
    except Exception as e:
        print(f"[automl] DBSCAN HATA: {e}")

    # Choose the best algorithm
    best_algo = max(results, key=lambda k: results[k]["silhouette"])
    best = results[best_algo]
    best_model = best["model"]
    best_labels = best["labels"]

    print(f"\n[automl] En iyi clustering: {best_algo} (silhouette={best['silhouette']:.4f})")

    # Cluster statistics
    cluster_stats = {}
    unique_labels = sorted(set(best_labels))
    for lbl in unique_labels:
        if lbl == -1:
            cluster_stats["noise"] = {"count": int((best_labels == -1).sum())}
            continue
        mask = best_labels == lbl
        cluster_stats[f"cluster_{lbl}"] = {
            "count": int(mask.sum()),
            "pct": round(float(mask.mean() * 100), 2),
        }

    # Additional metrics
    extra_metrics = {}
    try:
        valid_mask = best_labels != -1
        if valid_mask.sum() > 1 and len(np.unique(best_labels[valid_mask])) >= 2:
            extra_metrics["davies_bouldin"] = round(
                float(davies_bouldin_score(X_all[valid_mask], best_labels[valid_mask])), 4
            )
            extra_metrics["calinski_harabasz"] = round(
                float(calinski_harabasz_score(X_all[valid_mask], best_labels[valid_mask])), 4
            )
    except Exception:
        pass

    metrics: dict[str, Any] = {
        "task": "clustering",
        "best_algorithm": best_algo,
        "n_clusters": best["n_clusters"],
        "silhouette_score": round(float(best["silhouette"]), 4),
        "params": best["params"],
        "cluster_distribution": cluster_stats,
        "all_algorithm_scores": {
            k: round(float(v["silhouette"]), 4) for k, v in results.items()
        },
        **extra_metrics,
    }

    # Save separate labels for train and test
    n_train = len(X_train)
    metrics["train_labels"] = best_labels[:n_train].tolist()
    metrics["test_labels"] = best_labels[n_train:].tolist()

    return best_algo, best_model, metrics


# ANOMALY DETECTION
def _run_anomaly_detection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list[str],
) -> tuple[str, Any, dict]:
    """
    IsolationForest ile anomaly detection.
    Contamination otomatik tahmin edilir (veri setinin %5-15'i anomali varsayılır).
    """
    X_all = pd.concat([X_train, X_test], ignore_index=True).values

    # Test different contamination rates
    contaminations = [0.05, 0.10, 0.15]
    best_contamination = 0.10
    best_score = -np.inf
    results = {}

    print("[automl] IsolationForest — contamination taraması...")
    for cont in contaminations:
        try:
            iso = IsolationForest(
                contamination=cont,
                random_state=RANDOM_STATE,
                n_estimators=200,
                n_jobs=-1,
            )
            labels = iso.fit_predict(X_all)
            # -1 anomaly, 1 normal → convert for silhouette
            labels_bin = (labels == -1).astype(int)
            n_anomalies = (labels == -1).sum()

            if n_anomalies > 1 and n_anomalies < len(labels) - 1:
                try:
                    sil = silhouette_score(X_all, labels_bin)
                except Exception:
                    sil = -1
            else:
                sil = -1

            results[cont] = {
                "model": iso,
                "labels": labels,
                "n_anomalies": int(n_anomalies),
                "anomaly_pct": round(float(n_anomalies / len(labels) * 100), 2),
                "silhouette": sil,
            }
            print(f"[automl]   contamination={cont}: anomalies={n_anomalies} (%{results[cont]['anomaly_pct']})")

            if sil > best_score:
                best_score = sil
                best_contamination = cont
        except Exception as e:
            print(f"[automl]   contamination={cont}: HATA {e}")

    best = results.get(best_contamination, list(results.values())[0])
    best_model = best["model"]
    best_labels = best["labels"]

    print(f"\n[automl] En iyi contamination: {best_contamination} "
          f"({best['n_anomalies']} anomalies, %{best['anomaly_pct']})")

    # Anomaly scores (decision function)
    anomaly_scores = best_model.decision_function(X_all)
    score_threshold = float(np.percentile(anomaly_scores, best_contamination * 100))

    n_train = len(X_train)
    metrics: dict[str, Any] = {
        "task": "anomaly_detection",
        "algorithm": "isolation_forest",
        "best_contamination": best_contamination,
        "n_anomalies": best["n_anomalies"],
        "anomaly_pct": best["anomaly_pct"],
        "n_normal": int((best_labels == 1).sum()),
        "score_threshold": round(score_threshold, 4),
        "silhouette_score": round(float(best_score), 4) if best_score > -np.inf else None,
        "contamination_scan": {
            str(c): {
                "n_anomalies": r["n_anomalies"],
                "anomaly_pct": r["anomaly_pct"],
            }
            for c, r in results.items()
        },
        "train_labels": best_labels[:n_train].tolist(),
        "test_labels": best_labels[n_train:].tolist(),
    }

    # Feature importance - IsolationForest using mean depth
    try:
        fi_raw = best_model.feature_importances_
        fi = dict(zip(feature_names, fi_raw.tolist()))
        metrics["feature_importance"] = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        metrics["feature_importance"] = {}

    return "isolation_forest", best_model, metrics

# MAIN NODE
def automl_node(state: AgentState) -> AgentState:
    """
    Intent-based automatic model selection and training.

    Supported tasks:
        classification    → 6 model, 5-fold CV, ROC-AUC
        regression        → 6 model, 5-fold CV, R²
        clustering        → KMeans(k=2..8) + DBSCAN, silhouette
        anomaly_detection → IsolationForest, contamination taraması

    State inputs:
        X_train, X_test, y_train, y_test : Feature engineering output
        intent                           : Task type

    State outputs:
        model_name     : Selected model/algorithm name
        model_params   : Parameters
        model_artifact : Trained model object
        metrics        : Task-specific metrics
    """
    X_train_raw = state.get("X_train", [])
    X_test_raw = state.get("X_test", [])
    y_train_raw = state.get("y_train", [])
    y_test_raw = state.get("y_test", [])
    intent = state.get("intent", "classification")

    if not X_train_raw:
        return {**state, "error": "AutoML için veri bulunamadı.",
                "current_step": "automl_node"}

    X_train, X_test, y_train, y_test = _prep_data(
        X_train_raw, X_test_raw, y_train_raw, y_test_raw
    )
    feature_names = X_train.columns.tolist()

    print(f"\n[automl] Başlıyor:")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  Intent  : {intent}")

    # Task routing   
    if intent == "regression":
        model_name, model, metrics = _run_regression(
            X_train, y_train, X_test, y_test, feature_names
        )

    elif intent == "clustering":
        model_name, model, metrics = _run_clustering(
            X_train, X_test, feature_names
        )

    elif intent == "anomaly_detection":
        model_name, model, metrics = _run_anomaly_detection(
            X_train, X_test, feature_names
        )

    else:
        # classification (default) + eda/report fallback
        if not y_train_raw:
            return {**state, "error": "Classification için label (y) verisi bulunamadı.",
                    "current_step": "automl_node"}
        model_name, model, metrics = _run_classification(
            X_train, y_train, X_test, y_test, feature_names
        )

    # Print results   
    print(f"\n[automl] Sonuçlar ({model_name}):")
    for k, v in metrics.items():
        if k not in ("feature_importance", "all_model_scores", "all_algorithm_scores",
                     "cluster_distribution", "contamination_scan", "train_labels", "test_labels",
                     "per_class_report"):
            print(f"  {k}: {v}")

    fi = metrics.get("feature_importance", {})
    if fi:
        print(f"\n[automl] Top 5 feature:")
        for feat, imp in list(fi.items())[:5]:
            print(f"  {feat}: {imp:.4f}")

    return {
        **state,
        "model_name": model_name,
        "model_params": {"intent": intent},
        "model_artifact": model,
        "metrics": metrics,
        "current_step": "automl_node",
    }