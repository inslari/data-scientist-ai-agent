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

from src.core.config import settings
import time
import warnings
from typing import Any
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
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
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from scipy.optimize import minimize
from src.state import AgentState

from src.core.logging_config import get_logger
logger = get_logger(__name__)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


# Model Catalogs
def _get_classification_models() -> dict:
    models = {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        # AdaBoost: hızlı, dengesiz fraud verisinde rekabetçi bir zayıf-öğrenici topluluğu.
        "adaboost": AdaBoostClassifier(n_estimators=200, learning_rate=0.8, random_state=RANDOM_STATE),
        # şimdilik devre dışı: çok yavaş (~331s) + ROC-AUC olarak zayıf
        # "gradient_boosting": GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
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
    try:
        from catboost import CatBoostClassifier
        models["catboost"] = CatBoostClassifier(
            iterations=300, random_state=RANDOM_STATE, verbose=0, allow_writing_files=False,
        )
    except ImportError:
        pass
    return models


def _get_regression_models() -> dict:
    models = {
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        # şimdilik devre dışı: çok yavaş (~331s) + zayıf
        # "gradient_boosting": GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE),
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
    try:
        from catboost import CatBoostRegressor
        models["catboost"] = CatBoostRegressor(
            iterations=300, random_state=RANDOM_STATE, verbose=0, allow_writing_files=False,
        )
    except ImportError:
        pass
    return models


# Feature importance
def _unwrap_to_estimator(model: Any) -> Any:
    """Sarmalayıcıları (CalibratedClassifierCV, SMOTE/sklearn Pipeline) soyup
    içteki gerçek tahmin edici modele iner. Stacking, çağıran tarafça ayrıca
    ele alınır (estimators_ ortalaması)."""
    # CalibratedClassifierCV → kalibre edilmiş ilk alt-modelin estimator'ı
    cc_list = getattr(model, "calibrated_classifiers_", None)
    if cc_list:
        inner = getattr(cc_list[0], "estimator", None) or getattr(cc_list[0], "base_estimator", None)
        if inner is not None:
            return _unwrap_to_estimator(inner)
    # imblearn/sklearn Pipeline → "clf" veya son adım
    if hasattr(model, "named_steps"):
        steps = model.named_steps
        nxt = steps["clf"] if "clf" in steps else list(steps.values())[-1]
        return _unwrap_to_estimator(nxt)
    return model


def _importances_from(est: Any, feature_names: list[str]):
    """Tek bir tahmin ediciden feature_importances_ veya |coef_| dizisi (ya da None)."""
    arr = None
    if hasattr(est, "feature_importances_"):
        arr = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_, dtype=float)
        if coef.ndim > 1:
            coef = np.abs(coef).mean(axis=0)
        arr = np.abs(coef)
    if arr is not None and arr.shape[0] == len(feature_names):
        return arr
    return None


def _extract_feature_importance(model: Any, feature_names: list[str]) -> dict:
    try:
        base = _unwrap_to_estimator(model)
        # OOF ağırlıklı blend: fitted base'lerin importance'larını BLEND AĞIRLIĞIYLA ortala.
        if isinstance(base, (WeightedProbaBlend, WeightedMeanBlend)) and hasattr(base, "fitted_"):
            arrs = []
            for w, sub in zip(base.weights_, base.fitted_):
                a = _importances_from(_unwrap_to_estimator(sub), feature_names)
                if a is not None:
                    arrs.append(a * w)
            if arrs:
                avg = np.sum(arrs, axis=0)
                imp = dict(zip(feature_names, avg.tolist()))
                return dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
            return {}
        # StackingClassifier/Regressor: fitted base'lerin importance'larını ortala.
        # (RandomForest gibi 'estimators_'ı olan ama final_estimator_ olmayan
        #  modellerle karışmasın diye iki niteliği birlikte kontrol ederiz.)
        if hasattr(base, "estimators_") and hasattr(base, "final_estimator_"):
            arrs = []
            for sub in base.estimators_:
                a = _importances_from(_unwrap_to_estimator(sub), feature_names)
                if a is not None:
                    arrs.append(a)
            if arrs:
                avg = np.mean(arrs, axis=0)
                imp = dict(zip(feature_names, avg.tolist()))
                return dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
            return {}
        arr = _importances_from(base, feature_names)
        if arr is not None:
            imp = dict(zip(feature_names, arr.tolist()))
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


# Optuna tuning
def _tune_model_classification(model_name: str, X_train: pd.DataFrame, y_train: np.ndarray, is_binary: bool, strong_regularization: bool = False, is_imbalanced: bool = False) -> Any:
    """Tunes the selected model using Optuna. If strong_regularization is True, shrinks the search space to prevent overfitting."""
    reg_msg = " (STRONG REGULARIZATION)" if strong_regularization else ""
    logger.info(f"\n[automl] {model_name} için Optuna hiperparametre optimizasyonu başlıyor{reg_msg} (15 sn limiti)...")
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scoring = "roc_auc" if is_binary else "roc_auc_ovr_weighted"

    def objective(trial):
        params = {}
        if model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 5, 12) if strong_regularization else trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 20, 100) if strong_regularization else trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50) if strong_regularization else trial.suggest_int("min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]) if strong_regularization else "sqrt",
            }
            model = RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)
        elif model_name == "extra_trees":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 5, 12) if strong_regularization else trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 20, 100) if strong_regularization else trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50) if strong_regularization else trial.suggest_int("min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]) if strong_regularization else "sqrt",
            }
            model = ExtraTreesClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)
        elif model_name == "adaboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
            }
            model = AdaBoostClassifier(**params, random_state=RANDOM_STATE)
        elif model_name == "gradient_boosting":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True) if strong_regularization else trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 7) if strong_regularization else trial.suggest_int("max_depth", 3, 10),
            }
            model = GradientBoostingClassifier(**params, random_state=RANDOM_STATE)
        elif model_name == "xgboost":
            from xgboost import XGBClassifier
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True) if strong_regularization else trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 7) if strong_regularization else trial.suggest_int("max_depth", 3, 10),
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

        # Imbalanced ise SMOTE pipeline ile sar (her fold'da ayrı SMOTE, leakage-free)
        cv_model = _wrap_with_smote(model, y_train) if is_imbalanced else model
        scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        return scores.mean()

    sampler = optuna.samplers.TPESampler(seed=settings.RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    try:
        study.optimize(objective, n_trials=20, timeout=15)
        best_params = study.best_params
        logger.info(f"[automl] Optuna tamamlandı. En iyi parametreler: {best_params}")
        
        # Best model Instantiate
        if model_name == "random_forest":
            return RandomForestClassifier(**best_params, random_state=settings.RANDOM_SEED, n_jobs=-1)
        elif model_name == "extra_trees":
            return ExtraTreesClassifier(**best_params, random_state=settings.RANDOM_SEED, n_jobs=-1)
        elif model_name == "adaboost":
            return AdaBoostClassifier(**best_params, random_state=settings.RANDOM_SEED)
        elif model_name == "gradient_boosting":
            return GradientBoostingClassifier(**best_params, random_state=settings.RANDOM_SEED)
        elif model_name == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(**best_params, random_state=settings.RANDOM_SEED, eval_metric="logloss", verbosity=0, n_jobs=-1)
        elif model_name == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**best_params, random_state=settings.RANDOM_SEED, verbose=-1, n_jobs=-1)
    except Exception as e:
        logger.info(f"[automl] Optuna tuning başarısız oldu, default modele dönülüyor: {e}")
        
    return None

def _tune_model_regression(model_name: str, X_train: pd.DataFrame, y_train: np.ndarray, strong_regularization: bool = False) -> Any:
    """Tunes the selected regression model using Optuna."""
    reg_msg = " (STRONG REGULARIZATION)" if strong_regularization else ""
    logger.info(f"\n[automl] {model_name} için Optuna hiperparametre optimizasyonu başlıyor{reg_msg} (15 sn limiti)...")
    
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {}
        if model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 7) if strong_regularization else trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 10, 50) if strong_regularization else trial.suggest_int("min_samples_split", 2, 10),
            }
            model = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
        elif model_name == "extra_trees":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 7) if strong_regularization else trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 10, 50) if strong_regularization else trial.suggest_int("min_samples_split", 2, 10),
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
    sampler = optuna.samplers.TPESampler(seed= settings.RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    try:
        study.optimize(objective, n_trials=20, timeout=15)
        best_params = study.best_params
        logger.info(f"[automl] Optuna tamamlandı. En iyi parametreler: {best_params}")
        
        if model_name == "random_forest":
            return RandomForestRegressor(**best_params, random_state=settings.RANDOM_SEED, n_jobs=-1)
        elif model_name == "extra_trees":
            return ExtraTreesRegressor(**best_params, random_state=settings.RANDOM_SEED, n_jobs=-1)
        elif model_name == "gradient_boosting":
            return GradientBoostingRegressor(**best_params, random_state=settings.RANDOM_SEED)
        elif model_name == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(**best_params, random_state=settings.RANDOM_SEED, verbosity=0, n_jobs=-1)
        elif model_name == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(**best_params, random_state=settings.RANDOM_SEED, verbose=-1, n_jobs=-1)
    except Exception as e:
        logger.info(f"[automl] Optuna tuning başarısız oldu, default modele dönülüyor: {e}")
        
    return None


# Classification
def _get_class_weights(y: np.ndarray) -> dict:
    """Calculates class weights."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))
 
 
def _wrap_with_smote(model, y: np.ndarray):
    """
    İmbalanced binary ise modeli [SMOTE -> model] imblearn Pipeline'ı ile sarar.
    SMOTE CV sırasında her fold'un SADECE train kısmına uygulanır (leakage yok),
    final fit'te sadece eğitim verisine uygulanır, predict'te baypas edilir.
    Dengeli veya multi-class ise modeli olduğu gibi döndürür.
    """
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        return model
    minority_ratio = counts.min() / counts.max()
    if minority_ratio > 0.3:
        return model
    k = min(5, counts.min() - 1)
    if k < 1:
        return model  # azınlık çok az, SMOTE yapılamaz
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        return ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=k)),
            ("clf", model),
        ])
    except ImportError:
        return model
 
 
def _get_classification_models_balanced(class_weights: dict) -> dict:
    """Balanced versions of models supporting class_weight."""
    cw = "balanced"  # sklearn'in built-in balanced weight
 
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            n_jobs=-1, class_weight=cw
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            n_jobs=-1, class_weight=cw
        ),
        # AdaBoost class_weight parametresini desteklemez; dengesizlikte CV döngüsü
        # bunu otomatik SMOTE pipeline'ına sarar (fold-içi, leakage-free).
        "adaboost": AdaBoostClassifier(
            n_estimators=200, learning_rate=0.8, random_state=RANDOM_STATE
        ),
        # şimdilik devre dışı: çok yavaş (~331s) + ROC-AUC olarak zayıf
        # "gradient_boosting": GradientBoostingClassifier(
        #     n_estimators=200, random_state=RANDOM_STATE
        #     # GradientBoosting class_weight desteklemiyor, sample_weight kullanacağız
        # ),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE, class_weight=cw
            )),
        ]),
    }
 
    try:
        from xgboost import XGBClassifier
        # scale_pos_weight for XGBoost binary imbalance
        pos_weight = class_weights.get(0, 1) / max(class_weights.get(1, 1), 1e-9)
        models["xgboost"] = XGBClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            eval_metric="logloss", verbosity=0, n_jobs=-1,
            scale_pos_weight=pos_weight
        )
    except ImportError:
        pass
 
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            verbose=-1, n_jobs=-1, class_weight=cw
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier
        models["catboost"] = CatBoostClassifier(
            iterations=300, random_state=RANDOM_STATE, verbose=0,
            allow_writing_files=False, auto_class_weights="Balanced",
        )
    except ImportError:
        pass

    return models
 
 
def _build_stacking_candidate(models, results, X_train, y_train, scoring, is_imbalanced):
    """En iyi 2-3 modeli StackingClassifier ile birleştirip CV skorunu döndürür.

    Dönüş: (unfitted_stack | None, cv_score | None). Maliyeti sınırlamak için
    değerlendirme 3-fold yapılır. Imbalanced'da her base SMOTE pipeline'ına
    sarılır (leakage-free); GradientBoosting SMOTE pipeline desteklemediği için
    o durumda stack base'lerinden çıkarılır.
    """
    from sklearn.base import clone
    from sklearn.ensemble import StackingClassifier

    ranked = sorted(results, key=lambda k: results[k]["score"], reverse=True)
    base_names = []
    for n in ranked:
        if is_imbalanced and n == "gradient_boosting":
            continue
        base_names.append(n)
        if len(base_names) >= 3:
            break
    if len(base_names) < 2:
        return None, None

    estimators = []
    for n in base_names:
        m = clone(models[n])
        if is_imbalanced:
            m = _wrap_with_smote(m, y_train)
        estimators.append((n, m))

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        cv=3,
        n_jobs=-1,
    )
    eval_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    try:
        scores = cross_val_score(stack, X_train, y_train, cv=eval_cv, scoring=scoring, n_jobs=-1)
        logger.info(f"[automl] {'stacking_ensemble':25s} ROC-AUC: {scores.mean():.4f} "
              f"(base: {', '.join(base_names)})")
        return stack, float(scores.mean())
    except Exception as e:
        logger.info(f"[automl] Stacking ensemble değerlendirilemedi: {e}")
        return None, None


# ── OOF (out-of-fold) ağırlıklı blend ────────────────────────
# Grandmaster ensemble'ın özü: stacking gibi bir meta-model OGRENMEK yerine, en iyi
# K modelin OOF (cross_val_predict ile leakage-free) tahminleri DOGRUDAN optimize
# edilmis agirliklarla harmanlanir. Agirliklar OOF metrigini (AUC/R²) maksimize eder
# (scipy, w>=0 & Σw=1). Stacking'den farki: meta-model yok → asiri-uyum riski dusuk,
# yorumlanabilir (her modelin katki agirligi acik), grandmaster `preds_weights`'in
# tabular karsiligi. Blend YALNIZCA OOF'ta en iyi tek modeli gecerse secilir.
_BLEND_TOP_K = 4          # en fazla bu kadar modeli harmanla (maliyet siniri)
_BLEND_MIN_SAMPLES = 50   # bundan az ornekte OOF agirlik optimizasyonu kararsiz → atla


class WeightedProbaBlend(BaseEstimator, ClassifierMixin):
    """K sinitlandiricinin predict_proba ciktilarinin sabit-agirlikli ortalamasi.
    Modul-seviyesi sinif → joblib ile picklable. fit her base'i tum train'e fit eder;
    predict_proba agirlikli olasilik ortalamasi, predict argmax."""

    def __init__(self, estimators=None, weights=None):
        self.estimators = estimators      # [(name, unfitted_estimator), ...]
        self.weights = weights

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.fitted_ = []
        for _name, est in self.estimators:
            e = clone(est)
            e.fit(X, y)
            self.fitted_.append(e)
        w = np.asarray(self.weights, dtype=float)
        self.weights_ = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
        return self

    def predict_proba(self, X):
        P = None
        for w, e in zip(self.weights_, self.fitted_):
            p = e.predict_proba(X)
            P = w * p if P is None else P + w * p
        return P

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class WeightedMeanBlend(BaseEstimator, RegressorMixin):
    """K regresörün predict ciktilarinin sabit-agirlikli ortalamasi (regression blend)."""

    def __init__(self, estimators=None, weights=None):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y):
        self.fitted_ = []
        for _name, est in self.estimators:
            e = clone(est)
            e.fit(X, y)
            self.fitted_.append(e)
        w = np.asarray(self.weights, dtype=float)
        self.weights_ = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
        return self

    def predict(self, X):
        pred = None
        for w, e in zip(self.weights_, self.fitted_):
            p = e.predict(X)
            pred = w * p if pred is None else pred + w * p
        return pred


def _optimize_blend_weights(oof_list: list, y_true: np.ndarray, metric_fn) -> tuple[np.ndarray, float]:
    """OOF tahmin listesini metrigi maksimize edecek agirliklarla harmanla.
    Nelder-Mead kullanilir (AUC gibi parcali-sabit/turevsiz metrikte SLSQP'den saglam);
    her adimda agirliklar clip(>=0)+normalize edilir (basit simpleks projeksiyonu).
    oof_list elemanlari binary'de 1D (pozitif olasilik / tahmin), multiclass'ta 2D (n,c).
    Donus: (normalize agirliklar, en iyi metrik skoru)."""
    P = np.stack(oof_list, axis=0)   # (K, n) | (K, n, c)
    K = P.shape[0]

    def _blend(w):
        w = np.clip(w, 0, None)
        s = w.sum()
        w = w / s if s > 0 else np.ones(K) / K
        return np.tensordot(w, P, axes=1)

    def _neg(w):
        try:
            return -float(metric_fn(y_true, _blend(w)))
        except Exception:
            return 1e9

    x0 = np.ones(K) / K
    best_w, best_score = x0, -_neg(x0)
    # Birkac baslangic (esit + her modele tam agirlik) ile yerel-minimum riskini azalt
    starts = [x0] + [np.eye(K)[i] for i in range(K)]
    for s0 in starts:
        res = minimize(_neg, s0, method="Nelder-Mead",
                       options={"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-6})
        w = np.clip(res.x, 0, None)
        w = w / w.sum() if w.sum() > 0 else np.ones(K) / K
        score = -_neg(w)
        if score > best_score:
            best_w, best_score = w, score
    return best_w, float(best_score)


def _build_oof_blend_candidate(
    models, results, X_train, y_train, cv, is_binary, is_imbalanced,
) -> tuple[Any, float | None, np.ndarray | None]:
    """Top-K modelin OOF olasiliklarini optimize agirliklarla harmanlayan UNFITTED
    WeightedProbaBlend + OOF skoru dondurur. Imbalanced'da her base SMOTE pipeline'ina
    sarilir (fold-ici, leakage-free). Yetersiz veri/aday → (None, None, None)."""
    if len(X_train) < _BLEND_MIN_SAMPLES:
        return None, None, None
    ranked = [n for n in sorted(results, key=lambda k: results[k]["score"], reverse=True)
              if n in models]
    base_names = ranked[:_BLEND_TOP_K]
    if len(base_names) < 2:
        return None, None, None

    oof_list, used = [], []
    for n in base_names:
        m = _wrap_with_smote(clone(models[n]), y_train) if is_imbalanced else clone(models[n])
        try:
            oof = cross_val_predict(m, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)
        except Exception as e:
            logger.info(f"[automl] OOF blend: '{n}' atlandi ({e})")
            continue
        oof_list.append(oof[:, 1] if is_binary else oof)
        used.append(n)
    if len(used) < 2:
        return None, None, None

    if is_binary:
        metric = lambda y, p: roc_auc_score(y, p)
    else:
        metric = lambda y, p: roc_auc_score(y, p, multi_class="ovr", average="weighted")
    w, blend_score = _optimize_blend_weights(oof_list, y_train, metric)

    estimators = [
        (n, _wrap_with_smote(clone(models[n]), y_train) if is_imbalanced else clone(models[n]))
        for n in used
    ]
    blend = WeightedProbaBlend(estimators=estimators, weights=w)
    logger.info(f"[automl] {'oof_blend':25s} ROC-AUC: {blend_score:.4f} "
          f"(agirliklar={dict(zip(used, np.round(w, 3).tolist()))})")
    return blend, float(blend_score), w


def _build_oof_blend_candidate_reg(
    models, results, X_train, y_train, cv,
) -> tuple[Any, float | None, np.ndarray | None]:
    """Regression OOF blend: top-K regresörün OOF tahminlerini R²'yi maksimize eden
    agirliklarla harmanlar. UNFITTED WeightedMeanBlend + OOF R² dondurur."""
    if len(X_train) < _BLEND_MIN_SAMPLES:
        return None, None, None
    ranked = [n for n in sorted(results, key=lambda k: results[k]["score"], reverse=True)
              if n in models]
    base_names = ranked[:_BLEND_TOP_K]
    if len(base_names) < 2:
        return None, None, None

    oof_list, used = [], []
    for n in base_names:
        try:
            oof = cross_val_predict(clone(models[n]), X_train, y_train, cv=cv, n_jobs=-1)
        except Exception as e:
            logger.info(f"[automl] OOF blend (reg): '{n}' atlandi ({e})")
            continue
        oof_list.append(oof)
        used.append(n)
    if len(used) < 2:
        return None, None, None

    metric = lambda y, p: r2_score(y, p)
    w, blend_score = _optimize_blend_weights(oof_list, y_train, metric)
    estimators = [(n, clone(models[n])) for n in used]
    blend = WeightedMeanBlend(estimators=estimators, weights=w)
    logger.info(f"[automl] {'oof_blend':25s} R²: {blend_score:.4f} "
          f"(agirliklar={dict(zip(used, np.round(w, 3).tolist()))})")
    return blend, float(blend_score), w


def _calibrated_fit(model, X_train, y_train, sample_weight, n_samples):
    """Modeli CalibratedClassifierCV ile sarıp fit eder (olasılık kalibrasyonu).

    method: yeterli veri varsa 'isotonic', az veride 'sigmoid'. Estimator klonlanır
    (unfitted olmalı). Herhangi bir hatada kalibrasyonsuz fit'e güvenle düşer.
    Dönüş: (fitted_model, calibration_applied: bool).
    """
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV

    method = "isotonic" if n_samples >= 1000 else "sigmoid"
    try:
        cal = CalibratedClassifierCV(clone(model), method=method, cv=3)
        if sample_weight is not None:
            cal.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            cal.fit(X_train, y_train)
        logger.info(f"[automl] Olasılık kalibrasyonu uygulandı (method={method}, cv=3)")
        return cal, True
    except Exception as e:
        logger.info(f"[automl] Kalibrasyon başarısız ({e}); kalibrasyonsuz fit'e dönülüyor")
        m = clone(model)
        if sample_weight is not None:
            m.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            m.fit(X_train, y_train)
        return m, False


def _run_classification(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
    feature_names: list[str],
    top_models: list[str] = None,
    strong_regularization: bool = False,
) -> tuple[str, Any, dict]:
    """
    6 models, 5-fold stratified CV, ROC-AUC based selection.
    For class imbalance: SMOTE + class_weight automatically applied.
    """
    from sklearn.utils.class_weight import compute_sample_weight
 
    is_binary = len(np.unique(y_train)) == 2
 
    # Class imbalance check
    classes, counts = np.unique(y_train, return_counts=True)
    minority_ratio  = counts.min() / counts.max()
    is_imbalanced   = minority_ratio < 0.3
 
    if is_imbalanced:
        logger.info(f"[automl] Dengesiz sınıf tespit edildi: {dict(zip(classes.tolist(), counts.tolist()))}")
        logger.info(f"[automl] Minority ratio: {minority_ratio:.3f} → SMOTE pipeline (fold-içi, leakage-free)")

    # Class weights hesapla
    class_weights = _get_class_weights(y_train)
 
    # Model selection
    if is_imbalanced:
        models = _get_classification_models_balanced(class_weights)
    else:
        models = _get_classification_models()
        
    if top_models:
        logger.info(f"[automl] Sadece Top Modeller denenecek: {top_models}")
        models = {k: v for k, v in models.items() if k in top_models}
        
        if not models:
            logger.error("[automl] Top modeller bulunamadı! Liste fail-safe olarak sıfırlanıyor.")
            models = _get_classification_models_balanced(class_weights) if is_imbalanced else _get_classification_models()
 
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = "roc_auc" if is_binary else "roc_auc_ovr_weighted"
    results = {}
 
    for name, model in models.items():
        start = time.time()
        try:
            # SMOTE'u pipeline içine al: her fold'un sadece train kısmına uygulanır (leakage-free)
            wrapped = _wrap_with_smote(model, y_train) if is_imbalanced else model

            # GradientBoosting SMOTE pipeline desteklemiyor (sample_weight ile yapılır)
            if name == "gradient_boosting" and is_imbalanced:
                fold_scores = []
                for train_idx, val_idx in cv.split(X_train, y_train):
                    X_fold_train = X_train.iloc[train_idx]
                    y_fold_train = y_train[train_idx]
                    X_fold_val   = X_train.iloc[val_idx]
                    y_fold_val   = y_train[val_idx]
                    sw = compute_sample_weight("balanced", y_fold_train)
                    model_clone = GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE)
                    model_clone.fit(X_fold_train, y_fold_train, sample_weight=sw)
                    proba = model_clone.predict_proba(X_fold_val)[:, 1]
                    fold_scores.append(roc_auc_score(y_fold_val, proba))
                scores_mean = np.mean(fold_scores)
                scores_std  = np.std(fold_scores)
            else:
                scores = cross_val_score(
                    wrapped, X_train, y_train,
                    cv=cv, scoring=scoring, n_jobs=-1
                )
                scores_mean = scores.mean()
                scores_std  = scores.std()

            elapsed = time.time() - start
            results[name] = {"score": scores_mean, "std": scores_std, "time": elapsed}
            logger.info(f"[automl] {name:25s} ROC-AUC: {scores_mean:.4f} ± {scores_std:.4f} ({elapsed:.1f}s)")

        except Exception as e:
            logger.error(f"[automl] {name:25s} HATA: {e}")
 
    if not results:
        raise ValueError("Hiçbir model çalışmadı.")
 
    best_name_single = max(results, key=lambda k: results[k]["score"])
    logger.info(f"\n[automl] Baseline en iyi (tek model): {best_name_single} "
          f"(ROC-AUC: {results[best_name_single]['score']:.4f})")

    # ── Ensemble adayları: (1) stacking (meta-model), (2) OOF ağırlıklı blend ──
    ensemble_model, ensemble_score = _build_stacking_candidate(
        models, results, X_train, y_train, scoring, is_imbalanced
    )
    if ensemble_score is not None:
        results["stacking_ensemble"] = {"score": ensemble_score, "std": 0.0, "time": 0.0}

    blend_model, blend_score, blend_weights = _build_oof_blend_candidate(
        models, results, X_train, y_train, cv, is_binary, is_imbalanced
    )
    if blend_score is not None:
        results["oof_blend"] = {"score": blend_score, "std": 0.0, "time": 0.0}

    # ── Seçim: en yüksek OOF/CV skorlu aday (tek model / stacking / blend) ──
    best_name = max(results, key=lambda k: results[k]["score"])
    use_blend    = best_name == "oof_blend" and blend_model is not None
    use_stacking = best_name == "stacking_ensemble" and ensemble_model is not None
    use_ensemble = use_blend or use_stacking   # metrik bayrağı (ikisi de "ensemble")

    if use_blend:
        logger.info(f"[automl] OOF blend seçildi (ROC-AUC: {blend_score:.4f} > "
              f"tek model {results[best_name_single]['score']:.4f})")
        selected_model = blend_model             # unfitted; base'ler SMOTE'la sarılı
        gb_imbalanced_path = False
    elif use_stacking:
        logger.info(f"[automl] Stacking seçildi (ROC-AUC: {ensemble_score:.4f} > "
              f"tek model {results[best_name_single]['score']:.4f})")
        selected_model = ensemble_model          # unfitted; SMOTE base'lerin içinde
        gb_imbalanced_path = False
    else:
        best_name = best_name_single
        # Optuna tuning — temiz X_train/y_train alır, SMOTE pipeline içinde otomatik uygulanır
        tuned_model = _tune_model_classification(best_name, X_train, y_train, is_binary, strong_regularization, is_imbalanced)
        selected_model = tuned_model if tuned_model is not None else models[best_name]
        # Tuned modelı da SMOTE pipeline ile sar: final fit de leakage-free
        if is_imbalanced and best_name != "gradient_boosting":
            selected_model = _wrap_with_smote(selected_model, y_train)
        gb_imbalanced_path = best_name == "gradient_boosting" and is_imbalanced

    # ── Final training (+ opsiyonel olasılık kalibrasyonu) ──
    # GradientBoosting+imbalanced sample_weight ile eğitilir; diğerlerinde SMOTE
    # pipeline veya class_weight zaten dengeyi sağlar.
    sample_weight = compute_sample_weight("balanced", y_train) if gb_imbalanced_path else None
    # cv=3 kalibrasyon için her sınıfta yeterli örnek olmalı. Blend için kalibrasyon
    # atlanır: AUC-optimize harman zaten sıralama-temelli, kalibrasyon ek maliyet +
    # WeightedProbaBlend'i CV ile sarmak gereksiz pahalı.
    can_calibrate = is_binary and int(counts.min()) >= 6 and not use_blend
    if can_calibrate:
        best_model, calibration_applied = _calibrated_fit(
            selected_model, X_train, y_train, sample_weight, n_samples=len(y_train)
        )
    else:
        if sample_weight is not None:
            selected_model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            selected_model.fit(X_train, y_train)
        best_model, calibration_applied = selected_model, False
    
    y_prob = None
    try:
        y_pred = best_model.predict(X_test)
    except Exception:
        y_pred = np.zeros(len(X_test)) # Fallback if predict fails
        
    optimal_threshold = 0.5  # default

    try:
        y_prob = best_model.predict_proba(X_test)
        if y_prob is not None and is_binary:
            # F1-score maximize threshold
            precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_prob[:, 1])
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
            best_f1_idx = np.argmax(f1_scores)
            optimal_threshold = float(pr_thresholds[best_f1_idx]) if best_f1_idx < len(pr_thresholds) else 0.5

            y_pred = (y_prob[:, 1] >= optimal_threshold).astype(int)
            logger.info(f"[automl] Optimal threshold (F1-max): {optimal_threshold:.4f}")
    except Exception as e:
        logger.error(f"[automl] Threshold hesaplama hatası: {e}")

    # Metrics
    metrics: dict[str, Any] = {
        "accuracy":            round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_weighted":         round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "precision_weighted":  round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted":     round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "task":                "binary_classification" if is_binary else "multiclass_classification",
        "all_model_scores":    {k: round(v["score"], 4) for k, v in results.items()},
        "cv_roc_auc":          round(float(results[best_name]["score"]), 4),
        "cv_roc_auc_std":      round(float(results[best_name]["std"]), 4),
        "imbalance_handled":   is_imbalanced,
        "minority_ratio":      round(float(minority_ratio), 4),
        "smote_applied":       is_imbalanced,
        "optimal_threshold":   round(optimal_threshold, 4),
        "calibration_applied": calibration_applied,
        "ensemble_used":       use_ensemble,
        "ensemble_type":       "oof_blend" if use_blend else ("stacking" if use_stacking else None),
    }
    if use_blend and blend_weights is not None:
        _blend_bases = [name for name, _ in blend_model.estimators]
        metrics["blend_weights"] = dict(zip(_blend_bases, np.round(blend_weights, 4).tolist()))

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
 


# Regression
def _handle_skewed_target(y: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Hedef değişken skewed ise log1p transform uygular.
    Skewness > 1 ve tüm değerler >= 0 ise uygular.
    """
    try:
        from scipy.stats import skew as scipy_skew
        sk = float(scipy_skew(y))
    except ImportError:
        # no scipy -> manual calculation
        mean = np.mean(y)
        std  = np.std(y)
        sk   = float(np.mean(((y - mean) / (std + 1e-9)) ** 3)) if std > 0 else 0.0
 
    if abs(sk) > 1.0 and np.all(y >= 0):
        logger.info(f"[automl] Regression hedef skewed (skewness={sk:.2f}) → log1p transform uygulanıyor")
        return np.log1p(y), True
 
    logger.info(f"[automl] Regression hedef skewness={sk:.2f} — transform gerekmez")
    return y, False
 
 
def _run_regression(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
    feature_names: list[str],
    top_models: list[str] = None,
    strong_regularization: bool = False,
) -> tuple[str, Any, dict]:
    """
    6 model, 5-fold CV, R² bazlı seçim.
    Skewed hedef için log1p transform otomatik uygulanır.
    """
    # Skew handling
    y_train_transformed, log_applied = _handle_skewed_target(y_train)
    # test setini de aynı transform ile dönüştür
    y_test_transformed = np.log1p(y_test) if log_applied else y_test
 
    models  = _get_regression_models()
    
    if top_models:
        logger.info(f"[automl] Sadece Top Modeller denenecek: {top_models}")
        models = {k: v for k, v in models.items() if k in top_models}
        if not models:
            logger.error("[automl] Top modeller bulunamadı! Liste fail-safe olarak sıfırlanıyor.")
            models = _get_regression_models()
            
    cv      = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
 
    for name, model in models.items():
        start = time.time()
        try:
            scores = cross_val_score(
                model, X_train, y_train_transformed,
                cv=cv, scoring="r2", n_jobs=-1
            )
            elapsed = time.time() - start
            results[name] = {"score": scores.mean(), "std": scores.std(), "time": elapsed}
            logger.info(f"[automl] {name:25s} R²: {scores.mean():.4f} ± {scores.std():.4f} ({elapsed:.1f}s)")
        except Exception as e:
            logger.error(f"[automl] {name:25s} HATA: {e}")
 
    if not results:
        raise ValueError("Hiçbir regression modeli çalışmadı.")
 
    best_name_single = max(results, key=lambda k: results[k]["score"])
    logger.info(f"\n[automl] Baseline en iyi: {best_name_single} (R²: {results[best_name_single]['score']:.4f})")

    # ── OOF ağırlıklı blend adayı (top-K regresör, R² optimize ağırlık) ──
    blend_model, blend_score, blend_weights = _build_oof_blend_candidate_reg(
        models, results, X_train, y_train_transformed, cv
    )
    if blend_score is not None:
        results["oof_blend"] = {"score": blend_score, "std": 0.0, "time": 0.0}

    best_name = max(results, key=lambda k: results[k]["score"])
    use_blend = best_name == "oof_blend" and blend_model is not None

    if use_blend:
        logger.info(f"[automl] OOF blend seçildi (R²: {blend_score:.4f} > "
              f"tek model {results[best_name_single]['score']:.4f})")
        best_model = blend_model          # unfitted; aşağıda fit edilir
    else:
        best_name = best_name_single
        # Optuna tuning
        tuned_model = _tune_model_regression(best_name, X_train, y_train_transformed, strong_regularization)
        best_model  = tuned_model if tuned_model is not None else models[best_name]

    # Final training
    best_model.fit(X_train, y_train_transformed)
 
    # Prediction + inverse transform
    y_pred_transformed = best_model.predict(X_test)
 
    if log_applied:
        y_pred = np.expm1(y_pred_transformed)   
        y_test_orig = y_test                     
        logger.info(f"[automl] log1p inverse transform uygulandı (expm1)")
    else:
        y_pred      = y_pred_transformed
        y_test_orig = y_test
 
    # Negatif tahminleri 0'a kırp — YALNIZCA log1p uygulandıysa (orijinal hedef ≥0
    # idi, expm1 sonrası negatif anlamsız). log YOKSA hedef meşru negatif alabilir
    # (örn. kâr/zarar, sıcaklık farkı); koşulsuz kırpma o tahminleri bozar.
    if log_applied:
        y_pred = np.maximum(y_pred, 0)
 
    mse = mean_squared_error(y_test_orig, y_pred)
    r2  = r2_score(y_test_orig, y_pred)
 
    metrics: dict[str, Any] = {
        "rmse":         round(float(np.sqrt(mse)), 4),
        "mae":          round(float(mean_absolute_error(y_test_orig, y_pred)), 4),
        "r2":           round(float(r2), 4),
        "mse":          round(float(mse), 4),
        "task":         "regression",
        "log_transform_applied": log_applied,
        "all_model_scores": {k: round(v["score"], 4) for k, v in results.items()},
        "cv_r2":        round(float(results[best_name]["score"]), 4),
        "cv_r2_std":    round(float(results[best_name]["std"]), 4),
        "ensemble_used": use_blend,
        "ensemble_type": "oof_blend" if use_blend else None,
    }
    if use_blend and blend_weights is not None:
        _blend_bases = [name for name, _ in blend_model.estimators]
        metrics["blend_weights"] = dict(zip(_blend_bases, np.round(blend_weights, 4).tolist()))

    metrics["feature_importance"] = _extract_feature_importance(best_model, feature_names)
    return best_name, best_model, metrics



def _safe_silhouette(X, labels, cap: int = 5000) -> float:
    """silhouette_score with a row cap to stay O(cap²), not O(n²).

    silhouette builds a full pairwise-distance matrix, so on large datasets it
    is the single most expensive (and memory-hungry) call in the AutoML path —
    on a 100k+ row set it can stall the worker for minutes or OOM it. A long
    stall here is also what lets the idle LangGraph **checkpoint** connection get
    dropped by the server/pooler, surfacing later as a confusing
    ``aput_writes`` connection error. We therefore evaluate the score on a
    deterministic random subsample (seed-fixed) when ``n`` exceeds ``cap`` — a
    standard, unbiased estimate of the full-data silhouette.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    n = X.shape[0]
    if n > cap:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(n, size=cap, replace=False)
        X, labels = X[idx], labels[idx]
    # Need >=2 distinct labels among the (sub)sample, else silhouette is undefined.
    if len(np.unique(labels)) < 2:
        return -1.0
    return float(silhouette_score(X, labels))


# Clustering
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
    results: dict[str, dict[str, Any]] = {}

    # KMeans — k=2..8
    logger.info("[automl] KMeans deniyor (k=2..8)...")
    best_k = 2
    best_kmeans_score = -1

    for k in range(2, 9):
        try:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = km.fit_predict(X_all)
            if len(np.unique(labels)) < 2:
                continue
            sil = _safe_silhouette(X_all, labels)
            if sil > best_kmeans_score:
                best_kmeans_score = sil
                best_k = k
            logger.info(f"[automl]   k={k}: silhouette={sil:.4f}")
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
    logger.info(f"[automl] KMeans best: k={best_k}, silhouette={best_kmeans_score:.4f}")

    # DBSCAN
    logger.info("[automl] DBSCAN deniyor...")
    try:
        db = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
        db_labels = db.fit_predict(X_all)
        n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        noise_pct = round(float((db_labels == -1).mean() * 100), 2)

        if n_clusters_db >= 2:
            mask = db_labels != -1
            db_sil = _safe_silhouette(X_all[mask], db_labels[mask]) if mask.sum() > 1 else -1
        else:
            db_sil = -1

        results["dbscan"] = {
            "model": db,
            "labels": db_labels,
            "silhouette": db_sil,
            "n_clusters": n_clusters_db,
            "params": f"eps=0.5, min_samples=5, noise={noise_pct}%",
        }
        logger.info(f"[automl] DBSCAN: clusters={n_clusters_db}, silhouette={db_sil:.4f}, noise={noise_pct}%")
    except Exception as e:
        logger.error(f"[automl] DBSCAN HATA: {e}")

    # Choose the best algorithm
    best_algo = max(results, key=lambda k: results[k]["silhouette"])
    best = results[best_algo]
    best_model = best["model"]
    best_labels = best["labels"]

    logger.info(f"\n[automl] En iyi clustering: {best_algo} (silhouette={best['silhouette']:.4f})")

    # Cluster statistics
    cluster_stats: dict[str, dict[str, Any]] = {}
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


# Anomaly Detection

class GaussianDensityAnomalyDetector(BaseEstimator):
    """Multivariate-normal (independent-Gaussian product) anomaly detector.

    A from-scratch port of the credit-card anomaly notebook's density model:
    fit a univariate normal per feature, score a row by the *product* of those
    densities, and flag rows whose density falls below a threshold. We work in
    **log space** (sum of per-feature log-pdfs) instead of multiplying raw
    densities — mathematically identical for thresholding, but immune to the
    floating-point underflow that makes the raw product collapse to 0 once you
    have more than a handful of features (the notebook's threshold was already
    ~1e-19 at 9 features).

    The independence assumption is the same one the notebook leans on, and it is
    reasonable here because the upstream feature builder's columns are largely
    decorrelated aggregates. ``contamination`` sets the quantile of the training
    score distribution used as the decision threshold, matching the
    IsolationForest convention so both algorithms are directly comparable.

    sklearn-compatible surface (``fit`` / ``predict`` / ``score_samples`` /
    ``decision_function``) so it slots into the existing selection + persistence
    path with no special-casing.
    """

    def __init__(self, contamination: float = 0.10):
        self.contamination = contamination

    def fit(self, X, y=None):  # noqa: ANN001
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.mu_ = np.nanmean(X, axis=0)
        sigma = np.nanstd(X, axis=0)
        # Floor sigma so a constant feature can't divide by zero / dominate.
        floor = np.maximum(1e-6, 1e-6 * np.abs(self.mu_))
        self.sigma_ = np.where(sigma > floor, sigma, floor)
        scores = self.score_samples(X)
        # Low density = anomalous → threshold is the lower-tail quantile.
        self.offset_ = float(np.quantile(scores, self.contamination))
        return self

    def score_samples(self, X):  # noqa: ANN001
        """Log joint density (higher = more normal). NaNs treated as neutral."""
        X = np.asarray(X, dtype=float)
        z = (X - self.mu_) / self.sigma_
        log_pdf = -0.5 * z**2 - np.log(self.sigma_) - 0.5 * np.log(2 * np.pi)
        log_pdf = np.where(np.isfinite(log_pdf), log_pdf, 0.0)
        return log_pdf.sum(axis=1)

    def decision_function(self, X):  # noqa: ANN001
        """>= 0 → normal, < 0 → anomaly (sklearn outlier convention)."""
        return self.score_samples(X) - self.offset_

    def predict(self, X):  # noqa: ANN001
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def fit_predict(self, X, y=None):  # noqa: ANN001
        return self.fit(X, y).predict(X)


def _build_anomaly_candidates(cont: float) -> dict[str, Any]:
    """Candidate unsupervised detectors evaluated at a given contamination."""
    return {
        "isolation_forest": IsolationForest(
            contamination=cont, random_state=RANDOM_STATE, n_estimators=200, n_jobs=-1,
        ),
        "gaussian_density": GaussianDensityAnomalyDetector(contamination=cont),
    }


def _run_anomaly_detection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list[str],
) -> tuple[str, Any, dict]:
    """
    Anomaly detection with two complementary unsupervised detectors:
      - IsolationForest (tree-isolation depth)
      - GaussianDensityAnomalyDetector (multivariate-normal density — port of the
        credit-card anomaly notebook)
    Both are scanned over candidate contamination rates (%5–15 anomaly varsayımı)
    and the (algorithm, contamination) pair with the best silhouette wins.
    """
    X_all = pd.concat([X_train, X_test], ignore_index=True).values

    # Test different contamination rates × both algorithms
    contaminations = [0.05, 0.10, 0.15]
    best_key: tuple[str, float] | None = None
    best_score = -np.inf
    results: dict[tuple[str, float], dict[str, Any]] = {}

    logger.info("[automl] Anomaly detection — IsolationForest + GaussianDensity taraması...")
    for cont in contaminations:
        for algo, model in _build_anomaly_candidates(cont).items():
            try:
                labels = model.fit_predict(X_all)
                # -1 anomaly, 1 normal → convert for silhouette
                labels_bin = (labels == -1).astype(int)
                n_anomalies = int((labels == -1).sum())

                if 1 < n_anomalies < len(labels) - 1:
                    try:
                        sil = _safe_silhouette(X_all, labels_bin)
                    except Exception:
                        sil = -1
                else:
                    sil = -1

                results[(algo, cont)] = {
                    "algo": algo,
                    "model": model,
                    "labels": labels,
                    "n_anomalies": n_anomalies,
                    "anomaly_pct": round(float(n_anomalies / len(labels) * 100), 2),
                    "silhouette": sil,
                }
                logger.info(f"[automl]   {algo} contamination={cont}: "
                            f"anomalies={n_anomalies} (%{results[(algo, cont)]['anomaly_pct']}), "
                            f"silhouette={sil if sil == -1 else round(float(sil), 4)}")

                if sil > best_score:
                    best_score = sil
                    best_key = (algo, cont)
            except Exception as e:
                logger.error(f"[automl]   {algo} contamination={cont}: HATA {e}")

    if best_key is None:
        best_key = next(iter(results))
    best = results[best_key]
    best_algo, best_contamination = best_key
    best_model = best["model"]
    best_labels = best["labels"]

    logger.info(f"\n[automl] En iyi: {best_algo} (contamination={best_contamination}) "
          f"— {best['n_anomalies']} anomalies, %{best['anomaly_pct']}")

    # Anomaly scores (decision function)
    anomaly_scores = best_model.decision_function(X_all)
    score_threshold = float(np.percentile(anomaly_scores, best_contamination * 100))

    n_train = len(X_train)
    metrics: dict[str, Any] = {
        "task": "anomaly_detection",
        "algorithm": best_algo,
        "best_contamination": best_contamination,
        "n_anomalies": best["n_anomalies"],
        "anomaly_pct": best["anomaly_pct"],
        "n_normal": int((best_labels == 1).sum()),
        "score_threshold": round(score_threshold, 4),
        "silhouette_score": round(float(best_score), 4) if best_score > -np.inf else None,
        "contamination_scan": {
            f"{algo}@{c}": {
                "n_anomalies": r["n_anomalies"],
                "anomaly_pct": r["anomaly_pct"],
                "silhouette": r["silhouette"] if r["silhouette"] == -1 else round(float(r["silhouette"]), 4),
            }
            for (algo, c), r in results.items()
        },
        "train_labels": best_labels[:n_train].tolist(),
        "test_labels": best_labels[n_train:].tolist(),
    }

    # Feature importance — IsolationForest exposes mean-depth importances;
    # the Gaussian model ranks features by mean standardized deviation |z|.
    try:
        if best_algo == "isolation_forest":
            fi_raw = best_model.feature_importances_
            fi = dict(zip(feature_names, fi_raw.tolist()))
        else:
            z = np.abs((X_all - best_model.mu_) / best_model.sigma_)
            fi = dict(zip(feature_names, np.nanmean(z, axis=0).tolist()))
        metrics["feature_importance"] = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        metrics["feature_importance"] = {}

    return best_algo, best_model, metrics

# Main node
def train_model(state: AgentState) -> dict:
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
    from src.utils.dataset_store import load_raw, dump_y
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = load_raw(state)
    intent = state.get("intent", "classification")
    intent_norm = intent.strip().lower() if isinstance(intent, str) else "classification"

    if not X_train_raw:
        return {**state, "error": "AutoML için veri bulunamadı.",
                "current_step": "automl_node"}

    X_train, X_test, y_train, y_test = _prep_data(
        X_train_raw, X_test_raw, y_train_raw, y_test_raw
    )
    feature_names = X_train.columns.tolist()

    # Auto-correct intent if y is categorical string but intent is regression
    if intent_norm == "regression" and y_train.dtype.kind in ('O', 'U', 'S'):
        try:
            y_train.astype(float)
        except ValueError:
            logger.warning("[automl] Uyarı: y değişkeni string içeriyor, intent 'classification' olarak değiştirildi.")
            intent_norm = "classification"

    if intent_norm == "classification" or (intent_norm not in ("regression", "clustering", "anomaly_detection")):
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        known_classes = set(le.classes_)
        y_test = np.array([le.transform([val])[0] if val in known_classes else -1 for val in y_test])  # type: ignore

    logger.info(f"\n[automl] Başlıyor:")
    logger.info(f"  X_train : {X_train.shape}")
    logger.info(f"  X_test  : {X_test.shape}")
    logger.info(f"  Intent  : {intent_norm}")

    # Task routing   
    top_models = state.get("top_models", [])
    critic_feedback = state.get("critic_feedback", [])
    strong_regularization = any("Strong Regularization" in f or "Overfitting" in f for f in critic_feedback)
    
    if strong_regularization:
        logger.info("[automl] Critic Node'dan Strong Regularization (Finetuning) sinyali alındı!")

    if intent_norm == "classification":
        model_name, model, metrics = _run_classification(
            X_train, y_train, X_test, y_test, feature_names, top_models, strong_regularization
        )
    elif intent_norm == "regression":
        model_name, model, metrics = _run_regression(
            X_train, y_train, X_test, y_test, feature_names, top_models, strong_regularization
        )
    elif intent_norm == "clustering":
        model_name, model, metrics = _run_clustering(
            X_train, X_test, feature_names
        )
    elif intent_norm == "anomaly_detection":
        model_name, model, metrics = _run_anomaly_detection(
            X_train, X_test, feature_names
        )

    else:
        # classification (default) + eda/report fallback
        if not y_train_raw:
            return {**state, "error": "Classification için label (y) verisi bulunamadı.",
                    "current_step": "automl_node"}
        model_name, model, metrics = _run_classification(X_train, y_train, X_test, y_test, feature_names)  # type: ignore

    # Print results   
    logger.info(f"\n[automl] Sonuçlar ({model_name}):")
    for k, v in metrics.items():
        if k not in ("feature_importance", "all_model_scores", "all_algorithm_scores",
                     "cluster_distribution", "contamination_scan", "train_labels", "test_labels",
                     "per_class_report"):
            logger.info(f"  {k}: {v}")

    fi = metrics.get("feature_importance", {})
    if fi:
        logger.info(f"\n[automl] Top 5 feature:")
        for feat, imp in list(fi.items())[:5]:
            logger.info(f"  {feat}: {imp:.4f}")

    # Update top_models if first run
    if intent in ("classification", "regression") and not top_models:
        # Pseudo-ensemble anahtarlarini (oof_blend/stacking_ensemble) DISLA: bunlar
        # gercek model degil → top_models'a girerse retry'da model filtresini bozar.
        _pseudo = {"oof_blend", "stacking_ensemble"}
        all_scores = {k: v for k, v in metrics.get("all_model_scores", {}).items() if k not in _pseudo}
        if all_scores:
            # Sort models by score descending and take top 2
            sorted_models = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            top_models = [m[0] for m in sorted_models[:2]]
            logger.info(f"[automl] Top 2 model seçildi ve kaydedildi: {top_models}")

    # Save model to disk to avoid LangGraph serialization errors
    import joblib
    import os
    import uuid
    os.makedirs("data/models", exist_ok=True)
    model_path = f"data/models/model_{uuid.uuid4().hex}.pkl"
    try:
        joblib.dump(model, model_path)
    except Exception as e:
        logger.warning(f"[automl] UYARI: Model diske kaydedilemedi: {e}")
        model_path = None

    # Label-encode SONRASI y'yi diske yaz (downstream evaluator encoded y bekler);
    # inline yerine yol taşı (checkpoint payload şişmesini önler).
    y_paths = dump_y(y_train, y_test)

    return {
        **state,
        "intent": intent,
        **y_paths,
        "model_name": model_name,
        "model_params": {"intent": intent},
        "model_artifact": model_path,
        "metrics": metrics,
        "top_models": top_models,
        "current_step": "automl_node",
    }