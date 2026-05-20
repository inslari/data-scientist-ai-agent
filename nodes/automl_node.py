from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from state import AgentState

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


CANDIDATE_MODELS = {
    "random_forest": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "extra_trees": ExtraTreesClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200, random_state=RANDOM_STATE
    ),
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]),
}

try:
    from xgboost import XGBClassifier
    CANDIDATE_MODELS["xgboost"] = XGBClassifier(
        n_estimators=200, random_state=RANDOM_STATE,
        eval_metric="logloss", verbosity=0, n_jobs=-1
    )
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier
    CANDIDATE_MODELS["lightgbm"] = LGBMClassifier(
        n_estimators=200, random_state=RANDOM_STATE,
        verbose=-1, n_jobs=-1
    )
except ImportError:
    pass


def _evaluate(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob[:, 1])), 4)
        except Exception:
            pass
    return metrics


def _get_feature_importance(model, feature_names: list[str]) -> dict:
    try:
        est = model
        if hasattr(model, "named_steps"):
            est = model.named_steps.get("clf", model)
        if hasattr(est, "feature_importances_"):
            imp = dict(zip(feature_names, est.feature_importances_.tolist()))
            return dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        pass
    return {}


def automl_node(state: AgentState) -> AgentState:
    X_train_raw = state.get("X_train", [])
    X_test_raw = state.get("X_test", [])
    y_train_raw = state.get("y_train", [])
    y_test_raw = state.get("y_test", [])

    if not X_train_raw or not y_train_raw:
        return {**state, "error": "AutoML için eğitim verisi bulunamadı.",
                "current_step": "automl_node"}

    X_train = pd.DataFrame(X_train_raw)
    X_test = pd.DataFrame(X_test_raw)
    X_train.columns = [str(c) for c in X_train.columns]
    X_test.columns = [str(c) for c in X_test.columns]
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

    y_train = np.array(y_train_raw, dtype=int)
    y_test = np.array(y_test_raw, dtype=int)
    feature_names = X_train.columns.tolist()

    print(f"\n[automl] Başlıyor: {X_train.shape} | {len(CANDIDATE_MODELS)} model denenecek")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, model in CANDIDATE_MODELS.items():
        start = time.time()
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv,
                                     scoring="roc_auc", n_jobs=-1)
            elapsed = time.time() - start
            mean_score = scores.mean()
            results[name] = {"score": mean_score, "std": scores.std(), "time": elapsed}
            print(f"[automl] {name:25s} ROC-AUC: {mean_score:.4f} ± {scores.std():.4f} ({elapsed:.1f}sn)")
        except Exception as e:
            print(f"[automl] {name:25s} HATA: {e}")

    if not results:
        return {**state, "error": "Hiçbir model çalışmadı.", "current_step": "automl_node"}

    best_name = max(results, key=lambda k: results[k]["score"])
    best_model = CANDIDATE_MODELS[best_name]
    print(f"\n[automl] En iyi model: {best_name} (ROC-AUC: {results[best_name]['score']:.4f})")

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_prob = None
    try:
        y_prob = best_model.predict_proba(X_test)
    except Exception:
        pass

    metrics = _evaluate(y_test, y_pred, y_prob)
    metrics["cv_roc_auc"] = round(float(results[best_name]["score"]), 4)
    metrics["cv_roc_auc_std"] = round(float(results[best_name]["std"]), 4)
    metrics["all_model_scores"] = {k: round(v["score"], 4) for k, v in results.items()}
    metrics["feature_importance"] = _get_feature_importance(best_model, feature_names)

    print(f"\n[automl] Test seti sonuçları:")
    for k, v in metrics.items():
        if k not in ("feature_importance", "all_model_scores"):
            print(f"  {k}: {v}")

    if metrics["feature_importance"]:
        print(f"\n[automl] Top 5 feature:")
        for feat, imp in list(metrics["feature_importance"].items())[:5]:
            print(f"  {feat}: {imp:.4f}")

    return {
        **state,
        "model_name": best_name,
        "model_params": {},
        "model_artifact": best_model,
        "metrics": metrics,
        "current_step": "automl_node",
    }