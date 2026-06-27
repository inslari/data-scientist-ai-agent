from __future__ import annotations

"""
Dataset store
-------------
X_train/X_test/y_train/y_test gibi AĞIR split dizilerini AgentState'te inline
taşımak yerine diske (parquet) yazıp state'te yalnızca DOSYA YOLU tutar.

Neden: LangGraph checkpointer (AsyncPostgresSaver) her node geçişinde TÜM state'i
serialize eder. ~100k satırlık X_train list[dict]'i checkpoint'e yazmak devasa bir
payload yaratır ve Postgres backend'ini düşürür ("server closed the connection
unexpectedly"). model_artifact zaten aynı nedenle diske (joblib) yazılıyordu; bu
modül aynı disiplini X/y için uygular.

load_raw eski inline şemaya GERİ-UYUMLU düşer (path yoksa state'teki listeyi okur),
böylece önceki checkpoint'lerden resume bozulmaz.
"""

import os
import uuid

import numpy as np
import pandas as pd

from src.core.logging_config import get_logger

logger = get_logger(__name__)

_SPLIT_DIR = "data/splits"


def _ensure_dir() -> None:
    os.makedirs(_SPLIT_DIR, exist_ok=True)


def _as_df(X) -> pd.DataFrame:
    df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    df.columns = [str(c) for c in df.columns]
    return df


def _y_to_frame(y) -> pd.DataFrame:
    return pd.DataFrame({"y": np.asarray(y).ravel()})


def dump_split(X_train, X_test, y_train, y_test, *, tag: str = "") -> dict:
    """X/y bölünmesini parquet'e yazar; state'e konacak yol anahtarlarını döndürür.
    Herhangi bir yazma hatasında ilgili yol None döner — çağıran taraf inline
    fallback'e güvenebilir."""
    _ensure_dir()
    pfx = f"{_SPLIT_DIR}/{(tag + '_') if tag else ''}{uuid.uuid4().hex}"
    out: dict = {}
    targets = {
        "X_train_path": (f"{pfx}_xtrain.parquet", lambda: _as_df(X_train)),
        "X_test_path":  (f"{pfx}_xtest.parquet",  lambda: _as_df(X_test)),
        "y_train_path": (f"{pfx}_ytrain.parquet", lambda: _y_to_frame(y_train)),
        "y_test_path":  (f"{pfx}_ytest.parquet",  lambda: _y_to_frame(y_test)),
    }
    for key, (path, build) in targets.items():
        try:
            build().to_parquet(path, index=False)
            out[key] = path
        except Exception as e:
            logger.warning(f"[dataset_store] '{key}' diske yazılamadı, inline'a düşülecek: {e}")
            out[key] = None
    return out


def dump_y(y_train, y_test, *, tag: str = "y") -> dict:
    """Yalnızca y_train/y_test'i (örn. AutoML'de label-encode SONRASI) parquet'e
    yazar; güncel yol anahtarlarını döndürür."""
    _ensure_dir()
    pfx = f"{_SPLIT_DIR}/{tag}_{uuid.uuid4().hex}"
    out: dict = {}
    for key, path, y in (
        ("y_train_path", f"{pfx}_ytrain.parquet", y_train),
        ("y_test_path",  f"{pfx}_ytest.parquet",  y_test),
    ):
        try:
            _y_to_frame(y).to_parquet(path, index=False)
            out[key] = path
        except Exception as e:
            logger.warning(f"[dataset_store] '{key}' diske yazılamadı: {e}")
            out[key] = None
    return out


def _load_records(state: dict, path_key: str, inline_key: str) -> list:
    """X için: parquet yolundan records listesi döndürür; yoksa state'teki inline
    listeye düşer (geri-uyum). Aşağı akış kodu `pd.DataFrame(raw)` ve `if not raw`
    beklediğinden ham list[dict] formatı korunur."""
    path = state.get(path_key)
    if path and os.path.exists(path):
        try:
            return pd.read_parquet(path).to_dict(orient="records")
        except Exception as e:
            logger.warning(f"[dataset_store] '{path}' okunamadı, inline'a düşülüyor: {e}")
    return state.get(inline_key, []) or []


def _load_y(state: dict, path_key: str, inline_key: str) -> list:
    path = state.get(path_key)
    if path and os.path.exists(path):
        try:
            return pd.read_parquet(path)["y"].tolist()
        except Exception as e:
            logger.warning(f"[dataset_store] '{path}' okunamadı, inline'a düşülüyor: {e}")
    return state.get(inline_key, []) or []


def load_raw(state: dict) -> tuple[list, list, list, list]:
    """state'ten (X_train_raw, X_test_raw, y_train_raw, y_test_raw)'ı ham list
    formatında getirir. Disk-yolu şeması varsa parquet'ten okur, yoksa inline'a düşer."""
    return (
        _load_records(state, "X_train_path", "X_train"),
        _load_records(state, "X_test_path", "X_test"),
        _load_y(state, "y_train_path", "y_train"),
        _load_y(state, "y_test_path", "y_test"),
    )
