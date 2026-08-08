"""Katman 1 serialize katmanı (task #6) — Artifact -> disk round-trip + fail-closed kapılar.

Ölçüt (kullanıcı): round-trip parity TEK BAŞINA YETMEZ — bir alan kaybolsa da çıktı aynı
çıkabilir (o yol tetiklenmiyorsa, T5a dersi). Bu yüzden İKİ karşılaştırma: (1) whitelist'teki
HER alan tek tek, (2) transform_pipeline çıktısı hücre-hücre.
"""
import glob
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.services.inference.orchestrator import fit_pipeline, transform_pipeline
from src.services.inference.serialize import (
    ArtifactSchemaMismatch, _to_jsonable, _SERIALIZED_FIELDS, _ARTIFACT_JSON,
    load_artifact, save_artifact,
)
from src.services.inference.pickle_io import PickleRef, SklearnVersionMismatch


def _art(fx=None, y=None):
    if fx is None:
        rs = np.random.RandomState(0)
        n = 90
        fx = pd.DataFrame({
            "a": rs.rand(n), "b": rs.rand(n) * 100,
            "cat": (["x", "y", "z", "w"] * n)[:n],
            "hi": [f"c{i % 25}" for i in range(n)],   # >10 kardinalite -> TargetEncoder (pickle)
        })
        y = pd.Series(np.arange(n) % 2)
    _, art = fit_pipeline(fx, y, {"intent": "classification", "encode_exclude": [],
                                  "outlier_cols": ["b"], "ratio_pairs": []})
    return fx, y, art


def _find_csv(pat):
    m = glob.glob(os.path.join("data", "csvs", pat)) or glob.glob(os.path.join("data", pat))
    return m[0] if m else None


def _norm(o):
    """PickleRef -> (sklearn_version, kind); numpy -> yerel (karşılaştırma için)."""
    if isinstance(o, PickleRef):
        return {"__ref__": o.sklearn_version, "kind": o.kind}
    if isinstance(o, dict):
        return {k: _norm(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_norm(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


# ── (1) TİPLİ SANITIZE — fail-closed ────────────────────────────

def test_sanitize_numpy_types_explicit():
    """np.int64/float64/bool_/ndarray AÇIK çevrilir (default=str fallback YOK)."""
    out = _to_jsonable({"i": np.int64(7), "f": np.float64(1.5), "b": np.bool_(True),
                        "arr": np.array([1, 2, 3]), "nested": [np.int64(9)]})
    assert out == {"i": 7, "f": 1.5, "b": True, "arr": [1, 2, 3], "nested": [9]}
    assert type(out["i"]) is int and type(out["f"]) is float and type(out["b"]) is bool


def test_sanitize_rejects_unknown_type():
    """Tanınmayan tip -> TypeError (sessiz string'e ÇEVİRMEZ)."""
    class Yabanci:
        pass
    with pytest.raises(TypeError):
        _to_jsonable({"x": Yabanci()})


def test_json_rejects_nan_inf():
    """allow_nan=False: NaN/Inf taşıyan param serialize'da PATLAR (sessiz round-trip değil)."""
    with tempfile.TemporaryDirectory() as d:
        _, _, art = _art()
        art.param_store["_probe"] = {"bad": float("nan")}
        with pytest.raises(ValueError):
            save_artifact(art, base_dir=d)


# ── (2) ROUND-TRIP — alan-bazlı VE çıktı ────────────────────────

def test_roundtrip_all_fields_and_output():
    """İKİ karşılaştırma: whitelist'teki HER alan + transform_pipeline çıktısı."""
    with tempfile.TemporaryDirectory() as d:
        fx, y, art = _art()
        X_mem = transform_pipeline(fx.copy(), art)
        aid = save_artifact(art, base_dir=d)
        art2 = load_artifact(aid, base_dir=d)

        # (a) ALAN-BAZLI: her whitelist alanı tek tek (tetiklenmeyen yol da görünür olsun)
        for f in _SERIALIZED_FIELDS:
            v1, v2 = getattr(art, f), getattr(art2, f)
            if f == "param_store":
                assert set(v1) == set(v2), "param_store anahtarları farklı"
                for k in v1:
                    a = json.dumps(_norm(v1[k]), sort_keys=True)
                    b = json.dumps(_norm(v2[k]), sort_keys=True)
                    assert a == b, f"param_store['{k}'] round-trip'te değişti"
            else:
                assert v1 == v2, f"alan '{f}' round-trip'te değişti: {v1!r} != {v2!r}"

        # (b) ÇIKTI: disk turundan geçen artifact aynı çıktıyı üretmeli (hücre-hücre)
        X_disk = transform_pipeline(fx.copy(), art2)
        assert list(X_disk.columns) == list(X_mem.columns)
        for c in X_mem.columns:
            assert np.allclose(X_disk[c].to_numpy(dtype=float), X_mem[c].to_numpy(dtype=float),
                               equal_nan=True), f"'{c}' disk turunda değişti"


def test_pickle_blobs_written_separately():
    """3 sklearn estimator ayrı .pkl'e; param_store'da sentinel kalır (JSON'da estimator YOK)."""
    with tempfile.TemporaryDirectory() as d:
        _, _, art = _art()
        aid = save_artifact(art, base_dir=d)
        adir = os.path.join(d, aid)
        pkls = sorted(f for f in os.listdir(adir) if f.endswith(".pkl"))
        assert "target_encode.pkl" in pkls, f"target encoder blob'u yok: {pkls}"
        meta = json.load(open(os.path.join(adir, _ARTIFACT_JSON), encoding="utf-8"))
        te = meta["param_store"]["target_encode"]
        assert "__pickle_ref__" in te["ref"], "estimator JSON'a sızmış olmamalı (sentinel bekleniyor)"


# ── FAIL-CLOSED KAPILAR ─────────────────────────────────────────

def test_pickle_version_gate_on_disk_path():
    """Diskteki sklearn_version bozulursa yükleme abort (bellek yoluyla AYNI kapı)."""
    with tempfile.TemporaryDirectory() as d:
        _, _, art = _art()
        aid = save_artifact(art, base_dir=d)
        jpath = os.path.join(d, aid, _ARTIFACT_JSON)
        meta = json.load(open(jpath, encoding="utf-8"))
        meta["param_store"]["target_encode"]["ref"]["sklearn_version"] = "0.1"   # SAHTE minor
        json.dump(meta, open(jpath, "w", encoding="utf-8"), allow_nan=False)
        with pytest.raises(SklearnVersionMismatch):
            load_artifact(aid, base_dir=d)


def test_schema_version_gate():
    """Uyumsuz artifact_schema_version -> ArtifactSchemaMismatch (fail-closed)."""
    with tempfile.TemporaryDirectory() as d:
        _, _, art = _art()
        aid = save_artifact(art, base_dir=d)
        jpath = os.path.join(d, aid, _ARTIFACT_JSON)
        meta = json.load(open(jpath, encoding="utf-8"))
        meta["artifact_schema_version"] = "99.0"
        json.dump(meta, open(jpath, "w", encoding="utf-8"), allow_nan=False)
        with pytest.raises(ArtifactSchemaMismatch):
            load_artifact(aid, base_dir=d)


# ── WHITELIST GARANTİSİ ─────────────────────────────────────────

def test_whitelist_ignores_injected_field():
    """Artifact'e sızan non-serializable bir alan serialize'ı KIRMAZ ve diske GİTMEZ."""
    with tempfile.TemporaryDirectory() as d:
        _, _, art = _art()
        art.__dict__["kacak_observer"] = lambda x: x       # whitelist'te YOK
        aid = save_artifact(art, base_dir=d)                # patlamamalı
        meta = json.load(open(os.path.join(d, aid, _ARTIFACT_JSON), encoding="utf-8"))
        assert "kacak_observer" not in meta, "whitelist dışı alan diske sızdı"
        assert set(meta) <= set(_SERIALIZED_FIELDS)


# ── GERÇEK VERİ round-trip (3 ayrışan dataset) ──────────────────

@pytest.mark.parametrize("pat,tgt,intent", [
    ("*credit_risk_dataset.csv", "loan_status", "classification"),   # pickle: target/scaler
    ("housing.csv", "median_house_value", "regression"),             # sayısal, null test-tarafı
    ("*superstore*.csv", "Profit", "regression"),                    # drop listesi dolu (Row ID)
])
def test_roundtrip_real_dataset(pat, tgt, intent):
    csv = _find_csv(pat)
    if not csv:
        pytest.skip(f"CSV yok: {pat}")
    df = pd.read_csv(csv, nrows=1500)
    num = df.select_dtypes(include="number").columns.tolist()
    if tgt not in df.columns or len(num) < 2:
        pytest.skip("uygun değil")
    y = _det_y(df, tgt, intent)
    cut = int(len(df) * 0.7)
    Xtr, Xte = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
    ctx = {"intent": intent, "encode_exclude": [], "outlier_cols": num[:1], "ratio_pairs": []}
    with tempfile.TemporaryDirectory() as d:
        _, art = fit_pipeline(Xtr.drop(columns=[tgt]), y.iloc[:cut].reset_index(drop=True), ctx)
        X_mem = transform_pipeline(Xte.drop(columns=[tgt]).copy(), art)
        aid = save_artifact(art, base_dir=d)
        art2 = load_artifact(aid, base_dir=d)
        for f in _SERIALIZED_FIELDS:
            if f != "param_store":
                assert getattr(art, f) == getattr(art2, f), f"[{pat}] alan '{f}' değişti"
        X_disk = transform_pipeline(Xte.drop(columns=[tgt]).copy(), art2)
        assert list(X_disk.columns) == list(X_mem.columns), f"[{pat}] kolon seti"
        for c in X_mem.columns:
            assert np.allclose(X_disk[c].to_numpy(dtype=float), X_mem[c].to_numpy(dtype=float),
                               equal_nan=True), f"[{pat}] '{c}' disk turunda değişti"


def _det_y(df, tgt, intent):
    if intent == "classification":
        v = df[tgt].astype("category").cat.codes
        return (v % 2).astype(int)
    return pd.Series(np.arange(len(df)) % 100, dtype=float)
