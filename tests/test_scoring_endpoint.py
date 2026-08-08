"""Katman 1 Parça 2 — skorlama endpoint'i uçtan uca.

Fixture: churn (binary + binary kolonlu → domain testi mümkün) orchestrator modda eğitilip GERÇEK
complete artifact + model üretir, champion yapılır. Testler predict()'i DOĞRUDAN çağırır (JWT/HTTP
katmanı gerekmez; current_user stub yeterli — fonksiyon yalnız .id kullanır).

Kapsam: happy-path (transform_pipeline ile eşleşme) · 5 katman negatif (409×2, 422×2, 500) ·
fail-open (log patlar, skor+prediction_id yine döner) · input_row boyut ÖLÇÜMÜ (10k batch).
"""
import json
import os
import shutil
import types

os.environ.setdefault("API_DATABASE_URL", "sqlite:///./data/_sbx_scoring_test.sqlite")

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from api.core.database import Base, SessionLocal, engine
# mapper seti
from api.models.user import UserModel  # noqa: F401
from api.models.csv import CSVFileModel  # noqa: F401
from api.models.job import BackgroundJobModel  # noqa: F401
from api.models.data_group import DataGroupModel  # noqa: F401
from api.models.mcp import MCPConfigModel  # noqa: F401
from api.models.project import ProjectModel
from api.models.artifact import ArtifactModel
from api.models.prediction import PredictionModel

from src.core.config import settings
from src.services.inference.serialize import load_artifact
from src.services.inference.orchestrator import transform_pipeline
import api.core.serving_cache as sc
from api.core.worker import _upsert_artifact_row
from api.routes.artifacts import promote_artifact
from api.routes.scoring import predict, PredictRequest

CHURN = "data/csvs/7723fd96-87eb-4548-89e7-87899f05badf_Bank Customer Churn Prediction.csv"
_USER = types.SimpleNamespace(id=4242)


def _run_pipeline(csv, target, intent, nrows):
    import sys
    sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
    from scripts.offline_report import build_state
    from src.services.feature_builder import build_features
    from src.services.ml_trainer import train_model
    from src.services.evaluator import evaluate_model
    from src.services.data_quality import run_data_quality
    state = build_state(csv, target, intent, full=False, models=None, nrows=nrows)
    for fn in (build_features, run_data_quality, train_model, evaluate_model):
        out = fn(state) or {}
        assert not out.get("error"), out.get("error")
        state.update(out)
    return state


@pytest.fixture(scope="module")
def champion():
    Base.metadata.create_all(bind=engine)
    old_mode = settings.FEATURE_PIPELINE_MODE
    settings.FEATURE_PIPELINE_MODE = "orchestrator"      # runtime override (env'den bağımsız)
    state = _run_pipeline(CHURN, "churn", "classification", nrows=3000)
    aid = state["inference_artifact_id"]
    assert aid, "Faz A artifact yazılmadı"

    db = SessionLocal()
    try:
        u = UserModel(email="scoring-test@local.test", hashed_password="x")
        db.add(u); db.commit(); db.refresh(u)
        _USER.id = u.id                                  # FK için gerçek user (modül geneli)
        p = ProjectModel(title="scoring-test", status="completed", user_id=_USER.id)
        db.add(p); db.commit(); db.refresh(p)
        pid = p.id
    finally:
        db.close()
    _upsert_artifact_row(pid, state)                     # serving index'e kaydet (champion'a DOKUNMAZ)
    # Manuel promotion (otomatik promotion kaldırıldı) — scoring için champion elle yapılır.
    db = SessionLocal()
    try:
        promote_artifact(pid, aid, db=db, current_user=_USER)
    finally:
        db.close()

    req = load_artifact(aid).raw_input_signature.get("required") or []
    yield {"pid": pid, "aid": aid, "required": req}

    # teardown
    settings.FEATURE_PIPELINE_MODE = old_mode
    sc.get_serving_cache().clear()
    db = SessionLocal()
    try:
        db.query(PredictionModel).filter(PredictionModel.project_id == pid).delete()
        db.query(ArtifactModel).filter(ArtifactModel.project_id == pid).delete()
        db.query(ProjectModel).filter(ProjectModel.id == pid).delete()
        db.query(UserModel).filter(UserModel.id == _USER.id).delete()
        db.commit()
    finally:
        db.close()
    shutil.rmtree(os.path.join("data", "artifacts", aid), ignore_errors=True)


def _raw_rows(csv, required, n):
    df = pd.read_csv(csv, nrows=5000)
    return df[[c for c in required if c in df.columns]].head(n).to_dict(orient="records")


# ─────────────────────────── HAPPY PATH ───────────────────────────

def test_happy_path_matches_transform_pipeline(champion):
    rows = _raw_rows(CHURN, champion["required"], 5)
    db = SessionLocal()
    try:
        resp = predict(champion["pid"], PredictRequest(rows=rows), db=db, current_user=_USER)
    finally:
        db.close()
    assert resp["task"] == "binary_classification"
    assert resp["artifact_id"] == champion["aid"]
    assert resp["threshold"] is not None
    assert len(resp["predictions"]) == 5
    assert all(p["prediction_id"] for p in resp["predictions"])

    # transform_pipeline + model ile BİREBİR eşleşme
    art = load_artifact(champion["aid"])
    import joblib
    model = joblib.load(art.model_ref["model_path"])
    X = transform_pipeline(pd.DataFrame(rows), art, verify_input=False, verify_delta=False)
    exp = model.predict_proba(X)[:, 1]
    got = [p["probability"] for p in resp["predictions"]]
    assert np.allclose(got, exp, atol=1e-9)


def test_predictions_are_logged(champion):
    rows = _raw_rows(CHURN, champion["required"], 3)
    db = SessionLocal()
    try:
        resp = predict(champion["pid"], PredictRequest(rows=rows), db=db, current_user=_USER)
        ids = [p["prediction_id"] for p in resp["predictions"]]
        n = db.query(PredictionModel).filter(PredictionModel.prediction_id.in_(ids)).count()
    finally:
        db.close()
    assert n == 3


# ─────────────────────────── FAZ C KALİBRASYON KAPISI (Katman 2 Parça 1) ───────────────────────────

def test_faz_c_calibration_gate_in_baseline_metrics(champion):
    """baseline_metrics kalibrasyon kapısını taşır (churn binary → DOLU); yalnız kapı alanları, eğri yok."""
    art = load_artifact(champion["aid"])
    bm = (art.training_reference or {}).get("baseline_metrics") or {}
    assert "calibration" in bm, "Faz C kalibrasyon kapısı baseline_metrics'e eklenmedi"
    cal = bm["calibration"]
    assert cal is not None, "churn binary → calibration DOLU olmalı (reg/multiclass'ta None)"
    assert set(cal.keys()) == {"brier_score", "brier_baseline", "is_well_calibrated"}
    assert "prob_true" not in cal and "prob_pred" not in cal      # eğri KOPYALANMADI (CBPE'ye gereksiz)
    assert "roc_auc" in bm or "pr_auc" in bm                       # test_metrics ASIL metrikleri duruyor (shallow copy)


# ─────────────────────────── NEGATİF: 5 KATMAN ───────────────────────────

def test_409_no_champion(champion):
    # champion'ı olmayan yeni proje
    db = SessionLocal()
    try:
        p = ProjectModel(title="no-champ", status="completed", user_id=_USER.id)
        db.add(p); db.commit(); db.refresh(p)
        pid = p.id
        with pytest.raises(HTTPException) as ei:
            predict(pid, PredictRequest(rows=[{"a": 1}]), db=db, current_user=_USER)
        assert ei.value.status_code == 409
        db.query(ProjectModel).filter(ProjectModel.id == pid).delete(); db.commit()
    finally:
        db.close()


def test_409_model_file_missing(champion, tmp_path):
    # disk'te complete artifact ama model_path yok → load_for_serving ModelFileMissing → 409
    import uuid
    aid = uuid.uuid4().hex
    d = os.path.join("data", "artifacts", aid); os.makedirs(d, exist_ok=True)
    meta = {"phase": "complete", "supported_scope": {"deployable": True},
            "model_ref": {"model_path": str(tmp_path / "gone.pkl"), "model_name": "x"},
            "raw_input_signature": {"required": [], "columns": {}}, "final_features": [],
            "param_store": {}, "manifest": [], "artifact_schema_version": "1.2"}
    json.dump(meta, open(os.path.join(d, "artifact.json"), "w"))
    db = SessionLocal()
    try:
        p = ProjectModel(title="missing-model", status="completed", user_id=_USER.id)
        db.add(p); db.commit(); db.refresh(p); pid = p.id
        db.add(ArtifactModel(id=aid, project_id=pid, phase="complete", deployable=True,
                             model_path=str(tmp_path / "gone.pkl"), is_champion=True))
        db.commit()
        with pytest.raises(HTTPException) as ei:
            predict(pid, PredictRequest(rows=[{"a": 1}]), db=db, current_user=_USER)
        assert ei.value.status_code == 409
        db.query(ArtifactModel).filter(ArtifactModel.project_id == pid).delete()
        db.query(ProjectModel).filter(ProjectModel.id == pid).delete(); db.commit()
    finally:
        db.close()
        shutil.rmtree(d, ignore_errors=True)
        sc.get_serving_cache().clear()


def test_422_missing_required_column(champion):
    rows = _raw_rows(CHURN, champion["required"], 3)
    drop_col = champion["required"][0]
    for r in rows:
        r.pop(drop_col, None)                       # zorunlu kolonu düşür
    db = SessionLocal()
    try:
        with pytest.raises(HTTPException) as ei:
            predict(champion["pid"], PredictRequest(rows=rows), db=db, current_user=_USER)
        assert ei.value.status_code == 422
        assert drop_col in ei.value.detail["missing_required"]
    finally:
        db.close()


def test_422_binary_domain_violation(champion):
    # churn'de active_member/credit_card 0/1 binary → 0.7 gönder → domain ihlali
    art = load_artifact(champion["aid"])
    cols = art.raw_input_signature.get("columns", {})
    bin_col = next((c for c, k in cols.items() if k.get("kind") == "binary"), None)
    assert bin_col, "binary kolon bulunamadı (test churn'e bağlı)"
    rows = _raw_rows(CHURN, champion["required"], 3)
    rows[0][bin_col] = 0.7
    db = SessionLocal()
    try:
        with pytest.raises(HTTPException) as ei:
            predict(champion["pid"], PredictRequest(rows=rows), db=db, current_user=_USER)
        assert ei.value.status_code == 422
        assert any(d["column"] == bin_col for d in ei.value.detail["domain_violations"])
    finally:
        db.close()


def test_500_transform_integrity(champion, monkeypatch):
    from src.services.inference.orchestrator import MissingFeatureProduction
    def _boom(*a, **k):
        raise MissingFeatureProduction(never_produced=["x"], produced_then_dropped=[], policy_filled=[])
    monkeypatch.setattr("api.routes.scoring.transform_pipeline", _boom)
    rows = _raw_rows(CHURN, champion["required"], 2)
    db = SessionLocal()
    try:
        with pytest.raises(HTTPException) as ei:
            predict(champion["pid"], PredictRequest(rows=rows), db=db, current_user=_USER)
        assert ei.value.status_code == 500
    finally:
        db.close()


# ─────────────────────────── FAIL-OPEN ───────────────────────────

def test_fail_open_log_failure_still_returns(champion, monkeypatch):
    # PredictionModel construction patlat → log yazımı çöker ama skor + prediction_id döner
    def _boom(*a, **k):
        raise RuntimeError("SIMULE log hatası")
    monkeypatch.setattr("api.routes.scoring.PredictionModel", _boom)
    rows = _raw_rows(CHURN, champion["required"], 3)
    db = SessionLocal()
    try:
        resp = predict(champion["pid"], PredictRequest(rows=rows), db=db, current_user=_USER)
    finally:
        db.close()
    assert len(resp["predictions"]) == 3
    assert all(p["prediction_id"] for p in resp["predictions"])   # id yazımdan bağımsız üretildi


# ─────────────────────────── ALAN-BAZLI ÖRNEKLEME (Katman 2 Parça 1) ───────────────────────────

def test_sample_input_row_deterministic_unbiased():
    """Hash-Bernoulli: deterministik + sistematik-sapmasız + kenar oranlar. Fixture GEREKMEZ (saf fonksiyon)."""
    import uuid as _uuid
    from api.routes.scoring import _sample_input_row
    pid = _uuid.uuid4().hex
    assert _sample_input_row(pid, 0.3) == _sample_input_row(pid, 0.3)     # determinizm
    assert _sample_input_row(pid, 1.0) is True                            # tam sakla
    assert _sample_input_row(pid, 0.0) is False                           # hiç saklama
    # 20k uuid → oran ≈ 0.3 (uuid4 uniform → sapmasız). ±0.02 tolerans.
    ids = [_uuid.uuid4().hex for _ in range(20000)]
    frac = sum(_sample_input_row(i, 0.3) for i in ids) / len(ids)
    assert abs(frac - 0.3) < 0.02, f"örnekleme oranı sapması: {frac}"


def test_field_level_sampling_p03(champion):
    """p=0.3: input_row seyrekleşir AMA her satır kaydedilir + NULL satırlar diğer alanları TAM taşır."""
    old = settings.SERVING_INPUT_ROW_SAMPLE_RATE
    settings.SERVING_INPUT_ROW_SAMPLE_RATE = 0.3
    try:
        rows = _raw_rows(CHURN, champion["required"], 400)
        db = SessionLocal()
        try:
            resp = predict(champion["pid"], PredictRequest(rows=rows), db=db, current_user=_USER)
            ids = [p["prediction_id"] for p in resp["predictions"]]
            recs = db.query(PredictionModel).filter(PredictionModel.prediction_id.in_(ids)).all()
        finally:
            db.close()
    finally:
        settings.SERVING_INPUT_ROW_SAMPLE_RATE = old

    assert len(recs) == 400                                # HER satır kaydedildi (alan-bazlı ≠ satır-bazlı)
    sampled = [r for r in recs if r.input_row_sampled]
    not_sampled = [r for r in recs if not r.input_row_sampled]
    assert all(r.input_row is not None for r in sampled)       # sampled ⟺ input_row dolu
    assert all(r.input_row is None for r in not_sampled)       # değilse NULL
    # KÖRLEŞME YOK: input_row=NULL satırlar diğer monitör alanlarını TAM taşır
    for r in not_sampled:
        assert r.prediction is not None
        assert r.probability is not None                       # churn binary → float
        assert r.input_warnings is not None                    # unseen/out-of-range sayaçları (dict)
    frac = len(sampled) / len(recs)
    assert 0.2 < frac < 0.4, f"p=0.3 örnekleme sapması (n=400): {frac}"


# ─────────────────────────── input_row BOYUT ÖLÇÜMÜ (p=1.0 vs p=0.3) ───────────────────────────

def _measure_input_row(champion, rows, rate):
    old = settings.SERVING_INPUT_ROW_SAMPLE_RATE
    settings.SERVING_INPUT_ROW_SAMPLE_RATE = rate
    db = SessionLocal()
    try:
        resp = predict(champion["pid"], PredictRequest(rows=rows), db=db, current_user=_USER)
        ids = [p["prediction_id"] for p in resp["predictions"]]
        recs = db.query(PredictionModel).filter(PredictionModel.prediction_id.in_(ids)).all()
    finally:
        db.close()
        settings.SERVING_INPUT_ROW_SAMPLE_RATE = old
    stored = [r for r in recs if r.input_row is not None]
    total = sum(len(json.dumps(r.input_row)) for r in stored)
    return len(recs), len(stored), total


def test_input_row_size_measurement(champion):
    df = pd.read_csv(CHURN, nrows=10000)
    rows = df[[c for c in champion["required"] if c in df.columns]].to_dict(orient="records")
    n = len(rows)

    n1, s1, t1 = _measure_input_row(champion, rows, 1.0)          # bugünkü varsayılan
    n3, s3, t3 = _measure_input_row(champion, rows, 0.3)          # gelecekteki kararın zemini

    per_row = t1 / max(1, s1)
    print(f"\n[ÖLÇÜM] input_row ({n} satır batch):")
    print(f"  p=1.0: {s1}/{n1} kayıtta input_row dolu, JSON={t1/1024:.1f} KB "
          f"→ 10k ≈ {t1/1024/1024:.2f} MB (satır başı {per_row:.0f} B)")
    print(f"  p=0.3: {s3}/{n3} kayıtta input_row dolu, JSON={t3/1024:.1f} KB "
          f"→ 10k ≈ {t3/1024/1024:.2f} MB (~%{100*s3/max(1,n3):.0f} depolama)")

    assert n1 == n and n3 == n                    # HER İKİ oranda da tüm satırlar kaydedildi (alan-bazlı)
    assert s1 == n                                # p=1.0 → hepsi dolu
    assert 0.25 * n < s3 < 0.35 * n               # p=0.3 → ~%30 dolu


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-s"]))
