"""Katman 1 skorlama endpoint'i — ham satır(lar) → skor. Katman 1'in SON parçası.

MİMARİ: eğitimin TERSİ. Eğitim job kuyruğuna yazar (worker graph koşar); skorlama SENKRON,
API process'inde. Endpoint `def` (async değil) → FastAPI CPU-bound işi (joblib.load + transform +
predict) otomatik THREADPOOL'a atar, event loop bloklanmaz. DB sync Session (get_db) → threadpool'da,
async karışımı YOK.

DOĞRULAMA ZİNCİRİ (her katman fail-closed, actionable hata):
  1. champion lookup (artifacts tablosu)         → 409 (serve edilebilir artifact yok)
  2. load_for_serving (üç kapı, cache üzerinden)  → 409 (phase/deployable/model-dosyası)
  3. check_raw_input (domain-farkında imza)       → 422 (eksik kolon / binary domain; +warnings)
  4. transform_pipeline (delta-guard/üretim)      → 500 (artifact bütünlük hatası)
  5. model.predict + eşik                          → 200

BATCH HEP-YA-HİÇ: transform_pipeline frame-düzeyi vektörize (X=raw.copy() üstünde adım adım) →
satır izolasyonu YOK; bir satır kontratı bozarsa TÜM batch düşer (422/500). Satır-başı izolasyon ertelendi.

LOG FAIL-OPEN: prediction_id yazımdan BAĞIMSIZ üretilir → log patlasa da response'ta döner
(monitoring ≠ serving doğruluğu).
"""
from __future__ import annotations

import hashlib
import time
import uuid
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.core.deps import get_current_user
from api.core.serving_cache import get_serving_cache
from api.models.artifact import ArtifactModel
from api.models.prediction import PredictionModel
from api.models.project import ProjectModel
from api.models.user import UserModel
from src.core.config import settings
from src.core.logging_config import get_logger
from src.services.inference.orchestrator import (
    MissingFeatureProduction, RawInputSignatureViolation, StepDeltaDivergence,
    check_raw_input, transform_pipeline,
)
from src.services.inference.references import ModelFileMissing
from src.services.inference.serialize import ArtifactNotServiceable

logger = get_logger(__name__)

router = APIRouter(prefix="/projects", tags=["Scoring"])


class PredictRequest(BaseModel):
    rows: list[dict]                       # her satır: {ham_kolon: değer} (raw_input_signature.required)
    # KAYNAK (Katman 2): "ui_manual" = dashboard Scoring playground'dan MANUEL deneme → aggregator
    # monitoring'ten DIŞLAR (drift/perf gerçek trafiği yansıtsın). Varsayılan "api" = üretim çağrısı
    # (omit eden harici/programatik istemci üretim sayılır — güvenli taraf: dışlanmaz, körleşmez).
    source: Literal["api", "ui_manual"] = "api"


def _jsonable(v):
    """numpy skaler → yerel Python (JSON/response için)."""
    if isinstance(v, np.generic):
        return v.item()
    return v


def _sample_input_row(prediction_id: str, rate: float) -> bool:
    """Katman 2 alan-bazlı örnekleme kararı — bu satırın ham ``input_row``'u SAKLANSIN mı?

    Deterministik hash-Bernoulli, ``prediction_id`` (uuid4) anahtarlı:
      * DETERMİNİSTİK — id'nin saf fonksiyonu; aynı id hep aynı karar, aggregator offline doğrular.
      * SİSTEMATİK-SAPMASIZ — uuid4 içerikten/sıradan/zamandan bağımsız uniform → her satır eşit
        olasılık ``rate``; feature değeri / batch pozisyonu / istemci sıralaması tercih edilmez.
        ("ilk N" istemci sıralamasına, "her k'ıncı" periyoda yanlıydı; hash ikisinden de bağımsız.)
      * STATELESS — satır-başı, sayaç/kilit yok → eşzamanlı istek + çoklu worker'da yarış yok.

    ``rate >= 1.0`` → hep True (tam sakla, varsayılan); ``rate <= 0.0`` → hep False (hiç saklama).
    """
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    h = hashlib.blake2b(prediction_id.encode("utf-8"), digest_size=8).hexdigest()
    u = int(h, 16) / 2 ** 64           # u ∈ [0, 1)
    return u < rate


@router.post("/{project_id}/predict")
def predict(
    project_id: int,
    body: PredictRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    # ── Sahiplik + girdi sınırı ──
    project = db.query(ProjectModel).filter(
        ProjectModel.id == project_id, ProjectModel.user_id == current_user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    rows = body.rows or []
    if not rows:
        raise HTTPException(status_code=400, detail="rows boş — en az bir satır gerekli")
    if len(rows) > settings.SERVING_MAX_ROWS:
        raise HTTPException(status_code=413,
                            detail=f"batch çok büyük: {len(rows)} > SERVING_MAX_ROWS={settings.SERVING_MAX_ROWS}")

    # ── Katman 1: champion lookup ──
    champ = db.query(ArtifactModel).filter(
        ArtifactModel.project_id == project_id,
        ArtifactModel.is_champion.is_(True),
        ArtifactModel.deployable.is_(True),
        ArtifactModel.phase == "complete",
    ).first()
    if champ is None:
        raise HTTPException(status_code=409,
                            detail="Bu proje için serve edilebilir artifact yok "
                                   "(önce eğit / son koşu complete+deployable değil).")

    # ── Katman 2: load_for_serving (cache; üç kapı) ──
    try:
        artifact, model = get_serving_cache().get(champ.id)
    except (ArtifactNotServiceable, ModelFileMissing) as e:
        raise HTTPException(status_code=409, detail=f"Artifact serve edilemez: {e}")

    raw = pd.DataFrame(rows)

    # ── Katman 3: raw_input_signature (domain-farkında; warnings YAKALA) ──
    try:
        sig = check_raw_input(raw, artifact.raw_input_signature,
                              artifact.tolerated_extra_columns)
    except RawInputSignatureViolation as e:
        raise HTTPException(status_code=422, detail={
            "error": "raw_input_signature_violation",
            "missing_required": e.missing, "undeclared_extra": e.extra,
            "domain_violations": e.domain,
        })
    warnings = {"unseen_categories": sig.get("unseen_categories", {}),
                "out_of_range": sig.get("out_of_range", {})}
    row_flags = sig.get("row_flags")               # per-row {has_unseen, has_out_of_range} (Katman 2 kalite)

    # ── Katman 4: transform (imza ZATEN kontrol edildi → verify_input=False; delta-guard AÇIK) ──
    try:
        X = transform_pipeline(raw, artifact, verify_input=False, verify_delta=True)
    except (StepDeltaDivergence, MissingFeatureProduction) as e:
        # Artifact bütünlük hatası — geçerli complete artifact'te OLMAMALI → fail-closed 500.
        logger.error(f"[scoring] transform bütünlük hatası (artifact={champ.id}): {e}")
        raise HTTPException(status_code=500, detail=f"Artifact bütünlük hatası: {e}")

    # ── Katman 5: predict + eşik ──
    mref = artifact.model_ref or {}
    threshold = mref.get("optimal_threshold")
    is_classifier = hasattr(model, "predict_proba")
    preds: list = []
    probs: list = []
    if is_classifier:
        proba = model.predict_proba(X)
        classes = list(getattr(model, "classes_", range(proba.shape[1])))
        if proba.shape[1] == 2:
            task = "binary_classification"
            prob_pos = proba[:, 1]
            thr = threshold if threshold is not None else 0.5
            idx = (prob_pos >= thr).astype(int)
            preds = [_jsonable(classes[i]) for i in idx]
            probs = [float(p) for p in prob_pos]
        else:
            task = "multiclass_classification"
            ai = proba.argmax(axis=1)
            preds = [_jsonable(classes[i]) for i in ai]
            probs = [{str(_jsonable(classes[j])): float(proba[r, j]) for j in range(proba.shape[1])}
                     for r in range(proba.shape[0])]
            threshold = None
    else:
        task = "regression"
        yhat = model.predict(X)
        preds = [float(v) for v in yhat]
        probs = [None] * len(preds)
        threshold = None

    # ── prediction_id ÖNCE üretilir (log yazımından BAĞIMSIZ — fail-open gereği) ──
    now = int(time.time())
    pred_ids = [uuid.uuid4().hex for _ in range(len(rows))]
    predictions = [
        {"row_index": i, "prediction": preds[i], "probability": probs[i],
         "prediction_id": pred_ids[i]}
        for i in range(len(rows))
    ]

    # ── Prediction log (FAIL-OPEN: patlasa da skor döner; id zaten üretildi) ──
    # ALAN-BAZLI örnekleme: input_row yalnız örneklem satırlarda saklanır (PSI referansı); diğer
    # alanlar HER satırda tam → gerçekleşen/tahmini performans + veri kalitesi monitörleri körleşmez.
    sample_rate = settings.SERVING_INPUT_ROW_SAMPLE_RATE
    # İŞ ANAHTARI (Katman 2 gerçekleşen-perf v2 erken-şartı): artifact bir entity kolonu bildirdiyse VE
    # gelen satırda VARSA yaz. Genelde entity kolonu model girdisi değil → çoğu satırda NULL (v1 beklenen).
    bk_col = getattr(artifact, "business_key_column", None)
    try:
        for i in range(len(rows)):
            sampled = _sample_input_row(pred_ids[i], sample_rate)
            flags = ({"has_unseen": bool(row_flags["has_unseen"][i]),
                      "has_out_of_range": bool(row_flags["has_out_of_range"][i])}
                     if row_flags else None)          # None → check_raw_input bayrak hesabı patladı (fail-safe)
            bk_val = rows[i].get(bk_col) if bk_col else None
            db.add(PredictionModel(
                prediction_id=pred_ids[i], project_id=project_id, artifact_id=champ.id,
                created_at=now, source=body.source,
                input_row=(rows[i] if sampled else None),
                input_row_sampled=sampled, prediction=preds[i],
                probability=probs[i], threshold=threshold, input_warnings=warnings,
                input_warning_flags=flags,
                business_key=(str(bk_val) if bk_val is not None else None),
            ))
        db.commit()
    except Exception as e:
        db.rollback()
        logger.warning(f"[scoring] prediction log YAZILAMADI (fail-open, skor döndü): {e}")

    return {
        "artifact_id": champ.id, "model_name": mref.get("model_name"), "task": task,
        "threshold": threshold, "threshold_provenance": mref.get("threshold_provenance"),
        "predictions": predictions, "warnings": warnings,
    }
