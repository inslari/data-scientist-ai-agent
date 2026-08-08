"""Katman 2 — GERÇEKLEŞEN performans için ``actual_label`` backfill endpoint'i.

MİMARİ: skorlama gibi SENKRON, API process'inde (graph çalıştırmaz — yalnız ``predictions.actual_label``
yazımı; job kuyruğu gereksiz). İki kanal, İKİSİ de ``prediction_id``'ye çözülür:
  * ``POST /projects/{id}/labels``      — batch JSON {labels:[{prediction_id, actual_label}]}
  * ``POST /projects/{id}/labels/csv``  — CSV upload + ELLE kolon bildirimi (key_column/label_column)

EŞLEME ANAHTARI = ``prediction_id`` (bugünkü kontrat: scoring response'ta döner, çağıran saklar).
İŞ ANAHTARI (``business_key``, customer_id) v2'ye BLOKLU: bir anahtar → çok tahmin → belirsizlik-çözüm
kuralı ister. ``input_row`` alan-bazlı örnekli (çoğu satırda NULL) → ondan iş anahtarı çıkarmak da kör.

FAIL-TRANSPARENT: eşleşmeyen anahtar SESSİZ kalmaz → yanıt {matched, unmatched, updated, invalid,
unmatched_keys}. İZOLASYON: yalnız {project_id == bu proje} satırları güncellenir → çapraz-proje
etiket enjeksiyonu imkânsız. İDEMPOTANS: aynı etiket tekrar → üzerine yaz + label_backfilled_at yenilenir
(pencere yeniden kirlenir → aggregator yalnız gerçekleşen metrikleri değil TÜM pencereyi baştan hesaplar;
donmuş pencerede PSI/CBPE birebir aynı → israf ama zararsız).

DIRTY-KONTROL BAĞI: ``label_backfilled_at`` FLOAT epoch (aggregator computed_at de float) → geç etiketin
1-saniye sınır yarışı kapalı.
"""
from __future__ import annotations

import io
import time

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.core.deps import get_current_user
from api.models.prediction import PredictionModel
from api.models.project import ProjectModel
from api.models.user import UserModel
from src.core.config import settings
from src.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/projects", tags=["Labels"])

_CHUNK = 500                                    # IN(...) param sınırı (SQLite ~999) → parça parça sorgu


class LabelItem(BaseModel):
    prediction_id: str
    actual_label: object                        # tip-agnostik (sınıf etiketi / reg değeri)


class LabelBatchRequest(BaseModel):
    labels: list[LabelItem]


def _require_project(db: Session, project_id: int, user: UserModel) -> ProjectModel:
    project = db.query(ProjectModel).filter(
        ProjectModel.id == project_id, ProjectModel.user_id == user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _apply_labels(db: Session, project_id: int, pairs: dict) -> dict:
    """{prediction_id: actual_label} → matching satırlara yaz. Döner: rapor sözlüğü.

    YALNIZ project_id == bu proje satırları (izolasyon). label_backfilled_at = float epoch (dirty-kontrol).
    null etiket geçersiz sayılır (skorlama None → SQL NULL ayrımı korunur; JSON 'null' backfill'i engellenir)."""
    # null etiketleri ayıkla (geçersiz — actual_label IS NOT NULL sorgusunu bozar)
    valid = {pid: lbl for pid, lbl in pairs.items() if lbl is not None}
    invalid = len(pairs) - len(valid)
    ids = list(valid.keys())
    now = time.time()
    matched_ids: set = set()

    for i in range(0, len(ids), _CHUNK):
        chunk = ids[i:i + _CHUNK]
        rows = (db.query(PredictionModel)
                  .filter(PredictionModel.project_id == project_id,
                          PredictionModel.prediction_id.in_(chunk))
                  .all())
        for r in rows:
            r.actual_label = valid[r.prediction_id]
            r.label_backfilled_at = now
            matched_ids.add(r.prediction_id)
    db.commit()

    unmatched = [pid for pid in ids if pid not in matched_ids]
    logger.info(f"[labels] proje={project_id} backfill: matched={len(matched_ids)} "
                f"unmatched={len(unmatched)} invalid={invalid}")
    return {
        "matched": len(matched_ids),
        "updated": len(matched_ids),
        "unmatched": len(unmatched),
        "invalid": invalid,
        # eşleşmeyeni GÖSTER (fail-transparent; çağıran yanlış id sakladıysa görsün). İlk 50 ile sınırlı.
        "unmatched_keys": unmatched[:50],
    }


@router.post("/{project_id}/labels")
def submit_labels(
    project_id: int,
    body: LabelBatchRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """Batch etiket backfill (JSON). Eşleme anahtarı = prediction_id."""
    _require_project(db, project_id, current_user)
    items = body.labels or []
    if not items:
        raise HTTPException(status_code=400, detail="labels boş — en az bir etiket gerekli")
    if len(items) > settings.SERVING_MAX_ROWS:
        raise HTTPException(status_code=413,
                            detail=f"batch çok büyük: {len(items)} > SERVING_MAX_ROWS={settings.SERVING_MAX_ROWS}")
    # SON KAZANIR (aynı id birden çok kez → düzeltme): dict doğal olarak son değeri tutar.
    pairs = {it.prediction_id: it.actual_label for it in items}
    return _apply_labels(db, project_id, pairs)


@router.post("/{project_id}/labels/csv")
async def submit_labels_csv(
    project_id: int,
    file: UploadFile = File(...),
    key_column: str = Form(...),       # ELLE bildirim — isimden çıkarım kırılgan (yanlış kolon = sessiz yanlış etiket)
    label_column: str = Form(...),
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """CSV etiket backfill. Kullanıcı ELLE bildirir hangi kolon prediction_id, hangisi actual_label."""
    _require_project(db, project_id, current_user)
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV okunamadı: {e}")
    if key_column not in df.columns or label_column not in df.columns:
        raise HTTPException(status_code=422, detail={
            "error": "column_not_found",
            "key_column": key_column, "label_column": label_column,
            "available_columns": [str(c) for c in df.columns],
        })
    if len(df) > settings.SERVING_MAX_ROWS:
        raise HTTPException(status_code=413,
                            detail=f"CSV çok büyük: {len(df)} > SERVING_MAX_ROWS={settings.SERVING_MAX_ROWS}")
    # SON KAZANIR; NaN key/label atılır (invalid sayılır — _apply_labels null'ı ayıklar, key NaN'i burada).
    pairs: dict = {}
    for pid, lbl in zip(df[key_column], df[label_column]):
        if pd.isna(pid):
            continue
        pairs[str(pid)] = (None if pd.isna(lbl) else _py(lbl))
    if not pairs:
        raise HTTPException(status_code=400, detail="CSV'de geçerli (prediction_id, etiket) çifti yok")
    return _apply_labels(db, project_id, pairs)


def _py(v):
    """numpy/pandas skaler → JSON-serileşebilir yerel Python."""
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, AttributeError):
            return v
    return v
