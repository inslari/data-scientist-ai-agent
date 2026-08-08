"""Katman 2 — actual_label backfill endpoint'i (/projects/{id}/labels + /labels/csv).

Fonksiyonları DOĞRUDAN çağırır (JWT/HTTP gerekmez; current_user stub yalnız .id kullanır) — scoring
test deseni. Kapsam: batch eşleme + eşleşmeyen raporu · sahiplik (404) · çapraz-proje izolasyonu ·
idempotans/düzeltme (üzerine yaz + backfilled_at yenilenir) · null etiket ayıklama · CSV parse + kolon
doğrulama (422) · label_backfilled_at FLOAT.
"""
import asyncio
import io
import os
import time
import types
import uuid

os.environ.setdefault("API_DATABASE_URL", "sqlite:///./data/_sbx_labels_test.sqlite")

import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

from api.core.database import Base, SessionLocal, engine
from api.models.user import UserModel
from api.models.csv import CSVFileModel  # noqa: F401  (mapper seti)
from api.models.data_group import DataGroupModel  # noqa: F401
from api.models.mcp import MCPConfigModel  # noqa: F401
from api.models.job import BackgroundJobModel  # noqa: F401
from api.models.project import ProjectModel
from api.models.prediction import PredictionModel

from api.routes.labels import (
    submit_labels, submit_labels_csv, LabelBatchRequest, LabelItem,
)

_USER = types.SimpleNamespace(id=0)
_OTHER = types.SimpleNamespace(id=0)


def _mk_predictions(db, project_id, artifact_id, ids):
    now = int(time.time())
    db.bulk_insert_mappings(PredictionModel, [
        {"prediction_id": pid, "project_id": project_id, "artifact_id": artifact_id,
         "created_at": now, "prediction": 1, "probability": 0.5} for pid in ids])
    db.commit()


@pytest.fixture(scope="module")
def seeded():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        u = UserModel(email="labels-test@local.test", hashed_password="x")
        o = UserModel(email="labels-other@local.test", hashed_password="x")
        db.add_all([u, o]); db.commit(); db.refresh(u); db.refresh(o)
        _USER.id, _OTHER.id = u.id, o.id
        p = ProjectModel(title="labels-test", status="completed", user_id=u.id)
        p2 = ProjectModel(title="labels-other", status="completed", user_id=o.id)
        db.add_all([p, p2]); db.commit(); db.refresh(p); db.refresh(p2)
        pid, pid_other = p.id, p2.id
        ids = [uuid.uuid4().hex for _ in range(5)]
        _mk_predictions(db, pid, "artX", ids)
        other_ids = [uuid.uuid4().hex for _ in range(2)]
        _mk_predictions(db, pid_other, "artY", other_ids)
    finally:
        db.close()

    yield {"pid": pid, "pid_other": pid_other, "ids": ids, "other_ids": other_ids}

    db = SessionLocal()
    try:
        db.query(PredictionModel).filter(PredictionModel.project_id.in_([pid, pid_other])).delete(
            synchronize_session=False)
        db.query(ProjectModel).filter(ProjectModel.id.in_([pid, pid_other])).delete(
            synchronize_session=False)
        db.query(UserModel).filter(UserModel.id.in_([_USER.id, _OTHER.id])).delete(
            synchronize_session=False)
        db.commit()
    finally:
        db.close()


def _labeled(db, pid):
    return {r.prediction_id: (r.actual_label, r.label_backfilled_at)
            for r in db.query(PredictionModel).filter(
                PredictionModel.project_id == pid,
                PredictionModel.actual_label.isnot(None)).all()}


def test_batch_match_and_unmatched_report(seeded):
    ids = seeded["ids"]
    body = LabelBatchRequest(labels=[
        LabelItem(prediction_id=ids[0], actual_label=1),
        LabelItem(prediction_id=ids[1], actual_label=0),
        LabelItem(prediction_id="does-not-exist", actual_label=1),   # eşleşmez
        LabelItem(prediction_id=ids[2], actual_label=None),          # geçersiz (null)
    ])
    db = SessionLocal()
    try:
        res = submit_labels(seeded["pid"], body, db=db, current_user=_USER)
    finally:
        db.close()
    assert res["matched"] == 2 and res["updated"] == 2
    assert res["unmatched"] == 1 and res["unmatched_keys"] == ["does-not-exist"]
    assert res["invalid"] == 1                                       # null etiket ayıklandı

    db = SessionLocal()
    try:
        lab = _labeled(db, seeded["pid"])
        assert lab[ids[0]][0] == 1 and lab[ids[1]][0] == 0
        assert ids[2] not in lab                                     # null yazılmadı (SQL NULL korunur)
        assert isinstance(lab[ids[0]][1], float)                     # label_backfilled_at FLOAT
    finally:
        db.close()


def test_ownership_404(seeded):
    body = LabelBatchRequest(labels=[LabelItem(prediction_id=seeded["ids"][0], actual_label=1)])
    db = SessionLocal()
    try:
        with pytest.raises(HTTPException) as ei:
            submit_labels(seeded["pid"], body, db=db, current_user=_OTHER)   # proje _USER'ın
        assert ei.value.status_code == 404
    finally:
        db.close()


def test_cross_project_isolation(seeded):
    # _USER'ın projesine, DİĞER projenin prediction_id'siyle etiket → eşleşMEmeli (izolasyon)
    body = LabelBatchRequest(labels=[
        LabelItem(prediction_id=seeded["other_ids"][0], actual_label=1)])
    db = SessionLocal()
    try:
        res = submit_labels(seeded["pid"], body, db=db, current_user=_USER)
        assert res["matched"] == 0 and res["unmatched"] == 1
        # diğer projedeki satır etiketlenMEmiş olmalı
        lab_other = _labeled(db, seeded["pid_other"])
        assert seeded["other_ids"][0] not in lab_other
    finally:
        db.close()


def test_idempotent_correction_updates_backfill(seeded):
    pid, ids = seeded["pid"], seeded["ids"]
    db = SessionLocal()
    try:
        submit_labels(pid, LabelBatchRequest(labels=[LabelItem(prediction_id=ids[3], actual_label=1)]),
                      db=db, current_user=_USER)
        first = _labeled(db, pid)[ids[3]]
    finally:
        db.close()
    time.sleep(0.01)
    db = SessionLocal()
    try:
        # DÜZELTME: aynı id, farklı etiket → üzerine yaz + backfilled_at yenilenir (pencere yeniden dirty)
        submit_labels(pid, LabelBatchRequest(labels=[LabelItem(prediction_id=ids[3], actual_label=0)]),
                      db=db, current_user=_USER)
        second = _labeled(db, pid)[ids[3]]
    finally:
        db.close()
    assert first[0] == 1 and second[0] == 0                          # etiket düzeltildi
    assert second[1] >= first[1]                                     # backfilled_at yenilendi


def test_csv_parse_and_bad_column(seeded):
    pid, ids = seeded["pid"], seeded["ids"]
    csv = f"pred_id,label\n{ids[4]},1\nmissing-id,0\n".encode("utf-8")

    def _upload(data):
        return UploadFile(filename="labels.csv", file=io.BytesIO(data),
                          headers=Headers({"content-type": "text/csv"}))

    db = SessionLocal()
    try:
        res = asyncio.run(submit_labels_csv(
            pid, file=_upload(csv), key_column="pred_id", label_column="label",
            db=db, current_user=_USER))
        assert res["matched"] == 1 and res["unmatched"] == 1
        assert _labeled(db, pid)[ids[4]][0] in (1, "1")              # tip-toleranslı (CSV int/str)
    finally:
        db.close()

    # YANLIŞ kolon adı → 422 (isimden çıkarım YOK, elle bildirim; yanlış = sessiz yanlış etiket riski)
    db = SessionLocal()
    try:
        with pytest.raises(HTTPException) as ei:
            asyncio.run(submit_labels_csv(
                pid, file=_upload(csv), key_column="NOPE", label_column="label",
                db=db, current_user=_USER))
        assert ei.value.status_code == 422
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-s"]))
