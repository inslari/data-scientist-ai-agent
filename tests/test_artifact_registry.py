"""Katman 1 Parça 1 — ArtifactModel serving index + worker upsert (MANUEL promotion).

worker._upsert_artifact_row diskteki artifact.json'dan (doğruluk kaynağı) okuyup metadata'yı UPSERT
eder — ama ``is_champion``'a DOKUNMAZ (otomatik promotion KALDIRILDI; canlıya alma manuel/endpoint).
Yeni satır champion DEĞİL; var olan satırın champion durumu re-upsert'te KORUNUR. Best-effort:
eksik/bozuk artifact.json job'u düşürmez. (Promotion'ın kendisi test_artifact_promotion.py'de.)
"""
import json
import os
import shutil
import uuid

import pytest

from api.core.database import Base, SessionLocal, engine
# Mapper'ların çözülmesi için TÜM model setini kaydet (worker_main ile aynı disiplin —
# DataGroupModel→CSVFileModel gibi string ilişkiler ancak hepsi import'luyken çözülür).
from api.models.user import UserModel  # noqa: F401
from api.models.csv import CSVFileModel  # noqa: F401
from api.models.job import BackgroundJobModel  # noqa: F401
from api.models.data_group import DataGroupModel  # noqa: F401
from api.models.mcp import MCPConfigModel  # noqa: F401
from api.models.project import ProjectModel
from api.models.artifact import ArtifactModel
from api.core.worker import _upsert_artifact_row

_ADIR = "data/artifacts"


def _write_artifact_json(aid, *, phase, deployable, model_path, model_name="rf"):
    d = os.path.join(_ADIR, aid)
    os.makedirs(d, exist_ok=True)
    meta = {
        "phase": phase,
        "supported_scope": {"deployable": deployable},
        "model_ref": {"model_path": model_path, "model_name": model_name},
        "artifact_schema_version": "1.2",
    }
    with open(os.path.join(d, "artifact.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    return d


@pytest.fixture()
def project(tmp_path):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    pid = None
    created_dirs = []
    try:
        p = ProjectModel(title="art-test", status="completed")
        db.add(p); db.commit(); db.refresh(p)
        pid = p.id
    finally:
        db.close()
    yield pid, created_dirs
    # temizlik
    db = SessionLocal()
    try:
        db.query(ArtifactModel).filter(ArtifactModel.project_id == pid).delete()
        db.query(ProjectModel).filter(ProjectModel.id == pid).delete()
        db.commit()
    finally:
        db.close()
    for d in created_dirs:
        shutil.rmtree(d, ignore_errors=True)


def _get(pid, aid):
    db = SessionLocal()
    try:
        return db.query(ArtifactModel).filter(ArtifactModel.id == aid).first(), \
               [r.id for r in db.query(ArtifactModel).filter(
                   ArtifactModel.project_id == pid, ArtifactModel.is_champion.is_(True)).all()]
    finally:
        db.close()


def test_complete_artifact_recorded_but_not_champion(project, tmp_path):
    # Complete+deployable artifact bile OTOMATİK champion OLMAZ (manuel promotion). Metadata dolu.
    pid, dirs = project
    mp = str(tmp_path / "m.pkl"); open(mp, "wb").write(b"x")
    aid = uuid.uuid4().hex
    dirs.append(_write_artifact_json(aid, phase="complete", deployable=True, model_path=mp))
    _upsert_artifact_row(pid, {"inference_artifact_id": aid})
    row, champs = _get(pid, aid)
    assert row is not None and row.phase == "complete" and row.deployable is True
    assert row.is_champion is False and champs == []    # otomatik promotion YOK


def test_reupsert_preserves_champion(project, tmp_path):
    # Elle champion yapılmış bir artifact yeniden koşuda (re-upsert) champion KALIR; worker düşürmez.
    pid, dirs = project
    mp = str(tmp_path / "m.pkl"); open(mp, "wb").write(b"x")
    aid = uuid.uuid4().hex
    dirs.append(_write_artifact_json(aid, phase="complete", deployable=True, model_path=mp))
    _upsert_artifact_row(pid, {"inference_artifact_id": aid})
    # elle promote (worker DEĞİL — burada doğrudan DB; gerçek promotion endpoint testi ayrı dosyada)
    db = SessionLocal()
    try:
        db.query(ArtifactModel).filter(ArtifactModel.id == aid).update({ArtifactModel.is_champion: True})
        db.commit()
    finally:
        db.close()
    _upsert_artifact_row(pid, {"inference_artifact_id": aid})   # yeniden koşu (metadata tazeler)
    row, champs = _get(pid, aid)
    assert row.is_champion is True and champs == [aid]  # re-upsert champion'ı KORUR


def test_non_serviceable_recorded_but_not_champion(project, tmp_path):
    pid, dirs = project
    # deployable=False → kaydedilir ama champion OLMAZ
    a1 = uuid.uuid4().hex
    dirs.append(_write_artifact_json(a1, phase="complete", deployable=False, model_path="x"))
    _upsert_artifact_row(pid, {"inference_artifact_id": a1})
    r1, champs = _get(pid, a1)
    assert r1 is not None and r1.is_champion is False and champs == []
    # phase=features (model yok) → yine champion değil
    a2 = uuid.uuid4().hex
    dirs.append(_write_artifact_json(a2, phase="features", deployable=True, model_path=None))
    _upsert_artifact_row(pid, {"inference_artifact_id": a2})
    r2, champs2 = _get(pid, a2)
    assert r2.phase == "features" and r2.is_champion is False and champs2 == []


def test_missing_artifact_json_is_best_effort(project):
    pid, _ = project
    # disk'te yok → hata YUTULUR, exception YOK (job düşmez)
    _upsert_artifact_row(pid, {"inference_artifact_id": "nonexistent_" + uuid.uuid4().hex})
    # id yoksa erken döner
    _upsert_artifact_row(pid, {})


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
