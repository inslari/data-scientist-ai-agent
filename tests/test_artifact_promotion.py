"""Manuel champion promotion — endpoint + atomiklik + fail-closed + olay günlüğü + pencere-bölme.

Fonksiyonları DOĞRUDAN çağırır (JWT/HTTP gerekmez; current_user gerçek UserModel — actor FK'sı için).
Serve edilebilir artifact = disk artifact.json (complete+deployable+model_ref) + VAR OLAN dummy model
dosyası (resolve_model_path yalnız varlık kontrol eder → gerçek model gerekmez) + ArtifactModel satırı.

Kapsam:
  * promote happy (ilk model → previous_champion None) · eski champion'ı düşür (tek champion invariant).
  * ATOMİKLİK: event yazımı transaction ORTASINDA patlar (mock) → rollback → eski champion KORUNUR,
    yeni promote OLMAZ, event YAZILMAZ (yarım promotion = iki/sıfır champion, ikisi de bozuk → imkânsız).
  * FAIL-CLOSED: serve edilemez artifact (model dosyası yok) → 409, hiç değişiklik yok.
  * sahiplik 404 · projeye ait olmayan artifact 404.
  * TAZE champion lookup (memoize yok): promote sonrası _serving_champion yeniyi görür (d regresyonu).
  * olay günlüğü: kim/ne zaman/hangi sürümden hangisine + sayfalama.
  * PENCERE-BÖLME (ekleme a): gün-ortası promote → aggregator o günü İKİ satır (her artifact kendi n'i).

⚠️ EŞZAMANLILIK: with_for_update SQLite'ta no-op (test ortamı), Postgres'te gerçek kilit (prod). Yarış
yalnız SQLite'ta teorik mümkün; tek-kullanıcı + elle/nadir promotion → pratik risk yok. artifacts.py
modül docstring'inde belgeli. Buradaki "iki ardışık promote → tek champion" testi tutarlılığı çitler.
"""
import json
import os
import shutil
import time
import uuid

os.environ.setdefault("API_DATABASE_URL", "sqlite:///./data/_sbx_promotion_test.sqlite")

import pytest
from fastapi import HTTPException

from api.core.database import Base, SessionLocal, engine
# mapper seti (conftest tüm seti kaydeder; standalone çalıştırma için de burada)
from api.models.user import UserModel
from api.models.csv import CSVFileModel  # noqa: F401
from api.models.job import BackgroundJobModel  # noqa: F401
from api.models.data_group import DataGroupModel  # noqa: F401
from api.models.mcp import MCPConfigModel  # noqa: F401
from api.models.project import ProjectModel
from api.models.artifact import ArtifactModel
from api.models.artifact_event import ArtifactEventModel
from api.models.prediction import PredictionModel
from api.models.monitoring_summary import MonitoringSummaryModel

import api.routes.artifacts as artifacts_mod
from api.routes.artifacts import promote_artifact, artifact_events
from api.routes.monitoring import _serving_champion
from src.services.monitoring.aggregator import run_monitoring_aggregation

_ADIR = "data/artifacts"
_DAY = 86_400


def _write_serviceable(aid: str, *, model_present: bool = True) -> str:
    """Disk artifact: complete+deployable + (opsiyonel) var olan model dosyası. Dizin yolunu döner."""
    d = os.path.join(_ADIR, aid)
    os.makedirs(d, exist_ok=True)
    mp = os.path.join(d, "model.pkl")
    if model_present:
        with open(mp, "wb") as fh:
            fh.write(b"x")                          # resolve_model_path yalnız VARLIK kontrol eder
    meta = {"phase": "complete", "supported_scope": {"deployable": True},
            "model_ref": {"model_path": mp, "model_name": "rf"},
            "raw_input_signature": {"required": [], "columns": {}}, "final_features": [],
            "param_store": {}, "manifest": [], "artifact_schema_version": "1.2"}
    with open(os.path.join(d, "artifact.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    return mp


def _mk_artifact(db, pid, aid, *, champion=False, model_present=True):
    mp = _write_serviceable(aid, model_present=model_present)
    db.add(ArtifactModel(id=aid, project_id=pid, phase="complete", deployable=True,
                         model_path=mp, model_name="rf", schema_version="1.2",
                         is_champion=champion,
                         created_at=int(time.time()), updated_at=int(time.time())))
    db.commit()
    return aid


@pytest.fixture()
def env():
    """Taze proje + kullanıcı; teardown DB + disk temizler."""
    Base.metadata.create_all(bind=engine)
    dirs = []
    db = SessionLocal()
    try:
        u = UserModel(email=f"promo-{uuid.uuid4().hex[:8]}@local.test", hashed_password="x")
        db.add(u); db.commit(); db.refresh(u)
        p = ProjectModel(title="promo-test", status="completed", user_id=u.id)
        db.add(p); db.commit(); db.refresh(p)
        pid, uid = p.id, u.id
    finally:
        db.close()
    state = {"pid": pid, "uid": uid, "dirs": dirs}
    yield state
    db = SessionLocal()
    try:
        db.query(ArtifactEventModel).filter(ArtifactEventModel.project_id == pid).delete()
        db.query(MonitoringSummaryModel).filter(MonitoringSummaryModel.project_id == pid).delete()
        db.query(PredictionModel).filter(PredictionModel.project_id == pid).delete()
        db.query(ArtifactModel).filter(ArtifactModel.project_id == pid).delete()
        db.query(ProjectModel).filter(ProjectModel.id == pid).delete()
        db.query(UserModel).filter(UserModel.id == uid).delete()
        db.commit()
    finally:
        db.close()
    for aid in dirs:
        shutil.rmtree(os.path.join(_ADIR, aid), ignore_errors=True)


class _User:
    def __init__(self, uid): self.id = uid


def _champs(pid):
    db = SessionLocal()
    try:
        return [r.id for r in db.query(ArtifactModel).filter(
            ArtifactModel.project_id == pid, ArtifactModel.is_champion.is_(True)).all()]
    finally:
        db.close()


def _events(pid):
    db = SessionLocal()
    try:
        return db.query(ArtifactEventModel).filter(
            ArtifactEventModel.project_id == pid).order_by(ArtifactEventModel.at).all()
    finally:
        db.close()


# ─────────────────────────── HAPPY PATH ───────────────────────────

def test_first_promote_no_previous_champion(env):
    pid, uid, dirs = env["pid"], env["uid"], env["dirs"]
    db = SessionLocal()
    try:
        a1 = _mk_artifact(db, pid, uuid.uuid4().hex); dirs.append(a1)
    finally:
        db.close()
    db = SessionLocal()
    try:
        resp = promote_artifact(pid, a1, db=db, current_user=_User(uid))
    finally:
        db.close()
    assert resp["promoted_artifact_id"] == a1
    assert resp["previous_champion"] is None            # ilk model → önceki yok
    assert resp["actor"] == uid
    assert _champs(pid) == [a1]
    evs = _events(pid)
    assert len(evs) == 1 and evs[0].event_type == "promote"
    assert evs[0].from_artifact_id is None and evs[0].artifact_id == a1 and evs[0].actor == uid


def test_promote_demotes_old_single_champion(env):
    pid, uid, dirs = env["pid"], env["uid"], env["dirs"]
    db = SessionLocal()
    try:
        a1 = _mk_artifact(db, pid, uuid.uuid4().hex, champion=True); dirs.append(a1)
        a2 = _mk_artifact(db, pid, uuid.uuid4().hex); dirs.append(a2)
    finally:
        db.close()
    db = SessionLocal()
    try:
        resp = promote_artifact(pid, a2, db=db, current_user=_User(uid))
    finally:
        db.close()
    assert resp["previous_champion"] == a1
    assert _champs(pid) == [a2]                          # TEK champion (eski düştü)
    evs = _events(pid)
    assert len(evs) == 1 and evs[0].from_artifact_id == a1 and evs[0].artifact_id == a2


# ─────────────────────────── ATOMİKLİK ───────────────────────────

def test_atomicity_event_failure_rolls_back(env, monkeypatch):
    """Event yazımı transaction ORTASINDA patlar → HİÇBİRİ olmaz: eski champion KORUNUR, yeni promote OLMAZ."""
    pid, uid, dirs = env["pid"], env["uid"], env["dirs"]
    db = SessionLocal()
    try:
        a1 = _mk_artifact(db, pid, uuid.uuid4().hex, champion=True); dirs.append(a1)
        a2 = _mk_artifact(db, pid, uuid.uuid4().hex); dirs.append(a2)
    finally:
        db.close()

    def _boom(*a, **k):
        raise RuntimeError("SIMULE event yazımı hatası (transaction ortası)")
    monkeypatch.setattr(artifacts_mod, "ArtifactEventModel", _boom)

    db = SessionLocal()
    try:
        with pytest.raises(HTTPException) as ei:
            promote_artifact(pid, a2, db=db, current_user=_User(uid))
        assert ei.value.status_code == 500
    finally:
        db.close()

    # YARIM PROMOTION İMKÂNSIZ: eski champion hâlâ champion, yeni promote OLMADI, event YAZILMADI.
    assert _champs(pid) == [a1], "atomiklik ihlali: eski champion düştü ama yeni set edilmedi (sıfır champion)"
    assert len(_events(pid)) == 0, "rollback'e rağmen event yazıldı (atomiklik ihlali)"


# ─────────────────────────── FAIL-CLOSED ───────────────────────────

def test_fail_closed_non_serviceable_409(env):
    """Model dosyası yok → load_for_serving 409 → hiç değişiklik (champion olmaz, event yok)."""
    pid, uid, dirs = env["pid"], env["uid"], env["dirs"]
    db = SessionLocal()
    try:
        a1 = _mk_artifact(db, pid, uuid.uuid4().hex, model_present=False); dirs.append(a1)
    finally:
        db.close()
    db = SessionLocal()
    try:
        with pytest.raises(HTTPException) as ei:
            promote_artifact(pid, a1, db=db, current_user=_User(uid))
        assert ei.value.status_code == 409
    finally:
        db.close()
    assert _champs(pid) == []
    assert len(_events(pid)) == 0


def test_ownership_404(env):
    pid, uid, dirs = env["pid"], env["uid"], env["dirs"]
    db = SessionLocal()
    try:
        a1 = _mk_artifact(db, pid, uuid.uuid4().hex); dirs.append(a1)
    finally:
        db.close()
    db = SessionLocal()
    try:
        with pytest.raises(HTTPException) as ei:
            promote_artifact(pid, a1, db=db, current_user=_User(uid + 99999))   # başka kullanıcı
        assert ei.value.status_code == 404
    finally:
        db.close()
    assert _champs(pid) == []


def test_artifact_not_in_project_404(env):
    pid, uid = env["pid"], env["uid"]
    db = SessionLocal()
    try:
        with pytest.raises(HTTPException) as ei:
            promote_artifact(pid, "nonexistent_" + uuid.uuid4().hex, db=db, current_user=_User(uid))
        assert ei.value.status_code == 404
    finally:
        db.close()


# ─────────────────────────── TAZE CHAMPION LOOKUP (d regresyonu) ───────────────────────────

def test_fresh_champion_lookup_after_promote(env):
    """Promote sonrası _serving_champion (scoring ile aynı taze DB sorgusu) yeniyi görür — memoize yok."""
    pid, uid, dirs = env["pid"], env["uid"], env["dirs"]
    db = SessionLocal()
    try:
        a1 = _mk_artifact(db, pid, uuid.uuid4().hex, champion=True); dirs.append(a1)
        a2 = _mk_artifact(db, pid, uuid.uuid4().hex); dirs.append(a2)
    finally:
        db.close()
    db = SessionLocal()
    try:
        assert _serving_champion(db, pid).id == a1        # başlangıç
        promote_artifact(pid, a2, db=db, current_user=_User(uid))
    finally:
        db.close()
    db = SessionLocal()
    try:
        assert _serving_champion(db, pid).id == a2        # taze sorgu yeniyi döndürür
        promote_artifact(pid, a1, db=db, current_user=_User(uid))   # geri al (rollback senaryosu)
    finally:
        db.close()
    db = SessionLocal()
    try:
        assert _serving_champion(db, pid).id == a1
    finally:
        db.close()
    assert _champs(pid) == [a1]                            # her adımda tek champion


def test_two_sequential_promotes_single_champion(env):
    """Ardışık iki promote → her zaman TEK champion (tutarlılık; eşzamanlılık notu modül docstring'inde)."""
    pid, uid, dirs = env["pid"], env["uid"], env["dirs"]
    db = SessionLocal()
    try:
        a1 = _mk_artifact(db, pid, uuid.uuid4().hex); dirs.append(a1)
        a2 = _mk_artifact(db, pid, uuid.uuid4().hex); dirs.append(a2)
    finally:
        db.close()
    for aid in (a1, a2):
        db = SessionLocal()
        try:
            promote_artifact(pid, aid, db=db, current_user=_User(uid))
        finally:
            db.close()
    assert _champs(pid) == [a2]                            # son promote kazanır, tek champion


# ─────────────────────────── OLAY GÜNLÜĞÜ ───────────────────────────

def test_events_endpoint_shape_and_pagination(env):
    pid, uid, dirs = env["pid"], env["uid"], env["dirs"]
    db = SessionLocal()
    try:
        a1 = _mk_artifact(db, pid, uuid.uuid4().hex); dirs.append(a1)
        a2 = _mk_artifact(db, pid, uuid.uuid4().hex); dirs.append(a2)
    finally:
        db.close()
    for aid in (a1, a2):
        db = SessionLocal()
        try:
            promote_artifact(pid, aid, db=db, current_user=_User(uid))
            time.sleep(0.01)                              # at sıralaması ayrışsın
        finally:
            db.close()

    db = SessionLocal()
    try:
        res = artifact_events(pid, limit=None, offset=0, db=db, current_user=_User(uid))
    finally:
        db.close()
    assert res["total"] == 2 and len(res["events"]) == 2
    # at DESC → en yeni (a2 promote) önce
    assert res["events"][0]["artifact_id"] == a2 and res["events"][0]["from_artifact_id"] == a1
    assert res["events"][1]["artifact_id"] == a1 and res["events"][1]["from_artifact_id"] is None
    # kim: actor + actor_email çözülür
    assert res["events"][0]["actor"] == uid and "@local.test" in res["events"][0]["actor_email"]
    assert res["has_more"] is False

    # sayfalama: limit=1 → 1 satır + has_more
    db = SessionLocal()
    try:
        page1 = artifact_events(pid, limit=1, offset=0, db=db, current_user=_User(uid))
    finally:
        db.close()
    assert len(page1["events"]) == 1 and page1["has_more"] is True and page1["total"] == 2


# ─────────────────────────── PENCERE-BÖLME (ekleme a) ───────────────────────────

def test_midday_promotion_splits_window(env):
    """Gün-ortası promote → aynı UTC gün İKİ artifact_id taşır → aggregator İKİ satır (her artifact
    kendi n'i; ikisi de insufficient_n olabilir). Sahte drift önlenir (champion sınırında doğal kesim)."""
    pid = env["pid"]
    day = ((int(time.time()) // _DAY) - 2) * _DAY         # 2 gün önce → kesin kapanmış gün
    a1, a2 = "mid_a1_" + uuid.uuid4().hex[:6], "mid_a2_" + uuid.uuid4().hex[:6]
    db = SessionLocal()
    try:
        # aynı günde önce a1 (5 tahmin), sonra a2 (5 tahmin) — gün ortası promotion senaryosu
        rows = []
        for aid, k in ((a1, 0), (a2, 1)):
            for i in range(5):
                rows.append({"prediction_id": uuid.uuid4().hex, "project_id": pid, "artifact_id": aid,
                             "created_at": day + 100 + k * 500 + i, "prediction": 1, "probability": 0.5,
                             "input_row_sampled": False})
        db.bulk_insert_mappings(PredictionModel, rows)
        db.commit()
    finally:
        db.close()

    run_monitoring_aggregation()

    db = SessionLocal()
    try:
        wins = db.query(MonitoringSummaryModel).filter(
            MonitoringSummaryModel.project_id == pid,
            MonitoringSummaryModel.window_start == day).all()
    finally:
        db.close()
    by_art = {w.artifact_id: w for w in wins}
    assert set(by_art) == {a1, a2}, "gün-ortası promotion pencereyi artifact sınırında BÖLMEDİ"
    for w in wins:
        assert w.n_total == 5                             # her artifact kendi n'i (birleşmedi)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-s"]))
