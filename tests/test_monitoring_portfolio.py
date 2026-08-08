"""FAZ 1 — monitoring/portfolio endpoint'i (cross-project, AYRI router prefix="/monitoring").

🔴 EN KRİTİK: SAHİPLİK-SIZINTI NEGATİF TESTİ. Cross-project join'de kaçarsa GÜVENLİK AÇIĞI olur ve
sessizce çalışır görünür → B kullanıcısının projesi A'nın yanıtında HİÇBİR dalda çıkmamalı. Join'in HER
dalı ayrı seed'lenir (champion yok / pencere yok / tahmin yok / tam) → B'nin EN ZENGİN projesi (champion+
pencere+tahmin, join'e en çok "yem") bile A'ya sızmamalı.

Ayrıca: üç boş-reason ayrışması · reason KIRPMA (kullanıcı eklemesi b: kart {driver, code}, value YOK) ·
N+1 YOK (sorgu sayısı proje sayısından BAĞIMSIZ sabit — SQLAlchemy cursor sayacıyla kanıt).
"""
import os
import time
import types
import uuid

os.environ.setdefault("API_DATABASE_URL", "sqlite:///./data/_sbx_monitoring_portfolio_test.sqlite")

import pytest
from sqlalchemy import event

from api.core.database import Base, SessionLocal, engine
from api.models.user import UserModel  # noqa: F401
from api.models.csv import CSVFileModel  # noqa: F401
from api.models.data_group import DataGroupModel  # noqa: F401
from api.models.mcp import MCPConfigModel  # noqa: F401
from api.models.job import BackgroundJobModel  # noqa: F401
from api.models.project import ProjectModel
from api.models.artifact import ArtifactModel
from api.models.prediction import PredictionModel
from api.models.monitoring_summary import MonitoringSummaryModel

from api.routes.monitoring import monitoring_portfolio

_DAY = 86_400
_A = types.SimpleNamespace(id=0)
_B = types.SimpleNamespace(id=0)


def _mk_project(db, uid, title):
    p = ProjectModel(title=title, status="completed", user_id=uid)
    db.add(p); db.commit(); db.refresh(p)
    return p.id


def _mk_champion(db, pid, aid, deployable=True, phase="complete", champion=True):
    db.add(ArtifactModel(id=aid, project_id=pid, phase=phase, deployable=deployable,
                         model_path=f"data/models/{aid}.pkl", model_name="blend",
                         schema_version="1.2", is_champion=champion,
                         created_at=int(time.time()), updated_at=int(time.time())))
    db.commit()


def _mk_preds(db, pid, aid, n):
    now = int(time.time())
    db.bulk_insert_mappings(PredictionModel, [
        {"prediction_id": uuid.uuid4().hex, "project_id": pid, "artifact_id": aid,
         "created_at": now, "prediction": 1, "probability": 0.5} for _ in range(n)])
    db.commit()


def _mk_window(db, pid, aid, window_start):
    db.add(MonitoringSummaryModel(
        project_id=pid, artifact_id=aid, window_start=window_start,
        window_end=window_start + _DAY, window_kind="daily", computed_at=time.time(),
        n_total=600, n_sampled=600, status="ok",
        psi_overall=0.05, psi_max=0.19, psi_degenerate_count=0,
        cbpe_gate_status="ok", realized_gate_status="ok", label_coverage=0.4,
        overall_status="warning",
        overall_status_reason={"driver": "drift", "code": "psi_drift_warning",
                               "value": {"psi_max": 0.19, "feature": "balance"},
                               "suppressed_axes": {"realized": "critical"},
                               "measurement_coverage": {"adequate": True}},
    ))
    db.commit()


def _full_project(db, uid, title, aid):
    """champion + tahmin + kapanmış pencere (has_data dalı — join'e en zengin yem)."""
    pid = _mk_project(db, uid, title)
    _mk_champion(db, pid, aid)
    _mk_preds(db, pid, aid, 20)
    now_day = (int(time.time()) // _DAY) * _DAY
    _mk_window(db, pid, aid, now_day - _DAY)
    return pid


@pytest.fixture(scope="module")
def seeded():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        a = UserModel(email="portf-a@local.test", hashed_password="x")
        b = UserModel(email="portf-b@local.test", hashed_password="x")
        db.add_all([a, b]); db.commit(); db.refresh(a); db.refresh(b)
        _A.id, _B.id = a.id, b.id

        # A: dört dal
        a_full = _full_project(db, a.id, "A-full", "a_full_champ")
        a_nochamp = _mk_project(db, a.id, "A-nochamp")
        _mk_champion(db, a_nochamp, "a_nc", deployable=False, phase="features", champion=False)
        a_nopred = _mk_project(db, a.id, "A-nopred")
        _mk_champion(db, a_nopred, "a_np")
        a_noclosed = _mk_project(db, a.id, "A-noclosed")
        _mk_champion(db, a_noclosed, "a_ncw")
        _mk_preds(db, a_noclosed, "a_ncw", 10)             # tahmin var, pencere YOK

        # B: EN ZENGİN dahil dört dal (A'nın yanıtına sızmamalı)
        b_full = _full_project(db, b.id, "B-full", "b_full_champ")
        b_nopred = _mk_project(db, b.id, "B-nopred")
        _mk_champion(db, b_nopred, "b_np")
        b_noclosed = _mk_project(db, b.id, "B-noclosed")
        _mk_champion(db, b_noclosed, "b_ncw")
        _mk_preds(db, b_noclosed, "b_ncw", 7)
    finally:
        db.close()

    ids = {"a_full": a_full, "a_nochamp": a_nochamp, "a_nopred": a_nopred, "a_noclosed": a_noclosed,
           "b_full": b_full, "b_nopred": b_nopred, "b_noclosed": b_noclosed}
    yield ids

    db = SessionLocal()
    try:
        for pid in ids.values():
            db.query(MonitoringSummaryModel).filter(MonitoringSummaryModel.project_id == pid).delete()
            db.query(PredictionModel).filter(PredictionModel.project_id == pid).delete()
            db.query(ArtifactModel).filter(ArtifactModel.project_id == pid).delete()
            db.query(ProjectModel).filter(ProjectModel.id == pid).delete()
        db.query(UserModel).filter(UserModel.id.in_([_A.id, _B.id])).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


def _call(user):
    db = SessionLocal()
    try:
        return monitoring_portfolio(db=db, current_user=user)
    finally:
        db.close()


def test_ownership_no_cross_project_leak(seeded):
    """🔴 KRİTİK: A'nın portfolio'sunda YALNIZ A'nın projeleri; B'nin HİÇBİRİ (en zengin b_full dahil)."""
    res = _call(_A)
    got_ids = {p["project_id"] for p in res["projects"]}
    a_ids = {seeded["a_full"], seeded["a_nochamp"], seeded["a_nopred"], seeded["a_noclosed"]}
    b_ids = {seeded["b_full"], seeded["b_nopred"], seeded["b_noclosed"]}
    assert got_ids == a_ids                                # tam A kümesi
    assert not (got_ids & b_ids)                           # B'den SIFIR sızıntı
    # b_full champion+pencere+tahmin taşır (join'in üç dalı da dolu) → yine de A'ya giremez
    assert seeded["b_full"] not in got_ids


def test_ownership_symmetric(seeded):
    """Simetrik: B'nin portfolio'sunda A'nın hiçbiri yok."""
    res = _call(_B)
    got_ids = {p["project_id"] for p in res["projects"]}
    a_ids = {seeded["a_full"], seeded["a_nochamp"], seeded["a_nopred"], seeded["a_noclosed"]}
    assert got_ids == {seeded["b_full"], seeded["b_nopred"], seeded["b_noclosed"]}
    assert not (got_ids & a_ids)


def test_three_empty_reasons_distinct(seeded):
    res = _call(_A)
    by = {p["project_id"]: p for p in res["projects"]}
    # a_nochamp artifact phase=features (complete+deployable DEĞİL) → serve edilebilir aday yok → not_trained.
    assert by[seeded["a_nochamp"]]["reason"] == "not_trained"
    assert by[seeded["a_nochamp"]]["card"] is None
    assert by[seeded["a_nopred"]]["reason"] == "no_predictions"
    assert by[seeded["a_noclosed"]]["reason"] == "no_closed_window"
    assert by[seeded["a_full"]]["reason"] is None and by[seeded["a_full"]]["has_data"] is True


def test_card_reason_is_truncated(seeded):
    """Kullanıcı eklemesi (b): kart reason YALNIZ {driver, code} — value/suppressed_axes YOK (detayda)."""
    res = _call(_A)
    card = next(p["card"] for p in res["projects"] if p["project_id"] == seeded["a_full"])
    assert set(card["reason"].keys()) == {"driver", "code"}
    assert card["reason"]["code"] == "psi_drift_warning"
    assert "value" not in card["reason"] and "suppressed_axes" not in card["reason"]
    # kart psi_max'ı AYRI taşır (reason'da tekrar etmez)
    assert card["psi_max"] == 0.19
    assert card["total_predictions"] == 20
    assert card["overall_status"] == "warning"


def test_no_n_plus_1_constant_query_count(seeded):
    """N+1 YOK: sorgu sayısı proje sayısından BAĞIMSIZ. A (4 proje) portfolio'sunun SQL sayısı küçük
    bir sabit olmalı (owned→champions→ROW_NUMBER pencere→COUNT ~ birkaç sorgu), proje×sabit DEĞİL."""
    counter = {"n": 0}

    def _count(conn, cursor, statement, params, context, executemany):
        counter["n"] += 1

    event.listen(engine, "after_cursor_execute", _count)
    try:
        res = _call(_A)
    finally:
        event.remove(engine, "after_cursor_execute", _count)
    assert len(res["projects"]) == 4
    # 4 proje için ≤ 6 SQL: owned + champions + window(ROW_NUMBER) + pred COUNT (+ SessionLocal ısınma marjı).
    # N+1 olsaydı proje/champion başına ek sorgu → ≥ 8-12. Sabit sınır bunu yakalar.
    assert counter["n"] <= 6, f"N+1 şüphesi: {counter['n']} SQL (beklenen sabit ≤6)"


def test_empty_portfolio_for_user_with_no_projects(seeded):
    """Hiç projesi olmayan kullanıcı → boş liste (owned_ids boş → erken dönüş, sızıntı yüzeyi yok)."""
    db = SessionLocal()
    try:
        ghost = UserModel(email="portf-ghost@local.test", hashed_password="x")
        db.add(ghost); db.commit(); db.refresh(ghost)
        gid = ghost.id
    finally:
        db.close()
    res = monitoring_portfolio(db=SessionLocal(), current_user=types.SimpleNamespace(id=gid))
    assert res["projects"] == []
    db = SessionLocal()
    try:
        db.query(UserModel).filter(UserModel.id == gid).delete(); db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-s"]))
