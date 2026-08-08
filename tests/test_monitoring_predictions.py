"""FAZ 1 — monitoring/predictions endpoint'i (paginated log, KEYSET/cursor).

Predictions satırları DOĞRUDAN seed'lenir (endpoint yalnız okur; champion/artifact yükü gerekmez).
Kapsam: keyset sayfalama (örtüşme/atlama YOK, sıra desc) · bileşik (created_at, prediction_id) tie-break
(aynı saniyede çok tahmin) · HEAD-INSERT stabilitesi (cursor'dan sonra yeni satır sayfayı KAYDIRMAZ →
offset'in sessiz-atlama hatası kapalı) · filtreler (artifact_id/date/has_label/has_warning) · redaction
KANCASI (input_row_redacted=false; input_row yalnız sampled satırlarda).
"""
import os
import types
import uuid

os.environ.setdefault("API_DATABASE_URL", "sqlite:///./data/_sbx_monitoring_predictions_test.sqlite")

import pytest
from fastapi import HTTPException

from api.core.database import Base, SessionLocal, engine
from api.models.user import UserModel  # noqa: F401
from api.models.csv import CSVFileModel  # noqa: F401
from api.models.data_group import DataGroupModel  # noqa: F401
from api.models.mcp import MCPConfigModel  # noqa: F401
from api.models.job import BackgroundJobModel  # noqa: F401
from api.models.project import ProjectModel
from api.models.prediction import PredictionModel

from api.routes.monitoring import monitoring_predictions
from src.core.config import settings

_USER = types.SimpleNamespace(id=0)
_OTHER = types.SimpleNamespace(id=0)


def _add_pred(db, pid, *, aid="champ", created_at=1000, sampled=False, label=None,
              flags=None, bk=None):
    pid_hex = uuid.uuid4().hex
    kwargs = dict(
        prediction_id=pid_hex, project_id=pid, artifact_id=aid, created_at=created_at,
        input_row=({"x": 1} if sampled else None), input_row_sampled=sampled,
        prediction=1, probability=0.5, threshold=0.5,
        input_warnings={"unseen_categories": {}, "out_of_range": {}},
        input_warning_flags=flags,
        business_key=bk,
    )
    # actual_label: JSON kolonu → Python None ATANIRSA JSON 'null' saklanır (SQL NULL değil), IS NULL bozulur.
    # Üretimde scoring bu alanı OMIT eder (SQL NULL) → testte de yalnız gerçek etiketi set et (invariant birebir).
    if label is not None:
        kwargs["actual_label"] = label
    db.add(PredictionModel(**kwargs))
    return pid_hex


@pytest.fixture(scope="module")
def seeded():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        u = UserModel(email="mon-pred@local.test", hashed_password="x")
        o = UserModel(email="mon-pred-other@local.test", hashed_password="x")
        db.add_all([u, o]); db.commit(); db.refresh(u); db.refresh(o)
        _USER.id, _OTHER.id = u.id, o.id

        pid = ProjectModel(title="pred-main", status="completed", user_id=u.id)
        db.add(pid); db.commit(); db.refresh(pid); pid = pid.id

        ids_by_ts = {}
        # 20 satır artan created_at (1000..1019) → desc sırada 1019 ilk. 5'i sampled, birkaçı label/flag/bk.
        for t in range(1000, 1020):
            sampled = (t % 4 == 0)
            label = 1 if t % 5 == 0 else None
            flags = ({"has_unseen": True, "has_out_of_range": False} if t % 3 == 0
                     else {"has_unseen": False, "has_out_of_range": False})
            if t == 1007:
                flags = None                               # bayrak hesabı patlamış satır (has_warning=false sayılır)
            bk = f"cust{t}" if t % 6 == 0 else None
            aid = "champ" if t < 1015 else "old"           # artifact_id filtresi için ikinci sürüm
            ids_by_ts[t] = _add_pred(db, pid, aid=aid, created_at=t, sampled=sampled,
                                     label=label, flags=flags, bk=bk)
        # aynı saniyede 5 tahmin (bileşik tie-break testi) → created_at=2000
        tie_ids = [_add_pred(db, pid, created_at=2000) for _ in range(5)]
        db.commit()
    finally:
        db.close()

    yield {"pid": pid, "tie_ids": tie_ids}

    db = SessionLocal()
    try:
        db.query(PredictionModel).filter(PredictionModel.project_id == pid).delete()
        db.query(ProjectModel).filter(ProjectModel.id == pid).delete()
        db.query(UserModel).filter(UserModel.id.in_([_USER.id, _OTHER.id])).delete(
            synchronize_session=False)
        db.commit()
    finally:
        db.close()


def _call(pid, user=None, **kw):
    # Endpoint doğrudan çağrılıyor (FastAPI yok) → atlanan Query() defaultları objedir; tümünü None ver.
    params = {"cursor": None, "limit": None, "date_from": None, "date_to": None,
              "artifact_id": None, "has_label": None, "has_warning": None}
    params.update(kw)
    db = SessionLocal()
    try:
        return monitoring_predictions(pid, db=db, current_user=(user or _USER), **params)
    finally:
        db.close()


def test_ownership_404(seeded):
    with pytest.raises(HTTPException) as ei:
        _call(seeded["pid"], user=_OTHER)
    assert ei.value.status_code == 404


def test_keyset_pagination_no_overlap_no_skip(seeded):
    """Tüm satırları cursor'la sayfala (limit=7) → toplam 25 satır, örtüşme/atlama YOK, desc sıra."""
    seen = []
    cursor = None
    pages = 0
    while True:
        res = _call(seeded["pid"], cursor=cursor, limit=7)
        assert res["window_timezone"] == "UTC"
        seen.extend(res["rows"])
        pages += 1
        if not res["has_more"]:
            assert res["next_cursor"] is None
            break
        cursor = res["next_cursor"]
        assert cursor is not None
        assert pages < 10                                  # sonsuz döngü sigortası
    ids = [r["prediction_id"] for r in seen]
    assert len(ids) == 25                                  # 20 + 5 tie
    assert len(set(ids)) == 25                             # örtüşme YOK
    # desc sıra: (created_at, prediction_id) azalan
    keys = [(r["created_at"], r["prediction_id"]) for r in seen]
    assert keys == sorted(keys, reverse=True)              # atlama/sıra bozulması YOK


def test_composite_cursor_tie_break(seeded):
    """Aynı created_at=2000'de 5 satır → created_at tek başına tie-break edemez; prediction_id ile
    toplam-sıra → 5 satır limit=2 ile 3 sayfada tam ve tekrarsız gelir."""
    seen = []
    cursor = None
    for _ in range(5):
        res = _call(seeded["pid"], cursor=cursor, limit=2,
                    date_from=2000, date_to=2000)          # yalnız tie grubu
        seen.extend(r["prediction_id"] for r in res["rows"])
        if not res["has_more"]:
            break
        cursor = res["next_cursor"]
    assert sorted(seen) == sorted(seeded["tie_ids"])       # 5'i de, tekrarsız


def test_head_insert_does_not_shift_pages(seeded):
    """OFFSET'in sessiz-atlama hatası KAPALI: page1'den sonra YENİ (daha yeni) tahmin gelirse,
    cursor'la çekilen page2 etkilenmez (yeni satır page1 bölgesine ait, page2 kaymaz)."""
    p1 = _call(seeded["pid"], cursor=None, limit=5)
    assert p1["has_more"]
    cursor = p1["next_cursor"]

    # HEAD-INSERT: en yeni created_at'ten de büyük bir satır ekle
    db = SessionLocal()
    try:
        new_id = _add_pred(db, seeded["pid"], created_at=9999)
        db.commit()
    finally:
        db.close()
    try:
        p2 = _call(seeded["pid"], cursor=cursor, limit=5)
        p2_ids = {r["prediction_id"] for r in p2["rows"]}
        p1_ids = {r["prediction_id"] for r in p1["rows"]}
        assert new_id not in p2_ids                        # yeni satır page2'ye SIZMADI (offset olsaydı satır atlardı)
        assert not (p1_ids & p2_ids)                       # page1 ↔ page2 örtüşme yok
    finally:
        db = SessionLocal()                                # temizle (fixture sayımlarını bozmasın)
        try:
            db.query(PredictionModel).filter(PredictionModel.prediction_id == new_id).delete()
            db.commit()
        finally:
            db.close()


def test_filter_artifact_id(seeded):
    res = _call(seeded["pid"], artifact_id="old", limit=200)
    assert len(res["rows"]) == 5                            # t in 1015..1019
    assert all(r["artifact_id"] == "old" for r in res["rows"])


def test_filter_date_range(seeded):
    res = _call(seeded["pid"], date_from=1005, date_to=1009, limit=200)
    ts = sorted(r["created_at"] for r in res["rows"])
    assert ts == [1005, 1006, 1007, 1008, 1009]


def test_filter_has_label(seeded):
    with_label = _call(seeded["pid"], has_label=True, limit=200)
    assert all(r["actual_label"] is not None for r in with_label["rows"])
    # t%5==0 → 1000,1005,1010,1015 = 4 etiketli
    assert len(with_label["rows"]) == 4
    without = _call(seeded["pid"], has_label=False, limit=200)
    assert all(r["actual_label"] is None for r in without["rows"])


def test_filter_has_warning(seeded):
    """Per-row JSON bayrak filtresi (SQLite as_boolean yolu). has_unseen=true olanlar (t%3==0), ama
    t=1007 flags=None → uyarısız sayılır."""
    warned = _call(seeded["pid"], has_warning=True, limit=200)
    assert len(warned["rows"]) > 0
    for r in warned["rows"]:
        f = r["input_warning_flags"]
        assert f is not None and (f["has_unseen"] or f["has_out_of_range"])
    # t%3==0 → 1002,1005,1008,1011,1014,1017; t=1007 zaten %3!=0 → etkilenmez → 6 uyarı
    assert len(warned["rows"]) == 6
    unwarned = _call(seeded["pid"], has_warning=False, limit=200)
    # NULL bayraklı satır (t=1007) uyarısız TARAFTA olmalı
    assert any(r["created_at"] == 1007 for r in unwarned["rows"])
    assert all(not (r["input_warning_flags"] or {}).get("has_unseen") for r in unwarned["rows"])


def test_redaction_hook_and_sampling(seeded):
    """Redaction KANCASI v1-identity: her satır input_row_redacted=false; input_row yalnız sampled'da dolu."""
    res = _call(seeded["pid"], limit=200)
    assert all(r["input_row_redacted"] is False for r in res["rows"])
    for r in res["rows"]:
        if r["input_row_sampled"]:
            assert r["input_row"] is not None
        else:
            assert r["input_row"] is None


def test_limit_clamped_to_max(seeded):
    res = _call(seeded["pid"], limit=100000)
    assert res["limit_applied"] == settings.MONITORING_PREDICTIONS_PAGE_MAX


def test_invalid_cursor_400(seeded):
    with pytest.raises(HTTPException) as ei:
        _call(seeded["pid"], cursor="!!!not-base64!!!")
    assert ei.value.status_code == 400


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-s"]))
