"""Katman 2 — monitoring OKUMA API'si (summary / detail / deployments / latest).

Fonksiyonları DOĞRUDAN çağırır (JWT/HTTP gerekmez; current_user stub yalnız .id) — labels/scoring deseni.
Sentetik monitoring_summary + artifacts + predictions satırları → HESAP YOK (endpoint yalnız okur) → hızlı,
model eğitimi gerekmez. Kapsam: pencere listesi+sıra+config · days üst-sınır kırpma · per_feature YALNIZ
detay/latest'te · "boş" reason ayrışması (not_trained/no_predictions/no_closed_window) · deployments
champion+window_count+total_predictions+last_window · sahiplik 404 · window_timezone UTC · yanıt boyut ölçümü.
"""
import json
import os
import time
import types
import uuid

os.environ.setdefault("API_DATABASE_URL", "sqlite:///./data/_sbx_monitoring_api_test.sqlite")

import pytest
from fastapi import HTTPException

from api.core.database import Base, SessionLocal, engine
from api.models.user import UserModel
from api.models.csv import CSVFileModel  # noqa: F401  (mapper seti)
from api.models.data_group import DataGroupModel  # noqa: F401
from api.models.mcp import MCPConfigModel  # noqa: F401
from api.models.job import BackgroundJobModel  # noqa: F401
from api.models.project import ProjectModel
from api.models.artifact import ArtifactModel
from api.models.prediction import PredictionModel
from api.models.monitoring_summary import MonitoringSummaryModel

from api.routes.monitoring import (
    monitoring_summary, monitoring_window_detail, monitoring_deployments, monitoring_latest,
)
from src.core.config import settings

_DAY = 86_400
_USER = types.SimpleNamespace(id=0)
_OTHER = types.SimpleNamespace(id=0)


def _mk_project(db, user_id, title):
    p = ProjectModel(title=title, status="completed", user_id=user_id)
    db.add(p); db.commit(); db.refresh(p)
    return p.id


def _mk_artifact(db, project_id, aid, champion=True, deployable=True, phase="complete"):
    db.add(ArtifactModel(id=aid, project_id=project_id, phase=phase, deployable=deployable,
                         model_path=f"data/models/{aid}.pkl", model_name="blend",
                         schema_version="1.2", is_champion=champion,
                         created_at=int(time.time()), updated_at=int(time.time())))
    db.commit()


def _mk_predictions(db, project_id, aid, n):
    now = int(time.time())
    db.bulk_insert_mappings(PredictionModel, [
        {"prediction_id": uuid.uuid4().hex, "project_id": project_id, "artifact_id": aid,
         "created_at": now, "prediction": 1, "probability": 0.5} for _ in range(n)])
    db.commit()


def _per_feature(n_feat=20):
    return {f"col_{i}": {"psi": round(0.01 * i, 4), "n_bins": 6, "degenerate": False,
                         "reason": None, "importance_weight": round(1.0 / n_feat, 4)}
            for i in range(n_feat)}


def _mk_window(db, project_id, aid, window_start, n_total=600, n_sampled=600, per_feature=True,
               labeled_n=250):
    coverage = round(labeled_n / n_total, 6) if n_total else 0.0
    db.add(MonitoringSummaryModel(
        project_id=project_id, artifact_id=aid, window_start=window_start,
        window_end=window_start + _DAY, window_kind="daily", computed_at=time.time(),
        n_total=n_total, n_sampled=n_sampled, status="ok",
        psi_overall=0.05, psi_max=0.19, psi_degenerate_count=0,
        importance_attributed_ratio=1.0, degenerate_importance_mass=0.0,
        psi_per_feature=(_per_feature() if per_feature else None),
        cbpe_estimate={"estimated_roc_auc": 0.83, "estimated_accuracy": 0.79,
                       "baseline_roc_auc": 0.86, "baseline_accuracy": 0.80,
                       "assumes": "covariate_drift_only"},
        cbpe_gate_status="ok",
        quality_rates={"unseen": {"count": 6, "rate": 0.01},
                       "out_of_range": {"count": 0, "rate": 0.0}, "n": n_total, "alarm_eligible": True},
        label_coverage=coverage,
        realized_metrics={"accuracy": 0.80, "roc_auc": 0.84, "baseline_accuracy": 0.80,
                          "baseline_roc_auc": 0.86, "labeled_n": labeled_n,
                          "labeled_positive_rate": 0.32, "baseline_positive_rate": 0.30,
                          "positive_rate_deviation": 0.02, "assumes": "labels_missing_at_random"},
        realized_gate_status="ok",
        # Faz 0: triage rozeti + hafif scalars (list) + overlay distribution (detay, deferred)
        overall_status="warning",
        overall_status_reason={"driver": "drift", "code": "psi_drift_warning", "value": {"psi_max": 0.19},
                               "core_unmeasured": False, "suppressed_axes": {},
                               "measurement_coverage": {"importance_attributed_ratio": 1.0,
                                                        "degenerate_importance_mass": 0.0, "adequate": True}},
        psi_scalars=({f"col_{i}": round(0.01 * i, 4) for i in range(20)} if per_feature else None),
        psi_distribution=({f"col_{i}": {"kind": "continuous", "edges": [None, 10.0, 20.0, None],
                                        "ref_props": [0.25, 0.5, 0.25], "prod_counts": [150, 300, 150],
                                        "prod_total": 600} for i in range(20)} if per_feature else None),
    ))
    db.commit()


@pytest.fixture(scope="module")
def seeded():
    Base.metadata.create_all(bind=engine)
    now_day = (int(time.time()) // _DAY) * _DAY
    db = SessionLocal()
    try:
        u = UserModel(email="mon-api@local.test", hashed_password="x")
        o = UserModel(email="mon-api-other@local.test", hashed_password="x")
        db.add_all([u, o]); db.commit(); db.refresh(u); db.refresh(o)
        _USER.id, _OTHER.id = u.id, o.id

        # pid_full: champion + eski artifact + 30 günlük pencere + predictions
        pid_full = _mk_project(db, u.id, "full")
        _mk_artifact(db, pid_full, "champ", champion=True)
        _mk_artifact(db, pid_full, "old", champion=False, deployable=True)   # champion değil
        _mk_predictions(db, pid_full, "champ", 50)
        for d in range(1, 31):                       # son 30 kapanmış gün
            _mk_window(db, pid_full, "champ", now_day - d * _DAY, per_feature=True)
        # eski artifact'a da bir pencere (deployments'ta görünsün)
        _mk_window(db, pid_full, "old", now_day - 40 * _DAY, per_feature=False)

        # boş reason projeleri
        pid_nochamp = _mk_project(db, u.id, "nochamp")
        _mk_artifact(db, pid_nochamp, "nc", champion=False, deployable=False, phase="features")
        pid_nopred = _mk_project(db, u.id, "nopred")
        _mk_artifact(db, pid_nopred, "np", champion=True)
        pid_noclosed = _mk_project(db, u.id, "noclosed")
        _mk_artifact(db, pid_noclosed, "ncw", champion=True)
        _mk_predictions(db, pid_noclosed, "ncw", 10)
    finally:
        db.close()

    yield {"pid_full": pid_full, "pid_nochamp": pid_nochamp, "pid_nopred": pid_nopred,
           "pid_noclosed": pid_noclosed, "now_day": now_day}

    db = SessionLocal()
    try:
        for pid in (pid_full, pid_nochamp, pid_nopred, pid_noclosed):
            db.query(MonitoringSummaryModel).filter(MonitoringSummaryModel.project_id == pid).delete()
            db.query(PredictionModel).filter(PredictionModel.project_id == pid).delete()
            db.query(ArtifactModel).filter(ArtifactModel.project_id == pid).delete()
            db.query(ProjectModel).filter(ProjectModel.id == pid).delete()
        db.query(UserModel).filter(UserModel.id.in_([_USER.id, _OTHER.id])).delete(
            synchronize_session=False)
        db.commit()
    finally:
        db.close()


def _call(fn, *args, **kw):
    db = SessionLocal()
    try:
        return fn(*args, db=db, current_user=_USER, **kw)
    finally:
        db.close()


def test_summary_list_shape_and_order(seeded):
    res = _call(monitoring_summary, seeded["pid_full"], days=30, artifact_id=None)
    assert res["window_timezone"] == "UTC"
    assert res["days_applied"] == 30
    w = res["windows"]
    assert len(w) == 30                                    # eski artifact penceresi 40 gün önce → cutoff dışı
    # sıra artan window_start
    assert [x["window_start"] for x in w] == sorted(x["window_start"] for x in w)
    # per_feature YOK (yanıt şişmesin); ama overall/max VAR (trend)
    assert "per_feature" not in w[0]["psi"]
    assert "distribution" not in w[0]["psi"]               # ağır overlay list'te YOK (deferred)
    assert w[0]["psi"]["overall"] == 0.05 and w[0]["psi"]["max"] == 0.19
    # Faz 0 item 3: hafif scalars list'te VAR (sparkline); Faz 0 item 1: triage rozeti VAR
    assert w[0]["psi"]["scalars"] is not None and len(w[0]["psi"]["scalars"]) == 20
    assert w[0]["overall_status"] == "warning"
    assert w[0]["overall_status_reason"]["code"] == "psi_drift_warning"
    # disiplin 1: her blok kendi n'i
    assert w[0]["psi"]["n"] == 600 and w[0]["cbpe"]["n"] == 600 and w[0]["realized"]["n"] == 250
    # disiplin 2: cbpe ↔ realized AYRI blok + assumes damgası
    assert w[0]["cbpe"]["estimate"]["assumes"] == "covariate_drift_only"
    assert w[0]["realized"]["metrics"]["assumes"] == "labels_missing_at_random"
    # disiplin 4: durum rozetleri
    assert w[0]["status"] == "ok" and w[0]["cbpe"]["gate_status"] == "ok"
    # disiplin 3: config bloğu eşikleri
    assert res["config"]["min_n_realized"] == settings.MONITORING_MIN_N_REALIZED
    assert res["config"]["max_window_days"] == settings.MONITORING_MAX_WINDOW_DAYS
    # label_coverage HER ZAMAN
    assert w[0]["realized"]["label_coverage"] == round(250 / 600, 6)


def test_days_clamped_to_max(seeded):
    res = _call(monitoring_summary, seeded["pid_full"], days=3650, artifact_id=None)
    assert res["days_requested"] == 3650
    assert res["days_applied"] == settings.MONITORING_MAX_WINDOW_DAYS      # 90'a kırpıldı (hata değil)
    res2 = _call(monitoring_summary, seeded["pid_full"], days=None, artifact_id=None)
    assert res2["days_applied"] == settings.MONITORING_DEFAULT_WINDOW_DAYS  # varsayılan


def test_window_detail_has_per_feature(seeded):
    ws = seeded["now_day"] - 5 * _DAY
    res = _call(monitoring_window_detail, seeded["pid_full"], ws, artifact_id="champ")
    assert res["window_timezone"] == "UTC"
    assert len(res["windows"]) == 1
    pf = res["windows"][0]["psi"]["per_feature"]
    assert pf is not None and len(pf) == 20                # detay per_feature DAHİL
    assert pf["col_0"]["importance_weight"] == round(1 / 20, 4)
    # Faz 0 item 2: distribution overlay DETAYDA var (deferred → undefer ile yüklendi)
    dist = res["windows"][0]["psi"]["distribution"]
    assert dist is not None and len(dist) == 20
    assert dist["col_0"]["kind"] == "continuous"
    assert dist["col_0"]["edges"][0] is None               # ±inf → null (JSON-güvenli)


def test_window_detail_404(seeded):
    with pytest.raises(HTTPException) as ei:
        _call(monitoring_window_detail, seeded["pid_full"], 12345, artifact_id="champ")
    assert ei.value.status_code == 404


def test_deployments_champion_and_counts(seeded):
    res = _call(monitoring_deployments, seeded["pid_full"])
    assert res["window_timezone"] == "UTC"
    assert res["champion"]["artifact_id"] == "champ"
    by_id = {d["artifact_id"]: d for d in res["deployments"]}
    assert set(by_id) == {"champ", "old"}
    assert by_id["champ"]["is_champion"] is True
    assert by_id["champ"]["total_predictions"] == 50       # indexli COUNT
    assert by_id["champ"]["window_count"] == 30
    assert by_id["champ"]["last_window"]["status"] == "ok"
    assert by_id["champ"]["last_window"]["overall_status"] == "warning"   # portfolio triage rozeti
    assert by_id["old"]["is_champion"] is False
    assert by_id["old"]["window_count"] == 1


def test_latest_has_data(seeded):
    res = _call(monitoring_latest, seeded["pid_full"])
    assert res["has_data"] is True and res["reason"] is None
    assert res["champion_artifact_id"] == "champ"
    assert res["window"]["psi"]["per_feature"] is not None  # latest per_feature dahil
    assert res["window"]["psi"]["distribution"] is not None  # latest distribution dahil (deferred undefer)
    assert res["window"]["overall_status"] == "warning"      # hero triage rozeti
    # en son pencere = now_day - 1*DAY (en yakın kapanmış)
    assert res["window"]["window_start"] == seeded["now_day"] - _DAY


def test_latest_empty_reasons_distinct(seeded):
    r1 = _call(monitoring_latest, seeded["pid_nochamp"])
    # nochamp artifact phase=features (complete+deployable DEĞİL) → serve edilebilir aday YOK → not_trained.
    assert r1["has_data"] is False and r1["reason"] == "not_trained"
    assert r1["champion_artifact_id"] is None
    assert r1["promotable_artifact_id"] is None

    r2 = _call(monitoring_latest, seeded["pid_nopred"])
    assert r2["has_data"] is False and r2["reason"] == "no_predictions"
    assert r2["champion_artifact_id"] == "np"

    r3 = _call(monitoring_latest, seeded["pid_noclosed"])
    assert r3["has_data"] is False and r3["reason"] == "no_closed_window"
    assert r3["champion_artifact_id"] == "ncw"


def test_ownership_404(seeded):
    db = SessionLocal()
    try:
        with pytest.raises(HTTPException) as ei:
            monitoring_summary(seeded["pid_full"], days=30, artifact_id=None,
                               db=db, current_user=_OTHER)        # proje _USER'ın
        assert ei.value.status_code == 404
    finally:
        db.close()


def test_response_size_measurement(seeded, capsys):
    """per_feature'ı A'dan çıkarma kararının ÖLÇÜMÜ: 30 pencere summary (per_feature YOK) vs
    aynı 30 pencere per_feature DAHİL olsaydı. Karar gerekçesi sayı olsun."""
    res = _call(monitoring_summary, seeded["pid_full"], days=30, artifact_id=None)
    without = len(json.dumps(res).encode("utf-8"))

    # hipotetik: her pencereye 20-feature per_feature enjekte et → "A'da olsaydı" boyutu
    hypo = json.loads(json.dumps(res))
    for w in hypo["windows"]:
        w["psi"]["per_feature"] = _per_feature()
    with_pf = len(json.dumps(hypo).encode("utf-8"))

    with capsys.disabled():
        print(f"\n[monitoring-api] summary 30 pencere: per_feature YOK={without/1024:.1f}KB  "
              f"per_feature DAHİL(hipotetik)={with_pf/1024:.1f}KB  "
              f"oran={with_pf/without:.1f}×")
    assert without < with_pf                                # per_feature çıkarımı yanıtı küçültür (karar doğru)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-s"]))
