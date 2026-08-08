"""FAZ 1 — monitoring/schema endpoint'i (scoring playground formu).

SENTETİK artifact (gerçek pipeline eğitmez → hızlı): elle kurulmuş ``raw_input_signature`` içeren bir
``artifact.json`` diske yazılır + dummy model dosyası (load_for_serving yalnız VARLIĞINI kontrol eder) →
champion yapılır → monitoring_schema çağrılır. Kapsam: 409 (champion yok) · domain eşlemesi (binary/
categorical/continuous) · known_values_truncated ÜÇ dal (yüksek-kardinalite / >max_known / <=max_known) ·
domain_check_policy + tolerated_extra_columns geçişi · config bloğu.
"""
import json
import os
import shutil
import types
import uuid

os.environ.setdefault("API_DATABASE_URL", "sqlite:///./data/_sbx_monitoring_schema_test.sqlite")

import pytest
from fastapi import HTTPException

from api.core.database import Base, SessionLocal, engine
from api.models.user import UserModel  # noqa: F401
from api.models.csv import CSVFileModel  # noqa: F401
from api.models.data_group import DataGroupModel  # noqa: F401
from api.models.mcp import MCPConfigModel  # noqa: F401
from api.models.job import BackgroundJobModel  # noqa: F401
from api.models.project import ProjectModel
from api.models.artifact import ArtifactModel

from api.routes.monitoring import monitoring_schema
from src.core.config import settings

_USER = types.SimpleNamespace(id=0)
_ARTIFACTS_DIR = os.path.join("data", "artifacts")


def _synthetic_signature():
    """Her kind'ı + üç known_values dalını kapsayan elle imza. max_known=20 varsayımıyla:
    cat_small(3)<=20 → dolu · cat_medium(30)>20 ama <=50 → truncated · cat_huge(None)=imza yüksek-kard → truncated."""
    cols = {
        "flag01": {"dtype_family": "number", "nunique": 2, "null_ratio": 0.0,
                   "kind": "binary", "allowed_values": ["0", "1"]},
        "cat_small": {"dtype_family": "object", "nunique": 3, "null_ratio": 0.01,
                      "kind": "categorical", "known_values": ["A", "B", "C"]},
        "cat_medium": {"dtype_family": "object", "nunique": 30, "null_ratio": 0.0,
                       "kind": "categorical", "known_values": [f"v{i}" for i in range(30)]},
        "cat_huge": {"dtype_family": "object", "nunique": 5000, "null_ratio": 0.0,
                     "kind": "categorical", "known_values": None},        # imza >50 → vocab saklanmamış
        "income": {"dtype_family": "number", "nunique": 900, "null_ratio": 0.02,
                   "kind": "continuous", "range": [0.0, 1_000_000.0]},
    }
    return {
        "columns": cols,
        "required": sorted(cols.keys()),
        "policy": {"binary": "strict_domain_abort", "categorical": "unseen_allowed_counted",
                   "continuous": "range_warn_only", "missing_required": "abort",
                   "undeclared_extra": "abort_unless_tolerated"},
        "signature_version": "1.0",
    }


def _write_artifact(aid: str, signature: dict, tolerated: list) -> None:
    adir = os.path.join(_ARTIFACTS_DIR, aid)
    os.makedirs(adir, exist_ok=True)
    model_path = os.path.join(adir, "model.pkl")
    with open(model_path, "wb") as fh:
        fh.write(b"\x00")                                  # resolve_model_path yalnız VARLIK ister
    meta = {
        "phase": "complete",
        "supported_scope": {"deployable": True},
        "model_ref": {"model_path": model_path, "model_name": "synthetic"},
        "raw_input_signature": signature,
        "tolerated_extra_columns": tolerated,
        "final_features": [], "param_store": {}, "manifest": [],
        "training_reference": {"baseline_metrics": {"accuracy": 0.8}},
        "artifact_schema_version": "1.2",
    }
    with open(os.path.join(adir, "artifact.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)


@pytest.fixture(scope="module")
def champ():
    Base.metadata.create_all(bind=engine)
    aid = uuid.uuid4().hex
    _write_artifact(aid, _synthetic_signature(), tolerated=["Row ID"])

    db = SessionLocal()
    try:
        u = UserModel(email="mon-schema@local.test", hashed_password="x")
        db.add(u); db.commit(); db.refresh(u)
        _USER.id = u.id
        p = ProjectModel(title="schema-test", status="completed", user_id=_USER.id)
        db.add(p); db.commit(); db.refresh(p)
        pid = p.id
        db.add(ArtifactModel(id=aid, project_id=pid, phase="complete", deployable=True,
                             model_path=os.path.join(_ARTIFACTS_DIR, aid, "model.pkl"),
                             model_name="synthetic", schema_version="1.2", is_champion=True))
        db.commit()
        # champion'ı olmayan ayrı proje (409 testi)
        p2 = ProjectModel(title="schema-nochamp", status="completed", user_id=_USER.id)
        db.add(p2); db.commit(); db.refresh(p2)
        pid_nochamp = p2.id
    finally:
        db.close()

    yield {"pid": pid, "aid": aid, "pid_nochamp": pid_nochamp}

    db = SessionLocal()
    try:
        db.query(ArtifactModel).filter(ArtifactModel.project_id == pid).delete()
        db.query(ProjectModel).filter(ProjectModel.id.in_([pid, pid_nochamp])).delete()
        db.query(UserModel).filter(UserModel.id == _USER.id).delete()
        db.commit()
    finally:
        db.close()
    shutil.rmtree(os.path.join(_ARTIFACTS_DIR, aid), ignore_errors=True)


def _call(pid):
    db = SessionLocal()
    try:
        return monitoring_schema(pid, db=db, current_user=_USER)
    finally:
        db.close()


def test_409_no_champion(champ):
    with pytest.raises(HTTPException) as ei:
        _call(champ["pid_nochamp"])
    assert ei.value.status_code == 409


def test_schema_shape_and_domains(champ):
    res = _call(champ["pid"])
    assert res["window_timezone"] == "UTC"
    assert res["champion_artifact_id"] == champ["aid"]
    assert res["tolerated_extra_columns"] == ["Row ID"]
    assert res["domain_check_policy"]["binary"] == "strict_domain_abort"
    assert res["config"]["schema_max_known_values"] == settings.MONITORING_SCHEMA_MAX_KNOWN_VALUES
    by = {c["name"]: c for c in res["columns"]}
    assert set(by) == {"flag01", "cat_small", "cat_medium", "cat_huge", "income"}
    # her kolon required (imza required = tüm kolonlar)
    assert all(c["required"] for c in res["columns"])
    # binary → allowed_values, domain=binary
    assert by["flag01"]["domain"] == "binary"
    assert by["flag01"]["allowed_values"] == ["0", "1"]
    # continuous → range_hint, domain=continuous
    assert by["income"]["domain"] == "continuous"
    assert by["income"]["range_hint"] == [0.0, 1_000_000.0]


def test_known_values_truncation_three_branches(champ):
    """Kullanıcı eklemesi (a): known_values_truncated ÜÇ dal."""
    res = _call(champ["pid"])
    by = {c["name"]: c for c in res["columns"]}
    # (1) küçük kardinalite (<=max_known) → dolu, truncated=false
    assert by["cat_small"]["known_values"] == ["A", "B", "C"]
    assert by["cat_small"]["known_values_truncated"] is False
    # (2) orta kardinalite (30 > max_known=20 ama imzada saklı) → null + truncated=true (dropdown pratik-dışı)
    assert by["cat_medium"]["known_values"] is None
    assert by["cat_medium"]["known_values_truncated"] is True
    # (3) yüksek kardinalite (imza None, >_SIG_MAX_VOCAB) → null + truncated=true
    assert by["cat_huge"]["known_values"] is None
    assert by["cat_huge"]["known_values_truncated"] is True


def test_max_known_values_threshold_is_config_driven(champ):
    """Eşik config'ten: MAX'ı 40'a çıkarınca cat_medium(30) artık DROPDOWN olur (truncated=false)."""
    old = settings.MONITORING_SCHEMA_MAX_KNOWN_VALUES
    settings.MONITORING_SCHEMA_MAX_KNOWN_VALUES = 40
    try:
        res = _call(champ["pid"])
        by = {c["name"]: c for c in res["columns"]}
        assert by["cat_medium"]["known_values"] is not None and len(by["cat_medium"]["known_values"]) == 30
        assert by["cat_medium"]["known_values_truncated"] is False
        # cat_huge imzada None → eşikten bağımsız HÂLÂ truncated (vocab hiç yok)
        assert by["cat_huge"]["known_values_truncated"] is True
    finally:
        settings.MONITORING_SCHEMA_MAX_KNOWN_VALUES = old


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-s"]))
