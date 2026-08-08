"""Katman 1 wiring — serialize yaşam döngüsü testleri (phase, iki kapı, update whitelist, atomiklik).

Faz B/C aynı artifact_id boyunca yaz-oku-yaz yapar; bu testler o mekanizmanın invariant'larını çitler:
  * `phase` üç değerli + şema 1.2 round-trip
  * `update_artifact_fields` YALNIZ whitelist alanları; `.pkl` blob'una DOKUNMAZ (param_store Faz A'da donar)
  * `phase="complete"` INVARIANT: model_ref DOLU ∧ baseline_metrics VAR (status yalan söyleyemez)
  * `load_for_serving` iki kapı: complete + deployable + model dosyası var
"""
import os
import time

import numpy as np
import pytest

from src.services.inference.protocol import Artifact
from src.services.inference.pickle_io import make_pickle_ref
from src.services.inference.serialize import (
    ArtifactNotServiceable, load_artifact, load_for_serving, save_artifact,
    update_artifact_fields,
)


def _mk(tmp, *, phase="features", model_ref=None, tref=None, scope=None, param_store=None):
    art = Artifact(
        manifest=[], param_store=param_store or {},
        raw_input_signature={"required": ["a"]},
        supported_scope=scope if scope is not None else {"deployable": True},
        final_features=["a"], phase=phase,
        model_ref=model_ref or {}, training_reference=tref or {},
    )
    return save_artifact(art, base_dir=tmp)


def test_phase_field_roundtrip(tmp_path):
    aid = _mk(str(tmp_path), phase="features")
    art = load_artifact(aid, base_dir=str(tmp_path))
    assert art.phase == "features"
    assert art.artifact_schema_version == "1.2"


def test_update_rejects_non_whitelist_fields(tmp_path):
    aid = _mk(str(tmp_path))
    with pytest.raises(ValueError):
        update_artifact_fields(aid, {"final_features": ["x"]}, base_dir=str(tmp_path))
    with pytest.raises(ValueError):
        update_artifact_fields(aid, {"param_store": {"z": 1}}, base_dir=str(tmp_path))


def test_update_is_json_only_pkl_blob_untouched(tmp_path):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(np.array([[1.0], [2.0], [3.0]]))
    aid = _mk(str(tmp_path), param_store={"robust_scale": make_pickle_ref(sc)})
    blob = os.path.join(str(tmp_path), aid, "robust_scale.pkl")
    assert os.path.exists(blob)
    mtime0 = os.path.getmtime(blob)
    time.sleep(0.02)
    update_artifact_fields(aid, {"model_ref": {"model_path": "x"}, "phase": "model"},
                           base_dir=str(tmp_path))
    assert os.path.getmtime(blob) == mtime0          # blob YENİDEN YAZILMADI


def test_phase_complete_invariant(tmp_path):
    aid = _mk(str(tmp_path))
    # model_ref boş + baseline yok → complete REDDEDİLİR
    with pytest.raises(ValueError):
        update_artifact_fields(aid, {"phase": "complete"}, base_dir=str(tmp_path))
    # model_ref var ama baseline yok → yine RED
    update_artifact_fields(aid, {"model_ref": {"model_path": "p"}, "phase": "model"},
                           base_dir=str(tmp_path))
    with pytest.raises(ValueError):
        update_artifact_fields(aid, {"phase": "complete"}, base_dir=str(tmp_path))
    # baseline eklenince → complete KABUL
    update_artifact_fields(aid, {"training_reference": {"baseline_metrics": {"roc_auc": 0.9}}},
                           base_dir=str(tmp_path))
    update_artifact_fields(aid, {"phase": "complete"}, base_dir=str(tmp_path))
    assert load_artifact(aid, base_dir=str(tmp_path)).phase == "complete"


def test_load_for_serving_rejects_incomplete(tmp_path):
    aid = _mk(str(tmp_path), phase="features")
    with pytest.raises(ArtifactNotServiceable):
        load_for_serving(aid, base_dir=str(tmp_path))


def test_load_for_serving_rejects_not_deployable(tmp_path):
    mp = tmp_path / "m.pkl"; mp.write_bytes(b"x")
    aid = _mk(str(tmp_path), phase="complete", model_ref={"model_path": str(mp)},
              tref={"baseline_metrics": {"a": 1}}, scope={"deployable": False})
    with pytest.raises(ArtifactNotServiceable):
        load_for_serving(aid, base_dir=str(tmp_path))


def test_load_for_serving_rejects_missing_model_file(tmp_path):
    aid = _mk(str(tmp_path), phase="complete",
              model_ref={"model_path": str(tmp_path / "nope.pkl")},
              tref={"baseline_metrics": {"a": 1}}, scope={"deployable": True})
    with pytest.raises(ArtifactNotServiceable):
        load_for_serving(aid, base_dir=str(tmp_path))


def test_load_for_serving_accepts_complete_deployable(tmp_path):
    mp = tmp_path / "m.pkl"; mp.write_bytes(b"x")
    aid = _mk(str(tmp_path), phase="complete", model_ref={"model_path": str(mp)},
              tref={"baseline_metrics": {"a": 1}, "per_feature": {}}, scope={"deployable": True})
    art = load_for_serving(aid, base_dir=str(tmp_path))
    assert art.phase == "complete"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
