"""Katman 1 artifact içerik-üreticileri — model_ref + training_reference (task #5).

SINIR (§2.20): bu tur yalnız MEKANİZMA — üreten fonksiyonlar + Artifact alanları + round-trip.
Wiring (fit_pipeline'ın çağırması, evaluation'ın baseline beslemesi) AYRI iş.
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.services.inference.protocol import Artifact
from src.services.inference.references import (
    ModelFileMissing, _REF_MAX_VOCAB, build_model_ref, build_training_reference,
    resolve_model_path,
)
from src.services.inference.serialize import (
    _SERIALIZED_FIELDS, load_artifact, save_artifact,
)


# ── model_ref ───────────────────────────────────────────────────

def test_model_ref_is_reference_not_copy():
    """KARAR: REFERANS (model 147MB). Yol taşınır, dosya kopyalanmaz; byte_size kaydedilir."""
    with tempfile.TemporaryDirectory() as d:
        mp = os.path.join(d, "model.pkl")
        with open(mp, "wb") as fh:
            fh.write(b"x" * 2048)
        ref = build_model_ref(mp, "blend", 0.61,
                              {"value": 0.61, "source": "oof_cv_f1max", "n": 900}, "1.5.0")
        assert ref["model_path"] == mp and ref["byte_size"] == 2048
        assert ref["threshold_provenance"]["source"] == "oof_cv_f1max"
        assert ref["optimal_threshold"] == 0.61 and type(ref["optimal_threshold"]) is float


def test_model_missing_fails_closed():
    """Referansın kırılganlık kontrolü — dosya yoksa kullanım anında abort (yükleme değil)."""
    ref = build_model_ref("data/models/YOK_olmayan.pkl", "blend", 0.5, None, "1.5.0")
    with pytest.raises(ModelFileMissing):
        resolve_model_path(ref)


def test_model_ref_carries_sklearn_version_for_gate():
    """Dördüncü sürüm-kapısı müşterisi: model pickle'ı METADATA düzeyinde sürüm taşır
    (147MB estimator OKUNMADAN karşılaştırılabilsin)."""
    ref = build_model_ref(None, "lr", 0.5, None, "1.5.2")
    assert ref["sklearn_version"] == "1.5.2"


# ── training_reference ──────────────────────────────────────────

def _raw():
    rs = np.random.RandomState(0)
    return pd.DataFrame({
        "cont": rs.normal(10, 3, 200),
        "cat_low": (["a", "b", "c"] * 200)[:200],
        "cat_high": [f"v{i}" for i in range(200)],   # 200 benzersiz → cap tetikler
        "drop_me": range(200),                        # id-benzeri, columns'a KOYULMAZ
    })


def test_training_reference_two_orthogonal_halves():
    """per_feature (PSI, raw-space) + baseline_metrics (CBPE, prediction-space) AYRI."""
    ref = build_training_reference(_raw(), columns=["cont", "cat_low"],
                                   baseline_metrics={"roc_auc": 0.9, "prevalence": 0.3})
    assert ref["space"] == "raw"
    assert set(ref["per_feature"]) == {"cont", "cat_low"}, "columns dışı kolon (drop_me) girmemeli"
    assert ref["baseline_metrics"]["roc_auc"] == 0.9
    # ortogonal: baseline_metrics per_feature'a SIZMAZ
    assert "roc_auc" not in ref["per_feature"]


def test_continuous_and_categorical_stats():
    ref = build_training_reference(_raw(), columns=["cont", "cat_low"])
    c = ref["per_feature"]["cont"]
    assert c["kind"] == "continuous" and set(c["quantiles"]) == {"0.01", "0.25", "0.5", "0.75", "0.99"}
    assert abs(c["mean"] - _raw()["cont"].mean()) < 1e-9
    k = ref["per_feature"]["cat_low"]
    assert k["kind"] == "categorical"
    assert abs(sum(k["value_counts_normalized"].values()) - 1.0) < 1e-9


def test_categorical_cap_records_truncation():
    """Yüksek-kard cap: vocab_truncated + tail_mass KAYDEDİLİR (Katman 2 'other' kovası)."""
    ref = build_training_reference(_raw(), columns=["cat_high"])
    k = ref["per_feature"]["cat_high"]
    assert len(k["value_counts_normalized"]) == _REF_MAX_VOCAB, "cap uygulanmalı"
    assert k["vocab_truncated"] == [200, _REF_MAX_VOCAB], "kesilen sayı kaydedilmeli"
    assert k["tail_mass"] > 0, "kesilen kütle (PSI other kovası) kaydedilmeli"
    # tutulan + tail = 1.0 (normalize bütünlüğü)
    assert abs(sum(k["value_counts_normalized"].values()) + k["tail_mass"] - 1.0) < 1e-9


def test_baseline_metrics_none_accepted():
    """Fit anında baseline_metrics YOK (evaluation sonra) → None kabul, §2.20 kilidi korunur."""
    ref = build_training_reference(_raw(), columns=["cont"])
    assert ref["baseline_metrics"] is None


def test_signature_kind_is_type_source_not_dtype():
    """🔴 BUG FIX: tip-kararı imzadan (kardinalite/domain), dtype'tan DEĞİL. Ayrık int kolonu
    (products_number gibi) salt-dtype'ta continuous → dejenere quantile → SAHTE PSI şişmesi.
    İmza onu categorical der → value_counts referansı. binary de categorical'a düşer (0/1 quantile
    dejenere olurdu). Continuous olan yüksek-kard numerik (age gibi) imzada da continuous kalır."""
    rs = np.random.RandomState(1)
    raw = pd.DataFrame({
        "products_number": rs.randint(1, 5, 400),      # ayrık int, nunique=4 → categorical olmalı
        "active_member": rs.randint(0, 2, 400),        # 0/1 → binary → categorical stat
        "age": rs.randint(18, 92, 400),                # yüksek-kard int → continuous
    })
    sig = {"columns": {
        "products_number": {"kind": "categorical"},
        "active_member": {"kind": "binary"},
        "age": {"kind": "continuous"},
    }}
    ref = build_training_reference(raw, columns=list(raw.columns), signature=sig)
    pf = ref["per_feature"]
    # imza kararı dtype'ı EZER: iki ayrık int categorical (dejenere quantile YOK)
    assert pf["products_number"]["kind"] == "categorical"
    assert "quantiles" not in pf["products_number"]
    assert pf["active_member"]["kind"] == "categorical", "binary → value_counts (dejenere quantile önlenir)"
    assert pf["age"]["kind"] == "continuous", "yüksek-kard numerik continuous kalır"
    # imzasız AYNI kolon (dtype fallback) → products_number continuous olurdu (regresyonu kanıtla)
    ref_dtype = build_training_reference(raw, columns=["products_number"])
    assert ref_dtype["per_feature"]["products_number"]["kind"] == "continuous", \
        "imzasız fallback dtype'a düşer (bug'ın eski davranışı — fix imza yolunda)"


def test_reference_cap_matches_signature_cap():
    """_REF_MAX_VOCAB, raw_input_signature'ın _SIG_MAX_VOCAB'ıyla hizalı (ikisi ayrılırsa drift)."""
    from src.services.inference.orchestrator import _SIG_MAX_VOCAB
    assert _REF_MAX_VOCAB == _SIG_MAX_VOCAB


# ── round-trip: iki alan da dolu ────────────────────────────────

def test_roundtrip_covers_both_fields():
    """model_ref + training_reference dolu → save/load alan-bazlı birebir (whitelist otomatik kapsar)."""
    with tempfile.TemporaryDirectory() as d:
        mp = os.path.join(d, "m.pkl")
        open(mp, "wb").write(b"y" * 512)
        art = Artifact(
            manifest=[{"id": "x", "order": 0, "type": "rowwise", "delta": {"added": [], "removed": []}}],
            final_features=["f"],
            model_ref=build_model_ref(mp, "blend", 0.55,
                                      {"value": 0.55, "source": "oof_cv_f1max", "n": 100}, "1.5.0"),
            training_reference=build_training_reference(
                _raw(), columns=["cont", "cat_low", "cat_high"],
                baseline_metrics={"roc_auc": 0.88, "brier": 0.12, "prevalence": 0.3}),
        )
        aid = save_artifact(art, base_dir=d)
        art2 = load_artifact(aid, base_dir=d)
        assert art2.model_ref == art.model_ref, "model_ref round-trip'te değişti"
        assert art2.training_reference == art.training_reference, "training_reference round-trip'te değişti"
        # whitelist gerçekten kapsıyor mu
        assert "model_ref" in _SERIALIZED_FIELDS and "training_reference" in _SERIALIZED_FIELDS
