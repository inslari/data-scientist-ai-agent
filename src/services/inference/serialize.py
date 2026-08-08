from __future__ import annotations

"""Artifact ⇄ disk — Katman 1 serialize katmanı (task #6).

``pickle_io`` estimator-DÜZEYİ kapı (tek estimator + sürüm); bu modül artifact-DÜZEYİ:
tüm ``Artifact``'i JSON-metadata + ayrı ``.pkl`` blob'larına yazar/okur. Ayrı sorumluluk,
ayrı dosya.

DÜZEN (``dump_split`` deseniyle hizalı):
    data/artifacts/<artifact_id>/
    ├── artifact.json      # WHITELIST'teki tüm inline metadata (aşağıda)
    ├── target_encode.pkl  # PickleRef.estimator (run'a bağlı — yoksa dosya da yok)
    ├── robust_scale.pkl
    └── standard_scale.pkl

İKİ SÜRÜM KAPISI (ikisi de FAIL-CLOSED):
  * ``artifact_schema_version`` — yükleyici uyumsuz major/minor'da ``ArtifactSchemaMismatch``.
  * pickle ``sklearn_version`` — ``load_pickle_ref`` disk yolunda da AYNI kapıdan geçer
    (``enforce_sklearn_compat``); sahte sürüm → ``SklearnVersionMismatch``, estimator dönmez.

WHITELIST (kod düzeyinde garanti): ``Artifact``'in yalnız ``_SERIALIZED_FIELDS`` alanları diske
gider. İleride Artifact'e sızan bir alan (observer referansı, fit-only ctx, geçici state)
OTOMATİK dışarıda kalır — blacklist'te olduğu gibi "eklemeyi unutma" riski yok.

JSON SANITIZE FAIL-CLOSED (default= fallback YOK): numpy/pandas tipleri AÇIK ve TİPLİ çevrilir
(``np.int64→int`` vb.); tanınmayan tip → ``TypeError``. ``default=str`` gibi bir fallback numpy'ı
SESSİZCE string'e çevirir, disk turunda tip değişir, ``transform`` yanlış hesaplar. NaN/Inf de
``allow_nan=False`` ile reddedilir (JSON-standart-dışı; başka bir okuyucu kırılır).
"""

import json
import os
import pickle
import tempfile
import uuid
from dataclasses import fields

import numpy as np
import pandas as pd

from src.core.logging_config import get_logger
from src.services.inference.pickle_io import (
    PickleRef, enforce_sklearn_compat, make_pickle_ref,
)
from src.services.inference.protocol import Artifact

logger = get_logger(__name__)

_ARTIFACT_JSON = "artifact.json"
_PICKLE_SENTINEL = "__pickle_ref__"

# WHITELIST — diske giden Artifact alanları. Buraya EKLEMEDİĞİN alan serialize EDİLMEZ.
# (training_reference / model_ref şimdilik BOŞ ama şema yer tutar — şema önce, içerik sonra.)
_SERIALIZED_FIELDS = (
    "manifest", "param_store", "raw_input_signature", "supported_scope",
    "tolerated_extra_columns", "unseen_category_policy", "training_reference",
    "model_ref", "final_features", "phase", "business_key_column", "artifact_schema_version",
)

# update_artifact_fields'in DOKUNABİLECEĞİ alanlar — Faz B/C yalnız bunları değiştirir.
# param_store (dolayısıyla .pkl blob'ları) Faz A'da DONAR → burada YOK → blob yeniden-yazma
# yapısal olarak imkânsız. Başka alan patch'lenmeye çalışılırsa reddedilir (fail-closed).
_UPDATABLE_FIELDS = ("model_ref", "training_reference", "phase")


class ArtifactSchemaMismatch(RuntimeError):
    """Diskteki artifact şeması yükleyicinin desteklediğiyle uyumsuz — FAIL-CLOSED.

    ``sklearn`` sürüm kapısıyla aynı disiplin: yanlış yorumlanmış bir artifact'le
    skorlamaktansa DUR. Yalnız major.minor karşılaştırılır (patch geriye-uyumlu sayılır)."""


def _schema_compatible(stored: str, supported: str) -> bool:
    def mm(v):
        p = (v or "0").split(".")
        return (int(p[0]) if p and p[0].isdigit() else 0,
                int(p[1]) if len(p) > 1 and p[1].isdigit() else 0)
    return mm(stored) == mm(supported)


# ── JSON sanitize — TİPLİ, FAIL-CLOSED ─────────────────────────────

def _to_jsonable(o):
    """numpy/pandas → yerel Python; ``PickleRef`` → sentinel. Tanınmayan tip → TypeError.

    NaN/Inf burada YAKALANMAZ (değer olabilir); ``json.dumps(allow_nan=False)`` onları
    reddeder → serialize noktasında görünür olur (sessiz round-trip değil)."""
    if isinstance(o, PickleRef):
        return {_PICKLE_SENTINEL: o.kind, "sklearn_version": o.sklearn_version}
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    # ⚠️ numpy tipleri Python tiplerinden ÖNCE gelmeli: np.float64 `float`'ın ALT SINIFI,
    # np.bool_ değil ama int-benzeri numpy `int` alt sınıfı olabilir → `isinstance(o, float)`
    # numpy'ı sessizce yakalayıp OLDUĞU GİBİ döndürürdü (fail-closed'un tam tersi, disk turunda
    # tip kayması). Bu yüzden numpy kontrolleri yerel-tip kontrolünden ÖNCE.
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return [_to_jsonable(v) for v in o.tolist()]
    if isinstance(o, (str, bool, int, float, type(None))):
        return o
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    raise TypeError(
        f"serialize: JSON'a çevrilemeyen tip {type(o).__name__} (değer={o!r}). "
        f"Açık bir dönüşüm kuralı YOK — fail-closed (default= fallback numpy'ı sessizce "
        f"string'e çevirir, disk turunda tip kayar). Yeni tip → _to_jsonable'a kural ekle."
    )


def _extract_pickles(param_store: dict) -> "tuple[dict, dict]":
    """param_store'u gez: her ``PickleRef`` hücresini ayrı blob'a AL, yerine sentinel bırak.
    Dönüş: (json-temiz param_store, {step_id: PickleRef})."""
    blobs: dict = {}

    def walk(o, yol):
        if isinstance(o, PickleRef):
            blobs[yol] = o
            return {_PICKLE_SENTINEL: o.kind, "sklearn_version": o.sklearn_version}
        if isinstance(o, dict):
            return {k: walk(v, f"{yol}.{k}") for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [walk(v, f"{yol}[{i}]") for i, v in enumerate(o)]
        return o

    temiz = {sid: walk(v, sid) for sid, v in param_store.items()}
    return temiz, blobs


def _blob_name(step_id: str) -> str:
    """PickleRef step_id'sinden güvenli dosya adı (üst-düzey adım; iç içe pickle YOK bugün)."""
    return step_id.split(".")[0] + ".pkl"


# ── ATOMİK YAZMA — temp dosyaya yaz + os.replace (POSIX & Windows'ta atomik, üzerine yazar) ──
# Yazma ORTASINDA hata olursa hedef dosya BOZULMAZ (ya eski hali ya yeni hali, yarısı asla).
# Faz B/C aynı artifact.json'u yeniden yazdığı için (yaz-oku-yaz) bu şart: kesinti bir sonraki
# fazın yüklemesini kırmasın.

def _atomic_write_json(path: str, meta: dict) -> None:
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, allow_nan=False)  # NaN/Inf → ValueError (fail-closed)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _atomic_write_pickle(path: str, obj) -> None:
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".pkltmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_artifact(artifact: Artifact, base_dir: str = "data/artifacts") -> str:
    """Artifact'i ``base_dir/<id>/``'e yaz. Dönüş: ``artifact_id``.

    param_store'un PickleRef hücreleri ayrı ``.pkl``'e; kalan her şey ``artifact.json``'a
    (tipli sanitize + allow_nan=False). WHITELIST: yalnız ``_SERIALIZED_FIELDS``."""
    artifact_id = uuid.uuid4().hex
    adir = os.path.join(base_dir, artifact_id)
    os.makedirs(adir, exist_ok=True)

    meta: dict = {}
    for f in _SERIALIZED_FIELDS:
        val = getattr(artifact, f)
        if f == "param_store":
            temiz, blobs = _extract_pickles(val)
            meta[f] = _to_jsonable(temiz)
            for sid, ref in blobs.items():
                _atomic_write_pickle(os.path.join(adir, _blob_name(sid)), ref.estimator)
        else:
            meta[f] = _to_jsonable(val)

    _atomic_write_json(os.path.join(adir, _ARTIFACT_JSON), meta)   # NaN/Inf → ValueError (fail-closed)

    logger.info(f"[serialize] artifact yazıldı: {adir} "
                f"({len(meta['param_store'])} param, {len(os.listdir(adir))-1} pkl)")
    return artifact_id


def _restore_pickles(param_store: dict, adir: str) -> dict:
    """Sentinel'leri geri PickleRef'e çevir: ``.pkl``'i oku + sürüm-kapısını (fail-closed) uygula."""
    def walk(o, yol):
        if isinstance(o, dict) and _PICKLE_SENTINEL in o:
            sv = o.get("sklearn_version", "0")
            enforce_sklearn_compat(sv, __import__("sklearn").__version__)   # DİSK yolu, aynı kapı
            with open(os.path.join(adir, _blob_name(yol)), "rb") as fh:
                est = pickle.load(fh)
            ref = make_pickle_ref(est)     # runtime sürümüyle mühürle
            ref.sklearn_version = sv       # ...ama SAKLANAN sürümü koru (kapı zaten geçti)
            ref.kind = o[_PICKLE_SENTINEL]
            return ref
        if isinstance(o, dict):
            return {k: walk(v, f"{yol}.{k}") for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v, f"{yol}[{i}]") for i, v in enumerate(o)]
        return o
    return {sid: walk(v, sid) for sid, v in param_store.items()}


def load_artifact(artifact_id: str, base_dir: str = "data/artifacts") -> Artifact:
    """Diskten Artifact kur. İki fail-closed kapı: şema sürümü + pickle sklearn sürümü."""
    adir = os.path.join(base_dir, artifact_id)
    with open(os.path.join(adir, _ARTIFACT_JSON), "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    supported = Artifact().artifact_schema_version
    stored = meta.get("artifact_schema_version", "0")
    if not _schema_compatible(stored, supported):
        raise ArtifactSchemaMismatch(
            f"artifact şema {stored}, yükleyici {supported} destekliyor. FAIL-CLOSED: "
            f"yanlış yorumlanmış artifact'le skorlamaktansa DUR."
        )

    meta["param_store"] = _restore_pickles(meta.get("param_store") or {}, adir)
    kwargs = {f: meta[f] for f in _SERIALIZED_FIELDS if f in meta}
    return Artifact(**kwargs)


def update_artifact_fields(artifact_id: str, patch: dict, base_dir: str = "data/artifacts") -> str:
    """Var olan artifact'in YALNIZ ``_UPDATABLE_FIELDS`` (model_ref/training_reference/phase) alanlarını
    güncelle — Faz B (model_ref) ve Faz C (baseline_metrics + phase=complete) için.

    ``artifact.json``'u okur, patch'i üst-düzeyde overlay eder, ATOMİK yeniden yazar. **``.pkl``
    blob'larına ve ``param_store``'a DOKUNMAZ** (param_store Faz A'da donuyor; fonksiyon onu hiç
    görmüyor → "blob değişmez" garantisi kod-yapısal). İzin dışı alan patch'lenirse reddeder.

    STATUS YALAN SÖYLEYEMEZ: ``phase="complete"`` ancak ``model_ref`` DOLU **ve**
    ``training_reference.baseline_metrics`` VAR ise kabul edilir; değilse ``ValueError`` (fail-closed
    burada — çağıran fail-open sarar, artifact önceki fazda takılı kalır, servis kapısı reddeder)."""
    bad = set(patch) - set(_UPDATABLE_FIELDS)
    if bad:
        raise ValueError(f"update_artifact_fields: güncellenemez alan(lar) {sorted(bad)} "
                         f"(izinli: {list(_UPDATABLE_FIELDS)}) — param_store/manifest Faz A'da donar.")
    jpath = os.path.join(base_dir, artifact_id, _ARTIFACT_JSON)
    with open(jpath, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    for k, v in patch.items():
        meta[k] = _to_jsonable(v)                       # tipli sanitize (numpy → yerel)
    if meta.get("phase") == "complete":
        mref = meta.get("model_ref") or {}
        tref = meta.get("training_reference") or {}
        if not mref or tref.get("baseline_metrics") is None:
            raise ValueError(
                "phase=complete INVARIANT ihlali: model_ref DOLU ve baseline_metrics VAR olmalı "
                f"(model_ref={'dolu' if mref else 'BOŞ'}, "
                f"baseline_metrics={'var' if tref.get('baseline_metrics') is not None else 'YOK'})")
    _atomic_write_json(jpath, meta)
    logger.info(f"[serialize] artifact güncellendi: {artifact_id} alanlar={sorted(patch)} "
                f"phase={meta.get('phase')}")
    return artifact_id


class ArtifactNotServiceable(RuntimeError):
    """Artifact SERVİS için hazır değil → skorlama HİÇ yapılmaz (FAIL-CLOSED serving kapısı).

    ``load_artifact`` (permissive) Faz B/C'nin eksik artifact'i tamamlaması için her fazı yükler;
    bu ayrı kapı serving'e YALNIZ ``complete`` + model-dosyası-var + ``deployable`` artifact'i geçirir.
    "Güncellemek için yüklenebilir mi" ≠ "servis için hazır mı" — iki soru, iki kapı."""


def load_for_serving(artifact_id: str, base_dir: str = "data/artifacts") -> Artifact:
    """SERVİNG kapısı — üç şart (hepsi FAIL-CLOSED): (1) ``phase=="complete"``; (2)
    ``supported_scope.deployable``; (3) ``model_ref`` dosyası fiilen var. Biri düşerse
    ``ArtifactNotServiceable``. Şema kapısı ``load_artifact``'ta zaten işler."""
    from src.services.inference.references import resolve_model_path, ModelFileMissing
    art = load_artifact(artifact_id, base_dir)          # permissive load + şema kapısı
    reasons = []
    if art.phase != "complete":
        reasons.append(f"phase={art.phase!r} (complete değil — eksik faz)")
    if not (art.supported_scope or {}).get("deployable"):
        reasons.append("supported_scope.deployable=False (kapsam olgusu beyan edilmemiş/ihlal)")
    try:
        resolve_model_path(art.model_ref)
    except ModelFileMissing as e:
        reasons.append(str(e))
    if reasons:
        raise ArtifactNotServiceable(
            f"artifact {artifact_id} servis edilemez: " + " | ".join(reasons))
    return art
