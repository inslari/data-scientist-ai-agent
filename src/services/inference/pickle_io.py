from __future__ import annotations

"""Paylaşılan pickle-ref + sklearn sürüm-kapısı — TÜM pickle adımlarının tek kapısı.

§1.1 şemasının ``param_store`` "pickle-ref, sklearn versiyonuyla" alanının somutu. Üç pickle
adımı (TargetEncode, RobustScale, StandardScale) fitted sklearn estimator'ını ``PickleRef``
olarak saklar; kullanım-anında ``load_pickle_ref`` TEK sürüm-kapısından geçirir → sürüm-kontrol
tek yerde, üç tüketici. Yeni pickle adımı gelince gate'i yeniden yazmaz, buraya bağlanır.

SERIALIZE ÇAPASI (task #6): ``PickleRef`` şimdi in-memory dataclass; serialize katmanı onu
JSON-metadata (``kind`` + ``sklearn_version``) + ayrı ``.pkl`` blob'a yazacak/okuyacak →
serialize katmanı "PickleRef yaz/oku"a indirgenir. Bu yüzden sürüm alanı ilk günden PickleRef'te.

Sürüm-kapısı politikası (C, onaylı):
  * patch farkı      → "warn"  (devam; patch sınırında pickle-binary uyumu beklenir)
  * minor|major fark → "abort" (FAIL-CLOSED; model HİÇ skorlamaz — yanlış skorlamaktansa dur).
    sklearn pickle-binary uyumunu minor sınırında GARANTİ ETMEZ (public-API-stable != pickle-stable).
"""

import re
from dataclasses import dataclass
from typing import Any

import sklearn

from src.core.logging_config import get_logger

logger = get_logger(__name__)


class SklearnVersionMismatch(RuntimeError):
    """sklearn pickle sürüm uyumsuzluğu — fail-closed abort (model skorlamamalı)."""


def _parse_semver(v: str) -> tuple:
    nums = []
    for part in (v or "0").split(".")[:3]:
        m = re.match(r"\d+", part)
        nums.append(int(m.group()) if m else 0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)


def check_sklearn_compat(stored: str, runtime: str) -> str:
    """Saf karar: ``"ok"`` | ``"warn"`` | ``"abort"``. minor|major fark → abort, patch → warn."""
    smaj, smin, spat = _parse_semver(stored)
    rmaj, rmin, rpat = _parse_semver(runtime)
    if (smaj, smin) != (rmaj, rmin):
        return "abort"
    if spat != rpat:
        return "warn"
    return "ok"


def enforce_sklearn_compat(stored: str, runtime: str) -> None:
    """Fail-closed kapı: abort ise ``SklearnVersionMismatch`` FIRLAT (estimator dönmeden →
    model hiç skorlamaz); warn ise logla-devam et."""
    verdict = check_sklearn_compat(stored, runtime)
    if verdict == "abort":
        raise SklearnVersionMismatch(
            f"sklearn pickle uyumsuz: model sklearn {stored} ile egitildi, runtime {runtime}. "
            f"Minor/major sinirinda pickle-binary uyumu GARANTI DEGIL (public-API-stable != "
            f"pickle-stable). FAIL-CLOSED: yanlis skorlamaktansa DUR. Cozum: modeli sklearn "
            f"{runtime} ile YENIDEN EGIT ya da runtime'i {stored} ile ESLE."
        )
    if verdict == "warn":
        logger.warning(
            f"[pickle_io] sklearn patch farki: egitim {stored}, runtime {runtime}. "
            f"Devam (patch sinirinda pickle uyumu beklenir), ama izle."
        )


@dataclass
class PickleRef:
    """Fitted sklearn estimator + eğitildiği sklearn sürümü. param_store'da pickle-ref hücresi."""

    estimator: Any
    sklearn_version: str
    kind: str = "sklearn_pickle"


def make_pickle_ref(estimator: Any) -> PickleRef:
    """Fit-anı: estimator'ı o anki sklearn sürümüyle mühürle."""
    return PickleRef(estimator=estimator, sklearn_version=sklearn.__version__)


def load_pickle_ref(ref: PickleRef) -> Any:
    """Kullanım-anı: TEK sürüm-kapısı (fail-closed) sonra estimator'ı döndür. Sürüm uyumluysa
    ref.estimator'ı OLDUĞU GİBİ döndürür (kimlik korunur → soyutlama davranışa hiçbir şey eklemez;
    yalnız kapı). Uyumsuzsa ``SklearnVersionMismatch`` → estimator dönmez, skorlama olmaz."""
    enforce_sklearn_compat(ref.sklearn_version, sklearn.__version__)
    return ref.estimator
