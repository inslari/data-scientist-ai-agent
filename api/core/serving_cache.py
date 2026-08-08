"""Katman 1 skorlama model cache — process-içi LRU, per-key load-lock.

NEDEN: model 147MB'a kadar; her istekte `joblib.load` mümkün değil (I/O + RAM). Cache SICAK tutar.
artifact_id IMMUTABLE (uuid) → yeni champion = yeni anahtar → invalidation GEREKMEZ (eski LRU'dan düşer).

PER-KEY LOAD-LOCK (kritik): iki eşzamanlı istek AYNI soğuk artifact'i isterse, ikisi de yüklerse
147MB İKİ KEZ RAM'e gelir (patlar). Anahtar-başına kilit → o artifact'i YALNIZ BİR thread yükler,
diğeri bekler ve hazır sonucu alır. Global kilit yalnız dict/lock-registry işlemleri için (kısa) —
PAHALI yükleme global kilit DIŞINDA (başka artifact'ler bloklanmasın).

RAM UYARISI: `joblib.load` sıkıştırılmışı TAM boyutta açar → compress RAM'i DÜŞÜRMEZ. Bütçe
`settings.SERVING_CACHE_MAX_ARTIFACTS` (varsayılan 2) düşük tutulur; fitted_ prune (backlog) gerçek çare.
"""
from __future__ import annotations

import gc
import threading
from collections import OrderedDict
from typing import Callable, Tuple

import joblib

from src.core.config import settings
from src.core.logging_config import get_logger
from src.services.inference.serialize import load_for_serving
from src.services.inference.references import resolve_model_path

logger = get_logger(__name__)


def _default_loader(artifact_id: str, base_dir: str) -> Tuple[object, object]:
    """Diskten (Artifact üç-kapı + model). PAHALI — cache tarafından kilit ALTINDA, bir kez çağrılır."""
    art = load_for_serving(artifact_id, base_dir)          # phase/deployable/model-dosyası kapıları
    model = joblib.load(resolve_model_path(art.model_ref))  # compress otomatik algılanır
    return art, model


class ServingCache:
    def __init__(self, max_artifacts: int, loader: "Callable[[str, str], Tuple[object, object]]" = _default_loader):
        self._max = max(1, int(max_artifacts))
        self._loader = loader
        self._store: "OrderedDict[str, Tuple[object, object]]" = OrderedDict()
        self._glock = threading.Lock()               # _store + _key_locks'ı korur (KISA tutulur)
        self._key_locks: dict[str, threading.Lock] = {}

    def _key_lock(self, aid: str) -> threading.Lock:
        with self._glock:
            lk = self._key_locks.get(aid)
            if lk is None:
                lk = threading.Lock()
                self._key_locks[aid] = lk
            return lk

    def get(self, artifact_id: str, base_dir: str = "data/artifacts") -> "Tuple[object, object]":
        # HIZLI YOL: zaten cache'te (kısa global kilit)
        with self._glock:
            hit = self._store.get(artifact_id)
            if hit is not None:
                self._store.move_to_end(artifact_id)
                return hit
        # YAVAŞ YOL: per-key kilit → aynı soğuk artifact'i yalnız BİR thread yükler
        with self._key_lock(artifact_id):
            with self._glock:                        # kilidi beklerken başkası yüklemiş olabilir
                hit = self._store.get(artifact_id)
                if hit is not None:
                    self._store.move_to_end(artifact_id)
                    return hit
            loaded = self._loader(artifact_id, base_dir)   # PAHALI — global kilit DIŞINDA
            with self._glock:
                self._store[artifact_id] = loaded
                self._store.move_to_end(artifact_id)
                while len(self._store) > self._max:
                    old_id, _ = self._store.popitem(last=False)   # LRU eviction
                    logger.info(f"[serving_cache] eviction: {old_id} (kapasite={self._max})")
                    gc.collect()
            return loaded

    # Test/introspection yardımcıları
    def keys(self) -> list:
        with self._glock:
            return list(self._store.keys())

    def clear(self) -> None:
        with self._glock:
            self._store.clear()
            self._key_locks.clear()
        gc.collect()


# Process-içi singleton (API worker'ları paylaşır).
_cache: "ServingCache | None" = None
_singleton_lock = threading.Lock()


def get_serving_cache() -> ServingCache:
    global _cache
    if _cache is None:
        with _singleton_lock:
            if _cache is None:
                _cache = ServingCache(settings.SERVING_CACHE_MAX_ARTIFACTS)
    return _cache
