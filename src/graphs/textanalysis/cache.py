"""
İçerik-adresli cache yardımcıları (metin analizi).

classify ve embed sonuçlarını text+model anahtarıyla saklar/okur. Best-effort:
DB hatası analizi bozmaz (cache yoksa sadece yeniden hesaplanır).
"""
from __future__ import annotations

import hashlib

from src.core.logging_config import get_logger

logger = get_logger(__name__)


def _key(kind: str, model: str, text: str) -> str:
    return hashlib.sha256(f"{kind}|{model}|{text}".encode("utf-8")).hexdigest()


def get_cached(kind: str, model: str, texts: list[str]) -> dict[str, object]:
    """texts içindeki cache hit'lerini {text: payload} olarak döndürür."""
    if not texts:
        return {}
    try:
        from api.core.database import SessionLocal
        from api.models.text_cache import TextAnalysisCacheModel
    except Exception:
        return {}

    keys = {_key(kind, model, t): t for t in set(texts)}
    out: dict[str, object] = {}
    db = SessionLocal()
    try:
        rows = (
            db.query(TextAnalysisCacheModel)
            .filter(TextAnalysisCacheModel.cache_key.in_(list(keys.keys())))
            .all()
        )
        for r in rows:
            txt = keys.get(r.cache_key)
            if txt is not None:
                out[txt] = r.payload
    except Exception as e:  # pragma: no cover - cache best-effort
        logger.warning(f"[TA cache] okuma hatası ({type(e).__name__}: {e})")
    finally:
        db.close()
    return out


def set_cached(kind: str, model: str, items: dict[str, object]) -> None:
    """{text: payload} kayıtlarını cache'e yazar (varsa atlar)."""
    if not items:
        return
    try:
        from api.core.database import SessionLocal
        from api.models.text_cache import TextAnalysisCacheModel
    except Exception:
        return

    db = SessionLocal()
    try:
        new_keys = {_key(kind, model, t): (t, p) for t, p in items.items()}
        existing = {
            r.cache_key
            for r in db.query(TextAnalysisCacheModel.cache_key)
            .filter(TextAnalysisCacheModel.cache_key.in_(list(new_keys.keys())))
            .all()
        }
        for ck, (txt, payload) in new_keys.items():
            if ck in existing:
                continue
            db.add(TextAnalysisCacheModel(cache_key=ck, kind=kind, model=model, payload=payload))
        db.commit()
    except Exception as e:  # pragma: no cover - cache best-effort
        db.rollback()
        logger.warning(f"[TA cache] yazma hatası ({type(e).__name__}: {e})")
    finally:
        db.close()
