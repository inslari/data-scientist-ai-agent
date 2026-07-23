"""
Resolver that turns a role ("main" | "intelligence" | "council") into the effective
provider/model/credentials configured in the DB — or None to signal "use env".

This is the single bridge the LLM factory (src/core/llm.py) calls. Everything here is
defensive: any DB/decrypt problem returns None so a broken row can never abort a job,
it just falls back to the env defaults (legacy behaviour).

A short TTL cache keeps the hot path cheap and lets the separate worker process pick up
admin changes within ~TTL seconds without a restart. Write routes call ``invalidate()``
for immediate convergence within their own process.
"""
import time
import threading
from typing import Optional

from api.core.database import SessionLocal

_TTL_SECONDS = 30.0
_lock = threading.Lock()
# role -> (expires_at, config_or_None)
_cache: dict[str, tuple[float, Optional[dict]]] = {}


def invalidate() -> None:
    """Drop the cache so the next resolve re-reads the DB. Called after any write."""
    with _lock:
        _cache.clear()


def _resolve_from_db(role: str) -> Optional[dict]:
    # Local imports so a partially-initialised model registry / import cycle can't
    # break module import; this function is always called lazily at runtime.
    from api.models.ai_role_assignment import AiRoleAssignmentModel
    from api.models.ai_model import AiModelModel
    from api.models.ai_provider import AiProviderModel
    from api.core.secrets_crypto import decrypt

    db = SessionLocal()
    try:
        assignment = (
            db.query(AiRoleAssignmentModel)
            .filter(AiRoleAssignmentModel.role == role)
            .first()
        )
        if not assignment or assignment.model_id is None:
            return None

        model = db.query(AiModelModel).filter(AiModelModel.id == assignment.model_id).first()
        if not model:
            return None

        provider = db.query(AiProviderModel).filter(AiProviderModel.id == model.provider_id).first()
        if not provider:
            return None

        return {
            "provider_type": provider.provider_type,
            "model_name": model.model_name,
            "api_key": decrypt(provider.api_key_encrypted),
            "base_url": provider.base_url,
        }
    finally:
        db.close()


def get_role_config(role: str) -> Optional[dict]:
    """
    Effective config for ``role`` from the DB, or None if unassigned / on any error.
    Returns a dict: ``{provider_type, model_name, api_key, base_url}``.
    """
    now = time.monotonic()
    with _lock:
        cached = _cache.get(role)
        if cached and cached[0] > now:
            return cached[1]

    try:
        config = _resolve_from_db(role)
    except Exception:
        # Any DB failure → behave as "unassigned" so the caller uses env defaults.
        config = None

    with _lock:
        _cache[role] = (now + _TTL_SECONDS, config)
    return config
