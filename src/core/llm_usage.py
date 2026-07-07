"""Token/cost accounting + per-run budget for every LLM call (C3).

The pipeline fans out a lot of LLM work per project (intent → schema-analysis →
concurrent expert council → synthesizer → planner/feature/leakage/eval/report →
up to 3 critic retries). Before this there was no usage capture and no spend cap.

How it works:

  * :class:`UsageTracker` is a per-run accumulator, made active for the duration
    of a job via the :func:`track_usage` context manager (the worker enters it
    around each job, like ``bind_project``). It lives in a ``ContextVar`` so it is
    correct under the council's ``asyncio.gather`` fan-out.
  * :data:`usage_callback` is a single LangChain callback attached to every
    ``get_llm()`` instance. On each ``on_llm_end`` it reads the response's
    ``usage_metadata`` / ``llm_output`` and adds the tokens to the active tracker.
  * If ``LLM_TOKEN_BUDGET > 0`` and the run exceeds it, the callback raises
    :class:`TokenBudgetExceeded`, aborting the run (the ``@node`` safety net turns
    that into a structured error that lands in the report).
  * At context exit the tracker flushes one aggregated ``llm_usage`` row per
    (project, provider, model).

When no tracker is active (API, MCP, tests), the callback is a no-op — usage is
only recorded inside a ``track_usage`` block.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

from langchain_core.callbacks import BaseCallbackHandler

from src.core.config import settings
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class TokenBudgetExceeded(RuntimeError):
    """Raised when a run's cumulative token usage exceeds ``LLM_TOKEN_BUDGET``."""


class UsageTracker:
    """Thread-safe per-run token accumulator, keyed by (provider, model)."""

    def __init__(self, project_id: Optional[str] = None, budget: int = 0):
        self.project_id = str(project_id) if project_id is not None else None
        self.budget = budget or 0
        self._lock = threading.Lock()
        # (provider, model) -> {prompt, completion, total, calls}
        self._buckets: dict[tuple[str, str], dict[str, int]] = {}
        self.total_tokens = 0

    def add(self, provider: str, model: str, prompt: int, completion: int, total: int) -> None:
        with self._lock:
            key = (provider or "?", model or "?")
            b = self._buckets.setdefault(
                key, {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
            )
            b["prompt"] += int(prompt or 0)
            b["completion"] += int(completion or 0)
            b["total"] += int(total or 0)
            b["calls"] += 1
            self.total_tokens += int(total or 0)
            over = self.budget > 0 and self.total_tokens > self.budget
        if over:
            raise TokenBudgetExceeded(
                f"LLM token budget exceeded: {self.total_tokens} > {self.budget} "
                f"(project={self.project_id})"
            )

    def summary(self) -> dict:
        with self._lock:
            return {
                "project_id": self.project_id,
                "total_tokens": self.total_tokens,
                "buckets": {f"{p}/{m}": dict(v) for (p, m), v in self._buckets.items()},
            }

    def flush(self) -> None:
        """Persist aggregated buckets to the ``llm_usage`` table. Best-effort."""
        with self._lock:
            buckets = {k: dict(v) for k, v in self._buckets.items()}
        if not buckets:
            return
        try:
            from api.core.database import SessionLocal
            from api.models.llm_usage import LLMUsageModel
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("llm_usage flush skipped (model/db unavailable): %s", e)
            return
        db = SessionLocal()
        try:
            now = int(time.time())
            for (provider, model), b in buckets.items():
                db.add(LLMUsageModel(
                    created_at=now,
                    project_id=self.project_id,
                    provider=provider,
                    model=model,
                    prompt_tokens=b["prompt"],
                    completion_tokens=b["completion"],
                    total_tokens=b["total"],
                    calls=b["calls"],
                ))
            db.commit()
            logger.info(
                "[llm_usage] project=%s total_tokens=%s buckets=%s",
                self.project_id, self.total_tokens, len(buckets),
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("[llm_usage] flush failed: %s", e)
            db.rollback()
        finally:
            db.close()


_tracker_var: ContextVar[Optional[UsageTracker]] = ContextVar("llm_usage_tracker", default=None)


def get_active_tracker() -> Optional[UsageTracker]:
    return _tracker_var.get()


@contextmanager
def track_usage(project_id: Optional[str] = None, budget: Optional[int] = None):
    """Make a fresh :class:`UsageTracker` active for the enclosed block, then flush.

    Used by the worker around each job. ``budget`` defaults to
    ``settings.LLM_TOKEN_BUDGET``; pass 0 to disable the cap for a run.
    """
    tracker = UsageTracker(
        project_id=project_id,
        budget=settings.LLM_TOKEN_BUDGET if budget is None else budget,
    )
    token = _tracker_var.set(tracker)
    try:
        yield tracker
    finally:
        _tracker_var.reset(token)
        if settings.ENABLE_LLM_USAGE_TRACKING:
            tracker.flush()


def _extract_usage(response) -> tuple[int, int, int]:
    """Pull (prompt, completion, total) tokens from an LLMResult. 0s if absent."""
    prompt = completion = total = 0

    # Preferred: usage_metadata on the message generations (provider-normalised).
    try:
        for gen_list in getattr(response, "generations", []) or []:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                um = getattr(msg, "usage_metadata", None) if msg is not None else None
                if um:
                    prompt += int(um.get("input_tokens", 0) or 0)
                    completion += int(um.get("output_tokens", 0) or 0)
                    total += int(um.get("total_tokens", 0) or 0)
    except Exception:  # pragma: no cover - defensive
        pass

    if total or prompt or completion:
        return prompt, completion, total

    # Fallback: llm_output.token_usage (OpenAI-style).
    try:
        usage = (getattr(response, "llm_output", None) or {}).get("token_usage", {}) or {}
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", 0) or 0) or (prompt + completion)
    except Exception:  # pragma: no cover - defensive
        pass
    return prompt, completion, total


class UsageTrackingCallback(BaseCallbackHandler):
    """Attached to every get_llm() instance; records usage into the active tracker."""

    raise_error = True  # let TokenBudgetExceeded propagate instead of being swallowed

    def on_llm_end(self, response, **kwargs) -> None:
        tracker = _tracker_var.get()
        if tracker is None:
            return
        prompt, completion, total = _extract_usage(response)
        if not (prompt or completion or total):
            return
        # provider/model tagged onto the callback-owning llm via metadata kwargs.
        meta = kwargs.get("metadata") or {}
        provider = meta.get("ls_provider") or meta.get("provider") or "?"
        model = meta.get("ls_model_name") or meta.get("model") or "?"
        tracker.add(provider, model, prompt, completion, total)


# Singleton — safe to share across models (all state lives in the ContextVar tracker).
usage_callback = UsageTrackingCallback()
