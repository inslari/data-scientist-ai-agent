"""
Merkezi node sarmalayıcı (LangGraph adaptör katmanı için ortak politika).
=========================================================================

`@node(...)` dekoratörü her LangGraph node'una üç ortak davranışı tek yerden
kazandırır — node gövdeleri yalnızca kendi işine odaklanır:

  1. **Upstream-error short-circuit:** state'te `error` varsa node gövdesi hiç
     çalışmaz; sadece `{"current_step": name}` döndürülür. Böylece bir hata
     oluştuğunda zincirdeki sonraki node'lar kırık state üzerinde çalışmaz —
     hata `report_node`'a kadar (o `runs_on_error=True`) güvenle taşınır.
  2. **Exception güvenlik ağı:** node gövdesi exception fırlatırsa tüm graf
     çökmek yerine yapısal bir hata state'ine (`error` + `error_category`)
     dönüştürülür; pipeline `report_node`'da düzgünce sonlanır.
  3. **Yapısal zamanlama logları:** her node'un başlangıç/bitiş/hata kaydı tek
     formatta üretilir (proje bağlamıyla — bkz. logging_config).

Senkron ve async node'ların ikisini de sarar; `config` gibi ek argümanları
olduğu gibi geçirir.
"""
from __future__ import annotations

import functools
import inspect
import time
import traceback
from typing import Awaitable, Callable

from src.core.logging_config import get_logger

log = get_logger("ds_agent.node")


def classify_error(exc: BaseException) -> str:
    """Exception'ı kaba bir kategoriye ayırır (rapor/teşhis ve metrik için).

    Kategoriler: "data", "db", "llm", "timeout", "memory", "unknown".
    Bilinçli olarak basit — tip + mesaj örüntüsüne bakar, ağ bağımlılığı yok.
    """
    name = type(exc).__name__.lower()
    msg = str(exc).lower()

    if isinstance(exc, (TimeoutError,)) or "timeout" in name or "timed out" in msg:
        return "timeout"
    if isinstance(exc, MemoryError) or "memory" in msg:
        return "memory"
    if any(k in name for k in ("keyerror", "valueerror", "typeerror", "indexerror",
                               "attributeerror", "emptydata", "parser")):
        return "data"
    if any(k in msg for k in ("connection", "operationalerror", "psycopg", "sqlalchemy",
                              "could not connect", "deadlock")):
        return "db"
    if any(k in msg for k in ("openai", "anthropic", "rate limit", "api key",
                              "completion", "token", "llm")):
        return "llm"
    return "unknown"


def _on_exception(node_name: str, exc: BaseException, t0: float) -> dict:
    category = classify_error(exc)
    log.error(
        "✗ %s failed after %.2fs [%s]: %s\n%s",
        node_name, time.perf_counter() - t0, category, exc,
        traceback.format_exc(),
    )
    return {
        "error": f"[{node_name}] {type(exc).__name__}: {exc}",
        "error_category": category,
        "current_step": node_name,
    }


def _finalize(node_name: str, result: object, t0: float) -> dict:
    """Gövde sonucunu normalize eder: dict garantisi + current_step + done log."""
    if not isinstance(result, dict):
        result = {} if result is None else {"_result": result}
    result.setdefault("current_step", node_name)
    log.info("✓ %s (%.2fs)", node_name, time.perf_counter() - t0)
    return result


def node(name: str | None = None, *, runs_on_error: bool = False) -> Callable:
    """LangGraph node'unu ortak politikayla sarar.

    Args:
        name: Loglarda görünen ad (varsayılan: fonksiyon adı).
        runs_on_error: True ise upstream-error short-circuit'i atlar; node gövdesi
            hata olsa da çalışır (yalnızca `report_node` için — hatayı raporlamalı).
    """

    def decorator(fn: Callable) -> Callable:
        node_name = name or fn.__name__

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(state, *args, **kwargs):
                if not runs_on_error and state.get("error"):
                    log.info("⏭ %s atlandı (upstream error)", node_name)
                    return {"current_step": node_name}
                log.info("▶ %s", node_name)
                t0 = time.perf_counter()
                try:
                    result = await fn(state, *args, **kwargs)
                except Exception as exc:  # noqa: BLE001 — kasıtlı güvenlik ağı
                    return _on_exception(node_name, exc, t0)
                return _finalize(node_name, result, t0)

            return awrapper

        @functools.wraps(fn)
        def wrapper(state, *args, **kwargs):
            if not runs_on_error and state.get("error"):
                log.info("⏭ %s atlandı (upstream error)", node_name)
                return {"current_step": node_name}
            log.info("▶ %s", node_name)
            t0 = time.perf_counter()
            try:
                result = fn(state, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — kasıtlı güvenlik ağı
                return _on_exception(node_name, exc, t0)
            return _finalize(node_name, result, t0)

        return wrapper

    return decorator
