"""
Yapısal logging altyapısı.
=========================

Tüm süreçler (API, worker, MCP) `setup_logging()` çağırır. Loglar, içinde
bulunulan işin `project_id`'siyle zenginleştirilir; böylece paralel worker'larda
hangi log hangi projeye ait kolayca görülür.

`setup_logging()` **tek kontrol noktasıdır**: hangi handler'ların aktif olacağı
buradan yönetilir. Şu an iki hedef vardır:

* **Console** — her zaman açık (stdout).
* **Veritabanı** — opsiyonel; `DBLogQueueHandler` ile asenkron + toplu yazar
  (bkz. `src/core/log_handlers.py`). Worker bunu açık başlatır.

Yeni bir hedef (ör. dosya, uzak servis) eklemek için `_build_handlers()` içine
bir handler üretip listeye koymak yeterlidir — başka hiçbir yeri değiştirmeye
gerek yoktur.

Kullanım:
    from src.core.logging_config import setup_logging, bind_project, get_logger

    setup_logging(process="worker", db_logging=True)   # süreç başında bir kez
    log = get_logger(__name__)

    with bind_project(project_id):       # bir iş işlenirken
        log.info("graf başlıyor")        # → "... [project=42] graf başlıyor"

Env değişkenleri (logging'e özel; cycle'ı önlemek için doğrudan os.getenv):
    LOG_LEVEL            console + kök seviye (varsayılan INFO)
    LOG_DB_ENABLED       DB handler'ı aç (true/false; setup_logging argümanı önceliklidir)
    LOG_DB_LEVEL         DB'ye yazılacak min seviye (varsayılan INFO)
    LOG_DB_BATCH_SIZE    toplu INSERT boyutu (varsayılan 50)
    LOG_DB_FLUSH_INTERVAL  flush periyodu sn (varsayılan 2.0)
    LOG_DB_QUEUE_MAXSIZE   kuyruk üst sınırı (varsayılan 10000)
"""
from __future__ import annotations

import atexit
import logging
import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

from src.core.log_handlers import DBLogQueueHandler

# İçinde bulunulan işin proje kimliği — log kayıtlarına enjekte edilir.
_project_id_var: ContextVar[str] = ContextVar("project_id", default="-")

_CONFIGURED = False
# Aktif DB handler referansı (kapanışta flush/stop için).
_db_handler: Optional[DBLogQueueHandler] = None

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [project=%(project_id)s] %(message)s"
_DATEFMT = "%H:%M:%S"


class _ProjectContextFilter(logging.Filter):
    """Her log kaydına aktif `project_id`'yi ekler (contextvar'dan).

    Filtre **handler düzeyinde** takılır: logging'de logger filtreleri yalnızca
    o logger'a doğrudan yapılan çağrılara uygulanır, alt logger'lardan
    propagate olan kayıtlara uygulanmaz. Handler'a takınca her kayıt etiketlenir.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "project_id"):
            record.project_id = _project_id_var.get()
        return True


def _level_from(value: str | int | None, default: int = logging.INFO) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    return getattr(logging, str(value).upper(), default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _build_handlers(
    console_level: int,
    *,
    process: str,
    db_logging: bool,
    db_level: int,
) -> list[logging.Handler]:
    """Aktif handler'ları üretir. Yeni hedefler buraya eklenir (tek nokta)."""
    ctx_filter = _ProjectContextFilter()
    fmt = logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT)

    # 1) Console — her zaman.
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(fmt)
    console.addFilter(ctx_filter)
    handlers: list[logging.Handler] = [console]

    # 2) Veritabanı — opsiyonel, asenkron + toplu.
    global _db_handler
    if db_logging:
        db = DBLogQueueHandler(
            level=db_level,
            process=process,
            batch_size=int(os.getenv("LOG_DB_BATCH_SIZE", "50")),
            flush_interval=float(os.getenv("LOG_DB_FLUSH_INTERVAL", "2.0")),
            queue_maxsize=int(os.getenv("LOG_DB_QUEUE_MAXSIZE", "10000")),
        )
        db.addFilter(ctx_filter)
        _db_handler = db
        handlers.append(db)

    # 3) Gelecekte: dosya / uzak servis handler'ları buraya eklenebilir, ör.
    #    file = logging.handlers.RotatingFileHandler(...); handlers.append(file)

    return handlers


def setup_logging(
    level: str | int | None = None,
    *,
    process: str = "app",
    db_logging: bool | None = None,
    db_level: str | int | None = None,
) -> None:
    """Kök logger'ı yapılandırır. Idempotent — birden fazla çağrı zararsız.

    Args:
        level: console/kök seviye. Yoksa `LOG_LEVEL` env, o da yoksa INFO.
        process: log kaynağı etiketi ("worker"/"api"/"mcp") — DB kayıtlarına yazılır.
        db_logging: DB handler açılsın mı. None ise `LOG_DB_ENABLED` env'ine bakar.
        db_level: DB'ye yazılacak min seviye. Yoksa `LOG_DB_LEVEL` env, o da yoksa INFO.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    console_level = _level_from(level if level is not None else os.getenv("LOG_LEVEL"))
    if db_logging is None:
        db_logging = _env_bool("LOG_DB_ENABLED", default=False)
    db_lvl = _level_from(db_level if db_level is not None else os.getenv("LOG_DB_LEVEL"))

    handlers = _build_handlers(
        console_level, process=process, db_logging=db_logging, db_level=db_lvl
    )

    root = logging.getLogger()
    # Kök seviye, en düşük handler seviyesine inmeli ki kayıtlar handler'lara ulaşsın.
    active_levels = [console_level] + ([db_lvl] if db_logging else [])
    root.setLevel(min(active_levels))
    # Çift handler eklenmesini önle (reload/idempotency).
    root.handlers = handlers

    # SQLAlchemy'nin gürültülü INFO loglarını kıs — DB handler ile geri besleme
    # döngüsü ve fazlalık riskini azaltır.
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    if db_logging:
        atexit.register(shutdown_logging)

    _CONFIGURED = True


def shutdown_logging() -> None:
    """Kapanışta DB handler'ın kuyruğundaki kalan logları yazar ve thread'i durdurur."""
    global _db_handler
    if _db_handler is not None:
        _db_handler.stop()
        _db_handler = None


def get_logger(name: str) -> logging.Logger:
    """İsimlendirilmiş logger döndürür (setup_logging henüz çağrılmadıysa de güvenli)."""
    return logging.getLogger(name)


@contextmanager
def bind_project(project_id: object):
    """Verilen kod bloğu süresince log kayıtlarına `project_id` bağlar."""
    token = _project_id_var.set(str(project_id))
    try:
        yield
    finally:
        _project_id_var.reset(token)


def set_project(project_id: object) -> None:
    """Context manager kullanılamayan yerlerde (ör. uzun async akış) proje bağlar."""
    _project_id_var.set(str(project_id))
