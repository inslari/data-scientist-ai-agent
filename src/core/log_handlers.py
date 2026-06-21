"""Genişletilebilir log handler'ları.
================================

Bu modül, `setup_logging()` tarafından merkezî olarak takılan ek log
hedeflerini barındırır. Şu an tek hedef **veritabanı**dır (`DBLogQueueHandler`),
ama dosya/uzak servis gibi yeni handler'lar buraya eklenip `logging_config`
içinden tek noktadan kaydedilebilir.

Tasarım — neden kuyruk?
-----------------------
Worker ağır ML akışları çalıştırır; her log satırını anında DB'ye yazmak her
kayıtta bir DB round-trip demektir ve pipeline'ı yavaşlatır. Bunun yerine:

* **Üretici taraf** (`emit`) log çağrısının yapıldığı thread'de çalışır; sadece
  hafif bir dict üretip kuyruğa koyar (``put_nowait``) — asla bloklamaz.
* **Tüketici taraf** ayrı bir arka plan thread'idir; kuyruğu boşaltıp kayıtları
  **toplu** (`bulk_insert_mappings`) yazar. Boyut (``batch_size``) veya süre
  (``flush_interval``) eşiğinde flush eder.

Handler paylaşılan `LogRecord`'u **mutasyona uğratmaz** (exc_info/args temizlemez)
çünkü aynı kayıt console handler tarafından da formatlanır; bunun yerine `emit`
anında ihtiyaç duyulan alanları kopyalar.
"""
from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from typing import Callable, Optional

# Kuyruğa konan "artık dur ve kalanları yaz" işareti.
_SENTINEL = object()

# Aşırı uzun mesaj/traceback'lerin satırı şişirmesini engelle.
_MAX_MESSAGE = 20_000
_MAX_EXC = 50_000

# Traceback metnini üretmek için tek bir formatter yeter (durumsuz).
_EXC_FORMATTER = logging.Formatter()


class DBLogQueueHandler(logging.Handler):
    """Logları kuyruğa alıp arka planda toplu olarak DB'ye yazan handler.

    Kök logger'a takılabilir. `session_factory`/`model` enjekte edilebilir
    (test için); verilmezse uygulama DB'sinden tembelce yüklenir.
    """

    def __init__(
        self,
        *,
        level: int = logging.INFO,
        process: str = "app",
        batch_size: int = 50,
        flush_interval: float = 2.0,
        queue_maxsize: int = 10_000,
        session_factory: Optional[Callable] = None,
        model: Optional[type] = None,
    ) -> None:
        super().__init__(level=level)
        self.process = process
        self.batch_size = max(1, batch_size)
        self.flush_interval = max(0.1, flush_interval)
        self._session_factory = session_factory
        self._model = model
        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_maxsize)
        self._stopped = False
        self._dropped = 0
        self._thread = threading.Thread(
            target=self._run, name="db-log-writer", daemon=True
        )
        self._thread.start()

    # ── Üretici taraf (çağıran thread) ──────────────────────────────────────
    def emit(self, record: logging.LogRecord) -> None:
        try:
            row = self._to_row(record)
        except Exception:  # pragma: no cover - defansif
            self.handleError(record)
            return
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            # Aşırı yükte bloklamak yerine düşür; pipeline asla beklemesin.
            self._dropped += 1

    def _to_row(self, record: logging.LogRecord) -> dict:
        msg = record.getMessage()
        if len(msg) > _MAX_MESSAGE:
            msg = msg[:_MAX_MESSAGE] + "… (kısaltıldı)"

        exc_text: Optional[str] = None
        if record.exc_info:
            try:
                exc_text = _EXC_FORMATTER.formatException(record.exc_info)
            except Exception:  # pragma: no cover
                exc_text = None
        elif record.exc_text:
            exc_text = record.exc_text
        if exc_text and len(exc_text) > _MAX_EXC:
            exc_text = exc_text[:_MAX_EXC] + "… (kısaltıldı)"

        pid = getattr(record, "project_id", None)
        if pid in (None, "-", ""):
            pid = None
        else:
            pid = str(pid)

        return {
            "created_at": int(record.created),
            "level": record.levelname,
            "levelno": record.levelno,
            "logger": record.name,
            "process": self.process,
            "project_id": pid,
            "message": msg,
            "module": getattr(record, "module", None),
            "func": getattr(record, "funcName", None),
            "line": getattr(record, "lineno", None),
            "exc_text": exc_text,
        }

    # ── Tüketici taraf (arka plan thread) ───────────────────────────────────
    def _run(self) -> None:
        buffer: list[dict] = []
        while True:
            try:
                item = self._queue.get(timeout=self.flush_interval)
            except queue.Empty:
                item = None

            if item is _SENTINEL:
                self._flush(buffer)
                break

            if item is not None:
                buffer.append(item)

            # Zaman aşımı (item None) → elde ne varsa yaz; veya batch doldu.
            if (item is None and buffer) or len(buffer) >= self.batch_size:
                self._flush(buffer)
                buffer = []

    def _resolve_backend(self):
        """(session_factory, model) çiftini döndürür; tembel ve hataya dayanıklı."""
        if self._session_factory is not None and self._model is not None:
            return self._session_factory, self._model
        try:
            from api.core.database import SessionLocal
            from api.models.log_entry import LogEntryModel
        except Exception:
            return None, None
        return SessionLocal, LogEntryModel

    def _flush(self, records: list[dict]) -> None:
        if not records:
            return
        session_factory, model = self._resolve_backend()
        if session_factory is None or model is None:
            return
        db = session_factory()
        try:
            db.bulk_insert_mappings(model, records)
            db.commit()
        except Exception as exc:  # DB hatası logging'i çökertmemeli.
            try:
                db.rollback()
            except Exception:
                pass
            # Kendi hatamızı logger ile yazarsak döngü riski olur → doğrudan stderr.
            print(f"[log_db] flush failed ({len(records)} rec): {exc}", file=sys.__stderr__)
        finally:
            try:
                db.close()
            except Exception:
                pass

    # ── Yaşam döngüsü ───────────────────────────────────────────────────────
    def stop(self, timeout: float = 5.0) -> None:
        """Kalan logları yazıp arka plan thread'ini durdurur (idempotent)."""
        if self._stopped:
            return
        self._stopped = True
        try:
            self._queue.put_nowait(_SENTINEL)
        except queue.Full:
            try:
                self._queue.put(_SENTINEL, timeout=1.0)
            except Exception:
                pass
        self._thread.join(timeout=timeout)


def prune_logs(
    older_than_days: int,
    *,
    session_factory: Optional[Callable] = None,
    model: Optional[type] = None,
) -> int:
    """`older_than_days` günden eski log kayıtlarını siler; silinen satır sayısını döndürür.

    Otomatik çağrılmaz — retention politikası çağırana bırakılmıştır (örn. bir
    cron / scheduled task ya da worker açılışı). `LOG_RETENTION_DAYS` env'i bu
    fonksiyonu beslemek için bir kolaylıktır.
    """
    if session_factory is None or model is None:
        from api.core.database import SessionLocal
        from api.models.log_entry import LogEntryModel
        session_factory = session_factory or SessionLocal
        model = model or LogEntryModel

    cutoff = int(time.time()) - int(older_than_days) * 86_400
    db = session_factory()
    try:
        deleted = (
            db.query(model)
            .filter(model.created_at < cutoff)
            .delete(synchronize_session=False)
        )
        db.commit()
        return int(deleted or 0)
    finally:
        db.close()
