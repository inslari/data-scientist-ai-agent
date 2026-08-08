"""
Veritabanı bakım/temizlik servisi.

Zamanla şişen tabloları budar; WORKER sürecinde çalışır (API'yi bloklamaz):
  1. LangGraph **checkpoint** tabloları — terminal (completed/failed & eski) veya
     orphan (projesi silinmiş) thread'lerin checkpoint satırları. En büyük şişme
     kaynağı: her node geçişinde tüm state serialize edilir.
  2. **background_jobs** — tamamlanmış/başarısız ve eski job kayıtları.
  3. **app_logs** — retention'dan eski loglar (log_handlers.prune_logs yeniden kullanılır).
  4. **db_intelligence_cache** — projesi kalmamış orphan cache satırları (savunmacı).

Postgres'te checkpoint tabloları uygulama DB'siyle AYNI veritabanında (aynı bağlantı
dizesi + search_path) olduğundan doğrudan SQLAlchemy engine ile budanır. SQLite modunda
checkpoint'ler ayrı bir dosyadadır (data/checkpoints.sqlite) → sqlite3 ile budanır.

Tüm işlemler best-effort ve savunmacıdır: bir alt görev hata verse bile diğerleri çalışır.
Yalnızca terminal/orphan thread'ler silinir → çalışan/queue'daki projelerin checkpoint'lerine
DOKUNULMAZ.
"""
from __future__ import annotations

import os
import time
import sqlite3

from sqlalchemy import text

from api.core.database import engine, SessionLocal, SQLALCHEMY_DATABASE_URL
from src.core.config import settings

from src.core.logging_config import get_logger
logger = get_logger(__name__)

_CHECKPOINT_TABLES = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")
_SQLITE_CHECKPOINT_PATH = "data/checkpoints.sqlite"


# AYRIK ÇAĞRI YÜZEYİ: subtask'lar İSİMLE seçilebilir. Katman 2 monitoring aggregation'ı YIKICI GC'den
# (app_logs/background_jobs/checkpoints/intelligence_cache silme) BAĞIMSIZ çağırmak için — farklı kadanslar
# (aggregation SIK, GC SEYREK) ya da ad-hoc pencere kapatma (bkz. scripts/run_monitoring_aggregation.py).
GC_SUBTASKS = ("app_logs", "background_jobs", "checkpoints", "intelligence_cache_orphans")
MONITORING_SUBTASKS = ("monitoring_aggregate",)


def run_maintenance(only: "set[str] | None" = None) -> dict:
    """
    Temizlik alt görevlerini çalıştırır ve silinen satır sayılarını döner. Senkron; çağıran (worker)
    event loop'u bloklamamak için ayrı thread'de çağırmalıdır (asyncio.to_thread).

    ``only``: verilirse YALNIZ o isimli subtask'lar koşar (aksi hâlde hepsi). Örn.
    ``run_maintenance(only=MONITORING_SUBTASKS)`` = GC'siz yalnız Katman 2 aggregation → yıkıcı budama
    (log/job/checkpoint silme) tetiklenmez. Bilinmeyen isim atlanır (sessizce değil — log).
    """
    is_postgres = SQLALCHEMY_DATABASE_URL.startswith("postgresql")
    now = int(time.time())
    results: dict = {}

    all_tasks = (
        ("app_logs", _prune_app_logs),
        ("background_jobs", lambda: _prune_background_jobs(now)),
        ("checkpoints", lambda: _prune_checkpoints(now, is_postgres)),
        ("intelligence_cache_orphans", _prune_intelligence_cache_orphans),
        ("monitoring_aggregate", _run_monitoring_aggregate),   # Katman 2 — izole (patlarsa GC sürer)
    )
    if only is not None:
        unknown = set(only) - {n for n, _ in all_tasks}
        if unknown:
            logger.warning(f"[maintenance] bilinmeyen subtask(lar) atlandı: {sorted(unknown)}")
        tasks = [(n, f) for n, f in all_tasks if n in only]
    else:
        tasks = list(all_tasks)

    logger.info(f"[maintenance] Temizlik başlıyor (postgres={is_postgres}, subtask={[n for n,_ in tasks]})...")
    for name, fn in tasks:
        try:
            results[name] = fn()
        except Exception as e:
            logger.warning(f"[maintenance] '{name}' budama hatası: {str(e)[:200]}")
            results[name] = -1

    logger.info(f"[maintenance] Tamamlandı: {results}")
    return results


# ── 1. app_logs ───────────────────────────────────────────────────────────────

def _prune_app_logs() -> int:
    days = settings.LOG_RETENTION_DAYS
    if days is None or days <= 0:
        return 0
    from src.core.log_handlers import prune_logs
    deleted = prune_logs(days)
    logger.info(f"[maintenance] app_logs: {deleted} kayıt silindi (> {days}g).")
    return deleted


# ── 2. background_jobs ─────────────────────────────────────────────────────────

def _prune_background_jobs(now: int) -> int:
    days = settings.JOB_RETENTION_DAYS
    if days is None or days <= 0:
        return 0
    cutoff = now - days * 86_400
    from api.models.job import BackgroundJobModel
    db = SessionLocal()
    try:
        deleted = (
            db.query(BackgroundJobModel)
            .filter(
                BackgroundJobModel.status.in_(["completed", "failed"]),
                BackgroundJobModel.updated_at < cutoff,
            )
            .delete(synchronize_session=False)
        )
        db.commit()
        logger.info(f"[maintenance] background_jobs: {deleted} kayıt silindi (> {days}g).")
        return int(deleted or 0)
    finally:
        db.close()


# ── 3. checkpoints (LangGraph) ─────────────────────────────────────────────────

def _terminal_thread_ids(now: int) -> tuple[set[str], set[str]]:
    """
    Uygulama DB'sinden thread hedeflerini çözer:
      - all_ids: mevcut TÜM project id'leri (orphan tespiti için)
      - terminal_old: completed/failed + retention'dan eski project id'leri
    (thread_id = str(project_id))
    """
    days = settings.CHECKPOINT_RETENTION_DAYS
    cutoff = now - max(days, 0) * 86_400
    from api.models.project import ProjectModel
    db = SessionLocal()
    try:
        all_ids = {str(pid) for (pid,) in db.query(ProjectModel.id).all()}
        terminal_old = {
            str(pid)
            for (pid,) in db.query(ProjectModel.id)
            .filter(
                ProjectModel.status.in_(["completed", "failed"]),
                ProjectModel.updated_at < cutoff,
            )
            .all()
        }
        return all_ids, terminal_old
    finally:
        db.close()


def _prune_checkpoints(now: int, is_postgres: bool) -> int:
    if is_postgres:
        return _prune_checkpoints_postgres(now)
    return _prune_checkpoints_sqlite(now)


def _prune_checkpoints_postgres(now: int) -> int:
    """
    Postgres: checkpoint tabloları uygulama DB'siyle aynı bağlantıda (search_path).
    Silme hedefi (thread bazlı):
      - orphan:  thread_id, projects.id kümesinde YOK
      - terminal-old: thread_id, completed/failed & updated_at < cutoff projeye ait
    """
    days = settings.CHECKPOINT_RETENTION_DAYS
    cutoff = now - max(days, 0) * 86_400
    total = 0
    with engine.begin() as conn:
        for table in _CHECKPOINT_TABLES:
            # Tablo yoksa (henüz hiç run olmadıysa) atla
            exists = conn.execute(text("SELECT to_regclass(:t)"), {"t": table}).scalar()
            if not exists:
                continue
            res = conn.execute(
                text(
                    f"""
                    DELETE FROM {table}
                    WHERE thread_id NOT IN (SELECT CAST(id AS TEXT) FROM projects)
                       OR thread_id IN (
                            SELECT CAST(id AS TEXT) FROM projects
                            WHERE status IN ('completed','failed') AND updated_at < :cutoff
                       )
                    """
                ),
                {"cutoff": cutoff},
            )
            total += res.rowcount or 0
    logger.info(f"[maintenance] checkpoints (pg): {total} satır silindi (terminal>{days}g + orphan).")
    return total


def _prune_checkpoints_sqlite(now: int) -> int:
    """
    SQLite: checkpoint'ler ayrı dosyada (data/checkpoints.sqlite). Hedef thread'leri
    uygulama DB'sinden çözüp bu dosyada thread_id bazlı sileriz.
    """
    if not os.path.exists(_SQLITE_CHECKPOINT_PATH):
        return 0
    all_ids, terminal_old = _terminal_thread_ids(now)

    conn = sqlite3.connect(_SQLITE_CHECKPOINT_PATH)
    total = 0
    try:
        cur = conn.cursor()
        # Bu sqlite dosyasındaki mevcut checkpoint tablolarını bul
        tables = {
            r[0] for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for table in _CHECKPOINT_TABLES:
            if table not in tables:
                continue
            # o tablodaki thread'ler
            thread_ids = {r[0] for r in cur.execute(f"SELECT DISTINCT thread_id FROM {table}").fetchall()}
            to_delete = [tid for tid in thread_ids if tid not in all_ids or tid in terminal_old]
            for tid in to_delete:
                cur.execute(f"DELETE FROM {table} WHERE thread_id = ?", (tid,))
                total += cur.rowcount or 0
        conn.commit()
    finally:
        conn.close()
    logger.info(f"[maintenance] checkpoints (sqlite): {total} satır silindi.")
    return total


# ── 5. Katman 2 monitoring aggregator (metrik iskeleti) ────────────────────────

def _run_monitoring_aggregate() -> int:
    """Monitoring özet iskeletini koşturur (pencereleme + champion-bölme + idempotanslık).
    run_maintenance'in per-subtask try/except'i altında → patlarsa diğer budamalar sürer."""
    from src.services.monitoring.aggregator import run_monitoring_aggregation
    return run_monitoring_aggregation()


# ── 4. db_intelligence_cache orphans ───────────────────────────────────────────

def _prune_intelligence_cache_orphans() -> int:
    """Projesi kalmamış cache satırları (FK cascade genelde önler; savunmacı)."""
    from api.models.db_intelligence_cache import DbIntelligenceCacheModel
    from api.models.project import ProjectModel
    db = SessionLocal()
    try:
        subq = db.query(ProjectModel.id)
        deleted = (
            db.query(DbIntelligenceCacheModel)
            .filter(~DbIntelligenceCacheModel.project_id.in_(subq))
            .delete(synchronize_session=False)
        )
        db.commit()
        if deleted:
            logger.info(f"[maintenance] db_intelligence_cache: {deleted} orphan satır silindi.")
        return int(deleted or 0)
    finally:
        db.close()
