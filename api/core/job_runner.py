"""
Job Runner — nötr kuyruk altyapısı (domain-agnostik)
=====================================================
`background_jobs` tablosunu yoklayıp işleri çalıştıran GENEL polling altyapısı.
Hiçbir iş tipine (ML / text-analysis / TTD) bağlı DEĞİLDİR; her worker kendi
`dispatch` (iş çalıştırıcı) ve opsiyonel `reset_owner` (kurtarmada sahip kaydı
sıfırlama) callback'ini geçirir.

Böylece ML worker (`api/core/worker.py`) ve text-analysis worker
(`api/core/text_analysis_worker.py`) tamamen ayrışır: aynı kuyruk tablosunu
paylaşırlar ama birbirlerinin kodunu/işlerini asla görmezler. Bu modül yalnızca
"sıradaki işi al, çalıştır, durumu yaz" mekaniğini sağlar.
"""
import asyncio
import time
import traceback
from typing import Awaitable, Callable, Optional

from sqlalchemy.orm import Session

from api.core.database import SessionLocal, SQLALCHEMY_DATABASE_URL
from api.models.job import BackgroundJobModel

from src.core.logging_config import get_logger
logger = get_logger(__name__)

# Yapacak iş yokken poll döngüleri arası uyku süresi.
POLL_INTERVAL_SECONDS = 5

# İş çalıştırıcı imzası: bir job alır, çalıştırır (durumu kendi yazar/işaretler).
Dispatch = Callable[[BackgroundJobModel], Awaitable[None]]
# Kurtarma callback'i: (db, takılı_job_listesi) → sahip kayıtları 'queued' yapar
# (commit ETMEZ; dış fonksiyon commit eder).
ResetOwner = Callable[[Session, list[BackgroundJobModel]], None]


def mark_job(job_id: int, status: str, error: Optional[str] = None) -> None:
    """İş satırını nihai durumla günceller."""
    db = SessionLocal()
    try:
        job = db.query(BackgroundJobModel).filter(BackgroundJobModel.id == job_id).first()
        if job:
            job.status = status
            job.error = error
            job.updated_at = int(time.time())
            db.commit()
    finally:
        db.close()


async def run_job_loop(task_types: list[str], dispatch: Dispatch) -> None:
    """
    `task_types` lane'indeki pending işleri sürekli yoklayıp `dispatch` ile
    çalıştırır. Tek uzun-ömürlü asyncio.Task olarak çalışacak şekilde tasarlandı.
    """
    lane = "+".join(task_types) if task_types else "all"
    logger.info(f"[JobRunner] Job loop started (lane={lane}).")
    while True:
        try:
            await _process_next_job(task_types, dispatch)
        except asyncio.CancelledError:
            logger.info("[JobRunner] Loop cancelled — shutting down cleanly.")
            break
        except Exception as exc:
            logger.error(f"[JobRunner] Unexpected error in loop: {exc}")
            traceback.print_exc()
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def _process_next_job(task_types: list[str], dispatch: Dispatch) -> None:
    """En eski pending işi (lane'e kısıtlı) alır, 'running' yapar ve çalıştırır."""
    is_postgres = SQLALCHEMY_DATABASE_URL.startswith("postgresql")

    db = SessionLocal()
    try:
        q = db.query(BackgroundJobModel).filter(BackgroundJobModel.status == "pending")
        if task_types:
            q = q.filter(BackgroundJobModel.task_type.in_(task_types))
        q = q.order_by(BackgroundJobModel.created_at.asc())
        # Postgres'te eşzamanlı worker çakışmasını önlemek için satır kilidi.
        if is_postgres:
            q = q.with_for_update(skip_locked=True)

        job: BackgroundJobModel | None = q.first()
        if job is None:
            return  # Yapacak iş yok; çağıran uyuyacak.

        logger.info(f"[JobRunner] Picked job id={job.id} (task_type={job.task_type}, "
                    f"project_id={job.project_id}, text_analysis_id={job.text_analysis_id})")
        job.status = "running"
        job.updated_at = int(time.time())
        db.commit()
        db.refresh(job)
    finally:
        db.close()

    # Polling oturumu dışında çalıştır (DB açık tutulmasın).
    if job is not None:
        await dispatch(job)


def recover_jobs(task_types: list[str], reset_owner: Optional[ResetOwner] = None) -> int:
    """
    Başlangıç kurtarması: önceki çökmeden 'running'/'pending' kalmış işleri
    (yalnızca verilen lane) yeniden 'pending' yapar. `reset_owner` verilirse
    sahip kayıtları (Project/TextAnalysis) 'queued'a çeker. Her worker bunu
    YALNIZCA kendi lane'i için çağırır → birbirlerinin işlerini sıfırlamazlar.

    Returns: kurtarılan iş sayısı.
    """
    db = SessionLocal()
    try:
        q = db.query(BackgroundJobModel).filter(
            BackgroundJobModel.status.in_(["running", "pending"])
        )
        if task_types:
            q = q.filter(BackgroundJobModel.task_type.in_(task_types))
        stuck = q.all()

        for job in stuck:
            job.status = "pending"
            job.updated_at = int(time.time())

        if reset_owner and stuck:
            reset_owner(db, stuck)

        db.commit()
        return len(stuck)
    except Exception as e:
        db.rollback()
        logger.error(f"[JobRunner] Job recovery failed: {e}")
        return 0
    finally:
        db.close()
