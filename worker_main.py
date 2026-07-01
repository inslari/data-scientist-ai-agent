import asyncio
import time
from dotenv import load_dotenv

# Load env before importing DB models
load_dotenv(override=True)

from api.core.database import SessionLocal, Base, engine
from api.models.job import BackgroundJobModel
from api.models.project import ProjectModel
# Important: import all models so SQLAlchemy can initialize mappers properly
from api.models.csv import CSVFileModel
from api.models.data_group import DataGroupModel
from api.models.user import UserModel
from api.models.mcp import MCPConfigModel
from api.models.db_intelligence_cache import DbIntelligenceCacheModel  # table auto-created via create_all
from api.models.log_entry import LogEntryModel  # app_logs table — DB logging hedefi
from api.core.worker import job_worker_loop, maintenance_loop

def recover_interrupted_jobs():
    """
    On startup:
      Re-queue any jobs that were left in 'running' or 'pending' state from
      a previous worker crash (inbox recovery). We reset their status to
      'pending' so the worker loop picks them up cleanly.
    """
    print("[Worker Process] Running startup job recovery...")
    db = SessionLocal()
    try:
        # Recover interrupted jobs: set 'running' → 'pending' so the worker retries them
        stuck_jobs = (
            db.query(BackgroundJobModel)
            .filter(BackgroundJobModel.status.in_(["running", "pending"]))
            .all()
        )
        for job in stuck_jobs:
            print(f"[Worker Process] Re-queuing interrupted job id={job.id} for project_id={job.project_id}")
            job.status = "pending"
            job.updated_at = int(time.time())

        # Also make sure corresponding projects show 'queued' instead of 'running'
        if stuck_jobs:
            project_ids = [j.project_id for j in stuck_jobs if j.project_id]
            db.query(ProjectModel).filter(ProjectModel.id.in_(project_ids)).update(
                {"status": "queued"}, synchronize_session=False
            )

        db.commit()
        print(f"[Worker Process] Recovered {len(stuck_jobs)} jobs.")
    except Exception as e:
        print(f"[Worker Process] Error recovering jobs: {e}")
    finally:
        db.close()

def main():
    from src.core.logging_config import setup_logging
    # Worker'da console + DB logging açık (process="worker" etiketiyle app_logs'a yazar).
    setup_logging(process="worker", db_logging=True)

    print("==============================================")
    print("   Data Scientist AI Agent - Worker Process   ")
    print("==============================================")

    # Ensure tables exist
    Base.metadata.create_all(bind=engine)
    
    # Clean up any stuck jobs
    recover_interrupted_jobs()
    
    print("[Worker Process] Starting job loop...")
    try:
        import sys
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(_run_loops())
    except KeyboardInterrupt:
        print("\n[Worker Process] Shutting down cleanly due to KeyboardInterrupt.")


async def _run_loops():
    """
    Job işleme döngüsü + DB bakım döngüsünü aynı event loop'ta yan yana çalıştırır.
    maintenance_loop ağır işleri ayrı thread'e attığı için job polling'i bloklamaz.
    Bakım görevi çökse bile job döngüsü ayakta kalır (return_exceptions).
    """
    await asyncio.gather(
        job_worker_loop(),
        maintenance_loop(),
        return_exceptions=True,
    )

if __name__ == "__main__":
    main()
