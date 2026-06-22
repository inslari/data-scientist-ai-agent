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
from api.models.text_analysis import TextAnalysisModel  # text_analyses table — create_all
from api.models.text_cache import TextAnalysisCacheModel  # text_analysis_cache table — create_all
from api.core.worker import job_worker_loop, recover_interrupted_jobs, PROJECT_TASK_TYPES


def main():
    from src.core.logging_config import setup_logging
    # Worker'da console + DB logging açık (process="worker" etiketiyle app_logs'a yazar).
    setup_logging(process="worker", db_logging=True)

    print("==============================================")
    print("   Data Scientist AI Agent - Worker Process   ")
    print("   (lane: ML projects — text analysis ayrı)   ")
    print("==============================================")

    # Ensure tables exist
    Base.metadata.create_all(bind=engine)

    # Mevcut DB'lerde eksik kolonları idempotent ekle (background_jobs.text_analysis_id)
    from api.core.text_analysis_migrate import ensure_text_analysis_schema
    ensure_text_analysis_schema()

    # Sadece KENDİ lane'indeki (proje) işleri kurtar — text analizi işlerine dokunma.
    recovered = recover_interrupted_jobs(task_types=PROJECT_TASK_TYPES)
    print(f"[Worker Process] Recovered {recovered} project jobs.")

    print("[Worker Process] Starting job loop (project lane)...")
    try:
        import sys
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(job_worker_loop(task_types=PROJECT_TASK_TYPES))
    except KeyboardInterrupt:
        print("\n[Worker Process] Shutting down cleanly due to KeyboardInterrupt.")

if __name__ == "__main__":
    main()
