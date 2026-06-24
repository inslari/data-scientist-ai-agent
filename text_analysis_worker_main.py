"""
Text-Analysis Worker (ayrı süreç)
=================================
Yazılı geri bildirim analizi işleri (task_type="run_text_analysis") için
TAMAMEN AYRI bir worker sürecidir. ML proje worker'ından (worker_main.py)
bağımsızdır: kendi poll döngüsü, kendi crash-recovery'si vardır ve yalnızca
kendi lane'indeki işleri alır. İki worker aynı background_jobs tablosunu
paylaşır ama birbirlerinin işlerine asla dokunmaz.

Çalıştırma (kendi terminalinde):
    uv run python text_analysis_worker_main.py

Böylece uzun süren bir metin analizi, ağır ML projelerini bekletmez (ve tersi).
"""
import asyncio

from dotenv import load_dotenv

# Load env before importing DB models
load_dotenv(override=True)

from api.core.database import Base, engine
# Import all models so SQLAlchemy initializes mappers properly
from api.models.job import BackgroundJobModel  # noqa: F401
from api.models.project import ProjectModel  # noqa: F401
from api.models.csv import CSVFileModel  # noqa: F401
from api.models.data_group import DataGroupModel  # noqa: F401
from api.models.user import UserModel  # noqa: F401
from api.models.mcp import MCPConfigModel  # noqa: F401
from api.models.db_intelligence_cache import DbIntelligenceCacheModel  # noqa: F401
from api.models.log_entry import LogEntryModel  # noqa: F401
from api.models.text_analysis import TextAnalysisModel  # noqa: F401
from api.models.text_cache import TextAnalysisCacheModel  # noqa: F401
from api.core.text_analysis_worker import job_worker_loop, recover_interrupted_jobs, TEXT_ANALYSIS_TASK_TYPES


def main():
    from src.core.logging_config import setup_logging
    # Ayrı süreç → kendi log etiketi. Console + DB logging açık.
    setup_logging(process="text_worker", db_logging=True)

    print("==================================================")
    print("   Data Scientist AI Agent - Text Analysis Worker  ")
    print("   (lane: run_text_analysis — ML projeden ayrı)    ")
    print("==================================================")

    # Tablolar mevcut değilse oluştur (text_analyses, text_analysis_cache dahil)
    Base.metadata.create_all(bind=engine)

    from api.core.text_analysis_migrate import ensure_text_analysis_schema
    ensure_text_analysis_schema()

    # Sadece KENDİ lane'indeki (text analizi) işleri kurtar.
    recovered = recover_interrupted_jobs(task_types=TEXT_ANALYSIS_TASK_TYPES)
    print(f"[Text Worker] Recovered {recovered} text-analysis jobs.")

    print("[Text Worker] Starting job loop (text-analysis lane)...")
    try:
        import sys
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(job_worker_loop(task_types=TEXT_ANALYSIS_TASK_TYPES))
    except KeyboardInterrupt:
        print("\n[Text Worker] Shutting down cleanly due to KeyboardInterrupt.")


if __name__ == "__main__":
    main()
