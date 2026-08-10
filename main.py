from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.core.database import Base, engine
from api.models.mcp import MCPConfigModel
from api.models.csv import CSVFileModel
from api.models.data_group import DataGroupModel
from api.models.project import ProjectModel
from api.models.user import UserModel
from api.models.job import BackgroundJobModel  # Must be imported so the table is created
from api.models.data_group_query import DataGroupQueryModel  # Must be imported so the table is created
from api.models.db_intelligence_cache import DbIntelligenceCacheModel  # Must be imported so the table is created
from api.models.log_entry import LogEntryModel  # Must be imported so the app_logs table is created
from api.models.llm_usage import LLMUsageModel  # Must be imported so the llm_usage table is created (C3)
from api.models.ai_provider import AiProviderModel  # DB-managed LLM provider config tables
from api.models.ai_model import AiModelModel
from api.models.ai_role_assignment import AiRoleAssignmentModel
from api.models.password_reset_request import PendingPasswordResetModel  # mail-free reset requests
from api.models.artifact import ArtifactModel  # Katman 1 inference artifact index (serving)
from api.models.artifact_event import ArtifactEventModel  # Manuel promotion olay günlüğü
from api.models.prediction import PredictionModel  # Katman 2 temeli — skorlama kayıt tablosu
from api.models.monitoring_summary import MonitoringSummaryModel  # Katman 2 — pencereli monitoring özeti

# Yapısal logging — süreç başında bir kez. API'de DB logging varsayılan kapalı
# (LOG_DB_ENABLED=true ile açılabilir); tablo yine create_all ile oluşur.
from src.core.logging_config import setup_logging
setup_logging(process="api")

# Create all DB tables
Base.metadata.create_all(bind=engine)

# ŞEMA DRIFT KAPISI (create_all SONRASI): create_all mevcut tabloya kolon EKLEMEZ → modele kolon
# eklenip migration yazılmazsa DB geride kalır ve o tabloyu SELECT eden endpoint SESSİZCE 500'ler
# (monitoring detay prod'da tam böyle kırıldı). Fail-closed: eksik kolon → başlatma durur, migrate_db'ye
# yönlendirir. extra_in_db yalnız uyarı. Bypass: ALLOW_SCHEMA_DRIFT=1.
from api.core.db_safety import assert_schema_current
from src.core.logging_config import get_logger as _get_logger
assert_schema_current(engine, Base.metadata, logger=_get_logger("schema_drift"))

from api.routes.mcp import router as mcp_router
from api.routes.csv import router as csv_router
from api.routes.convert import router as convert_router
from api.routes.data_group import router as data_group_router
from api.routes.data_viz import router as data_viz_router, themes_router as viz_themes_router
from api.routes.project import router as project_router
from api.routes.scoring import router as scoring_router
from api.routes.artifacts import router as artifacts_router
from api.routes.labels import router as labels_router
from api.routes.monitoring import router as monitoring_router, portfolio_router as monitoring_portfolio_router
from api.routes.auth import router as auth_router
from api.routes.domain_pack import router as domain_pack_router
from api.routes.data_group_query import router as data_group_query_router
from api.routes.admin_ai import router as admin_ai_router
from api.routes.admin_users import router as admin_users_router, reset_router as admin_password_resets_router

from contextlib import asynccontextmanager
import asyncio
import time
from api.core.database import SessionLocal
from api.core.worker import job_worker_loop


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    """
    # Previously, we ran the worker and job recovery here. 
    # Now, the background worker is completely decoupled and runs via worker_main.py.
    yield  # Application is running


from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from api.core.limiter import limiter

app = FastAPI(
    title="Data Scientist AI Agent API",
    description="API for the Data Scientist AI Agent",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiter middleware
app.state.limiter = limiter
from slowapi import _rate_limit_exceeded_handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore
app.add_middleware(SlowAPIMiddleware)

app.include_router(data_group_router)
app.include_router(data_viz_router)
app.include_router(viz_themes_router)
app.include_router(data_group_query_router)
app.include_router(project_router)
app.include_router(scoring_router)
app.include_router(artifacts_router)
app.include_router(labels_router)
app.include_router(monitoring_router)
app.include_router(monitoring_portfolio_router)   # cross-project (prefix="/monitoring", AYRI router)
app.include_router(mcp_router)
app.include_router(csv_router)
app.include_router(convert_router)
app.include_router(auth_router)
app.include_router(domain_pack_router)
app.include_router(admin_ai_router)
app.include_router(admin_users_router)
app.include_router(admin_password_resets_router)


@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "ok", "message": "API is running successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

