from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.core.database import Base, engine
from api.models.mcp import MCPConfigModel
from api.models.csv import CSVFileModel
from api.models.data_group import DataGroupModel
from api.models.project import ProjectModel
from api.models.user import UserModel
from api.models.job import BackgroundJobModel  # Must be imported so the table is created

# Create all DB tables
Base.metadata.create_all(bind=engine)

from api.routes.mcp import router as mcp_router
from api.routes.csv import router as csv_router
from api.routes.data_group import router as data_group_router
from api.routes.project import router as project_router
from api.routes.auth import router as auth_router

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
app.include_router(project_router)
app.include_router(mcp_router)
app.include_router(csv_router)
app.include_router(auth_router)


@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "ok", "message": "API is running successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

