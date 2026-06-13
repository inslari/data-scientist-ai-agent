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

    On startup:
      1. Re-queue any jobs that were left in 'running' or 'pending' state from
         a previous server crash (inbox recovery).  We reset their status to
         'pending' so the worker loop picks them up cleanly.
      2. Start the background worker loop as a long-lived asyncio.Task.

    On shutdown:
      Cancel the worker task gracefully.
    """
    db = SessionLocal()
    try:
        # Recover interrupted jobs: set 'running' → 'pending' so the worker retries them
        stuck_jobs = (
            db.query(BackgroundJobModel)
            .filter(BackgroundJobModel.status.in_(["running", "pending"]))
            .all()
        )
        for job in stuck_jobs:
            print(f"[Startup] Re-queuing interrupted job id={job.id} for project_id={job.project_id}")
            job.status = "pending"
            job.updated_at = int(time.time())

        # Also make sure corresponding projects show 'queued' instead of 'running'
        if stuck_jobs:
            project_ids = [j.project_id for j in stuck_jobs if j.project_id]
            db.query(ProjectModel).filter(ProjectModel.id.in_(project_ids)).update(
                {"status": "queued"}, synchronize_session=False
            )

        db.commit()
    except Exception as e:
        print(f"[Startup] Error recovering jobs: {e}")
    finally:
        db.close()

    # Start the background worker as a persistent asyncio task
    worker_task = asyncio.create_task(job_worker_loop())
    print("[Startup] Background job worker task started.")

    yield  # Application is running

    # Graceful shutdown
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    print("[Shutdown] Background job worker stopped.")


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

